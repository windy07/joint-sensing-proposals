"""Exactly-once V73 operational cycle over the existing posterior authority.

The runtime is dependency-injected so the authoritative DynamicFactorGraph/JPDA,
physical counterfactual evaluator, availability boundary, safety shield, and
world object remain the ones already used by the project.  This module does not
instantiate or maintain a second posterior.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import copy, json, random
import numpy as np

from .v73_adaptive_belief_risk import (
    AdaptiveMissionConstrainedController, CandidateCertificate,
    ControllerVisibleState, AdaptiveDecision,
)

class RuntimeAuthorityError(RuntimeError): pass


def canonical_digest(obj:Any)->str:
    """Stable digest for authority/mutation checks; adapters may supply stronger digests."""
    def norm(x):
        # Enums expose class metadata in __dict__; walking that metadata can
        # recurse through the enum class instead of hashing its state value.
        if isinstance(x,Enum):
            return norm(x.value)
        # repr(ndarray) can truncate its middle and depends on print options.
        # A mutation fingerprint must cover every element and its interpretation.
        if isinstance(x,np.ndarray):
            if x.dtype.hasobject:
                return {"__ndarray__":True,"dtype":x.dtype.str,"shape":list(x.shape),
                        "values":norm(x.tolist())}
            return {"__ndarray__":True,"dtype":x.dtype.str,"shape":list(x.shape),
                    "bytes":x.tobytes(order="C").hex()}
        if isinstance(x,np.generic):
            return norm(x.item())
        if hasattr(x,"to_controller_digest"):
            return x.to_controller_digest()
        if hasattr(x,"state_dict") and callable(x.state_dict):
            return norm(x.state_dict())
        if hasattr(x,"__dict__"):
            return {k:norm(v) for k,v in sorted(vars(x).items()) if not k.startswith('_cache')}
        if isinstance(x,Mapping): return {str(k):norm(v) for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))}
        if isinstance(x,(list,tuple)): return [norm(v) for v in x]
        if isinstance(x,set): return sorted(norm(v) for v in x)
        if isinstance(x,float): return format(x,'.17g')
        if isinstance(x,(str,int,bool)) or x is None: return x
        return repr(x)
    raw=json.dumps(norm(obj),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return sha256(raw).hexdigest()


def derive_crn_seed(master_seed:int, cycle_index:int, replicate_index:int)->int:
    """Paired-CRN seed independent of candidate identity and enumeration order."""
    raw=f"v73-crn|{int(master_seed)}|{int(cycle_index)}|{int(replicate_index)}".encode()
    return int.from_bytes(sha256(raw).digest()[:8],'big') & ((1<<63)-1)


@dataclass(frozen=True)
class AvailabilityAudit:
    action_id: str
    arrived_information_only: bool
    physically_feasible: bool
    budget_feasible: bool
    spectrum_feasible: bool
    transport_feasible: bool
    emission_feasible: bool
    safety_precheck: bool
    reasons: Tuple[str,...] = ()
    digest: str = ""

    @property
    def feasible(self)->bool:
        return all((self.arrived_information_only,self.physically_feasible,self.budget_feasible,
                    self.spectrum_feasible,self.transport_feasible,self.emission_feasible,self.safety_precheck))


@dataclass(frozen=True)
class CycleTrace:
    cycle_index: int
    arrived_factor_count: int
    posterior_before_digest: str
    posterior_after_counterfactual_digest: str
    availability_digest: str
    candidate_ids: Tuple[str,...]
    crn_seeds_digest: str
    selected_action_id: str
    no_op_reason: Optional[str]
    action_commit_count: int
    world_transition_count: int
    dual_before: float
    dual_after: float
    adaptive_lambda: float
    safety_veto: bool
    trace_digest: str


class V73OperationalCycle:
    """Run one authoritative solve/evaluate/decide/commit/transition cycle."""
    def __init__(self, engine:Any, controller:AdaptiveMissionConstrainedController,
                 master_seed:int, counterfactual_replicates:int=64):
        if counterfactual_replicates<1: raise ValueError("counterfactual_replicates must be positive")
        self.engine=engine; self.controller=controller; self.master_seed=int(master_seed)
        self.counterfactual_replicates=int(counterfactual_replicates)

    def run(self, *, cycle_index:int, decision_time_s:float,
            admit_arrived_factors:Callable[[float],int],
            solve_sole_belief:Callable[[],Any],
            availability_state:Any,
            generate_candidates:Callable[[Any,Any],Sequence[Any]],
            candidate_id:Callable[[Any],str],
            audit_candidate:Callable[[Any,Any,Any],AvailabilityAudit],
            fork_belief:Callable[[Any],Any],
            evaluate_counterfactual:Callable[[Any,Any,Sequence[int],AvailabilityAudit],CandidateCertificate],
            build_controller_state:Callable[[Any,Any,Sequence[CandidateCertificate],str,str],ControllerVisibleState],
            safety_gate:Callable[[str,AdaptiveDecision],bool],
            commit_action:Callable[[str],None],
            advance_world_once:Callable[[],None],
            posterior_digest:Callable[[Any],str]=canonical_digest,
            availability_digest:Callable[[Any],str]=canonical_digest)->CycleTrace:
        arrived=int(admit_arrived_factors(float(decision_time_s)))
        if arrived<0: raise RuntimeAuthorityError("negative admitted-factor count")
        belief=solve_sole_belief()
        if belief is None: raise RuntimeAuthorityError("sole belief solver returned None")
        before=posterior_digest(belief); avail_digest=availability_digest(availability_state)
        candidates=list(generate_candidates(belief,availability_state))
        if not candidates: raise RuntimeAuthorityError("empty candidate set")
        ids=[str(candidate_id(c)) for c in candidates]
        if len(ids)!=len(set(ids)): raise RuntimeAuthorityError("duplicate candidate IDs")
        crn=[derive_crn_seed(self.master_seed,cycle_index,r) for r in range(self.counterfactual_replicates)]
        certs=[]; audits=[]
        for c,aid in zip(candidates,ids):
            audit=audit_candidate(c,belief,availability_state)
            if audit.action_id!=aid: raise RuntimeAuthorityError("candidate/audit action ID mismatch")
            fork=fork_belief(belief)
            if fork is belief: raise RuntimeAuthorityError("counterfactual fork aliases live belief")
            if posterior_digest(fork)!=before:
                raise RuntimeAuthorityError(f"counterfactual fork for {aid} does not start at the live posterior")
            cert=evaluate_counterfactual(fork,c,crn,audit)
            if cert.action_id!=aid: raise RuntimeAuthorityError("certificate action ID mismatch")
            if cert.feasible!=audit.feasible:
                # Feasibility is owned by the availability/physics audit, not a learned scorer.
                cert=CandidateCertificate(action_id=cert.action_id,feasible=audit.feasible,is_noop=cert.is_noop,
                    info_utility=cert.info_utility,set_risk_improvement=cert.set_risk_improvement,
                    set_risk_se=cert.set_risk_se,mission_loss_delta=cert.mission_loss_delta,
                    mission_loss_se=cert.mission_loss_se,resource_cost_normalized=cert.resource_cost_normalized,
                    emission_cost_normalized=cert.emission_cost_normalized,
                    physical_trace_digest=cert.physical_trace_digest,
                    infeasible_reasons=tuple(audit.reasons),metadata=cert.metadata)
            certs.append(cert); audits.append(audit)
            if posterior_digest(belief)!=before:
                raise RuntimeAuthorityError(f"live belief mutated while scoring {aid}")
        after_cf=posterior_digest(belief)
        if before!=after_cf: raise RuntimeAuthorityError("live belief changed during counterfactual evaluation")
        noops=[c for c in certs if c.is_noop]
        if len(noops)!=1: raise RuntimeAuthorityError("candidate universe must contain exactly one explicit no-op")
        state=build_controller_state(belief,availability_state,certs,before,avail_digest)
        if state.posterior_digest!=before or state.availability_digest!=avail_digest:
            raise RuntimeAuthorityError("controller view authority digest mismatch")
        decision=self.engine.v73_adaptive_decision(state,certs,controller=self.controller,commit_dual=False)
        veto=not bool(safety_gate(decision.selected_action_id,decision))
        if veto:
            noop=noops[0]
            decision=self.engine.v73_adaptive_decision(state,[noop],controller=self.controller,commit_dual=False)
            # The safety authority sees the actual fallback as a second decision.
            # A no-op emits no sensing command, so a second veto is recorded but
            # cannot be converted into an unreviewed non-noop action.
            safety_gate(decision.selected_action_id,decision)
        commits=0; transitions=0
        commit_action(decision.selected_action_id); commits+=1
        if commits!=1: raise RuntimeAuthorityError("action must be committed exactly once")
        advance_world_once(); transitions+=1
        if transitions!=1: raise RuntimeAuthorityError("world must advance exactly once")
        dual_before=self.controller.dual.value
        dual_after=self.controller.commit_certificate(decision,certs if not veto else [noops[0]])
        payload={"cycle_index":cycle_index,"arrived":arrived,"before":before,"after_cf":after_cf,
                 "availability":avail_digest,"ids":ids,"selected":decision.selected_action_id,
                 "commits":commits,"transitions":transitions,"dual_before":dual_before,
                 "dual_after":dual_after,"lambda":decision.lambda_t,"veto":veto}
        td=sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return CycleTrace(cycle_index,arrived,before,after_cf,avail_digest,tuple(ids),
            sha256(json.dumps(crn,separators=(',',':')).encode()).hexdigest(),
            decision.selected_action_id,decision.no_op_reason,commits,transitions,
            dual_before,dual_after,decision.lambda_t,veto,td)
