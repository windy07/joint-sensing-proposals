"""Matched common-posterior contracts for the ten supplied paper baselines.

P01--P06 are explicitly component/resource baselines.  P07--P10 are complete
policy/search baselines.  No baseline is allowed to see truth, world-family
labels, seed identity, future observations, or confirmation metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from hashlib import sha256
from math import isfinite, log
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple
from pathlib import Path
import json, random

from .v73_adaptive_belief_risk import reject_forbidden_features


@dataclass(frozen=True)
class MatchedAction:
    action_id: str
    feasible: bool
    resource_cost: float
    emission_cost: float
    transport_cost: float
    metadata: Mapping[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        reject_forbidden_features(self.metadata)
        if min(self.resource_cost,self.emission_cost,self.transport_cost)<0: raise ValueError("negative cost")


@dataclass(frozen=True)
class BaselineBudget:
    resource_limit: float
    emission_limit: float
    transport_limit: float
    compute_evaluations: int
    horizon: int
    def __post_init__(self):
        if min(self.resource_limit,self.emission_limit,self.transport_limit)<0: raise ValueError("negative budget")
        if self.compute_evaluations<1 or self.horizon<1: raise ValueError("invalid compute/horizon budget")


@dataclass(frozen=True)
class BaselineContext:
    posterior_digest: str
    availability_digest: str
    controller_view: Mapping[str,Any]
    actions: Tuple[MatchedAction,...]
    budget: BaselineBudget
    decision_time_s: float
    horizon: int
    common_random_numbers: Tuple[int,...]
    evaluators: Mapping[str,Callable[...,float]]=field(default_factory=dict,compare=False,repr=False)
    def __post_init__(self):
        reject_forbidden_features(self.controller_view)
        if not self.posterior_digest or not self.availability_digest: raise ValueError("authority digests required")
        if not self.actions: raise ValueError("empty matched action universe")
        if self.horizon!=self.budget.horizon: raise ValueError("horizon mismatch")
    @property
    def context_digest(self)->str:
        payload={"posterior":self.posterior_digest,"availability":self.availability_digest,
                 "actions":[asdict(a) for a in self.actions],"budget":asdict(self.budget),
                 "time":self.decision_time_s,"horizon":self.horizon}
        return sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


@dataclass(frozen=True)
class BaselineDecision:
    paper_id: str
    variant: str
    classification: str
    selected_action_id: str
    feasible: bool
    evaluations_used: int
    context_digest: str
    budget_digest: str
    trace: Mapping[str,Any]


class MatchedBaselinePolicy(Protocol):
    paper_id: str
    variant: str
    classification: str
    def decide(self,ctx:BaselineContext)->BaselineDecision: ...


def _budget_digest(b:BaselineBudget)->str:
    return sha256(json.dumps(asdict(b),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _feasible(a:MatchedAction,b:BaselineBudget)->bool:
    return bool(a.feasible and a.resource_cost<=b.resource_limit and a.emission_cost<=b.emission_limit and a.transport_cost<=b.transport_limit)

def _score(a:MatchedAction,key:str,default:float=0.0)->float:
    v=float(a.metadata.get(key,default))
    if not isfinite(v): raise ValueError(f"nonfinite {key} for {a.action_id}")
    return v

def _decision(policy,ctx,a,evals,trace):
    return BaselineDecision(policy.paper_id,policy.variant,policy.classification,a.action_id,_feasible(a,ctx.budget),evals,
                            ctx.context_digest,_budget_digest(ctx.budget),trace)

def _argmax_feasible(ctx,key,penalties=()):
    fs=[a for a in ctx.actions if _feasible(a,ctx.budget)]
    if not fs: raise RuntimeError("no feasible matched action")
    def s(a): return _score(a,key)-sum(w*_score(a,k) for k,w in penalties)
    return max(fs,key=lambda a:(s(a),-a.resource_cost,-a.emission_cost,a.action_id)),len(fs),s


class P01ADOptimalPlacement:
    paper_id='P01'; variant='common_posterior'; classification='component_geometry'
    def __init__(self,criterion='D'): self.criterion=criterion.upper()
    def decide(self,ctx):
        key={'D':'fim_logdet','A':'negative_crlb_trace'}.get(self.criterion,'fim_logdet')
        a,n,s=_argmax_feasible(ctx,key)
        return _decision(self,ctx,a,n,{"criterion":self.criterion,"matched_score":s(a),"scope":"component_not_full_policy"})

class P02CalibrationObject:
    paper_id='P02'; variant='common_posterior'; classification='component_calibration'
    def decide(self,ctx):
        a,n,s=_argmax_feasible(ctx,'schur_calibration_gain',(('calibration_cost',1.0),))
        return _decision(self,ctx,a,n,{"criterion":"nuisance_marginalized_information_gain","score":s(a),"scope":"component_not_full_policy"})

class P03HybridActivePassive:
    paper_id='P03'; variant='common_posterior'; classification='component_measurement_fusion'
    def decide(self,ctx):
        a,n,s=_argmax_feasible(ctx,'joint_active_passive_information',(('correlation_penalty',1.0),))
        return _decision(self,ctx,a,n,{"criterion":"joint_likelihood_information","score":s(a),"scope":"component_not_full_policy"})

class P04LPIPowerAllocation:
    paper_id='P04'; variant='common_posterior'; classification='resource_power_allocation'
    def decide(self,ctx):
        fs=[a for a in ctx.actions if _feasible(a,ctx.budget) and _score(a,'crlb_constraint_margin',0.0)>=0.0]
        if not fs: fs=[a for a in ctx.actions if _feasible(a,ctx.budget)]
        a=min(fs,key=lambda x:(_score(x,'intercept_probability',x.emission_cost),x.emission_cost,x.resource_cost,x.action_id))
        return _decision(self,ctx,a,len(fs),{"criterion":"minimum_LPI_exposure_subject_to_CRLB","crlb_margin":_score(a,'crlb_constraint_margin')})

class P05LPITargetAssignment:
    paper_id='P05'; classification='resource_target_assignment'
    def __init__(self,variant='exact_assignment'): self.variant=variant
    def decide(self,ctx):
        key='assignment_exact_utility' if self.variant=='exact_assignment' else 'assignment_neural_utility'
        a,n,s=_argmax_feasible(ctx,key,(('intercept_probability',1.0),))
        return _decision(self,ctx,a,n,{"criterion":key,"score":s(a),"neural_is_approximation":self.variant!='exact_assignment'})

class P06QoSDependencies:
    paper_id='P06'; variant='common_posterior'; classification='resource_task_graph'
    def decide(self,ctx):
        fs=[]
        for a in ctx.actions:
            if not _feasible(a,ctx.budget): continue
            if not bool(a.metadata.get('dependencies_satisfied',True)): continue
            if not bool(a.metadata.get('substitution_valid',True)): continue
            fs.append(a)
        if not fs: raise RuntimeError("no action satisfies QoS dependency contract")
        a=max(fs,key=lambda x:(_score(x,'qos_task_value')-_score(x,'qos_resource_penalty'),x.action_id))
        return _decision(self,ctx,a,len(fs),{"criterion":"task_dependency_QoS_value","scope":"resource_policy_on_declared_task_graph"})

class P07NonmyopicBernoulliGOSPA:
    paper_id='P07'; variant='common_posterior'; classification='policy_nonmyopic_set_risk'
    def decide(self,ctx):
        fs=[a for a in ctx.actions if _feasible(a,ctx.budget)]
        ev=ctx.evaluators.get('nonmyopic_gospa')
        vals=[]
        for a in fs[:ctx.budget.compute_evaluations]:
            v=float(ev(a,ctx) if ev else _score(a,'nonmyopic_gospa_risk',float('inf')))
            vals.append((v,a))
        if not vals: raise RuntimeError("P07 has no evaluable action")
        v,a=min(vals,key=lambda q:(q[0],q[1].action_id))
        return _decision(self,ctx,a,len(vals),{"criterion":"minimum_expected_horizon_GOSPA","risk":v,"horizon":ctx.horizon})

class P08VPWMCTS:
    paper_id='P08'; variant='common_posterior'; classification='policy_search'
    def decide(self,ctx):
        fs=[a for a in ctx.actions if _feasible(a,ctx.budget)]
        ev=ctx.evaluators.get('mcts_value')
        rng=random.Random(ctx.common_random_numbers[0] if ctx.common_random_numbers else 0)
        # Deterministic bounded progressive-widening surrogate over the common universe.
        ordered=sorted(fs,key=lambda a:sha256(f"{ctx.context_digest}|{a.action_id}".encode()).hexdigest())
        n=min(len(ordered),ctx.budget.compute_evaluations)
        vals=[]
        for a in ordered[:n]: vals.append((float(ev(a,ctx) if ev else _score(a,'vpw_value')) ,a))
        if not vals: raise RuntimeError("P08 has no evaluable action")
        v,a=max(vals,key=lambda q:(q[0],q[1].action_id))
        return _decision(self,ctx,a,n,{"criterion":"bounded_VPW_MCTS_value","value":v,"evaluations":n})

class P09DRLAdaptation:
    paper_id='P09'; classification='policy_learned_long_horizon'
    def __init__(self,checkpoint:Optional[str]=None,checkpoint_sha256:Optional[str]=None,
                 loader:Optional[Callable[[Path],Any]]=None):
        self.checkpoint=str(checkpoint) if checkpoint else None
        self.checkpoint_sha256=checkpoint_sha256
        self.model=None; self.checkpoint_error=None
        p=Path(self.checkpoint) if self.checkpoint else None
        if p is not None and p.is_file() and checkpoint_sha256 and callable(loader):
            actual=sha256(p.read_bytes()).hexdigest()
            if actual!=checkpoint_sha256:
                self.checkpoint_error='checkpoint_sha256_mismatch'
            else:
                try:self.model=loader(p)
                except Exception as e:self.checkpoint_error=f'checkpoint_loader_failed:{type(e).__name__}'
        elif self.checkpoint:
            self.checkpoint_error='checkpoint_missing_or_unverifiable'
        self.variant='faithful_checkpoint' if self.model is not None else 'common_posterior_no_verified_checkpoint'
    def decide(self,ctx):
        if self.model is not None and hasattr(self.model,'score_actions'):
            scores=self.model.score_actions(ctx)
            fs=[a for a in ctx.actions if _feasible(a,ctx.budget)]
            a=max(fs,key=lambda x:(float(scores[x.action_id]),x.action_id)); n=len(fs); score=float(scores[a.action_id])
            return _decision(self,ctx,a,n,{"criterion":"verified_checkpoint_q_value","score":score,
                "checkpoint_sha256":self.checkpoint_sha256,"checkpoint_loaded":True})
        key='drl_surrogate_q_value'; a,n,s=_argmax_feasible(ctx,key)
        return _decision(self,ctx,a,n,{"criterion":key,"checkpoint_available":False,
            "checkpoint_error":self.checkpoint_error,
            "fidelity_limit":"verified faithful checkpoint unavailable; common-posterior executable surrogate only"})

class P10RobustParticlePOMDP:
    paper_id='P10'; variant='common_posterior'; classification='policy_robust_pomdp'
    def decide(self,ctx):
        fs=[a for a in ctx.actions if _feasible(a,ctx.budget)]
        ev=ctx.evaluators.get('robust_pomdp_value')
        vals=[]
        for a in fs[:ctx.budget.compute_evaluations]:
            v=float(ev(a,ctx) if ev else _score(a,'robust_pomdp_value'))
            vals.append((v,a))
        if not vals: raise RuntimeError("P10 has no evaluable action")
        v,a=max(vals,key=lambda q:(q[0],q[1].action_id))
        return _decision(self,ctx,a,len(vals),{"criterion":"robust_particle_belief_POMDP_value","value":v,"horizon":ctx.horizon})


def default_ten_paper_policies()->Tuple[MatchedBaselinePolicy,...]:
    return (P01ADOptimalPlacement('D'),P02CalibrationObject(),P03HybridActivePassive(),P04LPIPowerAllocation(),
            P05LPITargetAssignment('exact_assignment'),P06QoSDependencies(),P07NonmyopicBernoulliGOSPA(),
            P08VPWMCTS(),P09DRLAdaptation(),P10RobustParticlePOMDP())


def run_matched_baseline_campaign(ctx:BaselineContext,policies:Optional[Sequence[MatchedBaselinePolicy]]=None):
    policies=tuple(policies or default_ten_paper_policies())
    out=[]
    for p in policies:
        d=p.decide(ctx)
        if d.context_digest!=ctx.context_digest: raise RuntimeError(f"{p.paper_id} context mismatch")
        if d.evaluations_used>ctx.budget.compute_evaluations: raise RuntimeError(f"{p.paper_id} exceeded compute budget")
        selected=next((a for a in ctx.actions if a.action_id==d.selected_action_id),None)
        if selected is None or not _feasible(selected,ctx.budget): raise RuntimeError(f"{p.paper_id} returned infeasible action")
        out.append(d)
    return tuple(out)
