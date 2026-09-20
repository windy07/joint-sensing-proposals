"""V73 adaptive, mission-constrained belief-risk decision policy.

This module is deliberately small and auditable.  It never owns a posterior and
never accepts simulator truth.  It consumes a read-only controller view derived
from the sole DynamicFactorGraph/JPDA posterior and candidate certificates
computed from frozen counterfactual forks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from hashlib import sha256
from math import exp, isfinite, log1p, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple
import json

FORBIDDEN_FEATURE_TOKENS = (
    "truth", "ground_truth", "world_family", "scenario_family", "family_label",
    "seed_identity", "seed_id", "future_observation", "future_truth",
    "realized_loss", "test_loss", "confirmation_metric", "oracle_action",
    "privileged", "simulator_state", "target_truth", "actual_state",
)

class ControllerFeatureError(ValueError):
    """Raised when a controller view is malformed or contains privileged data."""


def _canonical(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)): return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        if not isfinite(obj): raise ControllerFeatureError("non-finite controller feature")
        return float(obj)
    if isinstance(obj, (str, int, bool)) or obj is None: return obj
    raise ControllerFeatureError(f"unsupported controller feature type: {type(obj)!r}")


def reject_forbidden_features(features: Mapping[str, Any]) -> None:
    """Fail closed if any key could expose truth, family, seed, future, or test data."""
    stack=[features]
    while stack:
        cur=stack.pop()
        if isinstance(cur, Mapping):
            for k,v in cur.items():
                key=str(k).strip().lower()
                if any(tok in key for tok in FORBIDDEN_FEATURE_TOKENS):
                    raise ControllerFeatureError(f"forbidden controller feature: {k}")
                stack.append(v)
        elif isinstance(cur,(list,tuple)):
            stack.extend(cur)
        elif isinstance(cur,float) and not isfinite(cur):
            raise ControllerFeatureError("non-finite controller feature")


@dataclass(frozen=True)
class ControllerVisibleState:
    """Truth-isolated filtration-measurable summary z_t=phi(b_t,A_t,N_t).

    Dimensionless entries must be explicitly normalized before construction.
    Time quantities use seconds.  Entropies use nats.  Covariance summaries are
    normalized by preregistered physical reference variances.
    """
    cycle_index: int
    decision_time_s: float
    existence_mean: float
    existence_min: float
    cardinality_entropy_nats: float
    association_entropy_nats: float
    mixture_entropy_nats: float
    multimodality_index: float
    target_uncertainty_normalized: float
    nuisance_uncertainty_normalized: float
    normalized_innovation_sq: float
    newest_factor_age_s: float
    node_health_min: float
    link_health_min: float
    resource_remaining_fraction: float
    emission_remaining_fraction: float
    candidate_count: int
    posterior_digest: str
    availability_digest: str
    extra_observables: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        d=asdict(self)
        reject_forbidden_features(d)
        if self.cycle_index < 0 or self.candidate_count < 1:
            raise ControllerFeatureError("cycle_index must be nonnegative and candidate_count positive")
        for k,v in d.items():
            if isinstance(v,float) and not isfinite(v): raise ControllerFeatureError(f"{k} is non-finite")
        for name in ("existence_mean","existence_min","node_health_min","link_health_min",
                     "resource_remaining_fraction","emission_remaining_fraction"):
            v=float(getattr(self,name))
            if not (0.0 <= v <= 1.0): raise ControllerFeatureError(f"{name} outside [0,1]")
        if self.newest_factor_age_s < 0.0: raise ControllerFeatureError("negative factor age")
        if not self.posterior_digest or not self.availability_digest:
            raise ControllerFeatureError("posterior and availability digests are mandatory")
        # Validate extra-observable names at construction time, not only when a
        # downstream model eventually requests the feature vector.
        self.feature_vector()

    def feature_vector(self) -> Mapping[str,float]:
        vals={
          "existence_uncertainty": max(0.0,1.0-self.existence_mean),
          "weakest_existence_uncertainty": max(0.0,1.0-self.existence_min),
          "cardinality_entropy": max(0.0,self.cardinality_entropy_nats),
          "association_entropy": max(0.0,self.association_entropy_nats),
          "mixture_entropy": max(0.0,self.mixture_entropy_nats),
          "multimodality": max(0.0,self.multimodality_index),
          "target_uncertainty": max(0.0,self.target_uncertainty_normalized),
          "nuisance_uncertainty": max(0.0,self.nuisance_uncertainty_normalized),
          "innovation_stress": max(0.0,self.normalized_innovation_sq-1.0),
          "freshness_stress": log1p(max(0.0,self.newest_factor_age_s)),
          "node_health_deficit": max(0.0,1.0-self.node_health_min),
          "link_health_deficit": max(0.0,1.0-self.link_health_min),
          "resource_scarcity": max(0.0,1.0-self.resource_remaining_fraction),
          "emission_scarcity": max(0.0,1.0-self.emission_remaining_fraction),
          "candidate_log_count": log1p(float(self.candidate_count)),
        }
        for k,v in self.extra_observables.items():
            key=str(k).strip().lower()
            reject_forbidden_features({key:v})
            if key in vals:
                raise ControllerFeatureError(
                    f"extra observable cannot override reserved controller feature: {key}"
                )
            if not isfinite(float(v)): raise ControllerFeatureError(f"extra feature {k} non-finite")
            vals[key]=float(v)
        return MappingProxyType(vals)


@dataclass(frozen=True)
class HingeTerm:
    feature: str
    knot: float
    coefficient: float
    direction: int = 1

    def __post_init__(self):
        if self.coefficient < 0.0: raise ValueError("monotone coefficient must be nonnegative")
        if self.direction not in (-1,1): raise ValueError("direction must be +/-1")


@dataclass
class MonotoneAdaptiveLambdaGate:
    """Low-capacity monotone additive hinge model.

    Nonnegative coefficients make the declared monotonicity inspectable.  The
    output uses softplus, so lambda_t is finite and nonnegative.  Coefficients,
    knots, feature scales, and training split digest are serialized and frozen.
    """
    intercept: float = -1.5
    terms: Sequence[HingeTerm] = field(default_factory=lambda: (
        HingeTerm("association_entropy",0.15,0.28),
        HingeTerm("cardinality_entropy",0.10,0.24),
        HingeTerm("multimodality",0.10,0.35),
        HingeTerm("existence_uncertainty",0.10,0.25),
        HingeTerm("target_uncertainty",0.50,0.08),
        HingeTerm("nuisance_uncertainty",0.50,0.06),
        HingeTerm("freshness_stress",0.10,0.05),
        HingeTerm("resource_scarcity",0.40,0.05,direction=-1),
        HingeTerm("emission_scarcity",0.40,0.08,direction=-1),
    ))
    lambda_cap: float = 8.0
    training_split_digest: str = "unfitted-preregistered-initialization"
    version: str = "v73-monotone-hinge-1"

    @staticmethod
    def _softplus(x:float)->float:
        if x>35.0: return x
        if x<-35.0: return exp(x)
        return log1p(exp(x))

    def __call__(self, state:ControllerVisibleState)->Tuple[float,Dict[str,float]]:
        x=state.feature_vector(); linear=float(self.intercept); contributions={}
        for t in self.terms:
            if t.feature not in x: raise ControllerFeatureError(f"missing gate feature: {t.feature}")
            raw=(float(x[t.feature])-t.knot) if t.direction==1 else (t.knot-float(x[t.feature]))
            c=t.coefficient*max(0.0,raw)
            linear+=c; contributions[f"{t.feature}@{t.knot:g}"]=c
        lam=min(float(self.lambda_cap),self._softplus(linear))
        if not isfinite(lam) or lam<0.0: raise ControllerFeatureError("adaptive lambda invalid")
        return lam,{"linear":linear,"lambda":lam,**contributions}

    def frozen_digest(self)->str:
        payload={"intercept":self.intercept,"terms":[asdict(t) for t in self.terms],
                 "lambda_cap":self.lambda_cap,"training_split_digest":self.training_split_digest,
                 "version":self.version}
        return sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def load_monotone_gate_checkpoint(path:Path|str,expected_sha256:str|None=None):
    """Load a provenance-bearing gate checkpoint and verify every frozen field."""
    checkpoint_path=Path(path)
    raw=checkpoint_path.read_bytes()
    actual_sha256=sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha256!=expected_sha256:
        raise ControllerFeatureError("V73 gate checkpoint SHA-256 mismatch")
    doc=json.loads(raw)
    if doc.get("schema")!="v73-monotone-gate-checkpoint-2.0":
        raise ControllerFeatureError("unsupported or legacy V73 gate checkpoint schema")
    provenance=doc.get("provenance",{})
    for key in ("dataset_digest","split_registry_digest","train_seed_digest","validation_seed_digest"):
        if not isinstance(provenance.get(key),str) or not provenance[key]:
            raise ControllerFeatureError(f"V73 gate checkpoint lacks {key}")
    gate_doc=doc.get("gate",{})
    terms=tuple(HingeTerm(str(row["feature"]),float(row["knot"]),
                          float(row["coefficient"]),int(row.get("direction",1)))
                for row in gate_doc.get("terms",()))
    gate=MonotoneAdaptiveLambdaGate(
        intercept=float(gate_doc["intercept"]),terms=terms,
        lambda_cap=float(gate_doc["lambda_cap"]),
        training_split_digest=str(gate_doc["training_split_digest"]),
        version=str(gate_doc["version"]),
    )
    if gate.frozen_digest()!=gate_doc.get("frozen_digest"):
        raise ControllerFeatureError("V73 gate frozen digest mismatch")
    return gate,{"path":str(checkpoint_path.resolve()),"sha256":actual_sha256,
                 "gate_frozen_digest":gate.frozen_digest(),"provenance":dict(provenance)}


@dataclass(frozen=True)
class CandidateCertificate:
    """Paired counterfactual certificate relative to the explicit no-op.

    info_utility is dimensionless after preregistered normalization.
    set_risk_improvement and mission_loss_delta use declared loss units.
    Positive set_risk_improvement is beneficial; positive mission_loss_delta is
    degradation.  Standard errors use the same units as their means.
    """
    action_id: str
    feasible: bool
    is_noop: bool
    info_utility: float
    set_risk_improvement: float
    set_risk_se: float
    mission_loss_delta: float
    mission_loss_se: float
    resource_cost_normalized: float = 0.0
    emission_cost_normalized: float = 0.0
    physical_trace_digest: str = ""
    infeasible_reasons: Tuple[str,...] = ()
    metadata: Mapping[str,Any] = field(default_factory=dict)
    is_task_reference: bool = False
    track_update_probability: float = 0.0

    def __post_init__(self):
        reject_forbidden_features(self.metadata)
        for n in ("info_utility","set_risk_improvement","set_risk_se","mission_loss_delta",
                  "mission_loss_se","resource_cost_normalized","emission_cost_normalized"):
            if not isfinite(float(getattr(self,n))): raise ValueError(f"non-finite certificate field: {n}")
        if self.set_risk_se<0 or self.mission_loss_se<0: raise ValueError("standard errors must be nonnegative")
        if not (0.0 <= float(self.track_update_probability) <= 1.0):
            raise ValueError("track-update probability must lie in [0,1]")
        if self.is_task_reference and self.is_noop:
            raise ValueError("the task reference must be a physical non-noop candidate")
        if self.is_noop:
            # Relative estimands force the no-op reference to zero.
            if max(abs(self.info_utility),abs(self.set_risk_improvement),abs(self.mission_loss_delta),
                   abs(self.set_risk_se),abs(self.mission_loss_se))>1e-12:
                raise ValueError("no-op reference certificate must have zero relative utility, risk, and uncertainty")


@dataclass
class MissionDual:
    value: float = 0.0
    learning_rate: float = 0.05
    cap: float = 100.0

    def update(self, belief_certified_mission_delta:float, tolerance:float)->float:
        if not isfinite(belief_certified_mission_delta): raise ValueError("dual update requires finite belief certificate")
        self.value=min(self.cap,max(0.0,self.value+self.learning_rate*(belief_certified_mission_delta-tolerance)))
        return self.value


@dataclass(frozen=True)
class AdaptiveDecision:
    selected_action_id: str
    lambda_t: float
    nu_before: float
    selected_score: float
    no_op_reason: Optional[str]
    fallback_used: bool
    candidate_traces: Tuple[Mapping[str,Any],...]
    gate_trace: Mapping[str,float]
    posterior_digest: str
    availability_digest: str
    fallback_reason: Optional[str] = None


class AdaptiveMissionConstrainedController:
    """Final V73 candidate selector.  Exact certificates remain authoritative."""
    def __init__(self, gate:Optional[MonotoneAdaptiveLambdaGate]=None,
                 mission_tolerance:float=0.0, confidence_z:float=1.96,
                 dual:Optional[MissionDual]=None, min_positive_score:float=1e-10,
                 max_certificate_relative_se:float=1.0, hard_mission_gate:bool=True,
                 reference_noninferiority_gate:bool=False,
                 reference_mission_loss_margin:float=0.0,
                 reference_track_update_margin:float=0.0,
                 reference_fallback_on_risk_uncertainty:bool=False):
        if mission_tolerance<0.0: raise ValueError("mission tolerance must be nonnegative")
        if reference_mission_loss_margin < 0.0:
            raise ValueError("reference mission-loss margin must be nonnegative")
        if not (0.0 <= reference_track_update_margin <= 1.0):
            raise ValueError("reference track-update margin must lie in [0,1]")
        self.gate=gate or MonotoneAdaptiveLambdaGate()
        self.mission_tolerance=float(mission_tolerance)
        self.confidence_z=float(confidence_z)
        self.dual=dual or MissionDual()
        self.min_positive_score=float(min_positive_score)
        self.max_certificate_relative_se=float(max_certificate_relative_se)
        self.hard_mission_gate=bool(hard_mission_gate)
        self.reference_noninferiority_gate=bool(reference_noninferiority_gate)
        self.reference_mission_loss_margin=float(reference_mission_loss_margin)
        self.reference_track_update_margin=float(reference_track_update_margin)
        self.reference_fallback_on_risk_uncertainty=bool(
            reference_fallback_on_risk_uncertainty
        )
        if (
            self.reference_fallback_on_risk_uncertainty
            and not self.reference_noninferiority_gate
        ):
            raise ValueError(
                "task-reference fallback requires reference noninferiority"
            )

    def _bounds(self,c:CandidateCertificate)->Tuple[float,float]:
        return (c.set_risk_improvement-self.confidence_z*c.set_risk_se,
                c.mission_loss_delta+self.confidence_z*c.mission_loss_se)

    def decide(self,state:ControllerVisibleState,certificates:Sequence[CandidateCertificate])->AdaptiveDecision:
        if not certificates: raise ValueError("empty candidate set")
        ids=[c.action_id for c in certificates]
        if len(ids)!=len(set(ids)): raise ValueError("duplicate action IDs")
        noops=[c for c in certificates if c.is_noop and c.feasible]
        if len(noops)!=1: raise ValueError("exactly one feasible explicit no-op is required")
        physical_candidates=[c for c in certificates if not c.is_noop and c.feasible]
        physical_task_domain_empty=not physical_candidates
        task_references=[c for c in certificates if c.is_task_reference and c.feasible]
        if self.reference_noninferiority_gate:
            if physical_task_domain_empty and task_references:
                raise ValueError(
                    "an empty physical task domain cannot contain a task reference"
                )
            if not physical_task_domain_empty and len(task_references)!=1:
                raise ValueError(
                    "reference noninferiority requires exactly one feasible physical task reference"
                )
        task_reference=task_references[0] if task_references else None
        reference_mission_ucb=(
            self._bounds(task_reference)[1] if task_reference is not None else 0.0
        )
        reference_track_update=(
            float(task_reference.track_update_probability)
            if task_reference is not None
            else 0.0
        )
        lam,gate_trace=self.gate(state); nu_before=float(self.dual.value)
        traces=[]; admissible=[]
        for c in certificates:
            lcb,ucb=self._bounds(c)
            evidence_ok=True
            # Relative SE guard is scale-aware but does not reject exact zero-reference no-op.
            set_relative_se=0.0
            mission_relative_se=0.0
            if not c.is_noop:
                set_scale=max(abs(c.set_risk_improvement),1e-12)
                mission_scale=max(abs(c.mission_loss_delta),self.mission_tolerance,1e-12)
                set_relative_se=c.set_risk_se/set_scale
                mission_relative_se=c.mission_loss_se/mission_scale
                if (set_relative_se>self.max_certificate_relative_se or
                        mission_relative_se>self.max_certificate_relative_se):
                    evidence_ok=False
            if self.reference_noninferiority_gate:
                mission_excess=max(
                    0.0,
                    ucb
                    - reference_mission_ucb
                    - self.reference_mission_loss_margin,
                )
                update_shortfall=max(
                    0.0,
                    reference_track_update
                    - float(c.track_update_probability)
                    - self.reference_track_update_margin,
                )
                mission_safe=bool(mission_excess <= 1.0e-15)
                update_safe=bool(update_shortfall <= 1.0e-15)
            else:
                mission_excess=max(0.0,ucb-self.mission_tolerance)
                update_shortfall=0.0
                mission_safe=bool(c.is_noop or ucb <= self.mission_tolerance)
                update_safe=True
            regular_selection_allowed=bool(
                not (
                    self.reference_noninferiority_gate
                    and c.is_noop
                    and not physical_task_domain_empty
                )
            )
            score=c.info_utility+lam*lcb-nu_before*mission_excess
            feasible=bool(
                c.feasible
                and evidence_ok
                and regular_selection_allowed
                and ((mission_safe and update_safe) or not self.hard_mission_gate)
            )
            tr={"action_id":c.action_id,"is_noop":c.is_noop,"physical_feasible":c.feasible,
                "evidence_ok":evidence_ok,"mission_safe":mission_safe,"set_risk_lcb":lcb,"mission_loss_ucb":ucb,
                "is_task_reference":bool(c.is_task_reference),
                "reference_noninferiority_gate":self.reference_noninferiority_gate,
                "reference_fallback_on_risk_uncertainty":(
                    self.reference_fallback_on_risk_uncertainty
                ),
                "reference_mission_loss_ucb":reference_mission_ucb,
                "reference_mission_loss_margin":self.reference_mission_loss_margin,
                "track_update_probability":float(c.track_update_probability),
                "reference_track_update_probability":reference_track_update,
                "reference_track_update_margin":self.reference_track_update_margin,
                "track_update_noninferiority_safe":update_safe,
                "track_update_noninferiority_shortfall":update_shortfall,
                "physical_task_domain_empty":physical_task_domain_empty,
                "regular_selection_allowed":regular_selection_allowed,
                "info_utility":float(c.info_utility),
                "resource_cost_normalized":float(c.resource_cost_normalized),
                "emission_cost_normalized":float(c.emission_cost_normalized),
                "set_risk_relative_se":set_relative_se,"mission_loss_relative_se":mission_relative_se,
                "mission_excess":mission_excess,"score":score,"infeasible_reasons":list(c.infeasible_reasons),
                "physical_trace_digest":c.physical_trace_digest}
            traces.append(tr)
            if feasible: admissible.append((score,c,tr))
        fallback_reason=None
        if not admissible:
            if (
                self.reference_noninferiority_gate
                and self.reference_fallback_on_risk_uncertainty
                and task_reference is not None
            ):
                # The nonlocal-risk estimate can be too noisy to certify any
                # improvement.  That uncertainty is not evidence that the
                # physically feasible task reference should be replaced by a
                # track-breaking no-op.  Preserve the matched control action
                # and make no claim of nonlocal-risk improvement.
                chosen=task_reference
                reference_trace=next(
                    row for row in traces if row["action_id"]==chosen.action_id
                )
                score=float(reference_trace["score"])
                reason=None
                fallback=True
                fallback_reason=(
                    "risk_evidence_insufficient_fallback_to_task_reference"
                )
            else:
                chosen=noops[0]
                score=0.0
                reason="all_non_noop_candidates_infeasible_or_uncertified"
                fallback=True
                fallback_reason=reason
        else:
            admissible.sort(key=lambda q:(q[0], not q[1].is_noop, q[1].action_id),reverse=True)
            score,chosen,tr=admissible[0]
            fallback=False; reason=None
            non_noop=[q for q in admissible if not q[1].is_noop]
            if chosen.is_noop:
                if physical_task_domain_empty:
                    reason="no_physical_task_candidate_available"
                    fallback=True
                    fallback_reason=reason
                elif not non_noop: reason="no_feasible_non_noop_action"
                elif all(q[0] <= self.min_positive_score for q in non_noop):
                    reason="all_non_noop_lower_bound_scores_nonpositive"
                elif all(q[2]["mission_loss_ucb"]>self.mission_tolerance for q in non_noop):
                    reason="all_non_noop_actions_fail_mission_certificate"
                else: reason="no_op_is_best_certified_action"
            elif (
                score<=self.min_positive_score
                and not self.reference_noninferiority_gate
            ):
                # Anti-degeneracy: choosing an uncertain negative action is forbidden; explicit no-op wins.
                chosen=noops[0]; score=0.0; reason="best_non_noop_score_not_positive"; fallback=True
                fallback_reason=reason
        return AdaptiveDecision(chosen.action_id,lam,nu_before,float(score),reason,fallback,
                                tuple(MappingProxyType(dict(t)) for t in traces),MappingProxyType(dict(gate_trace)),
                                state.posterior_digest,state.availability_digest,
                                fallback_reason=fallback_reason)

    def commit_certificate(self,decision:AdaptiveDecision,certificates:Sequence[CandidateCertificate])->float:
        """Update the dual once, after the safety gate commits exactly one action.

        The input is the belief-derived UCB certificate, never realized truth loss.
        """
        selected=next((c for c in certificates if c.action_id==decision.selected_action_id),None)
        if selected is None: raise ValueError("selected action certificate missing")
        _,ucb=self._bounds(selected)
        if self.reference_noninferiority_gate:
            references=[c for c in certificates if c.is_task_reference and c.feasible]
            physical=[c for c in certificates if not c.is_noop and c.feasible]
            if not physical:
                if references or not selected.is_noop:
                    raise ValueError(
                        "empty physical task domain must commit the explicit no-op without a reference"
                    )
                return float(self.dual.value)
            if len(references)!=1:
                raise ValueError("committed decision lacks one task reference")
            _,reference_ucb=self._bounds(references[0])
            return self.dual.update(
                ucb-reference_ucb,
                self.reference_mission_loss_margin,
            )
        return self.dual.update(ucb,self.mission_tolerance)


def controller_view_digest(view:Mapping[str,Any])->str:
    reject_forbidden_features(view)
    return sha256(json.dumps(_canonical(view),sort_keys=True,separators=(",",":")).encode()).hexdigest()
