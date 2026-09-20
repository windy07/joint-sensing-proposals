from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AssociationBelief:
    target_origin: float
    clutter_origin: float
    ambiguous_origin: float
    missed_detection: float

    def __post_init__(self) -> None:
        values=(self.target_origin,self.clutter_origin,self.ambiguous_origin,self.missed_detection)
        if any((not math.isfinite(v)) or v<0.0 or v>1.0 for v in values):
            raise ValueError("association probabilities must lie in [0,1]")
        if abs(sum(values)-1.0)>1e-6:
            raise ValueError("association probabilities must sum to one")

    @property
    def wrong_origin_probability(self) -> float:
        return self.clutter_origin+self.ambiguous_origin

    @property
    def reliable_detection_probability(self) -> float:
        return self.target_origin

    @classmethod
    def from_unnormalized(cls, target: float, clutter: float, ambiguous: float, missed: float) -> "AssociationBelief":
        vals=[max(0.0,float(v)) for v in (target,clutter,ambiguous,missed)]
        total=sum(vals)
        if total<=0:
            raise ValueError("at least one association mass must be positive")
        return cls(*(v/total for v in vals))


@dataclass(frozen=True, slots=True)
class TemporalRiskState:
    cycle_index: int = 0
    unrooted_cumulative_hazard: float = 0.0
    root_cumulative_hazard: Mapping[str, float] = field(default_factory=dict)
    correlated_source_exposure: Mapping[str, float] = field(default_factory=dict)
    consecutive_unverified_updates: int = 0

    def update(
        self,
        belief: AssociationBelief,
        *,
        provenance_root_id: str | None,
        dependence: float,
        verified: bool,
        epsilon: float = 1e-8,
    ) -> "TemporalRiskState":
        if dependence<0.0 or dependence>1.0:
            raise ValueError("dependence must lie in [0,1]")
        wrong=min(1.0-epsilon,max(0.0,belief.wrong_origin_probability))
        hazard=-math.log1p(-wrong)
        exposures=dict(self.correlated_source_exposure)
        root_hazards=dict(self.root_cumulative_hazard)
        unrooted=float(self.unrooted_cumulative_hazard)
        if provenance_root_id is not None:
            prior=float(exposures.get(provenance_root_id,0.0))
            exposures[provenance_root_id]=prior+1.0
            old=float(root_hazards.get(provenance_root_id,0.0))
            # dependence=0 treats observations as independent. dependence=1
            # treats repetitions from the same root as fully correlated and
            # retains only the larger root-level hazard. Intermediate values
            # interpolate continuously without ever decreasing accumulated risk.
            independent=old+hazard
            fully_correlated=max(old,hazard)
            root_hazards[provenance_root_id]=(1.0-dependence)*independent+dependence*fully_correlated
        else:
            unrooted+=hazard
        return TemporalRiskState(
            cycle_index=self.cycle_index+1,
            unrooted_cumulative_hazard=unrooted,
            root_cumulative_hazard=root_hazards,
            correlated_source_exposure=exposures,
            consecutive_unverified_updates=0 if verified else self.consecutive_unverified_updates+1,
        )

    @property
    def accumulated_wrong_origin_probability(self) -> float:
        hazard=max(0.0,self.unrooted_cumulative_hazard+sum(self.root_cumulative_hazard.values()))
        return -math.expm1(-min(hazard,60.0))


@dataclass(frozen=True, slots=True)
class CandidateRiskVector:
    candidate_id: str
    feasible: bool
    association_wrong_origin_probability: float
    temporal_wrong_origin_probability: float
    geometry_certificate: float
    expected_task_loss: float
    tail_task_loss: float
    energy_j: float
    emission_exposure_j: float
    exact_information_gain_nat: float
    verification_value: float = 0.0
    abstains: bool = False
    diagnostics: Mapping[str, float | str | bool] = field(default_factory=dict)

    def validate(self) -> None:
        probabilities=(self.association_wrong_origin_probability,self.temporal_wrong_origin_probability)
        if any(not 0.0<=p<=1.0 for p in probabilities):
            raise ValueError("risk probabilities must lie in [0,1]")
        nonnegative=(self.expected_task_loss,self.tail_task_loss,self.energy_j,self.emission_exposure_j)
        if any((not math.isfinite(v)) or v<0 for v in nonnegative):
            raise ValueError("loss/resource terms must be finite and nonnegative")
        if not math.isfinite(self.geometry_certificate) or not math.isfinite(self.exact_information_gain_nat):
            raise ValueError("geometry/information terms must be finite")


@dataclass(frozen=True, slots=True)
class LexicographicRiskPolicy:
    max_association_wrong_origin: float = 0.35
    max_temporal_wrong_origin: float = 0.40
    max_tail_task_loss: float = math.inf
    min_geometry_certificate: float = 0.0
    require_scored_risk: bool = True

    def admissible(self, risk: CandidateRiskVector) -> bool:
        risk.validate()
        if not risk.feasible:
            return False
        if risk.abstains:
            return True
        return (
            risk.association_wrong_origin_probability<=self.max_association_wrong_origin
            and risk.temporal_wrong_origin_probability<=self.max_temporal_wrong_origin
            and risk.tail_task_loss<=self.max_tail_task_loss
            and risk.geometry_certificate>=self.min_geometry_certificate
        )

    def ordering_key(self, risk: CandidateRiskVector) -> tuple[float, ...]:
        """Noncompensatory first, scalar tradeoffs only within the safe set."""
        admissible=0.0 if self.admissible(risk) else 1.0
        return (
            admissible,
            risk.temporal_wrong_origin_probability,
            risk.association_wrong_origin_probability,
            risk.tail_task_loss,
            risk.expected_task_loss,
            risk.energy_j,
            risk.emission_exposure_j,
            -risk.exact_information_gain_nat,
            -risk.verification_value,
            -risk.geometry_certificate,
        )

    def select(self, risks: Sequence[CandidateRiskVector]) -> CandidateRiskVector:
        if not risks:
            raise ValueError("cannot select from an empty risk set")
        admissible=[risk for risk in risks if self.admissible(risk)]
        if not admissible:
            raise RuntimeError(
                "no admissible candidate; exact authority requires an explicit "
                "feasible abstention or verified safe fallback"
            )
        return min(admissible,key=self.ordering_key)


def geometry_reliability_reversal_example() -> tuple[CandidateRiskVector,CandidateRiskVector]:
    """A concrete counterexample to geometry-only ordering.

    Candidate A has the better directional certificate but its 0.8 wrong-origin
    mass creates a much larger Bayes task loss than Candidate B.  Thus ordering
    by geometry alone is reversed by the declared task risk.
    """
    a=CandidateRiskVector("geometry_only_winner",True,0.80,0.85,0.95,80.0,220.0,1.0,1.0,4.0)
    b=CandidateRiskVector("reliable_winner",True,0.05,0.06,0.55,3.0,8.0,1.2,1.1,2.0)
    return a,b
