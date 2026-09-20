"""Machine-readable experiment contracts and matched-world invariants.

The module deliberately separates proposal quality from closed-loop policy quality.
It contains no simulator-specific global state and performs no experiment by import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ReferenceSemantics(str, Enum):
    EXACT_ENUMERATION = "exact_enumeration"
    FINITE_COMMON_UNION = "finite_common_union"
    BRANCH_AND_BOUND = "branch_and_bound_reference"
    SAMPLED_REFERENCE = "sampled_reference"


class ExecutionTier(str, Enum):
    STATIC = "static"
    TINY = "tiny_deterministic"
    SERVER = "server_required"


@dataclass(frozen=True, slots=True)
class MatchedWorldContract:
    world_ids_sha256: str
    random_streams_sha256: str
    posterior_initialization_sha256: str
    feasible_universe_sha256: str
    resource_budget_sha256: str
    emission_constraint_sha256: str
    horizon_sha256: str
    exact_evaluation_budget: int

    def assert_matched(self, other: "MatchedWorldContract") -> None:
        if self != other:
            a, b = asdict(self), asdict(other)
            differences = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
            raise ValueError(f"unmatched experimental arms: {differences}")


@dataclass(frozen=True, slots=True)
class MethodArm:
    arm_id: str
    representation: str
    proposer: str
    authority: str
    candidate_budget: int
    ablations: tuple[str, ...] = ()
    checkpoint_sha256: str | None = None

    def validate(self, *, require_checkpoint: bool = True) -> None:
        if self.candidate_budget < 0:
            raise ValueError("candidate_budget must be nonnegative")
        learned = "learned" in self.proposer.lower()
        if require_checkpoint and learned and not self.checkpoint_sha256:
            raise ValueError("a learned arm requires an immutable checkpoint SHA-256")
        if self.authority not in {"none", "legacy_exact", "graph_fork_exact"}:
            raise ValueError(f"unknown authority {self.authority!r}")


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    experiment_id: str
    scientific_question: str
    family: str
    reference_semantics: ReferenceSemantics
    matched_world: MatchedWorldContract
    arms: tuple[MethodArm, ...]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    decision_rule: Mapping[str, Any]
    forbidden_claims: tuple[str, ...] = ()
    execution_tier: ExecutionTier = ExecutionTier.SERVER
    seeds: tuple[int, ...] = ()
    notes: str = ""
    binding_status: str = "template"
    proposal_modes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.experiment_id or not self.scientific_question:
            raise ValueError("experiment_id and scientific_question are required")
        if len({a.arm_id for a in self.arms}) != len(self.arms):
            raise ValueError("arm_id values must be unique")
        for arm in self.arms:
            arm.validate(require_checkpoint=self.binding_status == "materialized")
        if not self.primary_metrics:
            raise ValueError("at least one primary metric is required")
        if self.binding_status not in {"template", "materialized"}:
            raise ValueError("binding_status must be template or materialized")
        if self.proposal_modes:
            primary = self.proposal_modes.get("primary_scientific_estimand", {})
            secondary = self.proposal_modes.get("secondary_operational_estimand", {})
            if primary.get("mode") != "learned_only_top_k":
                raise ValueError("B1 primary proposal mode must be learned_only_top_k")
            if primary.get("representation_superiority_eligible") is not True:
                raise ValueError("B1 primary must explicitly own representation inference")
            if secondary.get("representation_superiority_eligible") is not False:
                raise ValueError("operational hybrid results cannot establish representation superiority")
            if secondary.get("reserve_may_read_exact_outcome_or_qualification_labels") is not False:
                raise ValueError("constructive reserve must be outcome-label blind")
            if any(arm.proposer != primary["mode"] for arm in self.arms):
                raise ValueError("all B1 primary arms must use the registered learned-only mode")
        if self.reference_semantics is ReferenceSemantics.SAMPLED_REFERENCE:
            joined = " ".join(self.forbidden_claims).lower()
            if "oracle" not in joined:
                raise ValueError("sampled references must explicitly forbid full-oracle language")

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_contract(path: str | Path) -> ExperimentContract:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    mw = MatchedWorldContract(**raw.pop("matched_world"))
    arms = tuple(MethodArm(**{**a, "ablations": tuple(a.get("ablations", ()))}) for a in raw.pop("arms"))
    raw["primary_metrics"] = tuple(raw["primary_metrics"])
    raw["secondary_metrics"] = tuple(raw.get("secondary_metrics", ()))
    raw["forbidden_claims"] = tuple(raw.get("forbidden_claims", ()))
    raw["seeds"] = tuple(raw.get("seeds", ()))
    raw["reference_semantics"] = ReferenceSemantics(raw["reference_semantics"])
    raw["execution_tier"] = ExecutionTier(raw.get("execution_tier", "server_required"))
    contract = ExperimentContract(matched_world=mw, arms=arms, **raw)
    contract.validate()
    return contract


def write_contract(contract: ExperimentContract, path: str | Path) -> None:
    contract.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(contract), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
