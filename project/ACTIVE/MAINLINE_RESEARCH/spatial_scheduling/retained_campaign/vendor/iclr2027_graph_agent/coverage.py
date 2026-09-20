from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .schema import ActionCandidate, deduplicate_candidates


@dataclass(frozen=True, slots=True)
class CertificateSearchResult:
    found: bool
    candidate: ActionCandidate | None
    certificate_value: float | None
    evaluations: int
    exhausted_declared_domain: bool
    termination_reason: str
    reference_domain_size: int | None
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)


def _unit_rows(rows: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    rows=np.asarray(rows,dtype=float)
    if rows.ndim!=2 or rows.shape[0]==0 or rows.shape[1]==0:
        raise ValueError("direction rows must be a nonempty rank-2 array")
    norms=np.linalg.norm(rows,axis=1)
    if np.any(norms<=epsilon):
        raise ValueError("direction rows must be nonzero")
    return rows/norms[:,None]


def frame_ratio(rows: np.ndarray, epsilon: float = 1e-12) -> float:
    u=_unit_rows(rows,epsilon)
    singular=np.linalg.svd(u,compute_uv=False)
    if singular.size==0:
        return 0.0
    maximum=float(np.max(singular))
    minimum=float(np.min(singular))
    if maximum<=epsilon:
        return 0.0
    return minimum/maximum


def robust_frame_ratio(rows: np.ndarray, erasures: int = 1, epsilon: float = 1e-12) -> float:
    """Registered C1 directional certificate.

    ``erasures`` is the number of sensing rows removed adversarially.  The
    function is deliberately named a frame ratio, not a covariance or global
    identifiability guarantee.
    """
    u=_unit_rows(rows,epsilon)
    m,d=u.shape
    if erasures<0:
        raise ValueError("erasures must be nonnegative")
    if m-erasures<d:
        return 0.0
    if erasures==0:
        return frame_ratio(u,epsilon)
    worst=math.inf
    for removed in combinations(range(m),erasures):
        keep=[i for i in range(m) if i not in removed]
        worst=min(worst,frame_ratio(u[keep],epsilon))
        if worst<=epsilon:
            return 0.0
    return float(worst)


def bounded_certificate_search(
    atomic_actions: Sequence[Mapping[str, object]],
    *,
    schedule_sizes: Iterable[int],
    certificate_threshold: float,
    erasures: int,
    is_feasible: Callable[[tuple[Mapping[str, object], ...]], bool],
    direction_rows: Callable[[tuple[Mapping[str, object], ...]], np.ndarray],
    make_candidate: Callable[[tuple[Mapping[str, object], ...], float], ActionCandidate],
    max_evaluations: int,
    declared_reference_domain_size: int | None = None,
) -> CertificateSearchResult:
    """Deterministic, finite fallback over a declared atomic action family.

    The search is exact only when it exhausts the declared family.  Otherwise
    the result is a bounded search witness and is never labelled a full oracle.
    Ordering uses a deterministic geometry-spread heuristic; correctness of a
    returned witness is independent of that ordering.
    """
    if max_evaluations<=0:
        raise ValueError("max_evaluations must be positive")
    actions=tuple(atomic_actions)
    if not actions:
        return CertificateSearchResult(False,None,None,0,True,"empty_reference_domain",0)
    ids=[str(a.get("action_id",i)) for i,a in enumerate(actions)]
    if len(ids)!=len(set(ids)):
        raise ValueError("atomic action ids must be unique")

    # Stable order first prioritizes declared analytic coverage score and then id.
    ordered=tuple(sorted(actions,key=lambda a:(-float(a.get("coverage_priority",0.0)),str(a.get("action_id","")))))
    evaluations=0
    feasible_seen=0
    best_value=-math.inf
    best: ActionCandidate | None=None
    total_domain=0
    sizes=tuple(sorted(set(int(s) for s in schedule_sizes if int(s)>0)))
    for size in sizes:
        if size<=len(ordered):
            total_domain += math.comb(len(ordered),size)
    for size in sizes:
        if size>len(ordered):
            continue
        for schedule in combinations(ordered,size):
            if evaluations>=max_evaluations:
                return CertificateSearchResult(
                    found=best is not None and best_value>=certificate_threshold,
                    candidate=best if best_value>=certificate_threshold else None,
                    certificate_value=best_value if best is not None else None,
                    evaluations=evaluations,
                    exhausted_declared_domain=False,
                    termination_reason="evaluation_budget_exhausted",
                    reference_domain_size=declared_reference_domain_size or total_domain,
                    diagnostics={"feasible_schedules_seen":feasible_seen,"best_value":best_value},
                )
            evaluations += 1
            if not is_feasible(schedule):
                continue
            feasible_seen += 1
            value=robust_frame_ratio(direction_rows(schedule),erasures=erasures)
            if value>best_value:
                best_value=value
                best=make_candidate(schedule,value)
            if value>=certificate_threshold:
                candidate=replace(best,protected=True,metadata={**best.metadata,"certificate_witness":True})
                return CertificateSearchResult(
                    True,candidate,value,evaluations,False,"qualifying_witness_found",
                    declared_reference_domain_size or total_domain,
                    {"feasible_schedules_seen":feasible_seen},
                )
    return CertificateSearchResult(
        False,None,(None if best is None else best_value),evaluations,True,
        "declared_reference_domain_exhausted_without_witness",
        declared_reference_domain_size or total_domain,
        {"feasible_schedules_seen":feasible_seen,"best_value":best_value},
    )


def reserve_certificate_slot(
    candidates: Sequence[ActionCandidate],
    witness: ActionCandidate | None,
    *,
    budget_k: int,
    model_scores: Mapping[str, float] | None = None,
) -> list[ActionCandidate]:
    """Keep at most ``budget_k`` unique candidates and reserve one C1 witness.

    This is a finite-domain coverage guarantee: if ``witness`` is a valid
    qualifying schedule in the declared fallback domain and K>=1, it is in the
    returned set.  It is not a population feasibility guarantee.
    """
    if budget_k<0:
        raise ValueError("budget_k must be nonnegative")
    if budget_k==0:
        return []
    scores=model_scores or {}
    unique=deduplicate_candidates(candidates)
    if witness is not None:
        witness=replace(witness,protected=True)
        unique=deduplicate_candidates([witness,*unique])
    protected=[c for c in unique if c.protected]
    ordinary=[c for c in unique if not c.protected]
    protected.sort(key=lambda c:(-float(scores.get(c.candidate_id,0.0)),c.candidate_id))
    ordinary.sort(key=lambda c:(-float(scores.get(c.candidate_id,0.0)),c.family,c.candidate_id))
    result=(protected+ordinary)[:budget_k]
    if witness is not None and all(c.canonical_key()!=witness.canonical_key() for c in result):
        result[-1]=witness
    return deduplicate_candidates(result)[:budget_k]


def qualifying_intersection(
    proposed: Sequence[ActionCandidate],
    reference: Sequence[ActionCandidate],
    qualifies: Callable[[ActionCandidate], bool],
) -> bool:
    qualifying={c.canonical_key() for c in reference if qualifies(c)}
    return any(c.canonical_key() in qualifying for c in proposed)
