from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
import math
import json

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch=None  # type: ignore[assignment]
    F=None  # type: ignore[assignment]

from .model import GraphAgentOutput


@dataclass(frozen=True, slots=True)
class FixedLossWeights:
    coverage: float = 1.0
    regret: float = 0.5
    feasibility: float = 0.5
    family_balance: float = 0.1
    diversity: float = 0.1
    # Auxiliary heads are architectural interfaces, not primary evidence.
    # They remain disabled until the registered source-native train and
    # validation labels pass the supervision-admissibility contract.
    association: float = 0.0
    provenance: float = 0.0
    temporal: float = 0.0
    verification: float = 0.0
    abstention: float = 0.0
    predictive_regression: float = 0.1


@dataclass(slots=True)
class ProposalBatchTargets:
    qualifying: "torch.Tensor"
    feasible: "torch.Tensor"
    exact_loss: "torch.Tensor"
    association_wrong_origin: "torch.Tensor"
    provenance_novelty: "torch.Tensor"
    temporal_wrong_origin: "torch.Tensor"
    verification_good: "torch.Tensor"
    abstention_good: "torch.Tensor"
    family_id: "torch.Tensor"
    local_action_proposal_budget_k: int
    association_mask: "torch.Tensor | None" = None
    provenance_mask: "torch.Tensor | None" = None
    temporal_mask: "torch.Tensor | None" = None
    verification_mask: "torch.Tensor | None" = None
    abstention_mask: "torch.Tensor | None" = None
    candidate_embedding: "torch.Tensor | None" = None


@dataclass(slots=True)
class JointScheduleBatchTargets:
    schedule_action_ids: tuple[tuple[int, ...], ...]
    qualifying: "torch.Tensor"
    feasible: "torch.Tensor"
    exact_loss: "torch.Tensor"
    joint_schedule_candidate_budget_k: int


def _budgeted_topk_coverage(logits,qualifying,budget_k:int):
    """Surrogate for the event that top-K contains a qualifying candidate.

    The loss compares the best qualifying logit with the K-th strongest
    nonqualifying competitor.  It is invariant to a uniform logit shift, so the
    model cannot reduce it by driving every proposal probability toward one.
    """
    if logits.numel()==0 or budget_k<=0:
        return logits.new_tensor(0.0)
    qualifying=qualifying.bool()
    if not qualifying.any() or budget_k>=logits.numel():
        return logits.new_tensor(0.0)
    nonqualifying=logits[~qualifying]
    if nonqualifying.numel()<budget_k:
        return logits.new_tensor(0.0)
    boundary=torch.topk(nonqualifying,k=budget_k).values[-1]
    # The event is existential: at least one qualifying item must cross the
    # top-K boundary.  A log-sum-exp over every qualifying item lets many weak
    # items lower the loss even when none is selected, so use the exact maximum
    # whose subgradient is well defined almost everywhere.
    best_qualifying=torch.max(logits[qualifying])
    return F.softplus(boundary-best_qualifying)


def _listwise_regret(logits,exact_loss,feasible,*,temperature:float=1.0):
    if logits.numel()==0:
        return logits.new_tensor(0.0)
    if not math.isfinite(float(temperature)) or temperature<=0:
        raise ValueError("listwise exact-loss temperature must be finite and positive")
    masked_loss=torch.where(feasible.bool(),exact_loss,torch.full_like(exact_loss,float("inf")))
    finite=torch.isfinite(masked_loss)
    if not finite.any():
        return logits.new_tensor(0.0)
    centered=masked_loss[finite]-masked_loss[finite].min()
    target=torch.softmax(-centered/float(temperature),dim=0)
    pred=torch.log_softmax(logits[finite],dim=0)
    return -(target*pred).sum()


def aggregate_local_action_logits(
    action_logits: Mapping[int, Sequence["torch.Tensor"]],
) -> dict[int, "torch.Tensor"]:
    """Fuse arrived local Agent scores without counting duplicated views twice."""

    result: dict[int, "torch.Tensor"] = {}
    for action_id, values in action_logits.items():
        if not values:
            continue
        stacked = torch.stack(tuple(values))
        result[int(action_id)] = torch.logsumexp(stacked, dim=0) - math.log(
            len(values)
        )
    return result


def aggregate_local_action_embeddings(
    action_embeddings: Mapping[int, Sequence["torch.Tensor"]],
) -> dict[int, "torch.Tensor"]:
    """Average graph-conditioned views after the same causal fusion audit."""

    result: dict[int, "torch.Tensor"] = {}
    for action_id, values in action_embeddings.items():
        if not values:
            continue
        stacked = torch.stack(tuple(values))
        result[int(action_id)] = stacked.mean(dim=0)
    return result


def joint_schedule_logits(
    action_logits: Mapping[int, "torch.Tensor"],
    schedule_action_ids: Sequence[Sequence[int]],
    *,
    action_embeddings: Mapping[int, "torch.Tensor"] | None = None,
    decoder=None,
) -> "torch.Tensor":
    """Lift local action evidence to the declared finite schedule domain."""

    if not action_logits:
        raise ValueError("joint schedule scoring requires local action logits")
    if decoder is not None:
        if action_embeddings is None:
            raise ValueError("the joint set decoder requires fused action embeddings")
        return decoder(action_logits, action_embeddings, schedule_action_ids)
    reference = next(iter(action_logits.values()))
    values: list[torch.Tensor] = []
    for schedule in schedule_action_ids:
        canonical = tuple(int(action_id) for action_id in schedule)
        if not canonical:
            values.append(reference.new_zeros(()))
            continue
        missing = [action_id for action_id in canonical if action_id not in action_logits]
        if missing:
            raise ValueError(
                f"joint schedule contains actions absent from all local views: {missing}"
            )
        # Under roughly standardized local scores, sqrt-cardinality scaling
        # avoids making larger schedules win solely because they contain more
        # additive terms while preserving constructive evidence accumulation.
        values.append(
            torch.stack([action_logits[action_id] for action_id in canonical]).sum()
            / math.sqrt(len(canonical))
        )
    return torch.stack(values)


def joint_schedule_training_loss(
    schedule_logits: "torch.Tensor",
    targets: JointScheduleBatchTargets,
    *,
    exact_loss_temperature: float = 1.0,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    if schedule_logits.numel() != len(targets.schedule_action_ids):
        raise ValueError("joint schedule logits do not align with schedule targets")
    terms = {
        "joint_coverage": _budgeted_topk_coverage(
            schedule_logits,
            targets.qualifying,
            targets.joint_schedule_candidate_budget_k,
        ),
        "joint_regret": _listwise_regret(
            schedule_logits,
            targets.exact_loss,
            targets.feasible,
            temperature=exact_loss_temperature,
        ),
    }
    return terms["joint_coverage"] + 0.5 * terms["joint_regret"], terms


def joint_schedule_transition_alignment_loss(
    current_logits: "torch.Tensor",
    current_targets: JointScheduleBatchTargets,
    previous_logits: "torch.Tensor",
    previous_targets: JointScheduleBatchTargets,
    *,
    exact_loss_temperature: float,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor | bool | int"]]:
    """Align score changes with exact relative-risk changes across causal cycles.

    The previous prediction is detached, so this is one-step causal supervision,
    not a future-reading recurrent objective.  Centering removes an irrelevant
    common logit shift.  Changed exact demand uses directional alignment; stable
    demand penalizes ranking drift.
    """

    if not math.isfinite(float(exact_loss_temperature)) or exact_loss_temperature<=0:
        raise ValueError("transition exact-loss temperature must be finite and positive")
    current_index={tuple(map(int,schedule)):index for index,schedule in enumerate(current_targets.schedule_action_ids)}
    previous_index={tuple(map(int,schedule)):index for index,schedule in enumerate(previous_targets.schedule_action_ids)}
    common=sorted(current_index.keys() & previous_index.keys())
    zero=current_logits.sum()*0.0
    if len(common)<2:
        return zero,{
            "joint_transition_alignment":zero,
            "joint_transition_stability":zero,
            "exact_demand_changed":False,
            "common_schedule_count":len(common),
        }
    ci=torch.as_tensor([current_index[s] for s in common],device=current_logits.device)
    pi=torch.as_tensor([previous_index[s] for s in common],device=previous_logits.device)
    common_feasible=(
        current_targets.feasible[ci].bool()
        & previous_targets.feasible[pi].to(current_logits.device).bool()
    )
    ci=ci[common_feasible]
    pi=pi[common_feasible.to(previous_logits.device)]
    feasible_common_count=int(ci.numel())
    if feasible_common_count<2:
        return zero,{
            "joint_transition_alignment":zero,
            "joint_transition_stability":zero,
            "exact_demand_changed":False,
            "common_schedule_count":feasible_common_count,
        }
    current_score=current_logits[ci]
    previous_score=previous_logits[pi].detach().to(current_score)
    current_loss=current_targets.exact_loss[ci]
    previous_loss=previous_targets.exact_loss[pi].to(current_loss)
    current_relative=current_loss-current_loss.min()
    previous_relative=previous_loss-previous_loss.min()
    target_delta=(previous_relative-current_relative)/float(exact_loss_temperature)
    target_delta=target_delta-target_delta.mean()
    prediction_delta=current_score-previous_score
    prediction_delta=prediction_delta-prediction_delta.mean()

    def demand_signature(targets: JointScheduleBatchTargets):
        feasible_indices=torch.nonzero(targets.feasible.bool(),as_tuple=False).flatten()
        if feasible_indices.numel()==0:
            return None,frozenset()
        losses=targets.exact_loss[feasible_indices]
        best_local=int(torch.argmin(losses).item())
        oracle=targets.schedule_action_ids[int(feasible_indices[best_local].item())]
        qualifying_indices=torch.nonzero(
            targets.qualifying.bool() & targets.feasible.bool(),as_tuple=False
        ).flatten().detach().cpu().tolist()
        qualifying=frozenset(
            targets.schedule_action_ids[int(index)] for index in qualifying_indices
        )
        return tuple(oracle),qualifying

    changed=demand_signature(current_targets)!=demand_signature(previous_targets)
    if changed and float(torch.linalg.vector_norm(target_delta).detach().cpu())>1.0e-9:
        direction=1.0-F.cosine_similarity(
            prediction_delta.unsqueeze(0),target_delta.unsqueeze(0),dim=1,eps=1.0e-8
        ).squeeze(0)
        target_scaled=target_delta/target_delta.abs().mean().clamp_min(1.0)
        magnitude=F.smooth_l1_loss(prediction_delta,target_scaled.clamp(-4.0,4.0))
        alignment=direction+0.1*magnitude
        stability=zero
    else:
        alignment=zero
        stability=F.smooth_l1_loss(prediction_delta,torch.zeros_like(prediction_delta))
    return alignment+stability,{
        "joint_transition_alignment":alignment,
        "joint_transition_stability":stability,
        "exact_demand_changed":bool(changed),
        "common_schedule_count":feasible_common_count,
    }


def _family_balance(logits,family_id):
    if logits.numel()==0:
        return logits.new_tensor(0.0)
    p=torch.softmax(logits,dim=0)
    families=torch.unique(family_id)
    masses=torch.stack([p[family_id==f].sum() for f in families])
    target=torch.full_like(masses,1.0/max(1,len(families)))
    return F.kl_div(masses.clamp_min(1e-8).log(),target,reduction="sum")


def _diversity_penalty(embedding,logits):
    if embedding is None or embedding.shape[0]<2:
        return logits.new_tensor(0.0)
    weight=torch.softmax(logits,dim=0)
    normalized=F.normalize(embedding,dim=-1)
    similarity=normalized@normalized.T
    pair_weight=weight[:,None]*weight[None,:]
    mask=~torch.eye(similarity.shape[0],dtype=torch.bool,device=similarity.device)
    return (pair_weight[mask]*similarity[mask].clamp_min(0.0)).sum()


def _masked_binary_cross_entropy(logits,target,mask):
    if logits.numel()==0:
        return logits.sum()*0.0
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits,target.float())
    mask=mask.float()
    if mask.numel()!=logits.numel():
        raise ValueError("auxiliary target mask must align with candidate logits")
    if mask.sum()<=0:
        return logits.sum()*0.0
    values=F.binary_cross_entropy_with_logits(logits,target.float(),reduction="none")
    return (values*mask).sum()/mask.sum()


def _heteroscedastic_regret_nll(mean_score, log_variance, exact_loss, feasible):
    """Calibrate a per-cycle normalized regret surrogate and its variance.

    This is predictive residual variance for the action-level surrogate.  It is
    not posterior-state uncertainty and it is not epistemic uncertainty from a
    single model.
    """
    if mean_score.numel() == 0:
        return mean_score.sum() * 0.0 + log_variance.sum() * 0.0
    valid = (
        feasible.bool()
        & torch.isfinite(mean_score)
        & torch.isfinite(log_variance)
        & torch.isfinite(exact_loss)
    )
    if not valid.any():
        return mean_score.sum() * 0.0 + log_variance.sum() * 0.0
    loss = exact_loss[valid]
    centered = loss - loss.min()
    scale = centered.max().clamp_min(1.0e-6)
    target = -centered / scale
    residual = mean_score[valid] - target
    log_var = log_variance[valid].clamp(-10.0, 10.0)
    return 0.5 * (torch.exp(-log_var) * residual.square() + log_var).mean()


def proposal_training_loss(output:GraphAgentOutput,targets:ProposalBatchTargets,weights:FixedLossWeights|None=None):
    if torch is None:
        raise RuntimeError("PyTorch is required for training")
    w=weights or FixedLossWeights()
    logits=output.candidate_score
    terms={
        "coverage":_budgeted_topk_coverage(
            logits,
            targets.qualifying,
            targets.local_action_proposal_budget_k,
        ),
        "regret":_listwise_regret(logits,targets.exact_loss,targets.feasible),
        "feasibility":_masked_binary_cross_entropy(output.feasibility_logit,targets.feasible,None),
        "family_balance":_family_balance(logits,targets.family_id),
        "diversity":_diversity_penalty(targets.candidate_embedding,logits),
        "association":_masked_binary_cross_entropy(output.association_risk_logit,targets.association_wrong_origin,targets.association_mask),
        "provenance":_masked_binary_cross_entropy(output.provenance_novelty_logit,targets.provenance_novelty,targets.provenance_mask),
        "temporal":_masked_binary_cross_entropy(output.temporal_risk_logit,targets.temporal_wrong_origin,targets.temporal_mask),
        "verification":_masked_binary_cross_entropy(output.verification_logit,targets.verification_good,targets.verification_mask),
        "abstention":_masked_binary_cross_entropy(output.abstention_logit,targets.abstention_good,targets.abstention_mask),
        "predictive_regression":_heteroscedastic_regret_nll(
            output.candidate_score,
            output.predictive_log_variance,
            targets.exact_loss,
            targets.feasible,
        ),
    }
    total=sum(getattr(w,name)*value for name,value in terms.items())
    return total,terms


@dataclass(frozen=True, slots=True)
class ValidationScore:
    conditional_joint_schedule_recall: float
    joint_schedule_regret: float
    association_brier_score: float | None
    checkpoint_path: str
    epoch: int
    metric_aggregation: str = "scenario_macro_over_world_macro"

    def selection_key(self)->tuple[float,float,int]:
        # Test metrics are deliberately absent.  Scientific estimands remain
        # fixed; validation only chooses a checkpoint.
        return (
            -self.conditional_joint_schedule_recall,
            self.joint_schedule_regret,
            self.epoch,
        )


class ValidationOnlyCheckpointSelector:
    def __init__(
        self,
        record_path: Path,
        *,
        minimum_acceptable_recall: float | None = None,
    ) -> None:
        if minimum_acceptable_recall is not None:
            minimum_acceptable_recall = float(minimum_acceptable_recall)
            if not math.isfinite(minimum_acceptable_recall):
                raise ValueError("minimum acceptable validation recall must be finite")
            if not 0.0 < minimum_acceptable_recall <= 1.0:
                raise ValueError(
                    "minimum acceptable validation recall must be in (0, 1]"
                )
        self.record_path=record_path
        self.best:ValidationScore|None=None
        self.minimum_acceptable_recall = minimum_acceptable_recall

    @property
    def selection_policy(self) -> str:
        if self.minimum_acceptable_recall is None:
            return "strict_lexicographic_recall_then_regret"
        return "minimum_regret_subject_to_recall_floor"

    def selection_key(self, score: ValidationScore) -> tuple[float, ...]:
        if self.minimum_acceptable_recall is None:
            return score.selection_key()
        if score.conditional_joint_schedule_recall >= self.minimum_acceptable_recall:
            return (
                0.0,
                score.joint_schedule_regret,
                -score.conditional_joint_schedule_recall,
                float(score.epoch),
            )
        # If no checkpoint satisfies the fixed recall constraint, retain the
        # highest-recall checkpoint and use exact regret only as a tiebreaker.
        return (
            1.0,
            -score.conditional_joint_schedule_recall,
            score.joint_schedule_regret,
            float(score.epoch),
        )

    def record_payload(self, score: ValidationScore) -> dict[str, object]:
        floor = self.minimum_acceptable_recall
        return {
            "schema": "iclr2027.validation_checkpoint_selection.v2",
            **asdict(score),
            "selection_policy": self.selection_policy,
            "minimum_acceptable_recall": floor,
            "recall_floor_met": (
                None
                if floor is None
                else score.conditional_joint_schedule_recall >= floor
            ),
            "fallback_if_no_epoch_meets_floor": (
                None
                if floor is None
                else "conditional_joint_schedule_recall:max_then_joint_schedule_regret:min"
            ),
        }

    def should_select(self, score: ValidationScore) -> bool:
        return self.best is None or self.selection_key(score) < self.selection_key(
            self.best
        )

    def commit(self, score: ValidationScore) -> None:
        self.best = score
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.record_path.with_suffix(self.record_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.record_payload(score), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.record_path)

    def consider(self,score:ValidationScore)->bool:
        if not self.should_select(score):
            return False
        self.commit(score)
        return True
