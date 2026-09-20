from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch=None  # type: ignore[assignment]
    nn=object  # type: ignore[assignment,misc]
    F=None  # type: ignore[assignment]

from .tensorize import TensorInformationGraph


@dataclass(frozen=True, slots=True)
class GraphAgentConfig:
    input_dim: int
    hidden_dim: int = 128
    num_node_types: int = 9
    num_roles: int = 37
    num_edge_types: int = 10
    num_hyperedge_types: int = 5
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    memory_dim: int = 128
    joint_decoder: str = "additive"
    joint_decoder_rank: int = 32

    def __post_init__(self) -> None:
        integer_fields = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_node_types": self.num_node_types,
            "num_roles": self.num_roles,
            "num_edge_types": self.num_edge_types,
            "num_hyperedge_types": self.num_hyperedge_types,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "memory_dim": self.memory_dim,
            "joint_decoder_rank": self.joint_decoder_rank,
        }
        for name, value in integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.joint_decoder not in {"additive", "low_rank_set"}:
            raise ValueError("joint_decoder must be additive or low_rank_set")


@dataclass(slots=True)
class GraphAgentOutput:
    candidate_score: "torch.Tensor"
    feasibility_logit: "torch.Tensor"
    association_risk_logit: "torch.Tensor"
    temporal_risk_logit: "torch.Tensor"
    provenance_novelty_logit: "torch.Tensor"
    communication_logit: "torch.Tensor"
    verification_logit: "torch.Tensor"
    abstention_logit: "torch.Tensor"
    predictive_log_variance: "torch.Tensor"
    node_embedding: "torch.Tensor"
    next_memory: "torch.Tensor"


if torch is not None:
    class LowRankJointSetDecoder(nn.Module):
        """Permutation-invariant schedule interactions over graph-conditioned actions.

        The additive score is retained as an exact residual path.  A low-rank
        symmetric pair term represents cooperation or conflict, while a set
        term represents effects that depend on the joint action as a whole.
        Both corrections start at zero, so an initialization control is exactly
        the historical additive decoder rather than a different random policy.
        """

        def __init__(self, hidden_dim: int, rank: int) -> None:
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.rank = int(rank)
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_dim, self.rank),
                nn.GELU(),
                nn.LayerNorm(self.rank),
            )
            self.pair_axis = nn.Parameter(torch.zeros(self.rank))
            self.set_head = nn.Sequential(
                nn.Linear(2 * self.rank + 1, self.rank),
                nn.GELU(),
                nn.Linear(self.rank, 1),
            )
            self.empty_schedule_bias = nn.Parameter(torch.zeros(()))
            nn.init.zeros_(self.set_head[-1].weight)
            nn.init.zeros_(self.set_head[-1].bias)

        def forward(
            self,
            action_logits: Mapping[int, "torch.Tensor"],
            action_embeddings: Mapping[int, "torch.Tensor"],
            schedule_action_ids: Sequence[Sequence[int]],
        ) -> "torch.Tensor":
            if not action_logits:
                raise ValueError("joint set decoding requires action logits")
            reference = next(iter(action_logits.values()))
            if set(action_logits) != set(action_embeddings):
                raise ValueError("joint set decoding requires one embedding per scored action")
            ordered_ids = sorted(action_logits)
            projected_tensor = self.projection(
                torch.stack([action_embeddings[action_id] for action_id in ordered_ids])
            )
            id_to_index = {
                action_id: index for index, action_id in enumerate(ordered_ids)
            }
            schedules = [tuple(int(action_id) for action_id in raw) for raw in schedule_action_ids]
            maximum_cardinality = max((len(schedule) for schedule in schedules), default=0)
            if not schedules:
                return reference.new_empty((0,))

            # Build a compact schedule-incidence tensor on the host, then carry
            # out every learned operation in one device batch.  The physical
            # candidate domain is finite and caller-owned; this changes only
            # execution granularity, not the permutation-invariant formula.
            padded_width = max(1, maximum_cardinality)
            schedule_indices: list[list[int]] = []
            schedule_masks: list[list[bool]] = []
            for schedule in schedules:
                if len(schedule) != len(set(schedule)):
                    raise ValueError("joint schedule contains duplicate action ids")
                missing = [
                    action_id for action_id in schedule if action_id not in id_to_index
                ]
                if missing:
                    raise ValueError(
                        "joint schedule contains actions absent from fused views: "
                        f"{missing}"
                    )
                indices = [id_to_index[action_id] for action_id in schedule]
                schedule_indices.append(indices + [0] * (padded_width - len(indices)))
                schedule_masks.append(
                    [True] * len(indices) + [False] * (padded_width - len(indices))
                )

            index = torch.as_tensor(
                schedule_indices, dtype=torch.long, device=projected_tensor.device
            )
            mask = torch.as_tensor(
                schedule_masks, dtype=torch.bool, device=projected_tensor.device
            )
            mask_float = mask.to(dtype=projected_tensor.dtype)
            cardinality = mask_float.sum(dim=1)
            gathered_logits = torch.stack(
                [action_logits[action_id] for action_id in ordered_ids]
            )[index]
            latent = projected_tensor[index]
            unary = (gathered_logits * mask_float).sum(dim=1) / cardinality.clamp_min(1.0).sqrt()

            masked_latent = latent * mask_float.unsqueeze(-1)
            latent_sum = masked_latent.sum(dim=1)
            latent_square_sum = masked_latent.square().sum(dim=1)
            pair_sum = 0.5 * (latent_sum.square() - latent_square_sum)
            pair_count = cardinality * (cardinality - 1.0) / 2.0
            pair = (pair_sum * self.pair_axis).sum(dim=-1) / pair_count.clamp_min(1.0)
            pair = torch.where(pair_count > 0.0, pair, torch.zeros_like(pair))

            latent_mean = latent_sum / cardinality.clamp_min(1.0).unsqueeze(-1)
            latent_max = latent.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            latent_max = torch.where(
                cardinality.unsqueeze(-1) > 0.0,
                latent_max,
                torch.zeros_like(latent_max),
            )
            set_features = torch.cat(
                [latent_mean, latent_max, torch.log1p(cardinality).unsqueeze(-1)],
                dim=-1,
            )
            set_effect = self.set_head(set_features).squeeze(-1)
            values = unary + pair + set_effect
            return torch.where(
                cardinality > 0.0,
                values,
                torch.zeros_like(values) + self.empty_schedule_bias,
            )


    def attach_joint_schedule_decoder(
        model: nn.Module, cfg: GraphAgentConfig | Mapping[str, object]
    ) -> nn.Module:
        def value(name: str, default: object) -> object:
            if isinstance(cfg, Mapping):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        mode = str(value("joint_decoder", "additive"))
        if mode == "low_rank_set":
            model.joint_schedule_decoder = LowRankJointSetDecoder(
                int(value("hidden_dim", 128)), int(value("joint_decoder_rank", 32))
            )
        elif mode == "additive":
            model.joint_schedule_decoder = None
        else:
            raise ValueError("joint_decoder must be additive or low_rank_set")
        return model


    class TypedMessageLayer(nn.Module):
        def __init__(self, hidden_dim: int, edge_types: int, dropout: float) -> None:
            super().__init__()
            self.edge_embedding=nn.Embedding(edge_types,hidden_dim)
            self.edge_numeric=nn.Linear(3,hidden_dim)
            self.message=nn.Sequential(nn.Linear(hidden_dim*3,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
            self.update=nn.GRUCell(hidden_dim,hidden_dim)
            self.norm=nn.LayerNorm(hidden_dim)
            self.dropout=nn.Dropout(dropout)

        def forward(self,x,edge_index,edge_type,edge_features):
            if edge_index.numel()==0:
                return x
            src,dst=edge_index
            e=self.edge_embedding(edge_type)+self.edge_numeric(edge_features)
            msg=self.message(torch.cat([x[src],x[dst],e],dim=-1))
            aggregate=torch.zeros_like(x)
            # Autocast may produce FP16/BF16 messages while the persistent node
            # state remains FP32.  Sparse reductions require identical dtypes;
            # accumulate in the node-state dtype for numerical stability.
            aggregate.index_add_(0,dst,msg.to(dtype=x.dtype))
            degree=torch.zeros(x.shape[0],dtype=x.dtype,device=x.device)
            degree.index_add_(0,dst,torch.ones_like(dst,dtype=x.dtype))
            aggregate=aggregate/degree.clamp_min(1.0).unsqueeze(-1)
            updated=self.update(aggregate,x)
            return self.norm(x+self.dropout(updated))


    class HyperedgeLayer(nn.Module):
        def __init__(self,hidden_dim:int,hyperedge_types:int,dropout:float)->None:
            super().__init__()
            self.type_embedding=nn.Embedding(hyperedge_types,hidden_dim)
            self.to_h=nn.Linear(hidden_dim,hidden_dim)
            self.to_n=nn.Linear(hidden_dim,hidden_dim)
            self.norm=nn.LayerNorm(hidden_dim)
            self.dropout=nn.Dropout(dropout)

        def forward(self,x,incidence,incidence_type):
            if incidence.numel()==0:
                return x
            node_idx,h_idx=incidence
            h_count=int(h_idx.max().item())+1
            h=torch.zeros((h_count,x.shape[-1]),dtype=x.dtype,device=x.device)
            contribution=self.to_h(x[node_idx])+self.type_embedding(incidence_type)
            h.index_add_(0,h_idx,contribution.to(dtype=x.dtype))
            h_degree=torch.zeros(h_count,dtype=x.dtype,device=x.device)
            h_degree.index_add_(0,h_idx,torch.ones_like(h_idx,dtype=x.dtype))
            h=h/h_degree.clamp_min(1.0).unsqueeze(-1)
            back=torch.zeros_like(x)
            back.index_add_(0,node_idx,self.to_n(h[h_idx]).to(dtype=x.dtype))
            n_degree=torch.zeros(x.shape[0],dtype=x.dtype,device=x.device)
            n_degree.index_add_(0,node_idx,torch.ones_like(node_idx,dtype=x.dtype))
            back=back/n_degree.clamp_min(1.0).unsqueeze(-1)
            return self.norm(x+self.dropout(back))


    class GPSBlock(nn.Module):
        """Local typed messages + bounded global attention + explicit hyperedges."""
        def __init__(self,cfg:GraphAgentConfig)->None:
            super().__init__()
            self.local=TypedMessageLayer(cfg.hidden_dim,cfg.num_edge_types,cfg.dropout)
            self.hyper=HyperedgeLayer(cfg.hidden_dim,cfg.num_hyperedge_types,cfg.dropout)
            self.attn=nn.MultiheadAttention(cfg.hidden_dim,cfg.num_heads,dropout=cfg.dropout,batch_first=True)
            self.ffn=nn.Sequential(nn.Linear(cfg.hidden_dim,cfg.hidden_dim*4),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.hidden_dim*4,cfg.hidden_dim))
            self.norm1=nn.LayerNorm(cfg.hidden_dim)
            self.norm2=nn.LayerNorm(cfg.hidden_dim)

        def forward(self,x,g:TensorInformationGraph):
            local=self.local(x,g.edge_index,g.edge_type,g.edge_features)
            local=self.hyper(local,g.hyperedge_index,g.hyperedge_type)
            # The physical graph is finite (roughly O(10) sensor nodes plus
            # factors/messages), so full within-graph attention is intentional.
            global_out,_=self.attn(local.unsqueeze(0),local.unsqueeze(0),local.unsqueeze(0),need_weights=False)
            x=self.norm1(local+global_out.squeeze(0))
            return self.norm2(x+self.ffn(x))


    class TypedProvenanceGraphAgent(nn.Module):
        """Permutation-respecting dynamic graph proposal model.

        The network emits diagnostics/proposals only.  It never receives truth,
        commits an action, or replaces the exact counterfactual evaluator.
        """
        def __init__(self,cfg:GraphAgentConfig)->None:
            super().__init__()
            self.cfg=cfg
            self.numeric=nn.Linear(cfg.input_dim,cfg.hidden_dim)
            self.node_type=nn.Embedding(cfg.num_node_types,cfg.hidden_dim)
            self.role=nn.Embedding(cfg.num_roles,cfg.hidden_dim)
            self.blocks=nn.ModuleList([GPSBlock(cfg) for _ in range(cfg.num_layers)])
            self.memory_in=nn.Linear(cfg.hidden_dim,cfg.memory_dim)
            self.memory=nn.GRUCell(cfg.memory_dim,cfg.memory_dim)
            self.memory_out=nn.Linear(cfg.memory_dim,cfg.hidden_dim)
            self.heads=nn.ModuleDict({
                "candidate_score":nn.Linear(cfg.hidden_dim,1),
                "feasibility":nn.Linear(cfg.hidden_dim,1),
                "association_risk":nn.Linear(cfg.hidden_dim,1),
                "temporal_risk":nn.Linear(cfg.hidden_dim,1),
                "provenance_novelty":nn.Linear(cfg.hidden_dim,1),
                "communication":nn.Linear(cfg.hidden_dim,1),
                "verification":nn.Linear(cfg.hidden_dim,1),
                "abstention":nn.Linear(cfg.hidden_dim,1),
                # Retain the module key for checkpoint compatibility.  Its
                # output is trained as heteroscedastic normalized-regret
                # residual variance, not as single-model epistemic uncertainty.
                "epistemic_log_variance":nn.Linear(cfg.hidden_dim,1),
            })
            attach_joint_schedule_decoder(self, cfg)

        def forward(self,g:TensorInformationGraph,previous_memory=None)->GraphAgentOutput:
            x=self.numeric(g.node_features)+self.node_type(g.node_type)+self.role(g.role_id)
            for block in self.blocks:
                x=block(x,g)
            if previous_memory is None:
                previous_memory=torch.zeros((x.shape[0],self.cfg.memory_dim),dtype=x.dtype,device=x.device)
            if previous_memory.shape!=(x.shape[0],self.cfg.memory_dim):
                raise ValueError("previous_memory must align with the current canonical node order")
            next_memory=self.memory(self.memory_in(x),previous_memory)
            x=x+self.memory_out(next_memory)
            action=x[g.action_node_index]
            if action.numel()==0:
                action=x.new_empty((0,x.shape[-1]))
            return GraphAgentOutput(
                candidate_score=self.heads["candidate_score"](action).squeeze(-1),
                feasibility_logit=self.heads["feasibility"](action).squeeze(-1),
                association_risk_logit=self.heads["association_risk"](action).squeeze(-1),
                temporal_risk_logit=self.heads["temporal_risk"](action).squeeze(-1),
                provenance_novelty_logit=self.heads["provenance_novelty"](action).squeeze(-1),
                communication_logit=self.heads["communication"](action).squeeze(-1),
                verification_logit=self.heads["verification"](action).squeeze(-1),
                abstention_logit=self.heads["abstention"](action).squeeze(-1),
                predictive_log_variance=self.heads["epistemic_log_variance"](action).squeeze(-1).clamp(-10.0,10.0),
                node_embedding=x,
                next_memory=next_memory,
            )
else:
    class LowRankJointSetDecoder:  # pragma: no cover
        def __init__(self,*args,**kwargs):
            raise RuntimeError("PyTorch is required for LowRankJointSetDecoder")

    def attach_joint_schedule_decoder(model, cfg):  # pragma: no cover
        raise RuntimeError("PyTorch is required for a joint schedule decoder")

    class TypedMessageLayer:  # pragma: no cover
        def __init__(self,*args,**kwargs):
            raise RuntimeError("PyTorch is required for TypedMessageLayer")

    class HyperedgeLayer:  # pragma: no cover
        def __init__(self,*args,**kwargs):
            raise RuntimeError("PyTorch is required for HyperedgeLayer")

    class TypedProvenanceGraphAgent:  # pragma: no cover
        def __init__(self,*args,**kwargs):
            raise RuntimeError("PyTorch is required for TypedProvenanceGraphAgent")
