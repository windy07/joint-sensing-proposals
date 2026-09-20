"""Matched-interface representation baselines for the proposal model.

These models consume the same tensorized arrived-information graph and emit the
same `GraphAgentOutput` contract as the proposed model.  They do not alter the
constructive reserve or exact authority.
"""
from __future__ import annotations
from dataclasses import MISSING, fields, is_dataclass
import inspect
from typing import Any, Iterable

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - exercised only without torch
    torch=None; Tensor=Any
    class _Missing:
        def __init__(self,*a:Any,**k:Any)->None: raise ImportError('PyTorch is required for learned representations') from exc
    class nn: Module=_Missing  # type: ignore

from .model import (
    GraphAgentOutput,
    HyperedgeLayer,
    TypedMessageLayer,
    TypedProvenanceGraphAgent,
    attach_joint_schedule_decoder,
)


def _attr(obj:Any,names:Iterable[str],default:Any=None)->Any:
    for name in names:
        if hasattr(obj,name): return getattr(obj,name)
        if isinstance(obj,dict) and name in obj:return obj[name]
    return default

def _cfg(config:Any,name:str,default:Any)->Any:
    return getattr(config,name,config.get(name,default) if isinstance(config,dict) else default)

def _explicit_cfg(config:Any,name:str)->Any:
    if isinstance(config,dict) and name in config:return config[name]
    if hasattr(config,name):return getattr(config,name)
    return None

def _as_tensor(x:Any,*,dtype:Any=None,device:Any=None)->Tensor:
    if torch is None: raise ImportError('PyTorch is required')
    if isinstance(x,Tensor): return x.to(device=device,dtype=dtype if dtype is not None else x.dtype)
    return torch.as_tensor(x,dtype=dtype,device=device)

def _extract(
    graph: Any,
    device: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    x=_attr(graph,('node_features','node_x','x'))
    if x is None: raise ValueError('tensor graph lacks node_features/node_x/x')
    x=_as_tensor(x,dtype=torch.float32,device=device)
    if x.ndim!=2: raise ValueError(f'expected node features [N,D], got {tuple(x.shape)}')
    types=_attr(
        graph,
        ('node_type','node_type_ids','node_types','type_ids'),
        torch.zeros(x.shape[0],dtype=torch.long),
    )
    types=_as_tensor(types,dtype=torch.long,device=device)
    roles=_attr(
        graph,
        ('role_id','role_ids','node_role_id','node_role_ids'),
        torch.zeros(x.shape[0],dtype=torch.long),
    )
    roles=_as_tensor(roles,dtype=torch.long,device=device)
    edge_index=_attr(graph,('edge_index','edges'),torch.empty((2,0),dtype=torch.long))
    edge_index=_as_tensor(edge_index,dtype=torch.long,device=device)
    if edge_index.ndim==2 and edge_index.shape[0]!=2 and edge_index.shape[1]==2:edge_index=edge_index.t()
    if edge_index.ndim!=2 or edge_index.shape[0]!=2:raise ValueError('edge_index must be [2,E]')
    action_index=_attr(graph,('action_node_index',))
    if action_index is not None:
        action_index=_as_tensor(action_index,dtype=torch.long,device=device)
    cx=_attr(graph,('candidate_features','candidate_x','action_features'))
    if cx is None:
        if action_index is not None:
            cx=x[action_index]
        else:
            cids=_attr(graph,('candidate_ids','action_ids'),())
            cx=torch.zeros((len(cids),x.shape[1]),dtype=x.dtype,device=device)
    cx=_as_tensor(cx,dtype=torch.float32,device=device)
    if cx.ndim!=2:raise ValueError('candidate features must be [C,D]')
    incidence=_attr(graph,('hyperedge_incidence','incidence'))
    if incidence is None: incidence=torch.empty((0,x.shape[0]),dtype=x.dtype,device=device)
    incidence=_as_tensor(incidence,dtype=torch.float32,device=device)
    return x,types,roles,edge_index,cx,incidence,action_index

def _segment_sum(messages:Tensor,index:Tensor,n:int)->Tensor:
    out=messages.new_zeros((n,messages.shape[-1]));
    if index.numel():out.index_add_(0,index,messages)
    return out

def _make_output(candidate_logits:Tensor,feasibility:Tensor,association:Tensor,temporal:Tensor,provenance:Tensor,communication:Tensor,verification:Tensor,abstain:Tensor,uncertainty:Tensor,memory:Tensor,node_embeddings:Tensor)->GraphAgentOutput:
    """Construct the source-native output dataclass by semantic field names."""
    values={
      'candidate_score':candidate_logits,
      'candidate_logits':candidate_logits,'candidate_scores':candidate_logits,'proposal_logits':candidate_logits,'proposal_scores':candidate_logits,
      'feasibility_logit':feasibility,
      'feasibility_logits':feasibility,'feasibility_scores':feasibility,'coverage_logits':feasibility,
      'association_risk_logit':association,
      'association_logits':association,'association_reliability':association.sigmoid(),'association_scores':association,
      'temporal_risk_logit':temporal,
      'provenance_novelty_logit':provenance,
      'provenance_logits':provenance,'provenance_novelty':provenance.sigmoid(),'novelty_logits':provenance,
      'communication_logit':communication,
      'communication_logits':communication,'coalition_logits':communication,'communication_scores':communication,
      'verification_logit':verification,
      'verification_logits':verification,'verification_scores':verification,
      'abstention_logit':abstain,
      'abstain_logits':abstain,'abstention_logits':abstain,'abstain_scores':abstain,
      'predictive_log_variance':uncertainty,
      'uncertainty':uncertainty,'uncertainty_log_variance':uncertainty,'predictive_uncertainty':uncertainty.exp(),
      'node_embedding':node_embeddings,
      'memory':memory,'next_memory':memory,'node_memory':memory,'node_embeddings':node_embeddings,'embeddings':node_embeddings,
    }
    if is_dataclass(GraphAgentOutput):
        kwargs={}
        for f in fields(GraphAgentOutput):
            if f.name in values:kwargs[f.name]=values[f.name];continue
            lname=f.name.lower()
            matched=None
            for key,val in values.items():
                if key in lname or lname in key:matched=val;break
            if matched is not None:kwargs[f.name]=matched
            elif f.default is not MISSING or f.default_factory is not MISSING:continue
            else:raise TypeError(f'unsupported required GraphAgentOutput field {f.name!r}')
        return GraphAgentOutput(**kwargs)
    sig=inspect.signature(GraphAgentOutput); kwargs={}
    for name,param in sig.parameters.items():
        if name=='self':continue
        if name in values:kwargs[name]=values[name]
        elif param.default is inspect._empty:raise TypeError(f'unsupported output parameter {name}')
    return GraphAgentOutput(**kwargs)


class _TypedNodeInput(nn.Module):
    """Give every representation the same node-type and role information."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_node_types: int,
        num_roles: int,
    ) -> None:
        super().__init__()
        self.numeric = nn.Linear(input_dim, hidden_dim)
        self.node_type = nn.Embedding(num_node_types, hidden_dim)
        self.role = nn.Embedding(num_roles, hidden_dim)

    def forward(self, x: Tensor, node_type: Tensor, role_id: Tensor) -> Tensor:
        return self.numeric(x) + self.node_type(node_type) + self.role(role_id)


class _Heads(nn.Module):
    def __init__(self,hidden:int,candidate_dim:int|None,memory_dim:int)->None:
        super().__init__();self.hidden_dim=hidden;self.candidate=(nn.Linear(candidate_dim,hidden) if candidate_dim is not None else None)
        self.memory_dim=memory_dim;self.memory_in=nn.Linear(hidden,memory_dim);self.memory_cell=nn.GRUCell(memory_dim,memory_dim);self.memory_out=nn.Linear(memory_dim,hidden)
        self.score=nn.Linear(hidden*2,1);self.feas=nn.Linear(hidden*2,1);self.assoc=nn.Linear(hidden*2,1);self.temporal=nn.Linear(hidden*2,1);self.prov=nn.Linear(hidden*2,1);self.comm=nn.Linear(hidden*2,1);self.verify=nn.Linear(hidden*2,1);self.abstain=nn.Linear(hidden*2,1);self.unc=nn.Linear(hidden*2,1)
    def forward(
        self,
        node_h: Tensor,
        cx: Tensor,
        memory: Tensor | None = None,
        action_index: Tensor | None = None,
    ) -> GraphAgentOutput:
        if memory is None:memory=node_h.new_zeros((node_h.shape[0],self.memory_dim))
        if memory.shape!=(node_h.shape[0],self.memory_dim):raise ValueError('memory must align with graph nodes')
        next_memory=self.memory_cell(self.memory_in(node_h),memory);node_h=node_h+self.memory_out(next_memory)
        graph_h=node_h.mean(0,keepdim=True) if node_h.numel() else cx.new_zeros((1,self.candidate.out_features))
        # In the source-native graph, candidates are action nodes.  Use their
        # post-encoder embeddings so every baseline receives the same typed
        # candidate context as the proposed model.  The explicit candidate
        # feature path remains for generic set inputs without action nodes.
        if action_index is not None:
            ch=node_h[action_index]
        else:
            if self.candidate is None:
                raise ValueError('candidate features require an explicit candidate_input_dim')
            ch=F.gelu(self.candidate(cx))
        joint=torch.cat([ch,graph_h.expand(ch.shape[0],-1)],-1)
        return _make_output(self.score(joint).squeeze(-1),self.feas(joint).squeeze(-1),self.assoc(joint).squeeze(-1),self.temporal(joint).squeeze(-1),self.prov(joint).squeeze(-1),self.comm(joint).squeeze(-1),self.verify(joint).squeeze(-1),self.abstain(joint).squeeze(-1),self.unc(joint).squeeze(-1),next_memory,node_h)


class DeepSetsAgent(nn.Module):
    def __init__(self,config:Any)->None:
        super().__init__();d=int(_cfg(config,'input_dim',_cfg(config,'node_input_dim',64)));h=int(_cfg(config,'hidden_dim',128));cd=_explicit_cfg(config,'candidate_input_dim');cd=(int(cd) if cd is not None else None);md=int(_cfg(config,'memory_dim',h));self.input=_TypedNodeInput(d,h,int(_cfg(config,'num_node_types',9)),int(_cfg(config,'num_roles',32)));self.phi=nn.Sequential(nn.GELU(),nn.Linear(h,h));self.rho=nn.Sequential(nn.Linear(h,h),nn.GELU());self.heads=_Heads(h,cd,md)
    def forward(self,graph:Any,memory:Tensor|None=None)->GraphAgentOutput:
        x,types,roles,_,cx,_,action_index=_extract(graph,next(self.parameters()).device);h=self.phi(self.input(x,types,roles));global_h=self.rho(h.sum(0,keepdim=True));h=h+global_h;return self.heads(h,cx,memory,action_index)

class GINAgent(nn.Module):
    def __init__(self,config:Any)->None:
        super().__init__();d=int(_cfg(config,'input_dim',_cfg(config,'node_input_dim',64)));h=int(_cfg(config,'hidden_dim',128));cd=_explicit_cfg(config,'candidate_input_dim');cd=(int(cd) if cd is not None else None);md=int(_cfg(config,'memory_dim',h));L=int(_cfg(config,'num_layers',3));self.input=_TypedNodeInput(d,h,int(_cfg(config,'num_node_types',9)),int(_cfg(config,'num_roles',32)));self.eps=nn.Parameter(torch.zeros(L));self.mlps=nn.ModuleList([nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,h)) for _ in range(L)]);self.heads=_Heads(h,cd,md)
    def encode(self,x:Tensor,node_type:Tensor,role_id:Tensor,edge_index:Tensor)->Tensor:
        h=F.gelu(self.input(x,node_type,role_id));src,dst=edge_index
        for eps,mlp in zip(self.eps,self.mlps):h=mlp((1+eps)*h+_segment_sum(h[src],dst,h.shape[0]))
        return h
    def forward(self,graph:Any,memory:Tensor|None=None)->GraphAgentOutput:
        x,types,roles,ei,cx,_,action_index=_extract(graph,next(self.parameters()).device);return self.heads(self.encode(x,types,roles,ei),cx,memory,action_index)

class GATAgent(nn.Module):
    def __init__(self,config:Any)->None:
        super().__init__();d=int(_cfg(config,'input_dim',_cfg(config,'node_input_dim',64)));h=int(_cfg(config,'hidden_dim',128));cd=_explicit_cfg(config,'candidate_input_dim');cd=(int(cd) if cd is not None else None);md=int(_cfg(config,'memory_dim',h));heads=max(1,int(_cfg(config,'num_heads',4)));layers=max(1,int(_cfg(config,'num_layers',3)));dropout=float(_cfg(config,'dropout',0.1));
        if h % heads:
            raise ValueError('GAT hidden_dim must be divisible by num_heads')
        self.input=_TypedNodeInput(d,h,int(_cfg(config,'num_node_types',9)),int(_cfg(config,'num_roles',32)));self.attn=nn.ModuleList([nn.MultiheadAttention(h,heads,dropout=dropout,batch_first=True) for _ in range(layers)]);self.ff=nn.ModuleList([nn.Sequential(nn.Linear(h,2*h),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*h,h)) for _ in range(layers)]);self.norm1=nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)]);self.norm2=nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)]);self.heads=_Heads(h,cd,md)
    def forward(self,graph:Any,memory:Tensor|None=None)->GraphAgentOutput:
        x,types,roles,ei,cx,_,action_index=_extract(graph,next(self.parameters()).device);h=F.gelu(self.input(x,types,roles));n=h.shape[0];mask=torch.ones((n,n),dtype=torch.bool,device=h.device);mask.fill_diagonal_(False)
        if ei.numel():mask[ei[1],ei[0]]=False
        for attn,ff,norm1,norm2 in zip(self.attn,self.ff,self.norm1,self.norm2):
            y,_=attn(h.unsqueeze(0),h.unsqueeze(0),h.unsqueeze(0),attn_mask=mask,need_weights=False)
            h=norm1(h+y.squeeze(0));h=norm2(h+ff(h))
        return self.heads(h,cx,memory,action_index)

def _typed_relations(
    graph: Any,
    edge_index: Tensor,
    device: Any,
) -> tuple[Tensor, Tensor]:
    edge_count = int(edge_index.shape[1])
    edge_type = _as_tensor(
        _attr(
            graph,
            ("edge_type", "edge_type_ids"),
            torch.zeros(edge_count, dtype=torch.long),
        ),
        dtype=torch.long,
        device=device,
    )
    edge_features = _as_tensor(
        _attr(
            graph,
            ("edge_features", "edge_attr"),
            torch.zeros((edge_count, 3), dtype=torch.float32),
        ),
        dtype=torch.float32,
        device=device,
    )
    if edge_type.shape != (edge_count,):
        raise ValueError("edge_type must align with edge_index")
    if edge_features.shape != (edge_count, 3):
        raise ValueError("edge_features must be [E,3]")
    return edge_type, edge_features


def _typed_hyperedges(graph: Any, device: Any) -> tuple[Tensor, Tensor]:
    incidence = _as_tensor(
        _attr(
            graph,
            ("hyperedge_index", "hyperedge_incidence"),
            torch.empty((2, 0), dtype=torch.long),
        ),
        dtype=torch.long,
        device=device,
    )
    if incidence.ndim == 2 and incidence.shape[0] != 2 and incidence.shape[1] == 2:
        incidence = incidence.t()
    if incidence.ndim != 2 or incidence.shape[0] != 2:
        raise ValueError("hyperedge incidence must be [2,I]")
    incidence_type = _as_tensor(
        _attr(
            graph,
            ("hyperedge_type", "hyperedge_type_ids"),
            torch.zeros(incidence.shape[1], dtype=torch.long),
        ),
        dtype=torch.long,
        device=device,
    )
    if incidence_type.shape != (incidence.shape[1],):
        raise ValueError("hyperedge_type must align with hyperedge incidence")
    return incidence, incidence_type


class GraphGPSAgent(nn.Module):
    """Typed-edge local messages plus bounded global attention, no hyperedges."""

    def __init__(self,config:Any)->None:
        super().__init__();d=int(_cfg(config,'input_dim',64));h=int(_cfg(config,'hidden_dim',128));md=int(_cfg(config,'memory_dim',h));cd=_explicit_cfg(config,'candidate_input_dim');cd=(int(cd) if cd is not None else None);heads=max(1,int(_cfg(config,'num_heads',4)));
        if h % heads:
            raise ValueError('GraphGPS hidden_dim must be divisible by num_heads')
        dropout=float(_cfg(config,'dropout',0.1));layers=max(1,int(_cfg(config,'num_layers',3)));self.input=_TypedNodeInput(d,h,int(_cfg(config,'num_node_types',9)),int(_cfg(config,'num_roles',32)));self.local=nn.ModuleList([TypedMessageLayer(h,int(_cfg(config,'num_edge_types',1)),dropout) for _ in range(layers)]);self.global_attn=nn.ModuleList([nn.MultiheadAttention(h,heads,dropout=dropout,batch_first=True) for _ in range(layers)]);self.global_ff=nn.ModuleList([nn.Sequential(nn.Linear(h,4*h),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*h,h)) for _ in range(layers)]);self.norm1=nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)]);self.norm2=nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)]);self.heads=_Heads(h,cd,md)
    def forward(self,graph:Any,memory:Tensor|None=None)->GraphAgentOutput:
        device=next(self.parameters()).device;x,types,roles,ei,cx,_,action_index=_extract(graph,device);edge_type,edge_features=_typed_relations(graph,ei,device);h=F.gelu(self.input(x,types,roles))
        for local_layer,attn,ff,norm1,norm2 in zip(self.local,self.global_attn,self.global_ff,self.norm1,self.norm2):
            local=local_layer(h,ei,edge_type,edge_features);global_h,_=attn(h.unsqueeze(0),h.unsqueeze(0),h.unsqueeze(0),need_weights=False);h=norm1(local+global_h.squeeze(0));h=norm2(h+ff(h))
        return self.heads(h,cx,memory,action_index)


class TypedHypergraphMPAgent(nn.Module):
    """Typed pairwise and hyperedge messages without global self-attention."""

    def __init__(self,config:Any)->None:
        super().__init__();d=int(_cfg(config,'input_dim',64));h=int(_cfg(config,'hidden_dim',128));md=int(_cfg(config,'memory_dim',h));cd=_explicit_cfg(config,'candidate_input_dim');cd=(int(cd) if cd is not None else None);dropout=float(_cfg(config,'dropout',0.1));layers=max(1,int(_cfg(config,'num_layers',3)));self.input=_TypedNodeInput(d,h,int(_cfg(config,'num_node_types',9)),int(_cfg(config,'num_roles',32)));self.local=nn.ModuleList([TypedMessageLayer(h,int(_cfg(config,'num_edge_types',1)),dropout) for _ in range(layers)]);self.hyper=nn.ModuleList([HyperedgeLayer(h,int(_cfg(config,'num_hyperedge_types',1)),dropout) for _ in range(layers)]);self.ff=nn.ModuleList([nn.Sequential(nn.Linear(h,4*h),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*h,h)) for _ in range(layers)]);self.norm=nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)]);self.heads=_Heads(h,cd,md)

    def forward(self,graph:Any,memory:Tensor|None=None)->GraphAgentOutput:
        device=next(self.parameters()).device;x,types,roles,ei,cx,_,action_index=_extract(graph,device);edge_type,edge_features=_typed_relations(graph,ei,device);incidence,incidence_type=_typed_hyperedges(graph,device);h=F.gelu(self.input(x,types,roles))
        for local_layer,hyper_layer,ff,norm in zip(self.local,self.hyper,self.ff,self.norm):
            updated=local_layer(h,ei,edge_type,edge_features);updated=hyper_layer(updated,incidence,incidence_type);h=norm(updated+ff(updated))
        return self.heads(h,cx,memory,action_index)

class SetTransformerAgent(nn.Module):
    def __init__(self,config:Any)->None:
        super().__init__();d=int(_cfg(config,'input_dim',_cfg(config,'node_input_dim',64)));h=int(_cfg(config,'hidden_dim',128));cd=_explicit_cfg(config,'candidate_input_dim');cd=(int(cd) if cd is not None else None);md=int(_cfg(config,'memory_dim',h));heads=max(1,int(_cfg(config,'num_heads',4)));
        if h % heads:
            raise ValueError('Set Transformer hidden_dim must be divisible by num_heads')
        self.input=_TypedNodeInput(d,h,int(_cfg(config,'num_node_types',9)),int(_cfg(config,'num_roles',32)));self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(h,heads,4*h,dropout=float(_cfg(config,'dropout',0.1)),batch_first=True,activation='gelu') for _ in range(int(_cfg(config,'num_layers',3)))]);self.heads=_Heads(h,cd,md)
    def forward(self,graph:Any,memory:Tensor|None=None)->GraphAgentOutput:
        x,types,roles,_,cx,_,action_index=_extract(graph,next(self.parameters()).device);h=F.gelu(self.input(x,types,roles)).unsqueeze(0)
        for b in self.blocks:h=b(h)
        return self.heads(h.squeeze(0),cx,memory,action_index)


def make_graph_agent(representation:str,config:Any)->nn.Module:
    name=representation.strip().lower().replace('-','_').replace(' ','_')
    if name in {'proposed','typed_provenance_dynamic_graph','typed_provenance_dynamic_factor_hypergraph'}:return TypedProvenanceGraphAgent(config)
    table={'deepsets':DeepSetsAgent,'deep_sets':DeepSetsAgent,'gin':GINAgent,'mpnn':GINAgent,'gat':GATAgent,'graphgps':GraphGPSAgent,'graph_gps':GraphGPSAgent,'typed_hypergraph_mp':TypedHypergraphMPAgent,'hypergraph_mp':TypedHypergraphMPAgent,'settransformer':SetTransformerAgent,'set_transformer':SetTransformerAgent}
    if name not in table:raise ValueError(f"unknown representation {representation!r}; choose {sorted(set(table) | {'proposed'})}")
    model=table[name](config)
    # Evaluation and runtime adapters need one representation-independent source
    # of memory and decoder dimensions.  Proposed models already expose cfg;
    # baselines must do the same for a genuinely interchangeable interface.
    model.cfg=config
    return attach_joint_schedule_decoder(model,config)
