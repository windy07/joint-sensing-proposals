from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import DynamicInformationGraph, NodeKind

try:
    import torch
except Exception:  # pragma: no cover - import contract tested when torch exists
    torch=None  # type: ignore[assignment]


@dataclass(slots=True)
class TensorInformationGraph:
    node_features: "torch.Tensor"
    node_type: "torch.Tensor"
    role_id: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_type: "torch.Tensor"
    edge_features: "torch.Tensor"
    hyperedge_index: "torch.Tensor"
    hyperedge_type: "torch.Tensor"
    action_node_index: "torch.Tensor"
    graph_index: "torch.Tensor"
    node_ids: tuple[str,...]
    feature_names: tuple[str,...]


def tensorize_graph(
    graph: DynamicInformationGraph,
    *,
    feature_names: Sequence[str],
    role_vocabulary: Mapping[str,int],
    edge_vocabulary: Mapping[str,int],
    hyperedge_vocabulary: Mapping[str,int],
    device: str | None = None,
) -> TensorInformationGraph:
    if torch is None:
        raise RuntimeError("PyTorch is required to tensorize graphs")
    graph.validate()
    nodes=sorted(graph.nodes,key=lambda n:n.node_id)
    index={n.node_id:i for i,n in enumerate(nodes)}
    x=torch.tensor([[float(n.features.get(f,0.0)) for f in feature_names] for n in nodes],dtype=torch.float32,device=device)
    type_vocab={kind.value:i for i,kind in enumerate(NodeKind)}
    node_type=torch.tensor([type_vocab[n.kind.value] for n in nodes],dtype=torch.long,device=device)
    role_id=torch.tensor([role_vocabulary.get(n.role,role_vocabulary.get("<unk>",0)) for n in nodes],dtype=torch.long,device=device)
    edges=sorted(graph.edges,key=lambda e:(e.kind.value,e.source_id,e.target_id,e.timestamp_s))
    if edges:
        edge_index=torch.tensor([[index[e.source_id] for e in edges],[index[e.target_id] for e in edges]],dtype=torch.long,device=device)
        edge_type=torch.tensor([edge_vocabulary.get(e.kind.value,0) for e in edges],dtype=torch.long,device=device)
        edge_features=torch.tensor([[float(e.features.get("age_s",0.0)),float(e.features.get("delay_s",0.0)),float(e.features.get("dependence",0.0))] for e in edges],dtype=torch.float32,device=device)
    else:
        edge_index=torch.empty((2,0),dtype=torch.long,device=device)
        edge_type=torch.empty((0,),dtype=torch.long,device=device)
        edge_features=torch.empty((0,3),dtype=torch.float32,device=device)
    incidences=[]; htypes=[]
    for hidx,h in enumerate(sorted(graph.hyperedges,key=lambda h:h.hyperedge_id)):
        ht=hyperedge_vocabulary.get(h.kind,0)
        for member in h.member_ids:
            incidences.append((index[member],hidx)); htypes.append(ht)
    hyperedge_index=(torch.tensor(incidences,dtype=torch.long,device=device).t().contiguous() if incidences else torch.empty((2,0),dtype=torch.long,device=device))
    hyperedge_type=torch.tensor(htypes,dtype=torch.long,device=device) if htypes else torch.empty((0,),dtype=torch.long,device=device)
    action_node_index=torch.tensor([i for i,n in enumerate(nodes) if n.kind==NodeKind.ACTION],dtype=torch.long,device=device)
    return TensorInformationGraph(x,node_type,role_id,edge_index,edge_type,edge_features,hyperedge_index,hyperedge_type,action_node_index,torch.zeros(len(nodes),dtype=torch.long,device=device),tuple(n.node_id for n in nodes),tuple(feature_names))
