from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch=None  # type: ignore[assignment]


@dataclass(slots=True)
class MemoryEntry:
    timestamp_s: float
    cycle_index: int
    state: "torch.Tensor"
    provenance_seen: frozenset[str]=frozenset()


class CausalNodeMemory:
    """Bounded per-node memory keyed by a caller-owned causal stream and node id."""
    def __init__(self,memory_dim:int,max_entries:int=4096)->None:
        if memory_dim<=0 or max_entries<=0: raise ValueError("memory_dim and max_entries must be positive")
        self.memory_dim=memory_dim; self.max_entries=max_entries; self._entries:dict[tuple[str,str],MemoryEntry]={}

    def gather(self,episode_id:str,node_ids:Sequence[str],*,now_s:float,cycle_index:int,device=None,dtype=None):
        if torch is None: raise RuntimeError("PyTorch is required for causal memory")
        rows=[]
        for node_id in node_ids:
            entry=self._entries.get((episode_id,node_id))
            if entry is None: rows.append(torch.zeros(self.memory_dim,dtype=dtype))
            else:
                if entry.timestamp_s>now_s+1e-12 or entry.cycle_index>cycle_index:
                    raise RuntimeError("future memory entry would violate the node filtration")
                rows.append(entry.state.to(dtype=dtype))
        return torch.stack(rows,dim=0).to(device=device) if rows else torch.empty((0,self.memory_dim),device=device,dtype=dtype)

    def update(self,episode_id:str,node_ids:Sequence[str],state,*,now_s:float,cycle_index:int,provenance_by_node:Mapping[str,Sequence[str]]|None=None)->None:
        if torch is None: raise RuntimeError("PyTorch is required for causal memory")
        if state.shape!=(len(node_ids),self.memory_dim): raise ValueError("state must align with node ids")
        # One transfer for the complete detached state, preserving row values.
        detached=state.detach().cpu()
        for i,node_id in enumerate(node_ids):
            key=(episode_id,node_id); prior=self._entries.get(key)
            if prior is not None and (now_s<prior.timestamp_s-1e-12 or cycle_index<prior.cycle_index):
                raise RuntimeError("causal memory cannot be written backwards")
            roots=frozenset((provenance_by_node or {}).get(node_id,()))
            self._entries[key]=MemoryEntry(float(now_s),int(cycle_index),detached[i],roots)
        self._evict()

    def _evict(self)->None:
        overflow=len(self._entries)-self.max_entries
        if overflow<=0:return
        oldest=sorted(self._entries,key=lambda k:(self._entries[k].cycle_index,self._entries[k].timestamp_s,k))[:overflow]
        for key in oldest: del self._entries[key]

    def clear_episode(self,episode_id:str)->None:
        for key in [k for k in self._entries if k[0]==episode_id]: del self._entries[key]
