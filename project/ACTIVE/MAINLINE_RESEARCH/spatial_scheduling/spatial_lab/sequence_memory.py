"""Online memory for semantic nodes continuously present in adjacent epochs.

The batched trainer uses the identical alignment rule and differentiates across
its registered chunks. Online updates detach because runtime performs no training.
"""
from __future__ import annotations
from .binding import bootstrap_vendor
bootstrap_vendor()
from iclr2027_graph_agent.memory import CausalNodeMemory
from .core import ContractError


class SequenceMemory:
    def __init__(self, width: int, max_entries: int = 4096):
        self.width = width
        self.max_entries = max_entries
        self.store = CausalNodeMemory(width, max_entries)
        self.stream = None
        self.last_epoch = -1
        self.live_tracks = set()
        self.retired_tracks = set()
        self.audit = []

    def begin(self, stream: str) -> None:
        self.store = CausalNodeMemory(self.width, self.max_entries)
        self.stream = str(stream)
        self.last_epoch = -1
        self.live_tracks = set()
        self.retired_tracks = set()
        self.audit = []

    def before(self, view, G, device):
        if self.stream is None:
            raise ContractError('MEMORY_WORLD_NOT_STARTED')
        if view['epoch'] != self.last_epoch + 1:
            raise ContractError('MEMORY_NONCONSECUTIVE_DECISION')
        live = {int(t['id']) for t in view['tracks']}
        if live & self.retired_tracks:
            raise ContractError('REUSED_RETIRED_TRACK_ID')
        retired = self.live_tracks - live
        self.retired_tracks.update(retired)
        # Track IDs are monotone within a world; a deleted track cannot return.
        removed = []
        present=set(G.node_ids)
        for key in list(self.store._entries):
            node = key[1]
            if node not in present or any(node == f'track/{i}' or
                   (node.startswith('joint/') and f'track/{i}' in node[6:].split('|')) or
                   node.endswith(f'/task=track/object={i}') for i in retired):
                removed.append(node)
                del self.store._entries[key]
        self.live_tracks = live
        mem = self.store.gather(self.stream, G.node_ids, now_s=view['decision_s'],
                               cycle_index=view['epoch'], device=device,
                               dtype=G.node_features.dtype)
        self.audit.append({'epoch': view['epoch'], 'retired_tracks': sorted(retired),
                           'removed_nodes': sorted(removed),
                           'history_nonzero_entries': int(mem.count_nonzero().item())})
        return mem

    def after(self, view, G, state):
        self.store.update(self.stream, G.node_ids, state, now_s=view['decision_s'],
                          cycle_index=view['epoch'])
        self.last_epoch = view['epoch']


    def snapshot(self):
        """Detached, weights_only-safe training continuation, not hidden truth."""
        return {'width':self.width,'max_entries':self.max_entries,'stream':self.stream,
                'last_epoch':self.last_epoch,'live_tracks':sorted(self.live_tracks),
                'retired_tracks':sorted(self.retired_tracks),'audit':self.audit,
                'entries':[{'stream':stream,'node':node,'timestamp_s':entry.timestamp_s,
                  'cycle_index':entry.cycle_index,'state':entry.state.detach().cpu().clone(),
                  'provenance_seen':sorted(entry.provenance_seen)}
                  for (stream,node),entry in sorted(self.store._entries.items())]}

    def restore(self,document):
        from iclr2027_graph_agent.memory import MemoryEntry
        if document['width']!=self.width or document['max_entries']!=self.max_entries:
            raise ContractError('MEMORY_SNAPSHOT_DIMENSION')
        self.begin(document['stream']);self.last_epoch=document['last_epoch']
        self.live_tracks=set(document['live_tracks']);self.retired_tracks=set(document['retired_tracks']);self.audit=document['audit']
        for row in document['entries']:
            if row['stream']!=self.stream or row['cycle_index']>self.last_epoch or row['state'].shape!=(self.width,):
                raise ContractError('MEMORY_SNAPSHOT_CAUSALITY')
            self.store._entries[row['stream'],row['node']]=MemoryEntry(row['timestamp_s'],row['cycle_index'],row['state'].detach().cpu(),frozenset(row['provenance_seen']))
