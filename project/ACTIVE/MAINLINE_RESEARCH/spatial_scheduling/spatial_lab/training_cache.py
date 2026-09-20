"""Verified per-world tensors, bounded CPU preparation and one-world prefetch.

No physics or teacher search is rerun when an accepted full-graph export exists.
Tensor dtypes, semantic order, labels and training-world order are preserved.
"""
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
import gzip,hashlib,json,time
import torch
from .core import ContractError,Task,digest,read_json,write_json,atomic_save
from .binding import source_digest,FEATURE_BINDING,LABEL_BINDING
from .spatial_graph import graph_digest,FEATURES
from .feature_state import WIDTH,FEEDBACK_START,feedback_values
from .training_compatibility import accept_export
from iclr2027_graph_agent.tensorize import TensorInformationGraph

SCHEMA='ain.stage5.prepared.training.v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def restore_graph(document):
    values={}
    for f in fields(TensorInformationGraph):
        name=f.name;v=document[name]
        if name in ('node_ids','feature_names'):values[name]=tuple(v)
        else:
            dtype=torch.float32 if name in ('node_features','edge_features') else torch.long
            t=torch.tensor(v,dtype=dtype)
            if name=='node_features':t=t.reshape(-1,WIDTH)
            if name=='edge_features':t=t.reshape(-1,3)
            if name in ('edge_index','hyperedge_index'):t=t.reshape(2,-1)
            values[name]=t
    G=TensorInformationGraph(**values)
    if G.feature_names!=FEATURES or len(G.node_ids)!=len(G.node_features):
        raise ContractError('PREPARED_GRAPH_LAYOUT')
    return G

def graph_values(G):
    return {f.name:getattr(G,f.name) for f in fields(G)}

def graph_on_device(values,device):
    return TensorInformationGraph(**{k:v.to(device=device) if torch.is_tensor(v) else v for k,v in values.items()})

def restore_feedback(G,row):
    values=graph_values(G);values['node_features']=G.node_features.clone()
    F=TensorInformationGraph(**values);feedback=row.get('feedback_object')
    common=torch.as_tensor(feedback_values(feedback,row['view'],None),dtype=torch.float32)
    F.node_features[:,FEEDBACK_START:]=common
    for i,a in enumerate(row['actions']):
        node=int(G.action_node_index[i]);ident=Task(**a).semantic_id()
        if G.node_ids[node]!=ident:raise ContractError('PREPARED_ACTION_ORDER')
        F.node_features[node,FEEDBACK_START:]=torch.as_tensor(feedback_values(feedback,row['view'],ident),dtype=torch.float32)
    return F

def _prepare_one(args):
    root,entry=args;root=Path(root);raw=root/entry['case'];export=root/entry['full_graph_sequence']
    if not raw.resolve().is_relative_to((root/'data/cases').resolve()) or not export.resolve().is_relative_to((root/'data/sequences').resolve()):
        raise ContractError('CACHE_INPUT_OUTSIDE_DATA')
    if sha(raw)!=entry['case_sha256'] or sha(export)!=entry['sequence_sha256']:
        raise ContractError('CACHE_INPUT_BYTES_CHANGED')
    body=json.loads(gzip.decompress(export.read_bytes()))
    if body['content_digest']!=digest({k:v for k,v in body.items() if k!='content_digest'}):
        raise ContractError('EXPORT_CONTENT_DIGEST')
    accept_export(body['source_digest'])
    seq=body['sequence'];manifest=body['dataset_manifest']
    if (body['case_sha256']!=entry['case_sha256'] or seq['world_seed']!=entry['world_seed'] or len(manifest)!=1 or
        manifest[0]['file_sha256']!=entry['case_sha256'] or manifest[0]['world_seed']!=entry['world_seed'] or
        manifest[0]['contexts']!=len(seq['contexts'])):raise ContractError('EXPORT_CASE_BINDING')
    configs={}
    for epoch,row in enumerate(seq['contexts']):
        if row['view']['epoch']!=epoch:raise ContractError('EXPORT_CONTEXT_ORDER')
        G=restore_graph(row.pop('full_input_graph'))
        if graph_digest(G)!=row['graph_digest']:raise ContractError('RESTORED_GRAPH_DIGEST')
        row['prepared_graph']=graph_values(G)
        if row.get('feedback_object'):
            F=restore_feedback(G,row)
            if graph_digest(F)!=row['feedback_graph_digest']:raise ContractError('RESTORED_FEEDBACK_DIGEST')
            row['prepared_feedback_graph']=graph_values(F)
        cfg=row['policy_configuration'];configs[digest(cfg)]=cfg
    packed={'schema':SCHEMA,'sequence':seq,'dataset_manifest':manifest,'export_entry':entry,
            'prepared_source_digest':source_digest(),'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING}
    rel=f'data/prepared/{seq["world_seed"]}.pt';dest=root/rel
    atomic_save(dest,lambda f:torch.save(packed,f))
    return {'world_seed':seq['world_seed'],'path':rel,'sha256':sha(dest),'dataset_manifest':manifest,
            'configurations':configs,'contexts':len(seq['contexts'])}

def prepare_cache(root,workers=1):
    root=Path(root);export=read_json(root/'data/DATASET_MANIFEST.json');accept_export(export['source_digest'])
    if not export['worlds']:raise ContractError('EMPTY_PREPARED_DATA')
    identity={'schema':SCHEMA,'source_digest':source_digest(),'export_manifest_digest':digest(export),
              'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING}
    index_path=root/'data/PREPARED_INDEX.json'
    if index_path.exists():
        old=read_json(index_path)
        if old.get('identity')!=identity or old.get('content_digest')!=digest({k:v for k,v in old.items() if k!='content_digest'}):
            raise ContractError('PREPARED_INDEX_BINDING')
        if all(sha(root/e['path'])==e['sha256'] for e in old['worlds']):return old
        raise ContractError('PREPARED_CACHE_BYTES_CHANGED')
    started=time.monotonic();results=[]
    def progress(status):
        row={'status':status,'phase':'PREPARING_TENSORS','worlds_prepared':len(results),'worlds_total':len(export['worlds']),
             'elapsed_s':time.monotonic()-started,'updated_ns':time.time_ns(),'workers':workers,
             'actual_optimizer_updates':0,'actual_world_advances':0}
        write_json(root/'data/PREPARE_RECEIPT.json',row)
        print(json.dumps(row),flush=True)
    progress('RUNNING')
    args=[(str(root),entry) for entry in export['worlds']]
    try:
        if workers==1:
            for arg in args:results.append(_prepare_one(arg));progress('RUNNING')
        else:
            with ProcessPoolExecutor(max_workers=min(workers,len(args))) as pool:
                for result in pool.map(_prepare_one,args):results.append(result);progress('RUNNING')
        body={'identity':identity,'worlds':results};body['content_digest']=digest(body)
        write_json(index_path,body);progress('COMPLETE');return body
    except Exception:
        progress('FAILED');raise

class PreparedSequences:
    """One active world and one next world, never the full dataset in each trainer."""
    def __init__(self,root):
        self.root=Path(root);self.index=read_json(self.root/'data/PREPARED_INDEX.json')
        if self.index.get('content_digest')!=digest({k:v for k,v in self.index.items() if k!='content_digest'}):
            raise ContractError('PREPARED_INDEX_CONTENT')
        identity=self.index['identity']
        if identity['source_digest']!=source_digest() or identity['feature_binding']!=FEATURE_BINDING or identity['label_binding']!=LABEL_BINDING:
            raise ContractError('PREPARED_INDEX_SOURCE')
        self.entries=self.index['worlds'];self.pool=ThreadPoolExecutor(max_workers=1)
        self.current=None;self.current_index=None;self.pending=None;self.pending_index=None
    def __len__(self):return len(self.entries)
    def _load(self,i):
        entry=self.entries[i];path=self.root/entry['path']
        if not path.resolve().is_relative_to((self.root/'data/prepared').resolve()) or sha(path)!=entry['sha256']:
            raise ContractError('PREPARED_TENSORS_CONTENT')
        body=torch.load(path,map_location='cpu',weights_only=True)
        if body['schema']!=SCHEMA or body['prepared_source_digest']!=source_digest() or body['dataset_manifest']!=entry['dataset_manifest']:
            raise ContractError('PREPARED_TENSORS_BINDING')
        return body['sequence']
    def __getitem__(self,i):
        if i>=len(self) or i<0:raise IndexError(i)
        if i!=self.current_index:
            self.current=self.pending.result() if self.pending_index==i else self._load(i)
            self.current_index=i;self.pending_index=(i+1)%len(self)
            self.pending=self.pool.submit(self._load,self.pending_index)
        return self.current
    @property
    def manifest(self):return [m for e in self.entries for m in e['dataset_manifest']]
    @property
    def configurations(self):return {k:v for e in self.entries for k,v in e['configurations'].items()}
    @property
    def world_seeds(self):return [e['world_seed'] for e in self.entries]
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
