"""One verified tensor shard per world; causal sequences are never shuffled within a world."""
from pathlib import Path
import hashlib, math
import torch
from .core import ContractError, Task, digest, read_json, write_json, atomic_save
from .binding import source_digest, FEATURE_BINDING, LABEL_BINDING
from .stage6_model import numeric_features,pack_graphs

SCHEMA='ain.stage6.compact.sequence.v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare_world(raw_path,dest):
    from .sequence_training import load_sequences
    from .training_cache import restore_graph, graph_values, restore_feedback
    sequences,manifest=load_sequences([raw_path]);seq=sequences[0];rows=[];configs={}
    for row in seq['contexts']:
        G=restore_graph(row['full_input_graph']);labels=row['paid_labels'];view=row['view'];cfg=row['policy_configuration']
        groups=[x['action_indices'] for x in labels]
        features=[numeric_features([Task(**a) for a in x['schedule']],view,cfg) for x in labels]
        executed=next((i for i,x in enumerate(labels) if digest(x['schedule'])==digest(row['executed'])),None)
        if executed is None:raise ContractError('EXECUTED_GROUP_NOT_IN_PAID_LABELS')
        costs=[float(x['proxy_m2']) for x in labels]
        if not all(math.isfinite(x) and x>=0 for x in costs):raise ContractError('INVALID_PROXY_LABEL')
        quantiles=torch.tensor(costs,dtype=torch.float64).quantile(torch.tensor([.25,.75],dtype=torch.float64))
        temperature=.5*max(float(quantiles[1]-quantiles[0]),100.)
        rows.append({'graph':graph_values(G),'adjacency':pack_graphs([G])['adj'][0],
          'feedback_graph':graph_values(restore_feedback(G,row)) if row.get('feedback_object') else None,'temperature':temperature,
          'groups':groups,'group_features':features,'costs':costs,'executed':executed,
          'numeric_labels':row['numeric_labels'],'future':row['realized_future_target_m2'],'epoch':view['epoch']})
        configs[digest(cfg)]=cfg
    body={'schema':SCHEMA,'source_digest':source_digest(),'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,
          'world_seed':seq['world_seed'],'contexts':rows,'manifest':manifest,'configurations':configs}
    atomic_save(dest,lambda f:torch.save(body,f))
    return {'world_seed':seq['world_seed'],'path':str(Path(dest).name),'sha256':sha(dest),
            'manifest':manifest,'contexts':len(rows),'configurations':configs}

def seal_index(root,entries):
    root=Path(root)
    if len({e['world_seed'] for e in entries})!=len(entries):raise ContractError('DUPLICATE_PREPARED_WORLD')
    body={'schema':SCHEMA,'source_digest':source_digest(),'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,'worlds':entries}
    body['content_digest']=digest(body);write_json(root/'INDEX.json',body);return body

class Worlds(torch.utils.data.Dataset):
    def __init__(self,root):
        self.root=Path(root);self.index=read_json(self.root/'INDEX.json');d=self.index
        if (d.get('schema')!=SCHEMA or d.get('source_digest')!=source_digest() or
            d.get('feature_binding')!=FEATURE_BINDING or d.get('label_binding')!=LABEL_BINDING or
            d.get('content_digest')!=digest({k:v for k,v in d.items() if k!='content_digest'})):
            raise ContractError('PREPARED_INDEX_IDENTITY')
        self.entries=d['worlds'];self.checked=set()
        if not self.entries:raise ContractError('EMPTY_DATASET')
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        entry=self.entries[i];path=self.root/entry['path']
        if not path.resolve().is_relative_to(self.root.resolve()):raise ContractError('CACHE_PATH')
        if i not in self.checked:
            if sha(path)!=entry['sha256']:raise ContractError('PREPARED_CONTENT_CHANGED')
            self.checked.add(i)
        d=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        if d['source_digest']!=self.index['source_digest'] or d['manifest']!=entry['manifest'] or d['world_seed']!=entry['world_seed']:
            raise ContractError('PREPARED_WORLD_BINDING')
        return d
    @property
    def manifest(self):return [r for e in self.entries for r in e['manifest']]
    @property
    def world_seeds(self):return [e['world_seed'] for e in self.entries]
    @property
    def configurations(self):return {k:v for e in self.entries for k,v in e['configurations'].items()}

def collate_worlds(items):return items

def targets(rows,device):
    b=len(rows);c=max(len(r['groups']) for r in rows);k=max(1,max(len(g) for r in rows for g in r['groups']))
    a=max(1,max(len(r['graph']['action_node_index']) for r in rows))
    ix=torch.zeros(b,c,k,dtype=torch.long);nf=torch.zeros(b,c,k,16);members=torch.zeros(b,c,k)
    valid=torch.zeros(b,c,dtype=torch.bool);cost=torch.zeros(b,c);controls=torch.zeros(b,a,8);cm=torch.zeros_like(controls)
    atom=torch.zeros(b,a);am=torch.zeros(b,a);executed=[];future=[]
    for i,r in enumerate(rows):
        count=len(r['groups']);valid[i,:count]=True;cost[i,:count]=torch.tensor(r['costs'])
        for j,(group,features) in enumerate(zip(r['groups'],r['group_features'])):
            n=len(group)
            if n:ix[i,j,:n]=torch.tensor(group);nf[i,j,:n]=torch.tensor(features);members[i,j,:n]=1
        for v in r['numeric_labels']:
            controls[i,v['index']]=torch.tensor(v['target']);cm[i,v['index']]=torch.tensor(v['mask'])
        for j in range(a):
            present=[v for v,g in zip(r['costs'],r['groups']) if j in g]
            if present:atom[i,j]=math.log1p(min(present)/10000.);am[i,j]=1
        executed.append(r['executed']);future.append(math.log1p(r['future']/10000.))
    return {name:value.to(device,non_blocking=True) for name,value in
       dict(indices=ix,features=nf,members=members,valid=valid,cost=cost,controls=controls,control_mask=cm,
            atom=atom,atom_mask=am,executed=torch.tensor(executed),future=torch.tensor(future),temperature=torch.tensor([r['temperature'] for r in rows])).items()}
