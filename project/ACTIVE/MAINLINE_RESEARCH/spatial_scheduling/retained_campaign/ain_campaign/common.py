from __future__ import annotations
import hashlib,json,os,time,math,tempfile
from pathlib import Path
from dataclasses import dataclass,asdict,field
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=('ID','spectrum','availability','topology','association','provenance')
ARMS=('original_agent','integrated_agent','deep_sets','set_transformer')
CONTROLS=('analytic_constructive','beam_block','random')
SCHEMA='ain.v11.1'
class ContractError(ValueError):pass
class BudgetExhausted(RuntimeError):pass

def plain(v):
 import numpy as np
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if hasattr(v,'__dataclass_fields__'):return plain(asdict(v))
 if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [plain(x) for x in v]
 if isinstance(v,Path):return str(v)
 return v

def canonical(v):return json.dumps(plain(v),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(canonical(v));f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def read_json(p):
 def obj(xs):
  d={}
  for k,v in xs:
   if k in d:raise ContractError('DUPLICATE_JSON_KEY')
   d[k]=v
  return d
 return json.loads(Path(p).read_text(encoding='utf-8-sig'),object_pairs_hook=obj,parse_constant=lambda v:(_ for _ in ()).throw(ContractError('NONFINITE_JSON')))
def source_digest():
 return digest({p.relative_to(ROOT).as_posix():filehash(p) for folder in ('ain_campaign','vendor') for p in sorted((ROOT/folder).rglob('*.py'))}|{'run.py':filehash(ROOT/'run.py')})
def seed_for(namespace,*parts):return int(hashlib.sha256(canonical([namespace,*parts])).hexdigest()[:15],16)
def rng(namespace,*parts):
 import numpy as np
 return np.random.default_rng(seed_for(namespace,*parts))
def require_finite(v,name):
 import numpy as np
 if not np.isfinite(v).all():raise ContractError('NONFINITE_'+name)
def seal(body):
 b=dict(body);b.pop('digest',None);return {**b,'digest':digest(b)}
def unseal(v):
 b=dict(v);s=b.pop('digest',None)
 if digest(b)!=s:raise ContractError('STALE_CONTENT_BINDING')
 return b

def completed(path,binding):
 """A receipt alone is insufficient: validate every committed output byte."""
 p=Path(path)
 if not p.exists():return False
 r=read_json(p)
 if r.get('binding')!=binding:raise ContractError('INCOMPATIBLE_RESUME:'+str(p))
 if r.get('status')!='COMPLETE':return False
 for item in r.get('outputs',[]):
  f=p.parent/item['path']
  if not f.is_file() or filehash(f)!=item['sha256']:raise ContractError('CORRUPT_COMPLETED_OUTPUT:'+str(f))
 return True

def receipt(path,binding,outputs,status='COMPLETE',**extra):
 p=Path(path);r={'schema':SCHEMA,'status':status,'binding':binding,'outputs':[{'path':os.path.relpath(Path(x),p.parent).replace('\\','/'),'sha256':filehash(x)} for x in outputs],**extra}
 atomic_json(p,r);return r

@dataclass
class Meter:
 score_cap:int
 construct_cap:int
 scores:int=0
 completed_scores:int=0
 atomic_scores:int=0
 cache_hits:int=0
 construction:int=0
 rejected:int=0
 descriptor_entries:int=0
 priority_candidates:int=0
 linear_solves:int=0
 matrix_factorizations:int=0
 pseudoinverses:int=0
 affinity_products:int=0
 outcome_rows:int=0
 quadrature_nodes:int=0
 phase16_models:int=0
 phase16_conditional_calls:int=0
 reference_failures:int=0
 score_seconds:float=0.
 prepare_seconds:float=0.
 priority_seconds:float=0.
 model_seconds:float=0.
 actual_graph_forks:int=0
 actual_graph_solves:int=0
 kernel_calls:int|None=0
 def construct(self):
  if self.construction>=self.construct_cap:raise BudgetExhausted('CONSTRUCTION_CAP')
  self.construction+=1
 def charge(self):
  if self.scores>=self.score_cap:raise BudgetExhausted('SCORE_CAP')
  self.scores+=1
 def report(self):return plain(self)
