"""Metadata-only split/collision check. It never constructs a world or opens outcomes."""
from .common import *
def validate_seed_plan(config):
 history=read_json(ROOT/'HISTORICAL_SEED_EXCLUSIONS.json');old=set(history['explicit_integer_exclusions']);ranges=history['conservative_ranges'];sets={}
 ns=config['seed_namespace']
 for split in ('train','development'):
  sets[split]=[seed_for(ns+':'+split+':'+('finite' if i%2==0 else 'dynamic')+':ID',i) for i in range(config['size'][split])]
 for track in ('finite','dynamic'):
  for family in config['families']:sets['final:'+track+':'+family]=[seed_for(ns+':final:'+track+':'+family,i) for i in range(config['size']['final_per_family'])]
 flat=[x for v in sets.values() for x in v]
 if len(flat)!=len(set(flat)) or set(flat)&old or set(config['model_seeds'])&old:raise ContractError('SEED_COLLISION')
 if any(r['lower']<=x<=r['upper'] for x in flat for r in ranges):raise ContractError('HISTORICAL_SEED_RANGE')
 return {'status':'COMPLETE','worlds':len(flat),'union_hash':digest(sorted(flat)),'split_counts':{k:len(v) for k,v in sets.items()},'historical_manifest_sha256':filehash(ROOT/'HISTORICAL_SEED_EXCLUSIONS.json'),'metadata_only':True}
