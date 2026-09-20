from pathlib import Path
import sys, json, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from ain_campaign.common import read_json, Meter, atomic_json, source_digest
from ain_campaign.contexts import finite_context
from ain_campaign.checks import declared_complementarity
from ain_campaign.dataset import label_context
from ain_campaign.graph_adapter import graph_context
from ain_campaign.models import make_model,objective

torch.set_num_threads(1)
cfg=read_json(ROOT/'configs/smoke.json')
ctx=declared_complementarity(finite_context(2197999057,4,'ID',1))
rec=label_context(ctx,cfg)
# Declared adverse anchor: aliased singleton, not an invented target label.
rec['aux']['protected_action']=[1]
acts=[tuple(a) for a in rec['candidates']]
model=make_model('integrated_agent',cfg,2197001);model.eval()
g=graph_context(ctx,Meter(0,0))
loss,logs,(scores,_),_=objective(model,g,acts,rec['scores'],rec['aux'],ctx,'integrated_agent')
base,_,_,_=objective(model,g,acts,rec['scores'],rec['aux'],ctx,'no_residual_priority')
assert logs['improving_candidate_count']>0 and logs['residual_witness_loss']>0
assert abs(float((loss-base).detach())-.1*logs['residual_witness_loss'])<1e-5
params=[p for p in model.parameters() if p.requires_grad]
grad=torch.autograd.grad(loss-base,params,allow_unused=True)
norm=sum(float(v.square().sum()) for v in grad if v is not None)**.5
assert norm>0
r={'status':'COMPLETE','checks':1,'description':'actual residual objective has nonzero gradient when verified candidates beat a declared adverse anchor','complete_risk_labels':True,'artificial_score_labels':False,'improving_candidates':logs['improving_candidate_count'],'residual_loss':logs['residual_witness_loss'],'parameter_gradient_norm':norm,'paid_scores':rec['meter']['scores'],'optimizer_updates':0,'source_digest':source_digest()}
atomic_json(ROOT/'evidence/RESIDUAL_GRADIENT_CHECK.json',r)
print(json.dumps(r,indent=2))
