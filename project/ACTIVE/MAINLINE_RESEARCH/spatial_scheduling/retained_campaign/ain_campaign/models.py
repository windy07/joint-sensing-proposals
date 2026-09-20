"""Retained B1 graph/backbones, joint set decoder, memory and checkpoint format."""
from __future__ import annotations
from dataclasses import asdict
import math,time
import numpy as np
import torch
import torch.nn.functional as F
from iclr2027_graph_agent.baselines import make_graph_agent
from iclr2027_graph_agent.model import GraphAgentConfig
from iclr2027_graph_agent.training import JointScheduleBatchTargets,joint_schedule_training_loss,joint_schedule_logits,joint_schedule_transition_alignment_loss
from iclr2027_graph_agent.memory import CausalNodeMemory
from .common import *
from .graph_adapter import graph_context,FEATURES,FEATURE_BINDING

REP={'original_agent':'proposed','integrated_agent':'proposed','no_assoc_supervision':'proposed','no_assoc_aux':'proposed','no_prov_aux':'proposed','no_aux':'proposed','no_residual_priority':'proposed','deep_sets':'deep_sets','set_transformer':'set_transformer'}
def config_for(arm,config):
 m=config['model'];h=m['widths'][REP[arm]]
 return GraphAgentConfig(input_dim=len(FEATURES),hidden_dim=h,num_layers=m['layers'],num_heads=m['heads'],memory_dim=h,dropout=m['dropout'],joint_decoder='low_rank_set',joint_decoder_rank=m['rank'])
def make_model(arm,config,seed,device='cpu'):
 torch.manual_seed(int(seed));np.random.seed(int(seed)%2**32)
 cfg=config_for(arm,config);model=make_graph_agent(REP[arm],cfg).to(device)
 return model

def schedule_scores(model,g,actions,previous_memory=None):
 out=model(g,previous_memory)
 logits={i:out.candidate_score[j] for j,i in enumerate(range(len(g.action_node_index)))}
 embeddings={i:out.node_embedding[g.action_node_index[j]] for j,i in enumerate(range(len(g.action_node_index)))}
 scores=joint_schedule_logits(logits,actions,action_embeddings=embeddings,decoder=model.joint_schedule_decoder)
 return scores,out

class LearnedPriority:
 def __init__(self,model,episode):self.model=model;self.episode=episode;self.memory=CausalNodeMemory(model.cfg.memory_dim);self.ctx_digest=None
 def prepare(self,ctx,meter):
  if self.ctx_digest==ctx.digest:return
  t=time.perf_counter();d=ctx.doc();device=next(self.model.parameters()).device;g=graph_context(ctx,meter,str(device));mem=self.memory.gather(self.episode,g.node_ids,now_s=d['decision_s'],cycle_index=d['epoch'],device=device,dtype=g.node_features.dtype)
  self.model.eval()
  with torch.no_grad():
   out=self.model(g,mem)
  self.memory.update(self.episode,g.node_ids,out.next_memory,now_s=d['decision_s'],cycle_index=d['epoch']);self.out=out;self.g=g;self.ctx_digest=ctx.digest;self.meter=meter
  meter.model_seconds+=time.perf_counter()-t
 def rank(self,actions):
  t=time.perf_counter();out=self.out;g=self.g
  with torch.no_grad():
   logits={i:out.candidate_score[i] for i in range(len(g.action_node_index))};emb={i:out.node_embedding[g.action_node_index[i]] for i in range(len(g.action_node_index))}
   scores=joint_schedule_logits(logits,actions,action_embeddings=emb,decoder=self.model.joint_schedule_decoder)
  ans=(-scores).detach().cpu().tolist();self.meter.model_seconds+=time.perf_counter()-t;self.meter.priority_candidates+=len(actions);return ans

def objective(model,g,actions,rows,aux,ctx,arm,previous=None,memory=None):
 scores,out=schedule_scores(model,g,actions,memory)
 L2=ctx.doc()['length_m']**2;risks=torch.tensor([r['risk_m2']/L2 for r in rows],dtype=scores.dtype,device=scores.device)
 if arm=='no_assoc_supervision':risks=torch.tensor([r['moment_proxy_m2']/L2 for r in rows],dtype=scores.dtype,device=scores.device)
 target=JointScheduleBatchTargets(tuple(actions),risks<=ctx.doc()['threshold_m2']/L2,torch.ones_like(risks,dtype=torch.bool),risks,4)
 base,terms=joint_schedule_training_loss(scores,target,exact_loss_temperature=.15)
 regression=F.smooth_l1_loss(scores,-torch.log1p(risks));loss=base+.2*regression
 logs={k:float(v.detach()) for k,v in terms.items()};logs['risk_regression']=float(regression.detach())
 # Complete task-risk labels exist for all arms; original keeps its disabled auxiliaries.
 if arm in ('integrated_agent','deep_sets','set_transformer','no_assoc_aux','no_prov_aux','no_residual_priority'):
  idx=torch.tensor(aux['indices'],dtype=torch.long,device=scores.device)
  if len(idx):
   y=torch.tensor(aux['association'],device=scores.device,dtype=scores.dtype);p=torch.tensor(aux['novelty'],device=scores.device,dtype=scores.dtype)
   la=F.binary_cross_entropy_with_logits(out.association_risk_logit[idx],y);lp=F.binary_cross_entropy_with_logits(out.provenance_novelty_logit[idx],p)
   wa=0. if arm=='no_assoc_aux' else .2;wp=0. if arm=='no_prov_aux' else .2
   loss=loss+wa*la+wp*lp;logs.update(association_loss=float(la.detach()),provenance_loss=float(lp.detach()),association_weight=wa,provenance_weight=wp)
 if arm in ('integrated_agent','deep_sets','set_transformer','no_assoc_aux','no_prov_aux','no_aux') and aux.get('protected_action') is not None:
  anchor=tuple(aux['protected_action']);idxa=next((i for i,a in enumerate(actions) if tuple(a)==anchor),None)
  if idxa is not None:
   eligible=torch.tensor([bool(a) and tuple(a)!=anchor for a in actions],device=scores.device,dtype=torch.bool)
   improving=eligible & (risks < risks[idxa]-1e-6)
   if bool(improving.any()):
    residual=torch.logsumexp(scores[eligible],0)-torch.logsumexp(scores[improving],0)
    loss=loss+.1*residual;logs['residual_witness_loss']=float(residual.detach());logs['improving_candidate_count']=int(improving.sum())
   else:logs['residual_witness_loss']=0.;logs['improving_candidate_count']=0
 if previous is not None:
  t,tt=joint_schedule_transition_alignment_loss(scores,target,previous[0],previous[1],exact_loss_temperature=.15)
  loss=loss+.1*t;logs['temporal_alignment']=float(t.detach())
 return loss,logs,(scores,target),out.next_memory
