"""Eight-cycle declared affine radar receiver on the actual Phase15 joint DFG.

The law is a range/clock-bias development approximation, not legacy R2 physics.
The sole simulation posterior is DynamicFactorGraph; scoring never publishes it.
"""
from __future__ import annotations
from dataclasses import asdict
import math,time
from fractions import Fraction
import numpy as np
from wideband_info_geometry.physics_v58.dynamic_factor_graph import DynamicFactorGraph,DynamicVariable,VariableKey,VariableKind
from wideband_info_geometry.physics_v58.joint_association import JointOperationBudget
from wideband_info_geometry.physics_v58.schema import RadarAction
from wideband_info_geometry.operational_runtime.joint_return_set import FrozenAffineReturnLaw,JointReturnSetPayload
from wideband_info_geometry.operational_runtime.joint_receipt import JointContextRegistry,JointReceiptQueue,issue_joint_transport_event,commit_joint_batch
from .common import *
from .contexts import Context,from_doc

class DynamicWorld:
 def __init__(self,seed:int,n:int=8,family='ID',horizon=8,index=1):
  self.seed=int(seed);self.n=n;self.family=family;self.horizon=horizon;self.index=index;self.epoch=0;self.history=[];self.arrived=[];self.resource=12.;self.energy_used=0.;self.setup_seconds=0.
  t=time.perf_counter();rr=rng('V11_DYN_LAW',seed)
  self.length=float(rr.choice([60.,100.,150.]));L=self.length
  self.dt=1.;self.mean=np.array([2000.+rr.uniform(-200,200),rr.uniform(2,12),0.])
  # Complete positive covariance in (range, radial velocity, shared delay bias).
  sd=np.array([L,.08*L,.8*L]);corr=np.array([[1,0,.35],[0,1,0],[.35,0,1]])
  self.P=sd[:,None]*corr*sd[None,:]
  self.F=np.array([[1,1,0],[0,1,0],[0,0,1]],float)
  self.Q=np.diag((np.array([.025,.012,.03])*L)**2)
  self.truth=self.mean+np.linalg.cholesky(self.P)@rng('V11_DYN_INITIAL',seed).normal(size=3)
  self.key=VariableKey(VariableKind.TARGET,'range_velocity_bias',0)
  self.graph=DynamicFactorGraph(fixed_lag_steps=horizon+2,retained_history_dimension_cap=3*(horizon+2))
  self.graph.add_variable(DynamicVariable(self.key,3,self.mean,self.P,('m','m/s','m'),anchored=True))
  self.ops=JointOperationBudget({k:200000 for k in JointOperationBudget.NAMES})
  self.graph.enable_joint_returns(existence_probability=1.,branch_cap=2,acquisition_cap=horizon+4,recipient_id='fusion',budget=self.ops)
  self.registry=JointContextRegistry(max_contexts=(horizon+2)*16,max_return_count=2)
  self.queue=JointReceiptQueue('fusion',self.registry,max_pending=32,max_consumed=64)
  self.base_channels=[]
  roots=max(4,n-n//5)
  for i in range(n):
   j=i if i<roots else i-roots
   kind=('direct','mixed','bias','mixed')[j%4]
   h=([1.,0.,0.] if kind=='direct' else [0.,0.,1.] if kind=='bias' else [1.,0.,(-1. if j%2 else 1.)])
   sd_noise=(.24 if kind=='direct' else .08 if kind=='bias' else .10)*L
   self.base_channels.append({'id':f'port-{i:02d}','root':f'rx-{j:02d}','H':h,'noise_sd_m':sd_noise,'common_loading':(-1. if j%2 else 1.)*.025*L,'energy_J':float(rr.choice([.4,.6,.8])),'available':True,'carrier_Hz':3e9,'bandwidth_Hz':20e6,'kind':kind})
  if family=='provenance':
   for i in range(n//2,n):
    source=self.base_channels[i%(n//3)];energy=self.base_channels[i]['energy_J'];ident=self.base_channels[i]['id'];self.base_channels[i]={**source,'energy_J':energy,'id':ident}
  if family=='spectrum':
   for c in self.base_channels:
    old_bandwidth=c['bandwidth_Hz'];c['carrier_Hz']=9e9;c['bandwidth_Hz']=5e6;c['noise_sd_m']*=old_bandwidth/c['bandwidth_Hz'];c['common_loading']*=2.
    c['noise_scaling']='fixed-SNR delay-error standard deviation proportional to reciprocal bandwidth'
    c['common_drift_scaling']='separate declared twofold calibration-drift intervention'
    c['carrier_effect']='range-only affine law; no Doppler term is inferred from carrier metadata'
  self._remaining_energy=Fraction('12');self._spent_energy=Fraction(0)
  self.packet_records=[];self.registered_actions={}
  # An actual unlabelled two-return calibration creates the association mixture.
  # Clutter is centered at the causal prior-predictive range, not at truth.
  # A zero clutter mean would put it ~2000 m from the target and erase the
  # intended association ambiguity before any policy action is selected.
  rc=(2.5 if family=='association' else .8)*L
  calibration_H=np.array([[1.,0.,-1.]])
  law=self._law(action_id=1000000,H=calibration_H,R=np.array([[.04*L*L]]),epoch=0,rho=(0.,1.),Rc=np.array([[rc*rc]]),clutter_offset=calibration_H@self.mean)
  self.registry.register(law,action_id=law.action_id,observation_epoch=0)
  payload=law.sample(rng('V11_DYN_CALIBRATION',seed),self.truth,True,provenance_id=f'calibration:{seed}')
  self._deliver(law,payload)
  self.setup_seconds=time.perf_counter()-t
  self.initial_graph_digest=self.graph._model_digest()

 def _law(self,*,action_id,H,R,epoch,rho=(1.,),Rc=None,clutter_offset=None):
  m=len(H)
  offset=(0.,)*m if clutter_offset is None else tuple(float(x) for x in clutter_offset)
  return FrozenAffineReturnLaw(action_id,epoch,(self.key.token(),),(3,),(('m','m/s','m'),),tuple(f'range_{j}' for j in range(m)),('m',)*m,tuple(map(tuple,H)),(0.,)*m,tuple(map(tuple,R)),tuple(map(tuple,np.zeros_like(H))),offset,tuple(map(tuple,R if Rc is None else Rc)),1.,tuple(rho))

 def _deliver(self,law,payload):
  event=issue_joint_transport_event(registry=self.registry,capture_id=payload.provenance_id,action_id=law.action_id,observation_epoch=law.observation_epoch,recipient_id='fusion',arrival_epoch=law.observation_epoch,delivered=True,law_digest=law.digest)
  self.queue.enqueue(event,payload);batch=self.queue.prepare(epoch=self.epoch)
  commit_joint_batch(self.queue,self.graph,batch)
  self.packet_records.append({'epoch':self.epoch,'action_id':law.action_id,'law_digest':law.digest,'payload_digest':payload.digest,'return_count':len(payload.measurements),'arrival_epoch':self.epoch})

 def begin_cycle(self):
  if self.epoch>=self.horizon:raise ContractError('HORIZON')
  old=self.key;self.epoch+=1
  self.truth=self.F@self.truth+np.linalg.cholesky(self.Q)@rng('V11_DYN_PROCESS',self.seed,self.epoch).normal(size=3)
  self.key=VariableKey(VariableKind.TARGET,'range_velocity_bias',self.epoch)
  new=DynamicVariable(self.key,3,np.zeros(3),np.eye(3),('m','m/s','m'),prior_enabled=False)
  self.graph.joint_predict(new_variable=new,previous_keys=(old,),transition_matrix=self.F,offset=np.zeros(3),covariance=self.Q,provenance_id=f'process:{self.seed}:{self.epoch}')
  # Keep the source's complete retained-history semantics; no posterior pruning.
  self.graph.joint_cut(eliminate_epoch=self.epoch-1,boundary_keys=(self.key,))
  return self.context()

 def context(self):
  snapshot=self.graph.joint_snapshot(capture_keys=(self.key.token(),),capture_dimensions=(3,),capture_units=(('m','m/s','m'),))
  self._snapshot=snapshot
  channels=[];rr=rng('V11_DYN_AVAIL',self.seed,self.epoch)
  for c0 in self.base_channels:
   c=dict(c0);c['acquire_s']=self.epoch+.05;c['arrival_s']=self.epoch+.35
   c['available']=bool(rr.random()>(.45 if self.family=='availability' else .04))
   if self.family=='availability' and rr.random()<.15:c['arrival_s']=self.epoch+1.2
   channels.append(c)
  cap=min(self.resource,2.4 if self.family!='topology' else 1.0)
  if self.family in ('topology','availability') and self.index%8==0:cap=0.
  body={'schema':'ain.v11.context.v1','track':'dynamic','origin':'DECLARED_AFFINE_RADAR_ON_PHASE15_DFG','truth_access':False,'context_id':f'V11-D-{self.seed}-{self.epoch}','world_id':f'V11-D-{self.seed}','world_seed':self.seed,'family':self.family,'epoch':self.epoch,'decision_s':float(self.epoch),'terminal_s':self.epoch+.75,'units':{'position':'m','velocity':'m/s','time':'s','energy':'J','risk':'m2'},'coordinate_units':['m','m/s','m'],'length_m':self.length,'resource_J':cap,'remaining_world_energy_J':self.resource,'max_size':2 if self.family=='topology' else 4,'threshold_m2':(-.01 if cap==0 else .12)*self.length**2,'lambda_energy':.01*self.length**2,'lambda_tail_m2':0.,'rho_m':2*self.length,'channels':channels,'weights':[m['weight'] for m in snapshot['modes']],'log_weights':[float(m['log_weight']) for m in snapshot['modes']],'remaining_energy_exact_J':str(self._remaining_energy),'means':[m['mean'].tolist() for m in snapshot['modes']],'covariances':[m['covariance'].tolist() for m in snapshot['modes']],'mode_ids':[m['mode_id'] for m in snapshot['modes']],'posterior_authority':'DynamicFactorGraph','posterior_digest':snapshot['graph_digest'],'admitted_factor_digest':snapshot['admitted_factor_digest'],'acquisitions':[list(x) for x in snapshot['acquisitions']],'arrived_history':list(self.history[-3:]),'positive_modes_retained':len(snapshot['modes']),'native_mapping':'RadarAction + registered FrozenAffineReturnLaw; declared affine receiver, not R2 extractor'}
  if len(snapshot['modes'])!=2:raise ContractError('POSITIVE_ASSOCIATION_BRANCH_LOSS')
  self.current=from_doc(body);return self.current

 def apply(self,ctx,action):
  if ctx.digest!=self.current.digest or ctx.doc()['posterior_digest']!=self.graph._model_digest():raise ContractError('STALE_RUNTIME_CONTEXT')
  ctx.legal(action);d=ctx.doc();ids=ctx.groups(action);start=time.perf_counter()
  debit=sum((Fraction(str(d['channels'][i]['energy_J'])) for i in action),Fraction(0))
  if debit>self._remaining_energy:raise ContractError('EXACT_WORLD_ENERGY_EXHAUSTED')
  if ids:
   ch=[d['channels'][i] for i in ids];H=np.asarray([c['H'] for c in ch]);sd=np.asarray([c['noise_sd_m'] for c in ch]);loading=np.asarray([c['common_loading'] for c in ch]);R=np.diag(sd**2)+np.outer(loading,loading)
   aid=1+sum(1<<i for i in action)
   native=RadarAction(action_id=aid,kind='active_bistatic',waveform_id='affine-radar',tx_node_id='tx',rx_node_ids=tuple(d['channels'][i]['id'] for i in action),resource_slot=self.epoch,energy_cost_j=float(debit),allowed=True,metadata={'schema':'v11.registered_joint_schedule','context_digest':ctx.digest,'ordered_ports':list(action),'deduplicated_roots':[c['root'] for c in ch],'declared_receiver':'affine-range-bias'})
   native.validate({'tx',*[c['id'] for c in d['channels']]},{'affine-radar'})
   law=self._law(action_id=aid,H=H,R=R,epoch=self.epoch)
   self.registry.register(law,action_id=aid,observation_epoch=self.epoch)
   self.registered_actions[(self.epoch,aid)]=(native,law)
   # Counter-based exogenous streams are shared across methods, not observations.
   common=float(rng('V11_DYN_COMMON_INNOVATION',self.seed,self.epoch).normal())
   noise=np.asarray([c['noise_sd_m']*float(rng('V11_DYN_PRIVATE_INNOVATION',self.seed,self.epoch,c['root']).normal())+c['common_loading']*common for c in ch])
   observation=H@self.truth+noise
   payload=JointReturnSetPayload(aid,self.epoch,'capture:'+digest([self.seed,self.epoch,[c['root'] for c in ch]]),law.digest,(tuple(observation),))
   self._deliver(law,payload);en=native.energy_cost_j;self._remaining_energy-=debit;self._spent_energy+=debit;self.resource=float(self._remaining_energy);self.energy_used=float(self._spent_energy)
  else:en=0.
  snapshot=self.graph.joint_snapshot(capture_keys=(self.key.token(),),capture_dimensions=(3,),capture_units=(('m','m/s','m'),))
  if len(snapshot['modes'])!=2:raise ContractError('POSITIVE_ASSOCIATION_BRANCH_LOSS')
  w=np.array([m['weight'] for m in snapshot['modes']]);mu=np.array([m['mean'] for m in snapshot['modes']]);est=float(w@mu[:,0]);var=float(sum(m['weight']*(m['covariance'][0,0]+(m['mean'][0]-est)**2) for m in snapshot['modes']))
  err=(est-self.truth[0])**2
  row={'epoch':self.epoch,'estimate_m':est,'squared_error_m2':float(err),'large_error':bool(math.sqrt(err)>2*self.length),'energy_J':en,'normalized_loss':float(err/self.length**2+.01*en),'posterior_variance_m2':var,'posterior_entropy':float(-sum(p*math.log(p) for p in w if p>0)),'positive_modes':len(w),'registered_action_count':len(ids),'update_seconds':time.perf_counter()-start,'posterior_digest':snapshot['graph_digest']}
  # Only arrived posterior summaries become future features. Real error never does.
  self.history.append({k:row[k] for k in ('epoch','estimate_m','posterior_variance_m2','posterior_entropy','energy_J')})
  return row
