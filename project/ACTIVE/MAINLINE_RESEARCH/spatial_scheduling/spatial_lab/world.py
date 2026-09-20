"""Uniform measurement-level world for ALL targets and reception roles.

No primary-target special path; truth exists here and in offline evaluation only.
Returned packets contain no target IDs, ideal waveform, true frequency or true
state. The same physical functions define control prediction and simulation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
from .core import *
from .physics import *
from .actions import visits,resource,device

class World:
 def __init__(self,seed,config):
  self.seed=int(seed);self.config=config;self.sensors=sensor_list(config);self.epoch=-1
  self.F,self.Q=transition(config['dt_s'],config['acceleration_psd_m2_s3'])
  self.bias=np.array([18.,-22.,math.radians(.25),math.radians(-.2),math.radians(.3),math.radians(-.3),8.,-12.,2.,-3.,4.,-5.])
  self.bias+=rng(seed,'bias').normal(size=NBIAS)*BIAS_INITIAL_SD*.05
  templates=[(0,[-950.,-350.,53.,18.]),(8,[850.,-260.,-48.,22.]),(18,[-220.,1450.,18.,-49.])]
  if config.get('diverse_target_templates',False):
   random=rng(seed,'public_prior_target_templates')
   cells=cell_list(config);indices=random.permutation(len(cells))[:min(3,config['max_tracks'])]
   templates=[]
   for i,j in enumerate(indices):
    position=cells[j]+random.uniform(-.8,.8,2)*config.get('search_cell_halfwidth_m',350.)
    angle=random.uniform(-math.pi,math.pi);speed=random.uniform(15.,65.)
    templates.append((i*8,[*position,speed*math.cos(angle),speed*math.sin(angle)]))
  if config['scenario']=='single':templates=templates[:1]
  if config.get('birth_process')=='capped_geometric' and config['scenario']!='single':
   clock=-1;generated=[]
   for i,(_,x) in enumerate(templates):
    clock+=int(rng(seed,'birth_process',i).geometric(config.get('birth_probability_per_cycle',.08)))
    generated.append((clock,x))
   templates=generated
  birth_epochs=config.get('target_birth_epochs')
  if birth_epochs is not None:templates=[(b,templates[i][1]) for i,b in enumerate(birth_epochs)]
  self.templates={i:{'birth':b,'initial':np.asarray(x)+rng(seed,'initial',i).normal(size=4)*[30,30,2,2]} for i,(b,x) in enumerate(templates)}
  self.targets={};self.all_targets={};self.pending=[];self.link_free_s=0.;self.processing_free_s=[0.]*4;self.evaluation_events=[]
  self.audit={'generated_true_detections':0,'generated_clutter':0,'unresolved_merges':0,'generated_ghosts':0,
              'world_advances':0,'delay_capture_rejections':0,'frequency_capture_rejections':0,'reference_failures':0,'generated_packets':0,'layer':'one_measurement_level_for_all_targets'}
 def step(self):
  from .work_budget import charge_world
  charge_world(self.seed,self.epoch+1,self.config)
  self.epoch+=1;k=self.epoch;self.audit['world_advances']+=1
  for i,t in list(self.targets.items()):
   if k>=self.config.get('target_death_epochs',{}).get(str(i),10**9):
    del self.targets[i];continue
   t['state']=self.F@t['state']+np.linalg.cholesky(self.Q)@rng(self.seed,'process',k,i).normal(size=4)
   if self.config.get('maneuver',False) and i==0 and 16<=k<21:t['state'][3]+=2.
  for i,t in self.templates.items():
   if t['birth']==k:
    self.targets[i]={'state':t['initial'].copy(),'birth':k,'rcs':float(rng(self.seed,'rcs',i).uniform(.7,1.3))}
    self.all_targets[i]=self.targets[i]
  self.bias+=rng(self.seed,'bias_walk',k).normal(size=NBIAS)*BIAS_WALK_SD
  return self.targets
 def observe(self,schedule,view):
  if view['epoch']!=self.epoch:raise ContractError('WORLD_EPOCH')
  legal(schedule,view,self.config)
  packets=[];e=self.epoch;dt=self.config['dt_s']
  for action in schedule:
   for visit in visits(action,self.config):
    task=visit['task'];s=self.sensors[task.node];d=device(self.config,s.id);root=digest(['numeric_capture',self.seed,e,task.to_dict()])
    rr=rng(self.seed,'receiver',e,s.id,action.semantic_id(),visit['index'])
    end=e*dt+visit['end_s'];work=resource(task,self.config);proc=work['processing_ops']/d['processing_ops_s'];bits=work['communication_bytes']*8
    transmit_start=end+proc;arrival=transmit_start+bits/d['communication_bps']
    delivered=bool(rr.random()>=self.config.get('packet_loss_probability',0.))
    if e in self.config.get('force_delayed_epochs',[]):arrival+=dt
    if e in self.config.get('force_loss_epochs',[]):delivered=False
    p={'schema':'ain.spatial.measurement_packet.v4','node':s.id,'source':s.name,'root':root,
       'acquire_epoch':e,'acquire_s':e*dt+visit['mid_s'],'acquisition_end_s':end,'arrival_s':arrival,
       'processing_s':proc,'transmission_start_s':transmit_start,'transmission_s':bits/d['communication_bps'],
       'delivered':delivered,'kind':task.kind,'task':task.to_dict(),'parent_action_id':action.execution_id(),
       'visit_index':visit['index'],'boresight_rad':task.bearing_rad,'field_of_view_rad':task.beam_rad,
       'measurement_units':measurement_units(task),'returns':[],
       'receiver_layer':'DECLARED_MEASUREMENT_LEVEL_ALL_TARGETS',
       'geometry_implementation':'receiver_revisions.passive_geometry' if task.role!='monostatic' else 'spatial_lab.physics',
       'reference_id':task.reference_id,'reference_acquired':True,'reference_capture_root':None,
       'reference_detected_count':0,'reference_channel_semantics':'actual two receiver records' if task.role=='direct' else 'own signal or acquired external reference',
       'communication_bytes':work['communication_bytes']}
    ref=reference_response(task,e,self.config)
    if task.role=='direct':
     p['reference_acquired']=availability(e,task.reference_node,self.config)
     p['reference_capture_root']=digest([root,'reference_receiver_capture',task.reference_node]) if p['reference_acquired'] else None
    if task.role=='bistatic' or task.kind=='calibrate':
     p['reference_acquired']=bool(ref['available'] and (task.kind=='calibrate' or rr.random()<ref['pd']))
     p['reference_capture_root']=digest([root,'measured_reference']) if p['reference_acquired'] else None
    if not p['reference_acquired']:
     self.audit['reference_failures']+=1;p['receiver_status']='NO_REFERENCE';packets.append(p);continue
    if task.kind=='calibrate':
     mu,H,R,pd=calibration_model(task,self.bias,NBIAS,e,self.config)
     if rr.random()<pd:
      p['returns']=[(mu+np.linalg.cholesky(R)@rr.normal(size=3)).tolist()]
     p['receiver_status']='CAPTURED' if p['returns'] else 'NOT_DETECTED';packets.append(p);continue
    common=None
    for tid,target in self.targets.items():
     x=target['state'].copy();x[:2]+=x[2:]*visit['mid_s']
     if not covers(x[:2],task,view,s) or blocked(x[:2],s,self.config):continue
     # Direct emission is an environmental event, common to the two receivers.
     if task.role=='direct' and rng(self.seed,'emission',e,tid).random()>=self.config.get('emission_probability',.7):continue
     if task.role=='direct' and not availability(e,task.reference_node,self.config):continue
     pd=detection_probability(x[:2],s,task,e,self.config,target['rcs'])
     # Emission and the reference channel were actually acquired above;
     # joint detection here includes the independent per-emitter reference SNR.
     if task.role=='direct':pd/=max(self.config.get('emission_probability',.7),1e-12)
     if task.role=='bistatic':pd/=max(ref['pd'],1e-12)
     if rr.random()>=min(1.,pd):continue
     mu=measurement(target['state'],self.bias,s,task,visit['mid_s']);R,Rc=noise_parts(x[:2],s,task,e,self.config,rcs=target['rcs'])
     if common is None:common=np.sqrt(np.diag(Rc))*rr.normal(size=len(Rc))
     z=mu+np.linalg.cholesky(R)@rr.normal(size=len(mu))+common
     # Delay capture and slow-time Nyquist are receiver support, not a truth clamp.
     if task.role!='monostatic' and task.mode=='full':
      if abs(z[0])>d['passive_delay_window_s'] or abs(z[1])>=.5/task.pri_s:
       reason='delay_capture_rejections' if abs(z[0])>d['passive_delay_window_s'] else 'frequency_capture_rejections'
       self.audit[reason]+=1;self.evaluation_events.append({'target_id':tid,'root':root,'epoch':e,'kind':'not_captured','reason':reason});continue
     for a in angle_indices(task):z[a]=float(wrap(z[a]))
     p['returns'].append(z.tolist());self.audit['generated_true_detections']+=1
     self.evaluation_events.append({'target_id':tid,'root':root,'epoch':e,'acquire_s':p['acquire_s'],'kind':'detected_before_transport'})
     if task.role=='direct':p['reference_detected_count']+=1
    if self.config.get('ghost_probability_per_visit',0.)>0 and rr.random()<self.config['ghost_probability_per_visit']:
     phantom=np.array([250.,250.,0.,0.]);phantom[:2]+=phantom[2:]*visit['mid_s']
     if covers(phantom[:2],task,view,s):
      try:
       R=noise_cov(phantom[:2],s,task,e,self.config);z=measurement(phantom,self.bias,s,task)+np.linalg.cholesky(R)@rr.normal(size=len(R))
       p['returns'].append(z.tolist());self.audit['generated_ghosts']+=1
      except ContractError:pass
    # Resolution-based unresolved return retention: retain one measured cell,
    # not a noiseless centroid or a truth-resolved labelled observation.
    bins={}
    for z in p['returns']:
     steps=([C0/(2*task.bandwidth_Hz),task.beam_rad,C0/(2*task.carrier_Hz*task.dwell_s)] if task.role=='monostatic'
            else [1/task.bandwidth_Hz,1/task.dwell_s,task.beam_rad,*([task.beam_rad] if task.role=='direct' else [])])
     if task.mode=='aoa':steps=[task.beam_rad]
     key=tuple(int(np.floor(v/h)) for v,h in zip(z,steps))
     if key in bins:self.audit['unresolved_merges']+=1
     else:bins[key]=z
    p['returns']=list(bins.values())
    # One false-alarm definition: expected count = searched resolution cells * Pfa.
    cells=resolution_cell_count(task,self.config)
    p['search_cells']=cells;p['clutter_mean']=cells*self.config.get('pfa_cell',1e-5)
    for _ in range(int(rr.poisson(p['clutter_mean']))):
     angle=task.bearing_rad+rr.uniform(-task.beam_rad/2,task.beam_rad/2)
     if task.role=='monostatic':z=[rr.uniform(C0*task.pulse_width_s/2,s.max_range_m),angle,rr.uniform(-C0/(4*task.carrier_Hz*task.pri_s),C0/(4*task.carrier_Hz*task.pri_s))]
     elif task.mode=='aoa':z=[angle]
     else:z=[rr.uniform(-d['passive_delay_window_s'],d['passive_delay_window_s']),rr.uniform(-.5/task.pri_s,.5/task.pri_s),angle,*([rr.uniform(-math.pi,math.pi)] if task.role=='direct' else [])]
     p['returns'].append(z);self.audit['generated_clutter']+=1
    rr.shuffle(p['returns']);p['reference_detected_count']=len(p['returns']) if task.role=='direct' else None
    p['receiver_status']='CAPTURED' if p['returns'] else 'EMPTY_CAPTURE';packets.append(p)
  # Per-node processor queue. Capture can overlap processing, but a processor
  # never handles two datagrams simultaneously. All targets share this queue.
  for p in sorted(packets,key=lambda p:(p['acquisition_end_s'],p['node'],p['visit_index'])):
   n=p['node'];start=max(p['acquisition_end_s'],self.processing_free_s[n])
   p['processing_start_s']=start;p['processing_end_s']=start+p['processing_s']
   self.processing_free_s[n]=p['processing_end_s']
  # Shared transport FIFO is ordered by completed receiver processing, not
  # by action-list position. All datagrams, including lost ones, consume airtime.
  for p in sorted(packets,key=lambda p:(p['processing_end_s'],p['node'],p['visit_index'])):
   start=max(p['processing_end_s'],self.link_free_s)
   p['transmission_start_s']=start;p['arrival_s']=start+p['transmission_s'];self.link_free_s=p['arrival_s']
   if e in self.config.get('force_delayed_epochs',[]):p['arrival_s']+=dt
  self.audit['generated_packets']+=len(packets)
  if self.config.get('copied_packets',False) and packets:
   duplicate=deepcopy(packets[0]);duplicate['transport_copy']='relay';packets.append(duplicate)
  return packets
 def queue(self,packets):self.pending.extend(deepcopy(packets))
 def arrived(self,cutoff_s):
  due=[p for p in self.pending if p['arrival_s']<=cutoff_s]
  self.pending=[p for p in self.pending if p['arrival_s']>cutoff_s]
  return sorted(due,key=lambda p:(p['arrival_s'],p['root']))
