"""Explicit causal mapping into the retained InformationGraph/tensorizer."""
from __future__ import annotations
import math,time
import numpy as np
from iclr2027_graph_agent.schema import GraphNode,GraphEdge,GraphHyperedge,DynamicInformationGraph,NodeKind,EdgeKind
from iclr2027_graph_agent.tensorize import tensorize_graph
from v10_finite import model as v10
from v10_finite.search import Features
from .contexts import Context
from .common import *
FEATURES=tuple(f'f{i:03d}' for i in range(96))
ROLES={'<unk>':0,'channel':1,'receiver':2,'position':3,'hypothesis':4,'shared_source':5,'capacity':6,'arrived':7}
EDGES={e.value:i for i,e in enumerate(EdgeKind)}
HYPERS={'provenance':0,'association':1,'schedule_resource':2}
FEATURE_BINDING=digest({'schema':'v11.causal_features.v1','features':FEATURES,'roles':ROLES,'edges':EDGES,'hyperedges':HYPERS,'position_normalization':'context.length_m','no_label_statistics':True})

def graph_context(ctx:Context,meter:Meter,device='cpu'):
 t=time.perf_counter();d=ctx.doc();cs=d['channels'];L=d['length_m'];L2=L*L;epoch=d['epoch'];now=d['decision_s'];nodes=[];edges=[];hyper=[]
 def add(ident,kind,role,values):
  ar=np.zeros(96,np.float32);v=np.asarray(values,dtype=float).ravel()
  if v.size>96 or not np.isfinite(v).all():raise ContractError('GRAPH_FEATURES')
  ar[:len(v)]=v;nodes.append(GraphNode(ident,kind,role,now,features={f:float(x) for f,x in zip(FEATURES,ar)},units={f:'1' for f in FEATURES}))
 def edge(a,b,kind,dep=0.):
  edges.append(GraphEdge(a,b,kind,timestamp_s=now,features={'age_s':0.,'delay_s':0.,'dependence':float(dep)}))
 if ctx.track=='finite':
  old=v10.load_context(d['finite_law']);om=v10.Meter(0,0);f=Features(old,om,affinity=False);channel_features=f.nodes
  meter.descriptor_entries+=om.preprocessing_entries
  p=np.asarray([float(x) for x in old.prior]);x=np.asarray([float(x) for x in old.x]);center=float(p@x);var=float(p@(x-center)**2)
  for j,(pp,xx,aa,ss) in enumerate(zip(p,x,old.association,old.source)):
   add(f'hypothesis/{j:02d}',NodeKind.NUISANCE_VARIABLE,'hypothesis',[pp,(xx-center)/L,aa,ss])
  prior=[center/L,var/L2,float(sum(pp*aa for pp,aa in zip(p,old.association))),float(sum(pp*ss for pp,ss in zip(p,old.source)))]
 else:
  p=np.asarray(d['weights']);mu=np.asarray(d['means']);P=np.asarray(d['covariances']);center=float(p@mu[:,0]);var=float(p@(P[:,0,0]+(mu[:,0]-center)**2));prior=[center/L,var/L2,*p.tolist()]
  for j,pp in enumerate(p):
   mm=mu[j].copy();mm[0]-=center
   add(f'hypothesis/{j:02d}',NodeKind.NUISANCE_VARIABLE,'hypothesis',[pp,*list(mm/L),*list((P[j]/L2).ravel())])
  cf=[]
  for c in cs:
   vv=np.zeros(64);h=np.asarray(c['H']);pred=mu@h;vv[:len(p)]=p;vv[4:4+len(p)]=(pred-center*h[0])/L;vv[8:11]=h;vv[12]=(h@P[0]@h)/L2;vv[13]=c['noise_sd_m']/L;vv[14]=c['common_loading']/L;vv[16:25]=(P[0]/L2).ravel();vv[32:35]=(p@mu-np.array([center,0,0]))/L;vv[48]=1;cf.append(vv)
  channel_features=np.asarray(cf)
 add('track',NodeKind.TRACK_VARIABLE,'position',prior)
 add('resource',NodeKind.RESOURCE,'capacity',[d['resource_J']/4,d['max_size']/4,d['threshold_m2']/L2,d['lambda_energy']/L2,d['rho_m']/L,d['terminal_s']-now,epoch/8])
 roots=sorted(set(c['root'] for c in cs));root_ids={g:f'root/{i:03d}' for i,g in enumerate(roots)}
 for g in roots:
  members=[i for i,c in enumerate(cs) if c['root']==g]
  add(root_ids[g],NodeKind.PROVENANCE_ROOT,'shared_source',[len(members)/len(cs),*np.asarray(channel_features[members]).mean(0).tolist()])
 for i,c in enumerate(cs):
  vals=np.zeros(96);vals[:64]=channel_features[i];group=[j for j,x in enumerate(cs) if x['root']==c['root']]
  vals[64:76]=[float(c['available']),c['energy_J']/max(1,d['resource_J']),c['acquire_s']-now,c['arrival_s']-now,len(group)/len(cs),math.log1p(c['carrier_Hz']/1e9),math.log1p(c['bandwidth_Hz']/1e6),d['max_size']/4,d['resource_J']/4,epoch/8,float(ctx.track=='dynamic'),var/L2]
  vals[80:88]=np.mean(channel_features[group,:8],axis=0)
  add(f'action/{i:03d}',NodeKind.ACTION,'channel',vals);add(f'sensor/{i:03d}',NodeKind.SENSOR,'receiver',vals)
  edge(f'sensor/{i:03d}',f'action/{i:03d}',EdgeKind.OWNS_ACTION);edge(f'action/{i:03d}','track',EdgeKind.OBSERVES);edge('track',f'action/{i:03d}',EdgeKind.CONDITIONS)
  edge(f'action/{i:03d}','resource',EdgeKind.CONSUMES_RESOURCE);edge(root_ids[c['root']],f'action/{i:03d}',EdgeKind.PROVENANCE,len(group)-1);edge(f'action/{i:03d}',root_ids[c['root']],EdgeKind.PROVENANCE,len(group)-1)
 for g in roots:
  members=tuple([root_ids[g]]+[f'action/{i:03d}' for i,c in enumerate(cs) if c['root']==g])
  hyper.append(GraphHyperedge('provenance/'+g,'provenance',members,now,root_ids[g]))
 hs=tuple(n.node_id for n in nodes if n.kind==NodeKind.NUISANCE_VARIABLE)
 hyper.append(GraphHyperedge('association','association',('track',*hs),now))
 for i,msg in enumerate(d.get('arrived_history',[])):
  if msg['epoch']>=epoch:raise ContractError('FUTURE_MEMORY')
  ident=f'message/{msg["epoch"]:03d}'
  add(ident,NodeKind.MESSAGE,'arrived',[(msg['estimate_m']-center)/L,msg['posterior_variance_m2']/L2,msg['posterior_entropy'],msg['energy_J'],(epoch-msg['epoch'])/8]);edge(ident,'track',EdgeKind.ARRIVED_MESSAGE)
 graph=DynamicInformationGraph('g:'+ctx.digest,d['world_id'],epoch,now,tuple(nodes),tuple(edges),tuple(hyper),system_posterior_ref=d.get('posterior_digest',ctx.digest),metadata={'adapter':'V11','context_digest':ctx.digest,'source_scope':d['origin']})
 tg=tensorize_graph(graph,feature_names=FEATURES,role_vocabulary=ROLES,edge_vocabulary=EDGES,hyperedge_vocabulary=HYPERS,device=device)
 meter.descriptor_entries+=len(nodes)*96+3*len(edges);meter.prepare_seconds+=time.perf_counter()-t
 return tg
