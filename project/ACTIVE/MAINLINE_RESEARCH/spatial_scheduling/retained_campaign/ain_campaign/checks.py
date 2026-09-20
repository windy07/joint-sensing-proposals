"""Focused changed-path checks only; no historical suite imports or replays."""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import patch
import tempfile,subprocess,sys,time,os,copy
import numpy as np
import torch
from .common import *
from .contexts import Context,finite_context,from_doc
from .native import DynamicWorld
from .risk import Scorer
from .graph_adapter import graph_context,FEATURE_BINDING
from .models import make_model,schedule_scores,config_for
from .search import run_policy,freeze_reference,audit
from .processes import run_child
from iclr2027_graph_agent.checkpoint import CheckpointMetadata,save_atomic_checkpoint,load_verified_checkpoint
from v10_finite import model as old


def fail(fn):
 try:fn()
 except Exception as e:return type(e).__name__+':'+str(e)
 raise AssertionError('expected structured failure')

def declared_complementarity(base):
 d=base.doc();b=copy.deepcopy(d['finite_law']);b.pop('digest');b.update(length_m='1',decisions_m=['-1','1'],resource_J='10',max_size=4,lambda_tail_m2='0',lambda_energy_m2_per_J='0',rho_m='1',threshold_m2='1/2')
 for z in b['states']:z['x_m']='-1' if z['target_index']<2 else '1';z['p']='1/16'
 for i,c in enumerate(b['channels']):
  c.update(available=True,energy_J='0',acquire_s='0',arrival_s='1/2',provenance=['direct','alias','beacon','alias'][i]);q=[]
  for z in b['states']:
   x=int(z['x_m']);a=2*z['association']-1
   q.append(str(F(3,4) if x==1 else F(1,4)) if i==0 else str(int((x*a if i in (1,3) else a)>0)))
  c['p_plus_given_z']=q
 b=old.seal(b);law=old.load_context(b);d.update(finite_law=b,length_m=1.,resource_J=10.,max_size=4,lambda_tail_m2=0.,lambda_energy=0.,rho_m=1.,threshold_m2=.5)
 for i,c in enumerate(d['channels']):c.update(q=[float(q) for q in law.channels[i].q],root=law.channels[i].provenance,available=True,energy_J=0.,acquire_s=0.,arrival_s=.5)
 return from_doc(d)

def checks(config,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();results=[];generated=0;score_attempts=0
 def ok(name,**kw):results.append({'name':name,'status':'PASS',**kw})
 base=finite_context(1117999001,4,'ID',1);generated+=1;ctx=declared_complementarity(base)
 # Distinct model assembled from a reviewed schema, not a copied risk table.
 m=Meter(16,0);s=Scorer(ctx,m);r={a:s.evaluate(a)['risk_m2'] for a in [(0,),(1,),(1,2),(1,3)]};score_attempts+=m.scores
 assert r[(0,)]==1. and r[(1,)]==2. and r[(1,2)]==0. and r[(1,3)]==2.
 # Independent Fraction enumeration includes all 16 rows, no scorer functions.
 d=ctx.doc();law=d['finite_law'];exact={}
 for action in r:
  ids=ctx.groups(action);total=F(0)
  for y in product((0,1),repeat=len(ids)):
   losses=[]
   for decision in (F(-1),F(1)):
    value=F(0)
    for z,st in enumerate(law['states']):
     mass=F(st['p'])
     for j,i in enumerate(ids):q=F(law['channels'][i]['p_plus_given_z'][z]);mass*=q if y[j] else 1-q
     value+=mass*(F(st['x_m'])-decision)**2
    losses.append(value)
   total+=min(losses)
  exact[str(action)]=str(total);assert float(total)==r[action]
 ok('exact complementary witness and copied provenance',risks=r if False else {str(k):v for k,v in r.items()},exact=exact)
 # Unit-coordinate change is not a physical spectrum intervention.
 dd=ctx.doc();bb=copy.deepcopy(dd['finite_law']);bb.pop('digest')
 for st in bb['states']:st['x_m']=str(2*F(st['x_m']))
 bb['length_m']='2';bb['decisions_m']=['-2','2'];bb['rho_m']='2';bb['threshold_m2']='2';dd.update(finite_law=old.seal(bb),length_m=2.,rho_m=2.,threshold_m2=2.)
 scaled=from_doc(dd);ms=Meter(1,0);rs=Scorer(scaled,ms).evaluate((0,));score_attempts+=1;assert rs['risk_m2']==4*r[(0,)];ok('unit scale risk covariance')
 # Compatible scalar and shared/chunk path match within declared criterion and RNG unchanged.
 legal=[a for k in range(3) for a in combinations(ctx.available(),k)];aa=Meter(len(legal),0);bbm=Meter(len(legal),0);state=np.random.get_state();before=digest([state[0],state[1].tolist(),state[2:]])
 arows=list(Scorer(ctx,aa,'compatibility').evaluate_many(legal,1));brows=list(Scorer(ctx,bbm,'production').evaluate_many(legal,4));after=np.random.get_state();assert before==digest([after[0],after[1].tolist(),after[2:]])
 diff=max(abs(a['risk_m2']-b['risk_m2']) for a,b in zip(arows,brows));assert diff<=1e-12;score_attempts+=aa.scores+bbm.scores
 ok('scalar versus chunk score/order/RNG equivalence',max_difference=diff,compatibility_label_hash=digest(arows),production_label_hash=digest(brows),hash_identity_claim=False)
 dd=ctx.doc();dd['truth']=[0];assert 'CAUSAL_LEAK' in fail(lambda:from_doc(dd));dd=ctx.doc();dd['units']['energy']='m2';fail(lambda:from_doc(dd));dd=ctx.doc();dd['channels'][0]['arrival_s']=2.;late=from_doc(dd);fail(lambda:late.legal((0,)));ok('causal/units/late-arrival rejection')
 w=DynamicWorld(1117999011,4,'ID',8,1);native=[];generated+=1
 first=None
 for _ in range(8):
  c=w.begin_cycle();generated+=1
  if first is None:first=c
  mm=Meter(1,0);sc=Scorer(c,mm,nodes=16);a=(c.available()[0],) if c.available() else ();row=sc.evaluate(a);score_attempts+=mm.scores;fp=w.graph._model_digest();assert fp==c.doc()['posterior_digest'];g1=graph_context(c,Meter(0,0));
  assert w.graph._model_digest()==fp
  native.append(w.apply(c,a))
 assert all(x['positive_modes']==2 for x in native);assert len(w.registered_actions)>0;assert len(w.history)==8
 ok('eight source DFG prediction/acquisition/update cycles',epochs=[x['epoch'] for x in native],mode_counts=[x['positive_modes'] for x in native],native_operations=w.ops.used)
 # Independent scalar projection of the ACTUAL source snapshot: compatible
 # target/bias coordinates, not velocity relabeled as metres.
 mm=Meter(1,0);ss=Scorer(first,mm,nodes=32);ref=ss.scalar_reference((0,));row=ss.evaluate((0,));mu=np.asarray(first.doc()['means']);y=float(np.asarray(first.doc()['weights'])@ref.eta);post=ref.conditional(y)
 weights=post['weights'];means=post['means'][:,0];v=float(weights@(post['covariances'][:,0,0]+(means-float(weights@means))**2));assert v>0 and row['mse_m2']>0;score_attempts+=1
 # Conditional mean from our own odds/linear update formula at the same y.
 logq=np.log(ss.w)-.5*(np.log(2*np.pi*ref.s)+(y-ref.eta)**2/ref.s);p=np.exp(logq-logq.max());p/=p.sum();est=float(p@(mu[:,0]+ref.k*(y-ref.eta)));assert abs(est-float(weights@means))<1e-9
 ok('Phase16 scalar projection agreement',posterior_mean_error=abs(est-float(weights@means)),phase16_meter=mm.report())
 # Tensorization sees only arrived summaries; changing private simulator truth
 # after snapshot cannot alter the already frozen policy input.
 frozen=first.digest;g=graph_context(first,Meter(0,0));w.truth+=np.array([10000.,0.,0.]);g2=graph_context(first,Meter(0,0));assert torch.equal(g.node_features,g2.node_features) and frozen==first.digest
 dd=first.doc();dd['arrived_history']=[{'epoch':dd['epoch'],'estimate_m':0,'posterior_variance_m2':1,'posterior_entropy':0,'energy_J':0}];fail(lambda:graph_context(from_doc(dd),Meter(0,0)));ok('no realized-truth or future-memory feature access')
 cm=run_policy(ctx,'analytic_constructive',2,4,64,pool_limit=16,nodes=16);assert len(cm['output'])<=2 and cm['meter']['scores']<=4
 emptydoc=ctx.doc();emptydoc['threshold_m2']=-1.;empty=from_doc(emptydoc);ce=run_policy(empty,'analytic_constructive',2,4,64,pool_limit=16);rr=freeze_reference(empty,[ce],32);ae=audit(empty,rr,[ce],32);assert ae['reports'][0]['Q_state']=='KNOWN_EMPTY_NUMERICAL' and ae['reports'][0]['coverage'] is None
 zero=run_policy(ctx,'analytic_constructive',2,0,64,pool_limit=16);assert zero['status']=='ABSTAINED' and zero['meter']['scores']==0;fail(lambda:freeze_reference(empty,[cm],32));score_attempts+=cm['meter']['scores']+ce['meter']['scores']+ae['meter']['scores'];ok('K/B, empty qualification, exhausted budget, cross-context reference')
 # A reference exception after entry preserves cost and unknown call count.
 mm=Meter(1,0);ss=Scorer(ctx,mm)
 with patch.object(ss,'_finite',side_effect=RuntimeError('injected_after_entry')):fail(lambda:ss.evaluate((0,)))
 assert mm.scores==1 and mm.kernel_calls is None;score_attempts+=1;ok('post-entry no-refund nullable failure accounting')
 # UNKNOWN must propagate when all point risks are threshold-ambiguous.
 cm=run_policy(ctx,'analytic_constructive',2,4,64,pool_limit=16);rr=freeze_reference(ctx,[cm],32);orig=Scorer.evaluate
 def ambiguous(self,action):
  row=orig(self,action);row['risk_m2']=ctx.doc()['threshold_m2'];row['numeric_error_estimate_m2']=1.;return row
 with patch.object(Scorer,'evaluate',ambiguous):ar=audit(ctx,rr,[cm],32)
 assert ar['reports'][0]['Q_state']=='UNKNOWN' and ar['reports'][0]['coverage'] is None and ar['reports'][0]['unconditional_miss'] is None
 score_attempts+=cm['meter']['scores']+ar['meter']['scores'];ok('ambiguous-reference tri-valued reporting')
 # Existing checkpoint writer/loader and original model class, no synthetic tensors.
 model=make_model('integrated_agent',config,config['model_seeds'][0]);model.eval();gg=graph_context(ctx,Meter(0,0));s0,_=schedule_scores(model,gg,[(),(0,),(1,2)]);cp=out/'roundtrip.pt';meta=CheckpointMetadata('v11.check',0,0,digest(asdict(config_for('integrated_agent',config))),FEATURE_BINDING,'check','check',source_digest(),'','','','',{})
 h=save_atomic_checkpoint(cp,model=model,optimizer=None,scheduler=None,metadata=meta);payload,_=load_verified_checkpoint(cp,expected={'source_tree_sha256':source_digest()},expected_file_sha256=h);clone=make_model('integrated_agent',config,config['model_seeds'][0]+1);clone.load_state_dict(payload['model_state']);clone.eval();s1,_=schedule_scores(clone,gg,[(),(0,),(1,2)]);assert torch.equal(s0,s1);ok('retained Agent checkpoint and structured-score identity',maximum_score_difference=float((s0-s1).abs().max()),backbone_class=type(model).__name__)
 # Atomic writes and task binding are actual public boundaries.
 target=out/'threaded.json'
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda i:atomic_json(target,{'i':i}),range(16)))
 assert read_json(target)['i'] in range(16);rp=out/'bound.receipt.json';receipt(rp,'binding',[target]);assert completed(rp,'binding');fail(lambda:completed(rp,'other'));atomic_json(target,{'changed':True});fail(lambda:completed(rp,'binding'));ok('concurrent atomic writes and resume content binding')
 # One success/failure/timeout stub each, including an inherited subprocess.
 command=[sys.executable,'-c'];a=run_child(command+['print("stub success")'],out/'stub_success.log',5);b=run_child(command+['import sys;sys.exit(7)'],out/'stub_failure.log',5)
 childpid=out/'child.pid';code='import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);pathlib.Path('+repr(str(childpid))+').write_text(str(p.pid));time.sleep(30)'
 c=run_child(command+[code],out/'stub_timeout.log',.5);assert a['exit_status']==0 and b['exit_status']==7 and c['exit_status']==124
 if os.name!='nt' and childpid.exists():
  pid=int(childpid.read_text());stat=Path(f'/proc/{pid}/stat');assert not stat.exists() or stat.read_text().split()[2]=='Z'
 ok('process success/failure/timeout tree cleanup',platform=os.name,windows_jobobject_executed=os.name=='nt')
 report={'status':'COMPLETE','groups':results,'group_count':len(results),'new_generated_context_calls':generated,'additional_declaration_variants':5,'paid_score_attempts':score_attempts,'optimizer_updates':0,'seconds':time.perf_counter()-start,'source_digest':source_digest()};atomic_json(out/'CHECKS.json',report);return report
