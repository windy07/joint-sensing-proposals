"""Focused correction checks. No historical suites, data or trained models run."""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
from itertools import combinations
import math,time
import numpy as np
import torch
from .common import *
from .stable_math import mixture_weights,conditional_position_variance,overlap_tail_refinement
from .contexts import finite_context,from_doc
from .native import DynamicWorld
from .risk import Scorer
from .search import run_policy,freeze_reference,audit
from .checks import declared_complementarity
from .analysis import paired_tail_upper
from .models import make_model,objective
from .dataset import label_context
from .graph_adapter import graph_context

def checks(config,out):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reports=[];score_calls=0
    def ok(name,**details):reports.append({'name':name,'status':'PASS',**details})
    def rejects(call):
        try:call()
        except Exception as exc:return type(exc).__name__+': '+str(exc)
        raise AssertionError('expected rejection')
    # Independent scalar identity: old subtractive formula becomes exactly zero.
    P=np.array([[1e16]]);H=np.ones((1,1));R=np.ones((1,1));k=np.linalg.solve(H@P@H.T+R,H@P[:,0])
    subtraction=float(P[0,0]-(H@P[:,0])@k);v=conditional_position_variance(P,H,R,k)
    assert subtraction==0. and v==1.
    ok('Joseph form avoids cancellation without jitter',subtractive_variance=subtraction,joseph_variance=v,exact_variance='10000000000000000/10000000000000001')
    w,lw=mixture_weights({'weights':[1.,0.],'log_weights':[0.,-1000.]})
    assert len(w)==2 and lw[1]==-1000. and np.isfinite(lw).all()
    rejects(lambda:mixture_weights({'weights':[1.,0.]}));rejects(lambda:mixture_weights({'weights':[.5,.5],'log_weights':[0.,-1000.]}))
    ok('positive source log branches retained and malformed bindings rejected',stored_log_weights=lw.tolist(),represented_weights=w.tolist())
    # Independent overlap inequality from posterior total variance, not GH copy.
    j,tail,err,te,calls=overlap_tail_refinement(np.array([.5,.5]),np.log([.5,.5]),1.,2.,.5,1.,1.)
    assert 0.<j<.25 and 0.<tail<1. and err<1e-6 and te<1e-6
    assert j<=.25*math.exp(-.5)+1e-9
    ok('scalar refinement and analytic overlap bound',J=j,tail=tail,error_estimate_m2=err,tail_error=te,evaluations=calls,formal_certificate=False)
    base=finite_context(2197999011,4,'ID',1);ctx=declared_complementarity(base)
    m=Meter(4,0);ss=Scorer(ctx,m);rr={a:ss.evaluate(a) for a in ((0,),(1,),(1,2),(1,3))};score_calls+=m.scores
    assert [rr[a]['risk_m2'] for a in rr]==[1.,2.,0.,2.]
    ok('complete law complementarity versus copied provenance',risk_m2={str(a):r['risk_m2'] for a,r in rr.items()})
    # Entire trajectory uses the source DFG. Inference never mutates that owner.
    world=DynamicWorld(2197999021,4,'spectrum',8,1);trajectory=[];first=None
    for cycle in range(8):
        cc=world.begin_cycle();first=first or cc;before=world.graph._model_digest();mm=Meter(1,0);s=Scorer(cc,mm,nodes=16)
        a=(cc.available()[0],) if cc.available() else ();value=s.evaluate(a);score_calls+=mm.scores
        assert before==world.graph._model_digest();trajectory.append(world.apply(cc,a))
        assert world._remaining_energy+world._spent_energy==Fraction(12) and world.resource>=0
    assert len(trajectory)==8 and all(a['positive_modes']==2 for a in trajectory)
    ok('eight-cycle source posterior isolation and exact resource conservation',epochs=[r['epoch'] for r in trajectory],remaining_J=str(world._remaining_energy),spent_J=str(world._spent_energy),native_operations=world.ops.used)
    d=first.doc();d['weights']=[1.,0.];d['log_weights']=[0.,-1000.];rare=from_doc(d);mm=Meter(1,0);row=Scorer(rare,mm,nodes=16).evaluate((rare.available()[0],));score_calls+=mm.scores
    assert math.isfinite(row['risk_m2']) and row.get('positive_mode_count')==2
    ok('source-shaped underflowed mixture scores without deleting a mode',risk_m2=row['risk_m2'])
    # Budgeted numerical failure does not erase other successfully verified actions.
    original=Scorer._finite
    def broken(self,a):
        if a:raise RuntimeError('deterministic injected after-entry candidate failure')
        return original(self,a)
    with patch.object(Scorer,'_finite',broken):partial=run_policy(ctx,'analytic_constructive',4,4,128,pool_limit=16)
    assert partial['status']=='PARTIAL_VERIFIED' and partial['selected']==[]
    assert partial['meter']['scores']==4 and partial['meter']['kernel_calls'] is None and len(partial['candidate_failures'])==3
    score_calls+=4;ok('candidate failure isolation with no refunds',commit=partial)
    # Capacity saturation is structural: opposite priorities cannot change K=2.
    class Hint:
        def __init__(self,sign):self.sign=sign
        def prepare(self,context,meter):self.m=meter
        def rank(self,actions):self.m.priority_candidates+=len(actions);return [self.sign*sum(a) for a in actions]
    aa=run_policy(ctx,'learned_test',2,6,128,learned=Hint(1),pool_limit=16)
    bb=run_policy(ctx,'learned_test',2,6,128,learned=Hint(-1),pool_limit=16)
    assert aa['output']==bb['output'] and aa['selected']==bb['selected'];score_calls+=aa['meter']['scores']+bb['meter']['scores']
    ok('two reserved slots saturate K=2 independently of learned priority',output=aa['output'],selected=aa['selected'])
    # Both orthogonal losses receive exactly the same full-risk rows.
    rec=label_context(ctx,config);score_calls+=rec['meter']['scores'];acts=[tuple(a) for a in rec['candidates']]
    g=graph_context(ctx,Meter(0,0));model=make_model('integrated_agent',config,config['model_seeds'][0]);model.eval()
    logs={}
    for arm in ('integrated_agent','no_assoc_aux','no_prov_aux','no_residual_priority'):
        loss,log,_,_=objective(model,g,acts,rec['scores'],rec['aux'],ctx,arm)
        assert torch.isfinite(loss);logs[arm]=log
    reg=[x['risk_regression'] for x in logs.values()];assert max(reg)-min(reg)==0.
    assert logs['no_assoc_aux']['association_weight']==0 and logs['no_prov_aux']['provenance_weight']==0
    assert logs['integrated_agent']['association_weight']==.2 and logs['no_assoc_aux']['provenance_weight']==.2
    ok('orthogonal auxiliaries preserve identical complete-risk supervision',loss_terms=logs)
    zero=paired_tail_upper([0.]*768)
    assert 0<zero['ci_high']<.02 and zero['mean']==0 and paired_tail_upper([])['mean'] is None
    ok('zero empirical excess retains nonzero population upper bound',report=zero)
    # Revalidate failed/empty qualification independently of the proposer.
    dd=ctx.doc();dd['threshold_m2']=-1.;empty=from_doc(dd);cc=run_policy(empty,'analytic_constructive',4,4,128,pool_limit=16)
    ref=freeze_reference(empty,[cc],128,128);ar=audit(empty,ref,[cc],128)
    assert ar['reports'][0]['Q_state']=='KNOWN_EMPTY_NUMERICAL' and ar['reports'][0]['coverage'] is None
    score_calls+=cc['meter']['scores']+ar['meter']['scores'];zero_commit=run_policy(ctx,'analytic_constructive',2,0,64,pool_limit=16)
    assert zero_commit['status']=='ABSTAINED' and zero_commit['meter']['scores']==0
    ok('independent audit empty qualification and zero-budget abstention')
    # Unit errors and content changes remain rejection, not normalizations.
    dd=first.doc();dd['units']['energy']='m2';rejects(lambda:from_doc(dd))
    dd=first.doc();dd['truth']=[1.];rejects(lambda:from_doc(dd))
    ok('causal and physical-unit rejection preserved')
    result={'status':'COMPLETE','groups':reports,'group_count':len(reports),'new_worlds':2,'source_dynamic_cycles':8,'paid_score_attempts':score_calls,'optimizer_updates':0,'seconds':time.perf_counter()-start,'source_digest':source_digest(),'contract_sha256':filehash(ROOT/'EXPERIMENT_CONTRACT.json')}
    atomic_json(out/'FINAL_CHECKS.json',result);return result
