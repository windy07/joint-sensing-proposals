"""Explicit bounded local checks; no historical auto-discovery or training.

Default: one thread, zero optimizer steps; finite matrix checks, one two-context
backward per backbone, and <=19 new world cycles. Previously accumulated local
cycles are charged in LOCAL_BUDGET.json, so the total cannot exceed 24.
"""
from __future__ import annotations
import argparse,sys,os,time,json,traceback,subprocess,hashlib,copy,io,contextlib,unittest
from pathlib import Path
from fractions import Fraction
import numpy as np
from .core import *
from .binding import source_manifest,source_digest,dependency_contract,policy_configuration,state_digest
from .preflight import preflight

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[2]
SCRIPTS=ROOT/'STAGE3_OUTPUT/scripts'

def config():
    c=read_json(PACKAGE/'spatial_configs/stage4_model.json');c['scenario']='late_multi';return c

def fixture(c=None):
    from .filtering import MultiTracker
    from .physics import NBIAS
    from .scheduling import bind
    c=c or config();t=MultiTracker(c);t.predict(0)
    t._add(np.array([-900.,-350.]),np.zeros((2,NBIAS)),np.eye(2)*10000.,velocity=np.array([40.,15.]))
    return t,bind({**t.view(Fraction(2),Fraction('.1'),[.4]*4),'search_age_cycles':[1]*4,'node_pointing_rad':[0.]*4,'communication_available_s':0.,'processing_available_s':[0.]*4})

class ConnectedContracts(unittest.TestCase):
    def setUp(self):self.c=config();self.t,self.v=fixture(self.c)
    def action(self,kind='track',node=0,role='monostatic'):
        from .actions import materialize
        return materialize(Task(node,kind,-1 if kind=='calibrate' else 0,role=role),self.v,self.c)
    def packet(self,root='r0',role='monostatic',node=0,kind='track'):
        from .physics import measurement,measurement_units,sensor_list,calibration_model,NBIAS
        from .actions import visits
        a=self.action(kind,node,role);visit=visits(a,self.c)[0];a=visit['task'];off=visit['mid_s'];s=sensor_list(self.c)[node]
        z=calibration_model(a,self.t.mean[:NBIAS],len(self.t.mean),0,self.c)[0] if kind=='calibrate' else measurement(self.t.mean[NBIAS:NBIAS+4],self.t.mean[:NBIAS],s,a,off)
        return {'schema':'ain.spatial.measurement_packet.v3','node':node,'source':s.name,'root':root,'acquire_epoch':0,'acquire_s':off,'arrival_s':off+.05,
                'kind':kind,'task':a.to_dict(),'measurement_units':measurement_units(a),'returns':[z.tolist()],
                'delivered':True,'reference_acquired':True,'reference_id':a.reference_id,'boresight_rad':a.bearing_rad,'field_of_view_rad':a.beam_rad,'clutter_mean':.01}
    def test_arrival_and_delivery_boundaries(self):
        p=self.packet();fp=self.t.fingerprint()
        self.assertEqual(self.t.receive(p,now_s=p['acquire_s'])['status'],'NOT_ARRIVED');self.assertEqual(fp,self.t.fingerprint())
        for key,status in [('delivered','TRANSMISSION_FAILED'),('reference_acquired','NO_REFERENCE')]:
            q=copy.deepcopy(p);q[key]=False
            r=self.t.receive(q,1.);self.t.initiate([q],[r]);self.assertEqual(r['status'],status);self.assertEqual(fp,self.t.fingerprint())
    def test_late_receipt_not_credited(self):
        from .scheduling import admitted_schedule
        p=self.packet();p['arrival_s']=2.;self.t.predict(1);fp=self.t.fingerprint()
        r=self.t.receive(p,2.);self.t.initiate([p],[r]);self.assertEqual(r['status'],'LATE_REJECTED');self.assertFalse(admitted_schedule([p],[r]));self.assertEqual(fp,self.t.fingerprint())
    def test_duplicate_and_conflicting_provenance(self):
        p=self.packet();self.t.receive(p,1.);fp=self.t.fingerprint()
        q=copy.deepcopy(p);q['transport_copy']='forwarded'
        self.assertEqual(self.t.receive(q,1.)['status'],'DUPLICATE');self.assertEqual(fp,self.t.fingerprint())
        q['returns'][0][0]+=.001
        self.assertRaises(ContractError,self.t.receive,q,1.)
    def test_source_and_unit_rejection(self):
        for key,value in [('source','invented'),('measurement_units',['km','rad','m/s'])]:
            p=self.packet();p[key]=value;self.assertRaises(ContractError,self.t.receive,p,1.)
    def test_joint_calibration_information_transfer(self):
        from .physics import NBIAS
        self.t.P[0,NBIAS]=self.t.P[NBIAS,0]=1500.
        before=self.t.P.copy();p=self.packet(kind='calibrate');r=self.t.receive(p,1.)
        self.assertEqual(r['status'],'CALIBRATED');self.assertLess(self.t.P[NBIAS,NBIAS],before[NBIAS,NBIAS]);self.assertGreater(self.t.P[0,0],0.)
    def test_shared_noise_joint_association(self):
        from .physics import NBIAS,measurement,sensor_list
        self.t._add(np.array([-700.,-300.]),np.zeros((2,NBIAS)),np.eye(2)*10000.,velocity=np.array([35.,12.]))
        p=self.packet(role='bistatic',node=2);a=Task(**p['task']);s=sensor_list(self.c)[2];j=self.t.tracks[1]['start']
        p['returns'].append(measurement(self.t.mean[j:j+4],self.t.mean[:NBIAS],s,a,p['acquire_s']).tolist())
        old=self.t.P.copy();r=self.t.receive(p,1.)
        self.assertEqual(r['status'],'GATED_JOINT_MOMENT_UPDATE');self.assertTrue(r['shared_capture_used']);self.assertGreater(r['hypotheses'],1)
        self.assertFalse(np.array_equal(old,self.t.P));np.linalg.cholesky(self.t.P)
    def test_aoa_has_no_birth_range(self):
        from .filtering import MultiTracker
        from .physics import sensor_list,NBIAS
        from dataclasses import replace
        p=self.packet(role='bistatic',node=2);p['task']['mode']='aoa';p['returns']=[[p['returns'][0][2]]];p['measurement_units']=['rad']
        t=MultiTracker(self.c);t.predict(0);r=t.receive(p,1.);t.initiate([p],[r]);self.assertEqual(len(t.tracks),0)
    def test_zero_signal_has_no_fabricated_variance(self):
        from .physics import noise_cov,sensor_list
        a=self.action();c={**self.c,'opaque_discs':[{'centre_m':[-1300.,-775.],'radius_m':500.}]}
        self.assertRaises(ContractError,noise_cov,[-900.,-350.],sensor_list(c)[0],a,0,c)
    def test_all_joint_blocks_and_semantic_roles(self):
        from .spatial_graph import graph,recover_joint_covariance
        from .scheduling import atoms,bind
        from .physics import NBIAS
        a=atoms(self.v,self.c);v=copy.deepcopy(self.v);P=np.array(v['covariance']);P[0,NBIAS]=P[NBIAS,0]=1500.;v['covariance']=P.tolist();v=bind(v)
        g1=graph(v,a,self.c);v2=copy.deepcopy(v);P[0,NBIAS]=P[NBIAS,0]=-1500.;v2['covariance']=P.tolist();v2=bind(v2);g2=graph(v2,a,self.c)
        self.assertFalse(np.array_equal(g1.node_features,g2.node_features));self.assertTrue(np.allclose(recover_joint_covariance(g2,v2),P,rtol=1e-6,atol=.001))
        self.t._add(np.array([400.,400.]),np.zeros((2,NBIAS)),np.eye(2)*10000.)
        v3=bind({**self.t.view(Fraction(2),Fraction('.1'),[.4]*4),'search_age_cycles':[1]*4})
        before={x.semantic_id():(x.node,x.kind,x.item) for x in a};after={x.semantic_id():(x.node,x.kind,x.item) for x in atoms(v3,self.c)}
        for k in before.keys()&after.keys():self.assertEqual(before[k],after[k])
    def test_processing_transport_changes_forecast_arrival(self):
        from .scheduling import predicted_receipts
        a=self.action();v=copy.deepcopy(self.v);v['processing_available_s'][0]=2.
        arrivals,_,_,audit=predicted_receipts((a,),v,self.c);self.assertFalse(arrivals);self.assertGreater(audit[0]['predicted_arrival_s'],2.)
    def test_memory_snapshot_exact_continuation(self):
        import torch
        from .spatial_graph import graph
        from .scheduling import atoms,bind
        from .sequence_memory import SequenceMemory
        G=graph(self.v,atoms(self.v,self.c),self.c);m=SequenceMemory(48);m.begin('world/1')
        m.before(self.v,G,'cpu');x=torch.ones((len(G.node_ids),48));m.after(self.v,G,x);snapshot=m.snapshot()
        second=SequenceMemory(48);second.restore(snapshot);v2=bind({**self.v,'epoch':1,'decision_s':1.})
        self.assertTrue(torch.equal(m.before(v2,G,'cpu'),second.before(v2,G,'cpu')))
        second.begin('world/2');self.assertEqual(int(second.before(self.v,G,'cpu').count_nonzero()),0)
    def test_same_observation_joseph_and_scipy_information(self):
        from scipy.linalg import cho_factor,cho_solve
        m=np.array([1.,2.]);P=np.array([[4.,1.],[1.,3.]]);H=np.array([[1.,.3]]);R=np.array([[.7]]);z=np.array([.4])
        mu,post,_,_=gaussian_update(m,P,H,R,z);info=cho_solve(cho_factor(P),np.eye(2))+H.T@cho_solve(cho_factor(R),H)
        reference=cho_solve(cho_factor(info),np.eye(2));reference_mean=m+reference@H.T@cho_solve(cho_factor(R),z)
        self.assertTrue(np.allclose(mu,reference_mean,rtol=1e-13,atol=1e-13));self.assertTrue(np.allclose(post,reference,rtol=1e-13,atol=1e-13))
    def test_empty_domain_uses_abstention_and_masked_numeric_loss(self):
        # A small tensor contract, no network forward/backward or world advance.
        import torch
        from types import SimpleNamespace
        from .numeric_model import NumericOutput
        from .spatial_graph import Priority,scores
        from .sequence_training import objective
        out=NumericOutput(torch.empty(0),torch.zeros((1,48)),torch.zeros((1,48)),
                          torch.empty((0,8)),torch.empty(0),torch.tensor(.4))
        class FakeModel:
            @property
            def joint_schedule_decoder(self):raise AssertionError('empty domain must not decode a joint action')
            def __call__(self,G,memory=None):return out
        G=SimpleNamespace(action_node_index=torch.empty(0,dtype=torch.long));model=FakeModel()
        row={'actions':[],'paid_labels':[{'action_indices':[],'proxy_m2':100.}],
             'numeric_labels':[],'future_target_mask':True,'realized_future_target_m2':100.}
        loss,terms=objective(model,out,G,row)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(terms['numeric'],0.)
        self.assertGreater(terms['future'],0.)
        values,_=scores(model,G,[],[()]);self.assertEqual(values.tolist(),[0.])
        priority=Priority.__new__(Priority);priority.device='cpu';priority.actions=();priority.out=out
        self.assertEqual(priority.priority(()),0.)
        self.assertRaises(ContractError,priority.priority,(Task(0,'search',0),))
        row['paid_labels'][0]['action_indices']=[0]
        self.assertRaises(ContractError,objective,model,out,G,row)

    def test_training_requires_explicit_server_execute(self):
        from .sequence_training import train
        self.assertRaises(ContractError,train,[],'.',execute=False)


def actual_weight_cache_check(out):
    """Two real untrained state files; no optimizer or new world."""
    out.mkdir(parents=True,exist_ok=True)
    import torch
    from .spatial_graph import make
    from .sequence_training import _checkpoint
    from .case_identity import checkpoint_identity,case_identity,seal_result
    from .run import main as run_main
    c=config();c['cycles']=1;c['scenario']='cache_fixture'
    allowed={digest(policy_configuration(c,'numeric_network')):policy_configuration(c,'numeric_network')};paths=[]
    for arm in ('integrated_agent','set_transformer'):
        m,mc=make(arm,seed=9300347);ck=_checkpoint(m,mc,arm,[],allowed,[],[],9300347,0)
        p=out/f'UNTRAINED_{arm}.pt';torch.save(ck,p);paths.append(p)
    ident=case_identity(9300348,'numeric_network',c,checkpoint_identity(paths[0]));body=seal_result({'schema':'ain.spatial.numeric.result.v3','status':'COMPLETE',
       'seed':9300348,'method':'numeric_network','configuration':c,'trace':[{'epoch':0}], 'checkpoint_sha256':ident['checkpoint']['sha256'],
       'fixture_only':True,'untrained_initialization':True},ident)
    plan={'model':{k:v for k,v in c.items() if k!='scenario'},'methods':['numeric_network'],'cases':[{'scenario':'cache_fixture','seeds':[9300348]}]}
    planpath=out/'PLAN.json';write_json(planpath,plan);results=[]
    for name,path,expected in [('correct',paths[0],0),('wrong_weights_same_source_config',paths[1],2)]:
        dest=out/name;write_json(dest/'cases/cache_fixture_9300348_numeric_network.json',body)
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer):code=run_main(['--config',str(planpath),'--output',str(dest),'--checkpoint',str(path),'--resume','--allow-untrained-checkpoint'])
        (dest/'cli.stdout.txt').write_text(buffer.getvalue(),encoding='utf-8');receipt=read_json(dest/'RUN_RECEIPT.json')
        assert code==expected,(name,code,receipt)
        if expected:assert 'CACHE_CASE_MISMATCH:checkpoint' in receipt['cases'][0]['error'],receipt
        else:assert receipt['cases'][0]['status']=='REUSED'
        results.append({'case':name,'status':receipt['status'],'checkpoint_sha256':checkpoint_identity(path)['sha256'],'expected_returncode':expected})
    write_json(out/'ACTUAL_CHECKPOINT_CACHE_CHECK.json',{'optimizer_updates':0,'world_cycles':0,'same_source':True,'same_config':True,'results':results})


def gradient_check(out):
    import torch
    from .spatial_graph import make,graph
    from .sequence_memory import SequenceMemory
    from .sequence_training import training_context,objective,architecture
    from .scheduling import bind,run_policy
    torch.set_num_threads(1)
    # Optimizer calls are a fatal contract error, including accidental calls in
    # helper functions. Backpropagation alone is not an optimizer update.
    def forbidden(*args,**kwargs):raise ContractError('LOCAL_OPTIMIZER_STEP_FORBIDDEN')
    for name in dir(torch.optim):
        cls=getattr(torch.optim,name)
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer):cls.step=forbidden
    c=policy_configuration({**config(),'forecast_horizon':1,'root_score_budget':3},'numeric_optimizer');t,_=fixture(c);rows=[]
    for epoch in range(2):
        if epoch:t.predict(epoch)
        view=bind({**t.view(Fraction(2),Fraction('.1'),[.4]*4),'search_age_cycles':[epoch+1]*4,'node_pointing_rad':[0.]*4,'processing_available_s':[0.]*4})
        _,commit=run_policy(view,c,'numeric_optimizer',9300349)
        # Two offline targets are fixed finite test labels, NOT effects claims.
        rows.append(training_context(view,commit,c,10000.-epoch*2000.))
    report={'optimizer_updates':0,'backwards_per_backbone':1,'contexts_per_backbone':2,'source_manifest':source_manifest(),'arms':{}}
    for arm in ('integrated_agent','deep_sets','set_transformer'):
        m,mc=make(arm,seed=9300350);m.eval();before=state_digest(m.state_dict());mem=SequenceMemory(mc.memory_dim);mem.begin('test/1');loss=0.;values=[];parts=[]
        for row in rows:
            G=graph(row['view'],tuple(Task(**a) for a in row['actions']),c);o=m(G,mem.before(row['view'],G,'cpu'));l,p=objective(m,o,G,row);loss+=l;parts.append(p);values.append(o.controls.detach().clone());mem.after(row['view'],G,o.next_memory)
        loss.backward();grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in m.named_parameters()}
        numeric=sum(v or 0 for n,v in grads.items() if n.startswith('numeric_head.'));history={n:v for n,v in grads.items() if n.endswith('memory.weight_hh') or n.endswith('memory_cell.weight_hh')};joint=sum(v or 0 for n,v in grads.items() if 'joint_schedule_decoder' in n)
        assert numeric>0 and any(v is not None and v>0 for v in history.values()) and joint>0,(arm,numeric,history,joint)
        assert state_digest(m.state_dict())==before
        reloaded,mc2=make(arm,seed=9300351);reloaded.load_state_dict(m.state_dict());reloaded.eval();mem2=SequenceMemory(mc.memory_dim);mem2.begin('test/1');diff=0.
        for row,value in zip(rows,values):
            G=graph(row['view'],tuple(Task(**a) for a in row['actions']),c)
            with torch.no_grad():o=reloaded(G,mem2.before(row['view'],G,'cpu'))
            diff=max(diff,float((o.controls-value).abs().max()));mem2.after(row['view'],G,o.next_memory)
        assert diff<=1e-7,(arm,diff)
        report['arms'][arm]={'numeric_gradient_norm_sum':numeric,'history_weight_gradients':history,'joint_decoder_gradient_norm_sum':joint,
            'all_gradients':grads,'weights_unchanged':True,'reload_max_difference':diff,'losses':parts,'architecture':architecture(m,mc)}
        write_json(out/'GRADIENTS_NO_TRAINING.json',report)
    write_json(out/'FIXED_TWO_CONTEXT_DATA.json',rows)
    return report


def short_closed_loops(out,budget):
    from .run import run_case
    from .spatial_graph import Priority
    from .scheduling import validate_view
    from .case_identity import case_identity,seal_result,validate_cached
    c=config();c['target_birth_epochs']=[0,1,2]
    cases=[('fixed_analytic',None,3,{}),('numeric_optimizer',None,3,{}),('fixed_network','integrated_agent',2,{}),
           ('numeric_network','integrated_agent',3,{}),('numeric_network','deep_sets',2,{}),('numeric_network','set_transformer',2,{}),
           ('numeric_optimizer',None,2,{'allowed_signal_roles':['direct'],'allow_calibration':False,'emission_probability':1.,'fixture_name':'direct_reception'}),
           ('random_execution',None,2,{'allowed_task_kinds':['calibrate'],'calibration_reference_off_epochs':[1],'fixture_name':'reference_failure'})]
    os.environ['RADAR_LOCAL_BUDGET_FILE']=str(budget);rows=[]
    for index,(method,arm,cycles,overrides) in enumerate(cases):
        cfg={**c,**overrides,'cycles':cycles};model=Priority.untrained(arm,seed=9300352) if arm else None
        checkpoint={'kind':'untrained_initialization','arm':arm,'seed':9300352,'sha256':state_digest(model.model.state_dict())} if model else None
        identity=case_identity(9300353,method,cfg,checkpoint);tag=f'{index:02d}_{method}_{arm or "analytic"}'
        calls={}
        def profile(frame,event,arg):
            if event!='call':return
            module=frame.f_globals.get('__name__','');name=frame.f_code.co_name
            if (module=='receiver_revisions.passive_geometry' and name=='measurement_and_jacobian') or (module=='spatial_lab.world' and name in ('step','observe')):
                scope='prediction_or_inference';f=frame
                while f:
                    if f.f_globals.get('__name__')=='spatial_lab.world' and f.f_code.co_name=='observe':scope='world_reception';break
                    f=f.f_back
                key=f'{module}.{name}/{scope}/{frame.f_locals.get("mode","")}'
                calls[key]=calls.get(key,0)+1
        try:
            sys.setprofile(profile);result=run_case(9300353,cfg,method,model);sys.setprofile(None)
            for f in result['trace']:
                validate_view(f['causal_view']);assert f['commit']['context_digest']==f['causal_view']['digest']
            assert result['world_advances']==cycles
            if model:assert state_digest(model.model.state_dict())==checkpoint['sha256']
            result['actual_profile_calls']=calls;result['local_interface_check_only']=True
            result=seal_result(result,identity);validate_cached(result,identity);write_json(out/'cases'/f'{tag}.json',result)
            rows.append({'tag':tag,'status':'COMPLETE','cycles':cycles,'method':method,'arm':arm,'summary':result['summary'],'profile_calls':calls,
                         'network_raw_proposals':sum(f['commit']['network_raw_output'] is not None for f in result['trace'])})
        except Exception as exc:
            sys.setprofile(None);row={'tag':tag,'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'profile_calls':calls};write_json(out/'failures'/f'{tag}.json',row);rows.append(row)
        write_json(out/'SHORT_LOOPS.json',{'results':rows,'budget':read_json(budget)})
    return rows


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--skip-closed-loop',action='store_true');ap.add_argument('--skip-backward',action='store_true');ap.add_argument('--budget-file');ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args(argv);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'LOCAL_CHECK_RECEIPT.json').exists():raise ContractError('USE_A_NEW_LOCAL_CHECK_OUTPUT')
    budget=Path(args.budget_file).resolve() if args.budget_file else out/'LOCAL_BUDGET.json'
    receipt={'status':'RUNNING','optimizer_updates':0,'phases':[],'source_manifest':source_manifest()};started=time.perf_counter()
    write_json(out/'LOCAL_CHECK_RECEIPT.json',receipt)
    try:
        receipt['preflight']=preflight(config())
        if args.verify_only:
            manifest=read_json(ROOT/'STAGE3_RELEASE_MANIFEST.json');bad=[]
            for row in manifest['files']:
                p=ROOT/row['path']
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:bad.append(row['path'])
            assert not bad,bad;receipt.update(status='VERIFIED_FILES_ONLY',verified_files=len(manifest['files']))
        else:
            for script in ('check_w0.py','check_physics_actions.py','check_time_conflict.py','check_server_receipts.py','check_fixed_physical_points.py','replay_existing_direct_capture.py'):
                p=SCRIPTS/script;log=out/(p.stem+'.log')
                r=subprocess.run([sys.executable,'-B',str(p)],capture_output=True,text=True,timeout=45,env={**os.environ,'RADAR_CHECK_OUTPUT':str(out/'time_conflict')})
                log.write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');receipt['phases'].append({'name':script,'returncode':r.returncode})
                write_json(out/'LOCAL_CHECK_RECEIPT.json',receipt)
                if r.returncode:raise ContractError('LOCAL_CHECK_FAILED:'+script)
            stream=io.StringIO();test=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConnectedContracts))
            (out/'CONNECTED_CONTRACTS.log').write_text(stream.getvalue(),encoding='utf-8');receipt['phases'].append({'name':'connected_contracts','tests':test.testsRun,'errors':len(test.errors),'failures':len(test.failures)})
            if not test.wasSuccessful():raise ContractError('CONNECTED_CONTRACT_TEST_FAILURE')
            actual_weight_cache_check(out/'cache_check');receipt['phases'].append({'name':'actual_weight_cache_check','passed':True})
            if not args.skip_backward:gradient_check(out);receipt['phases'].append({'name':'three_backbones_backward','backwards':3,'optimizer_updates':0})
            if not args.skip_closed_loop:
                rows=short_closed_loops(out,budget);receipt['phases'].append({'name':'short_loops','results':[{k:v for k,v in r.items() if k!='summary'} for r in rows]})
                if any(r['status']!='COMPLETE' for r in rows):raise ContractError('SHORT_LOOP_FAILURE')
            receipt['status']='PASSED'
    except Exception as exc:receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    finally:
        receipt['wall_seconds']=time.perf_counter()-started
        if budget.exists():receipt['local_compute_budget']=read_json(budget)
        write_json(out/'LOCAL_CHECK_RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_manifest','preflight','phases')},ensure_ascii=False))
    return 2 if receipt['status']=='FAILED' else 0
if __name__=='__main__':raise SystemExit(main())
