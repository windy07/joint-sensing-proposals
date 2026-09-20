"""Deterministic Stage4 regressions: no world advance, optimizer or backward.

All synthetic beliefs/packets below are explicitly interface fixtures. They are
not generated world results and cannot enter performance or training datasets.
"""
from __future__ import annotations
import copy,math,itertools,unittest,tempfile,json,hashlib,ast
from pathlib import Path
from fractions import Fraction
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from .core import Task,Meter,ContractError,BudgetExhausted,digest,read_json,write_json
from .actions import materialize,project,validate_numeric,control_label,control_target,resource,visits,device,teacher_control_labels
from .physics import NBIAS,CELLS,BIAS_WALK_SD,transition,sensor_list,measurement,measurement_units,conditional_detection_probability,noise_cov
from .scheduling import bind,atoms,forecast_once,advance_time,ProxyReference,criterion,discovery_prob,predicted_receipts,admitted_captures,run_policy,Generator
from .filtering import MultiTracker
from .local_checks import config,fixture,ConnectedContracts

DETAILS={}
OUTPUT=None

def note(name,body):
    DETAILS[name]=body
    if OUTPUT is not None:write_json(Path(OUTPUT)/'FIXED_DETAILS.json',DETAILS)

def roomy(c=None):
    c=c or config();t,v=fixture(c)
    v=bind({**v,'energy_remaining_exact_J':'20','tx_remaining_exact_s':'1'})
    return c,t,v

def packet_for(task,view,c,root,returns=None):
    sensor=sensor_list(c)[task.node];off=task.start_s+task.dwell_s/2
    return {'schema':'ain.spatial.deterministic_interface_packet.v4','node':task.node,'source':sensor.name,
      'root':root,'acquire_epoch':view['epoch'],'acquire_s':view['decision_s']+off,
      'arrival_s':view['decision_s']+off+.03,'kind':task.kind,'task':task.to_dict(),
      'measurement_units':measurement_units(task),'returns':returns or [],'delivered':True,
      'reference_acquired':True,'reference_id':task.reference_id,'boresight_rad':task.bearing_rad,
      'field_of_view_rad':task.beam_rad,'clutter_mean':.01}

class W1NumericDomain(unittest.TestCase):
    def setUp(self):self.c,self.t,self.v=roomy()
    def test_original_4_52_4_label_and_exact_resources(self):
        a=materialize(Task(0,'search',0),self.v,self.c,[.5,0,.5,.4,.5,.5,.5,0])
        a=replace(a,scan=tuple((beam[0],float(Fraction(n,10000))) for beam,n in zip(a.scan,[4,52,4])))
        label=control_label(a,self.v,self.c);b=materialize(a,self.v,self.c,label['target'])
        self.assertEqual([round(x[1]/b.pri_s) for x in b.scan],[4,52,4]);self.assertEqual(a,b)
        self.assertEqual(resource(a,self.c),resource(b,self.c));self.assertEqual(label['mask'],[1.]*8)
        note('V01_original_pulses',{'teacher':a.to_dict(),'label':label,'decoded':b.to_dict(),'resource':resource(b,self.c)})
    def test_endpoints_and_nonuniform_roundtrip(self):
        rows=[]
        for node,role in [(0,'monostatic'),(2,'direct')]:
            for kind in ('search','track'):
                for dwell,weights in [(0.,[0,1,0]),(1.,[1,0,0]),(.8,[.1,.9,.3]),(.5,[0,0,0])]:
                    a=materialize(Task(node,kind,0,role=role),self.v,self.c,[.5,dwell,.5,.4,*weights,0.])
                    label=control_label(a,self.v,self.c);b=materialize(a,self.v,self.c,label['target'])
                    self.assertEqual(a,b);self.assertEqual(resource(a,self.c),resource(b,self.c))
                    n=round(a.dwell_s/a.pri_s)
                    self.assertEqual(sum(round(x[1]/a.pri_s) for x in a.scan),n)
                    if dwell==1:self.assertEqual(a.dwell_s,.18);self.assertEqual(n,1800)
                    if dwell==0:self.assertEqual(a.dwell_s,.006);self.assertEqual(n,60)
                    rows.append({'node':node,'kind':kind,'control':dwell,'pulses':label['executable_pulses'],'dwell_s':a.dwell_s,'mask':label['mask']})
        note('V01_endpoints',rows)
    def test_wrap_hardware_boundaries(self):
        v=copy.deepcopy(self.v);v['mean'][NBIAS:NBIAS+2]=[-2500.,-1190.];v=bind(v)
        for offset in (0.,1.):
            a=materialize(Task(0,'track',0),v,self.c,[1.,1.,offset,0.,.5,.5,.5,0.])
            validate_numeric((a,),v,self.c);b=materialize(a,v,self.c,control_target(a,v,self.c)[0]);self.assertEqual(a,b)
        c={**self.c,'device_defaults':{'power_min_W':2.03,'power_max_W':99.97,'power_quantum_W':.1,'dwell_min_s':.00601,'dwell_max_s':.17999}}
        a=materialize(Task(0,'search',0),self.v,c,[0.,0.,.5,1.,1.,0.,0.,0.])
        b=materialize(Task(0,'search',0),self.v,c,[1.,1.,.5,1.,0.,0.,1.,0.])
        self.assertEqual(a.power_W,2.1);self.assertEqual(b.power_W,99.9)
        self.assertEqual(round(a.dwell_s/a.pri_s),61);self.assertEqual(round(b.dwell_s/b.pri_s),1799)
        for x in (a,b):validate_numeric((x,),self.v,c);self.assertEqual(x,materialize(x,self.v,c,control_target(x,self.v,c)[0]))
        note('V01_hardware_grid',{'min':a.to_dict(),'max':b.to_dict()})
    def test_projected_start_beyond_raw_domain_and_idempotence(self):
        raw=tuple(materialize(Task(node,'search',0),self.v,self.c,[.5,1,.5,1,.5,.5,.5,.25]) for node in (0,1))
        a,receipt=project(raw,self.v,self.c);self.assertIsNotNone(a,receipt)
        self.assertGreater(a[1].start_s,self.c['dt_s']*.25)
        aa,receipt2=project(a,self.v,self.c);self.assertEqual(a,aa);self.assertEqual(receipt2['corrections'],[])
        labels=teacher_control_labels(a,self.v,self.c)
        decoded=tuple(materialize(t,self.v,self.c,l['target']) for t,l in zip(a,labels))
        b,r=project(decoded,self.v,self.c)
        self.assertEqual(a,tuple(replace(t,requested_start_s=old.requested_start_s) for t,old in zip(b,a)))
        self.assertTrue(all(l['mask'][7]==1 for l in labels));note('V01_group_projection',{'receipt':receipt,'labels':labels,'repeat':receipt2})
    def test_final_domain_rejects_all_counterexamples(self):
        a=materialize(Task(0,'search',0),self.v,self.c,[.5,0,.5,.4,.5,.5,.5,0])
        span=math.radians(60);wrap=lambda x:(x+math.pi)%(2*math.pi)-math.pi
        invalid={
          'span60':replace(a,scan=((wrap(a.bearing_rad-span),.002),(a.bearing_rad,.002),(wrap(a.bearing_rad+span),.002))),
          'four_beams':replace(a,scan=tuple((a.bearing_rad,.0015) for _ in range(4))),
          'point_offset20':replace(a,bearing_rad=wrap(a.bearing_rad+math.radians(20)),scan=tuple((wrap(b+math.radians(20)),d) for b,d in a.scan)),
          'wrong_centre':replace(a,bearing_rad=a.scan[0][0]),
          'asymmetric_beams':replace(a,scan=(a.scan[0],a.scan[1],(a.scan[1][0],a.scan[2][1]))),
          'wrong_cell':replace(a,item=4),'wrong_hardware':replace(a,bandwidth_Hz=a.bandwidth_Hz*2),
          'one_pulse':replace(a,scan=((a.scan[0][0],.0001),(a.scan[1][0],.0058),(a.scan[2][0],.0001))),
          'unused_reference':replace(a,reference_node=2),'raw_start_outside':replace(a,start_s=self.c['dt_s']+.1,requested_start_s=self.c['dt_s']+.1)}
        direct=materialize(Task(2,'search',0,role='direct'),self.v,self.c)
        invalid['self_reference']=replace(direct,reference_node=2,reference_id='receiver/2')
        invalid['stale_track']=replace(materialize(Task(0,'track',0),self.v,self.c),item=99)
        outcomes={}
        for name,t in invalid.items():
            fixed,r=project((t,),self.v,self.c);self.assertIsNone(fixed,(name,r));self.assertEqual(r['status'],'REJECTED');outcomes[name]=r
        note('V02_domain_rejections',outcomes)
    def test_training_labels_rebuild_actual_projected_group(self):
        from .sequence_training import training_context
        c={**self.c,'forecast_horizon':1,'root_score_budget':2,'return_capacity':1}
        _,commit=run_policy(self.v,c,'numeric_optimizer',64001)
        row=training_context(self.v,commit,c)
        self.assertEqual(row['actions'],[t.to_dict() for t in atoms(self.v,c)])
        self.assertTrue(row['numeric_labels']);self.assertTrue(all('projected_start_s' in x and 'executable_pulses' in x for x in row['numeric_labels']))
        note('V01_training_connection',{'row':row,'commit':commit})

class W2SharedEvents(unittest.TestCase):
    def setUp(self):
        self.c,self.t,self.v=roomy({**config(),'search_cell_halfwidth_m':0.,'packet_loss_probability':0.})
        self.v['mean'][NBIAS:NBIAS+4]=[*CELLS[0],0.,0.];self.v=bind(self.v)
        self.a=materialize(Task(2,'search',0,role='direct'),self.v,self.c,[0,.5,.5,0,.5,.5,.5,0])
    def test_probability_tree_q_endpoints_and_duplicate_visits(self):
        from .shared_events import captures,any_detection_probability
        rows=captures((self.a,),self.c);sensor=sensor_list(self.c)[2];out=[]
        ds=[conditional_detection_probability(CELLS[0],sensor,r['task'],0,self.c,emission=1.) for r in rows]
        for q in (0.,.7,1.):
            c={**self.c,'emission_probability':q}
            for n in (1,3):
                actual=any_detection_probability(rows[:n],CELLS[0],self.v,c)
                expected=q*(1-math.prod(1-d for d in ds[:n]));self.assertAlmostEqual(actual,expected,places=14);self.assertLessEqual(actual,q)
                self.assertEqual(any_detection_probability(rows[:n]*2,CELLS[0],self.v,c),actual)
                out.append({'q':q,'captures':n,'conditional_receptions':ds[:n],'actual':actual,'exact_event_tree':expected})
            self.assertLessEqual(discovery_prob((self.a,),self.v,c)[0],q)
        note('V03_event_tree',out)
    def test_delivery_reference_failure_and_active_mixture(self):
        from .shared_events import captures,any_detection_probability
        direct=captures((self.a,),self.c)
        for field in ('delivered_condition','reference_condition'):
            rows=[{**r,field:False} for r in direct]
            self.assertEqual(any_detection_probability(rows,CELLS[0],self.v,self.c),0.)
        self.assertEqual(discovery_prob((self.a,),self.v,self.c,delivery_probability=0)[0],0.)
        active=materialize(Task(0,'search',0),self.v,self.c,[1.,1.,.5,0.,.5,.5,.5,0.])
        mixed=captures((self.a,active),self.c)
        actual=any_detection_probability(mixed,CELLS[0],self.v,self.c)
        self.assertGreater(actual,.7)
        s=sensor_list(self.c);miss_active=math.prod(1-conditional_detection_probability(CELLS[0],s[0],r['task'],0,self.c) for r in captures((active,),self.c))
        miss_direct=math.prod(1-conditional_detection_probability(CELLS[0],s[2],r['task'],0,self.c,emission=1) for r in direct)
        expected=1-miss_active*((1-.7)+.7*miss_direct)
        self.assertAlmostEqual(actual,expected,places=14)
        note('V03_active_mixture',{'direct_q':.7,'mixed_probability':actual,'reference_probability':expected,'failure_credit':0.})
    def test_conditional_posterior_and_known_missed_probability(self):
        results={}
        for q in (0.,.7,1.):
            c={**self.c,'emission_probability':q};results[q]=forecast_once(self.v,(self.a,),c,Meter())
        P=lambda q:np.asarray(results[q]['covariance'])
        np.testing.assert_allclose(P(.7),.3*P(0.)+.7*P(1.),rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal(P(0.),np.asarray(self.v['covariance']))
        expected=.3+.7*results[1.]['last_expected_missed_tracks']
        self.assertAlmostEqual(results[.7]['last_expected_missed_tracks'],expected,places=14)
        self.assertAlmostEqual(results[.7]['unseen_intensity'][0],self.v['unseen_intensity'][0]*(1-discovery_prob((self.a,),self.v,self.c)[0]),places=14)
        note('V03_joint_posterior',{'mixture_max_error':float(abs(P(.7)-(.3*P(0.)+.7*P(1.))).max()),
            'missed':{str(q):r['last_expected_missed_tracks'] for q,r in results.items()},'discovery':discovery_prob((self.a,),self.v,self.c).tolist()})
    def test_arrived_direct_misses_condition_one_persistent_emission(self):
        from .shared_events import captures
        t=MultiTracker(self.c);t.predict(0);t._add(CELLS[0].copy(),np.zeros((2,NBIAS)),np.eye(2)*10000.,velocity=np.zeros(2))
        rows=captures((self.a,),self.c);q=.7;product=1.;out=[]
        for k,r in enumerate(rows):
            task=r['task'];p=packet_for(task,self.v,self.c,f'empty_direct/{k}')
            d=conditional_detection_probability(CELLS[0],sensor_list(self.c)[2],task,0,self.c,emission=1.)
            product*=1-min(1-1e-12,d);result=t.receive(p,1.)
            expected=q*product/(1-q+q*product);actual=t.emission_summary()['0']
            self.assertAlmostEqual(actual,expected,delta=max(1e-25,abs(expected)*1e-10))
            fp=t.fingerprint();dup=t.receive(p,1.);self.assertEqual(dup['status'],'DUPLICATE');self.assertEqual(fp,t.fingerprint())
            out.append({'capture':k,'conditional_pd':d,'event_probability':actual,'Bayes_reference':expected,'receive':result})
        t.predict(1);self.assertEqual(t.emission_summary(),{});note('V03_filter_event_persistence',out)
    def test_only_admitted_root_gets_credit(self):
        from .shared_events import captures
        t=MultiTracker(self.c);t.predict(0);r=captures((self.a,),self.c)[0];p=packet_for(r['task'],self.v,self.c,'credit/root')
        accepted=t.receive(p,1.);duplicate=t.receive(p,1.)
        rows=admitted_captures([p,p],[accepted,duplicate],self.c);self.assertEqual(len(rows),1)
        x=discovery_prob((),self.v,self.c,capture_rows=rows)
        self.assertTrue(np.array_equal(x,discovery_prob((),self.v,self.c,capture_rows=rows*2)))
        for field,status in [('delivered','TRANSMISSION_FAILED'),('reference_acquired','NO_REFERENCE')]:
            bad=copy.deepcopy(p);bad[field]=False;result=t.receive(bad,1.)
            self.assertEqual(result['status'],status);self.assertFalse(admitted_captures([bad],[result],self.c))
        note('V03_credit',{'accepted':accepted,'duplicate':duplicate,'credit':x.tolist(),'actual_capture_count':len(rows)})
    def test_between_branch_mean_covariance_is_retained(self):
        from .shared_events import moment_match
        mean,P=moment_match([.3,.7],[np.array([-1.,2.]),np.array([3.,0.])],[np.eye(2),2*np.eye(2)])
        reference=1.7*np.eye(2)+.3*.7*np.outer(np.array([-4.,2.]),np.array([-4.,2.]))
        np.testing.assert_allclose(P,reference,rtol=1e-14,atol=1e-14);self.assertNotEqual(P[0,1],0)
        note('V03_full_moment',{'mean':mean.tolist(),'covariance':P.tolist()})

class W3PublicState(unittest.TestCase):
    def test_queue_capacity_probability_features_and_no_truth(self):
        from .spatial_graph import graph
        from .feature_state import PUBLIC_INDEX,WIDTH
        c,t,v=roomy();acts=atoms(v,c);v2=bind({**v,'processing_available_s':[2.,0.,0.,0.],'communication_available_s':2.})
        g=graph(v,acts,c);g2=graph(v2,acts,c);x=g.node_features.numpy();y=g2.node_features.numpy()
        self.assertEqual(g.node_ids,g2.node_ids);self.assertEqual(acts,atoms(v2,c));self.assertLess(abs(x).max(),100.)
        self.assertGreater(np.max(abs(x[:,PUBLIC_INDEX['processing_wait']]-y[:,PUBLIC_INDEX['processing_wait']])),0)
        self.assertGreater(np.max(abs(x[:,PUBLIC_INDEX['shared_link_wait']]-y[:,PUBLIC_INDEX['shared_link_wait']])),0)
        a=next(a for a in acts if a.node==0 and a.kind=='track')
        normal=predicted_receipts((a,),v,c)[3];late=predicted_receipts((a,),v2,c)[3]
        self.assertTrue(normal[0]['accepted_this_cycle']);self.assertFalse(late[0]['accepted_this_cycle'])
        c2={**c,'packet_loss_probability':.6,'emission_probability':.2,'device_defaults':{'communication_bps':2e6,'processing_ops_s':4e9}}
        z=graph(v,acts,c2).node_features.numpy()
        for name in ('processing_rate','communication_rate','emission_probability','delivery_probability'):
            self.assertGreater(np.max(abs(z[:,PUBLIC_INDEX[name]]-x[:,PUBLIC_INDEX[name]])),0,name)
        hidden={**c,'target_birth_epochs':[100,200,300],'force_loss_epochs':[0],'force_delayed_epochs':[0]}
        np.testing.assert_array_equal(x,graph(v,acts,hidden).node_features.numpy())
        no_queue=copy.deepcopy(v);no_queue.pop('processing_available_s');no_queue.pop('communication_available_s');no_queue=bind(no_queue)
        absent=graph(no_queue,acts,c).node_features.numpy();self.assertGreater(np.max(abs(absent[:,PUBLIC_INDEX['link_queue_present']]-x[:,PUBLIC_INDEX['link_queue_present']])),0)
        note('V04_queue_features',{'width':WIDTH,'same_semantic_graph':True,'max_feature_abs':float(abs(x).max()),'changed_columns':np.where(np.any(x!=y,axis=0))[0].tolist(),'normal_receipts':normal,'backlogged_receipts':late,'private_random_outcomes_not_encoded':True})
    def test_future_calibration_reference_and_localization_threshold(self):
        from .spatial_graph import graph
        c,t,v=roomy();acts=atoms(v,c);base=graph(v,acts,c).node_features.numpy()
        changed=graph(v,acts,{**c,'calibration_reference_off_epochs':[1,2],
            'birth_position_variance_limit_m2':2e6}).node_features.numpy()
        self.assertGreater(np.max(abs(base[:,90:92]-changed[:,90:92])),0.)
        self.assertGreater(np.max(abs(base[:,93]-changed[:,93])),0.)
        np.testing.assert_array_equal(base[:,92],changed[:,92])
        note('V04_future_reference_threshold',{'reserved_columns':[90,91,92,93,94,95],
          'future_known_reference_changes_tensor':True,'current_epoch_reference_unchanged':True,
          'model_input_width_unchanged':base.shape[1]})
    def test_complete_joint_covariance_and_atomic_mode_domain(self):
        from .spatial_graph import graph,recover_joint_covariance
        c,t,v=roomy();P=np.asarray(v['covariance']);P[0,NBIAS]=P[NBIAS,0]=1500.;v=bind({**v,'covariance':P.tolist()})
        c={**c,'allowed_measurement_modes':['full','aoa']};acts=atoms(v,c)
        self.assertTrue(any(a.mode=='aoa' for a in acts));self.assertTrue(any(a.mode=='full' for a in acts))
        for a in acts:validate_numeric((a,),v,c)
        G=graph(v,acts,c);np.testing.assert_allclose(recover_joint_covariance(G,v),P,rtol=1e-6,atol=.001)
        note('V04_preserved_joint',{'atoms':len(acts),'node_count':len(G.node_ids),'covariance_shape':list(P.shape),'all_declared_measurement_modes_present':True})

class W4TimeIndex(unittest.TestCase):
    def test_idle_one_two_three_and_pure_time_prediction(self):
        c,t,v=roomy();v=bind({**v,'communication_available_s':2.,'processing_available_s':[2.]*4})
        d=v;rows=[]
        for i in range(1,4):
            d=forecast_once(d,(),c,Meter());self.assertEqual(d['expected_missed_track_steps'],float(i));self.assertEqual(d['candidate_action_steps'],i)
            self.assertEqual(d['last_expected_missed_tracks'],1.);before=copy.deepcopy(d)
            d=advance_time(d,c);self.assertEqual(d['expected_missed_track_steps'],float(i));self.assertEqual(d['candidate_action_steps'],i)
            self.assertEqual(d['energy_remaining_exact_J'],v['energy_remaining_exact_J']);self.assertEqual(d['tx_remaining_exact_s'],v['tx_remaining_exact_s'])
            self.assertEqual(d['processing_available_s'],before['processing_available_s']);self.assertEqual(d['communication_available_s'],before['communication_available_s'])
            F=np.eye(len(before['mean']));Q=np.zeros_like(F);F[NBIAS:,NBIAS:],Q[NBIAS:,NBIAS:]=transition(c['dt_s'],c['acceleration_psd_m2_s3']);Q[:NBIAS,:NBIAS]=np.diag(BIAS_WALK_SD**2)
            np.testing.assert_allclose(d['covariance'],F@np.asarray(before['covariance'])@F.T+Q,rtol=1e-14,atol=1e-12)
            np.testing.assert_allclose(d['unseen_intensity'],np.asarray(before['unseen_intensity'])+c['birth_intensity_per_cell_per_cycle'],rtol=0,atol=1e-15)
            self.assertEqual(d['epoch'],i);self.assertEqual(d['decision_s'],i*c['dt_s']);self.assertEqual(d['emission_event_posterior'],{})
            rows.append({'actual_action_steps':i,'cumulative_missed':d['expected_missed_track_steps'],'next_epoch':d['epoch'],'energy_unchanged':True})
        note('V05_idle_index',rows)
    def test_proxy_mean_and_stage_coast_cost(self):
        c,t,v=roomy();c={**c,'tail_width':0};rows=[]
        for horizon in (1,2,3):
            result=ProxyReference(v,c,Meter(1),horizon).evaluate(())
            self.assertEqual(result['candidate_action_steps'],horizon);self.assertEqual(result['expected_missed_track_steps'],horizon)
            self.assertEqual(result['proxy_m2'],sum(result['forecast_trajectory_m2'])/horizon)
            self.assertEqual(result['proxy_sum_m2'],sum(result['forecast_trajectory_m2']))
            self.assertEqual(result['components']['expected_coast_cost_m2'],1000.)
            rows.append(result)
        note('V05_mean_and_sum',rows)
    def test_original_three_step_calibration_window_only(self):
        c=config();c.update(scenario='calibration_window',max_simultaneous=1,calibration_reference_off_epochs=[1,2],birth_intensity_per_cell_per_cycle=0.,coast_penalty_m2=0.,tentative_track_penalty_m2=0.,packet_loss_probability=0.)
        t=MultiTracker(c);t.predict(0);t.P[0,0]=300.**2;t._add(np.array([-900.,-350.]),np.zeros((2,NBIAS)),np.eye(2)*50.**2,velocity=np.zeros(2))
        v=bind({**t.view(Fraction('1.2'),Fraction('.02'),[0.]*4),'search_age_cycles':[0]*4})
        controls=[1.,1.,.5,0,.5,.5,.5,0.];meter=Meter(100,100);rows=[];failures=[]
        def step(view,name):
            a=() if name=='idle' else (materialize(Task(0,'calibrate' if name=='calibrate' else 'track',-1 if name=='calibrate' else 0,role='monostatic'),view,c,controls),)
            meter.construct();row=ProxyReference(view,c,meter,1).evaluate(a);post=forecast_once(view,a,c,Meter())
            return row['proxy_m2'],advance_time(post,c)
        for names in itertools.product(('idle','observe','calibrate'),repeat=3):
            d=v;vals=[]
            try:
                for name in names:value,d=step(d,name);vals.append(value)
                self.assertEqual(d['candidate_action_steps'],3)
                rows.append({'actions':names,'step_costs_m2':vals,'sum_cost_m2':sum(vals),'mean_cost_m2':sum(vals)/3})
            except ContractError as exc:failures.append({'actions':names,'error':repr(exc)})
        best=min(rows,key=lambda r:r['sum_cost_m2']);greedy=[];costs=[];d=v
        for _ in range(3):
            candidates=[]
            for name in ('idle','observe','calibrate'):
                try:value,nxt=step(d,name);candidates.append((value,name,nxt))
                except ContractError as exc:failures.append({'greedy':name,'error':repr(exc)})
            value,name,d=min(candidates,key=lambda x:x[0]);greedy.append(name);costs.append(value)
        self.assertNotEqual(greedy[0],best['actions'][0]);self.assertGreater(sum(costs),best['sum_cost_m2'])
        note('V05_original_calibration_window',{'configuration':c,'frozen_belief':v,'finite_domain':['idle','observe','calibrate'],'horizon':3,'best':best,'greedy':{'actions':greedy,'step_costs_m2':costs,'sum_cost_m2':sum(costs),'mean_cost_m2':sum(costs)/3},'all_rows':rows,'failures':failures,'meter':meter.report(),'world_advances':0})

class W5Feedback(unittest.TestCase):
    @staticmethod
    def feedbacks():
        from .feedback import create_feedback
        c,t,v=roomy({**config(),'illumination_off_epochs':[0],'forecast_horizon':1})
        v=bind({**v,'energy_remaining_exact_J':'.03'});m=Meter(4,96)
        a=materialize(Task(0,'search',0),v,c);b=materialize(Task(2,'search',0,role='bistatic'),v,c)
        _,rf=project((a,),v,c);_,ref=project((b,),v,c)
        assert rf['status']=='REJECTED' and ref['status']=='REJECTED'
        fb_rf=create_feedback(v,c,[rf],[],[],m);fb_ref=create_feedback(v,c,[ref],[],[],m)
        small=materialize(Task(0,'search',0),v,c,[0.,0.,.5,0.,.5,.5,.5,0.])
        row=ProxyReference(v,c,m,1).evaluate((small,));fb_geom=create_feedback(v,c,[],[row],[],m)
        return c,v,a,b,small,fb_rf,fb_ref,fb_geom
    def test_actual_reasons_values_reach_targeted_branches(self):
        from .feedback import targeted_proposals,validate_feedback
        c,v,a,b,small,rf,ref,geom=self.feedbacks();domain=atoms(v,c)
        outcomes={}
        for name,seed,feedback in [('resource',(a,),rf),('reference',(b,),ref),('geometry',(small,),geom)]:
            proposals=list(targeted_proposals(feedback,seed,v,c,domain));self.assertTrue(proposals,name)
            outcomes[name]={'feedback':feedback,'branches':[branch for _,branch,_ in proposals],
                'proposals':[[t.to_dict() for t in group] for group,_,_ in proposals]}
            if name=='resource':self.assertLess(resource(proposals[0][0][0],c)['energy_J'],resource(a,c)['energy_J'])
            if name=='reference':self.assertNotEqual(proposals[0][0][0].role,b.role)
            if name=='geometry':self.assertTrue(any(group!=seed for group,_,_ in proposals))
        self.assertNotEqual(outcomes['resource']['proposals'],outcomes['reference']['proposals'])
        bad=copy.deepcopy(rf);bad['records'][0]['value']+=1
        self.assertRaises(ContractError,validate_feedback,bad,v)
        note('V06_targeted_branches',outcomes)
    def test_feedback_value_changes_resource_action(self):
        from .feedback import create_feedback,targeted_proposals
        c,v,a,b,small,rf,ref,geom=self.feedbacks();v2=bind({**v,'energy_remaining_exact_J':'.015'})
        _,receipt=project((a,),v2,c);f2=create_feedback(v2,c,[receipt],[],[],Meter(4,96))
        p1=list(targeted_proposals(rf,(a,),v,c,atoms(v,c)))[0][0][0]
        p2=list(targeted_proposals(f2,(a,),v2,c,atoms(v2,c)))[0][0][0]
        self.assertLess(p2.power_W,p1.power_W);note('V06_value_path',{'budget_J':[.03,.015],'proposed_power_W':[p1.power_W,p2.power_W]})
    def test_feedback_bound_and_zero_K1_failure_contracts(self):
        c,v,*_=self.feedbacks();_,commit=run_policy(v,{**c,'root_score_budget':0,'return_capacity':1},'numeric_optimizer',64002)
        self.assertEqual(commit['output'],[]);self.assertEqual(commit['selected'],[]);self.assertEqual(commit['status'],'ABSTAINED');self.assertEqual(commit['meter']['score_calls'],0)
        _,one=run_policy(v,{**c,'root_score_budget':1,'return_capacity':1},'numeric_optimizer',64002)
        self.assertEqual(len(one['output']),1);self.assertEqual(one['meter']['score_calls'],1);self.assertEqual(one['meter']['regenerations'],0)
        with patch('spatial_lab.scheduling.forecast_once',side_effect=RuntimeError('deterministic scoring exception')):
            _,failed=run_policy(v,{**c,'root_score_budget':2,'return_capacity':1},'numeric_optimizer',64002)
        self.assertEqual(failed['output'],[]);self.assertEqual(failed['status'],'ABSTAINED');self.assertEqual(failed['meter']['score_failures'],2)
        self.assertTrue(failed['capacity_audit']['emergency_idle_used']);self.assertLessEqual(failed['meter']['regenerations'],1)
        self.assertLessEqual(failed['meter']['construction'],c['construction_cap'])
        note('V06_budget_edges',{'zero':commit,'K1':one,'scoring_exception':failed})
    def test_feedback_object_changes_numeric_input_same_context(self):
        from .spatial_graph import graph
        from .feature_state import FEEDBACK_START
        c,v,a,b,small,rf,ref,geom=self.feedbacks();domain=atoms(v,c)
        G1=graph(v,domain,c,feedback=rf);G2=graph(v,domain,c,feedback=ref)
        self.assertEqual(G1.node_ids,G2.node_ids)
        self.assertTrue(np.any(G1.node_features.numpy()[:,FEEDBACK_START:]!=G2.node_features.numpy()[:,FEEDBACK_START:]))
        np.testing.assert_array_equal(G1.node_features.numpy()[:,:FEEDBACK_START],G2.node_features.numpy()[:,:FEEDBACK_START])
        note('V06_feedback_graph',{'same_context':v['digest'],'resource_feedback_digest':rf['digest'],'reference_feedback_digest':ref['digest'],'different_feedback_tensor':True})

class W7ReleaseSafety(unittest.TestCase):
    def test_real_weight_cache_identity_and_zero_update_checkpoints(self):
        from .local_checks import actual_weight_cache_check
        out=Path(OUTPUT)/'actual_weight_cache' if OUTPUT else Path(tempfile.mkdtemp())
        actual_weight_cache_check(out);note('G06_actual_weights',read_json(out/'ACTUAL_CHECKPOINT_CACHE_CHECK.json'))
    def test_cycle_state_save_restore_and_tamper_rejection(self):
        from .state_io import save_cycle,restore_cycle
        from .world import World
        from .metrics import Evaluator
        c,t,v=roomy();t.epoch=-1;w=World(64003,c);ev=Evaluator(c)
        path=Path(OUTPUT or tempfile.mkdtemp())/'synthetic_cycle_state.json'
        before=t.fingerprint();save_cycle(path,64003,c,'synthetic_interface_only',w,t,ev,None,[],Fraction(2),Fraction('.1'),np.array([.4]*4),np.zeros(4,int),[0.]*4)
        t2=MultiTracker(c);w2=World(64003,c);ev2=Evaluator(c)
        d=restore_cycle(path,64003,c,'synthetic_interface_only',w2,t2,ev2,None);self.assertEqual(t2.fingerprint(),before);self.assertEqual(w2.audit['world_advances'],0)
        bad=read_json(path);bad['energy_exact_J']='1000';badpath=path.with_name('synthetic_tampered_state.json');write_json(badpath,bad)
        self.assertRaises(ContractError,restore_cycle,badpath,64003,c,'synthetic_interface_only',w2,t2,ev2,None)
        self.assertRaises(ContractError,restore_cycle,path,64004,c,'synthetic_interface_only',w2,t2,ev2,None)
        note('W7_state_roundtrip',{'fingerprint':before,'restored_fingerprint':t2.fingerprint(),'world_advances':0,'tamper_and_seed_rejected':True})
    def test_default_world_and_optimizer_refused(self):
        from .world import World
        from .sequence_training import train
        from .work_budget import LEDGER
        before=read_json(LEDGER)['world_advance_attempts'] if LEDGER.exists() else 0
        w=World(64005,config());self.assertRaises(ContractError,w.step);self.assertEqual(w.epoch,-1);self.assertEqual(w.audit['world_advances'],0)
        self.assertRaises(ContractError,train,[],'.',updates=1,execute=True)
        self.assertEqual(read_json(LEDGER)['world_advance_attempts'] if LEDGER.exists() else 0,before)
        note('W7_zero_work_guards',{'unregistered_world_refused_before_state_change':True,'local_optimizer_even_execute_true_refused':True})
    def test_formal_dependency_check_not_removed(self):
        from .preflight import preflight,dependency_contract
        with patch('spatial_lab.preflight.dependency_contract',return_value={**dependency_contract(),'numpy':'0.0.0'}):
            self.assertRaisesRegex(ContractError,'DEPENDENCY_PREFLIGHT',preflight,config())
        note('W7_dependency_mismatch_refused',{'forced_mismatch':'numpy=0.0.0','formal_gate_preserved':True})
    def test_mainline_loop_and_finite_module_syntax(self):
        from . import run
        tree=ast.parse(Path(run.__file__).read_text());function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_case')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step']
        self.assertEqual(len(calls),1)
        loops=[n for n in ast.walk(function) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch'];self.assertEqual(len(loops),1)
        count=0
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());count+=1
        note('W7_mainline_syntax',{'spatial_modules':count,'world_step_call_sites':len(calls),'epoch_loops':len(loops)})

GROUPS={'W1':W1NumericDomain,'W2':W2SharedEvents,'W3':W3PublicState,'W4':W4TimeIndex,'W5':W5Feedback,'W7':W7ReleaseSafety,'retained':ConnectedContracts}

def suite(groups=None):
    result=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(GROUPS[g]) for g in (groups or list(GROUPS)))
    if groups is None:
        from .stage5_tests import suite as completion_suite
        result.addTests(completion_suite())
    return result
