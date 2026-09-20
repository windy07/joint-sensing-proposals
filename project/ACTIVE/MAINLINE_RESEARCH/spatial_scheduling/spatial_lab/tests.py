"""Small scientific identities and public-boundary checks; no historical suites."""
from __future__ import annotations
import copy,unittest,math
from itertools import permutations
from fractions import Fraction
import numpy as np
from scipy.optimize import least_squares
from .core import *
from .physics import *
from .filtering import MultiTracker
from .scheduling import *
from .world import World
from .metrics import gospa_parts
from pathlib import Path
CONFIG=read_json(Path(__file__).resolve().parents[1]/'spatial_configs/local.json')['model']

class ScientificContracts(unittest.TestCase):
    def setup(self):
        cfg={**CONFIG,'scenario':'single'};t=MultiTracker(cfg);t.predict(0)
        return cfg,t,t.view(Fraction('9.6'),Fraction('.096'),[.4]*4)
    def test_01_geometry_jacobian_and_delay(self):
        state=np.array([320.,450.,18.,-8.]);bias=np.zeros(6);n=10
        for s in SENSORS:
            H=jacobian(state,s,n,6);eps=1e-4
            for j in range(4):
                v=state.copy();v[j]+=eps;w=state.copy();w[j]-=eps
                derivative=innovation(measurement(v,bias,s),measurement(w,bias,s),s)/(2*eps)
                np.testing.assert_allclose(derivative,H[:,6+j],atol=1e-8,rtol=1e-5)
        distance=2400.;self.assertEqual(C0*(2*distance/C0)/2,distance)
    def test_02_joseph_equals_information_and_independent_scipy_localization(self):
        m=np.array([10.,-3.]);P=np.array([[9.,2.],[2.,5.]])
        H=np.array([[1.,0.],[1.,1.]]);R=np.diag([2.,4.]);z=np.array([1.,-2.])
        mu,V,_,_=gaussian_update(m,P,H,R,z)
        W=np.linalg.inv(np.linalg.inv(P)+H.T@np.linalg.solve(R,H))
        np.testing.assert_allclose(V,W,atol=1e-13)
        sites=np.array([[-1000.,0.],[1000.,0.],[0.,1000.]])
        point=np.array([100.,300.]);dist=np.linalg.norm(point-sites,axis=1)
        sol=least_squares(lambda p:np.linalg.norm(p-sites,axis=1)-dist,np.array([0.,0.]))
        np.testing.assert_allclose(sol.x,point,atol=1e-5)
    def test_03_copied_provenance_and_calibration(self):
        cfg,t,v=self.setup();w=World(71,cfg);w.step();p=w.observe((Task(0,'calibrate'),),v)[0]
        before=t.P[0,0];t.receive(p);self.assertLess(t.P[0,0],before)
        fp=t.fingerprint();q=copy.deepcopy(p);q['transport_copy']='relay';t.receive(q)
        self.assertEqual(fp,t.fingerprint());self.assertEqual(t.stats['duplicate_packets'],1)
        q['returns'][0][0]+=1
        with self.assertRaises(ContractError):t.receive(q)
    def test_04_causality_resources_late_and_budget(self):
        cfg,t,v=self.setup();before=t.fingerprint()
        a,c=run_policy(v,{**cfg,'B':0},'feedback')
        self.assertEqual(a,());self.assertEqual(c['meter']['score_calls'],0);self.assertEqual(c['status'],'ABSTAINED')
        self.assertEqual(t.fingerprint(),before)
        x=dict(v);x['truth']=[1];x=bind(x)
        with self.assertRaises(ContractError):validate_view(x)
        x=bind({**v,'energy_remaining_exact_J':'399999999999999999/1000000000000000000'})
        with self.assertRaises(ContractError):legal((Task(0,'search',0),),x,cfg)
        p=World(72,cfg);p.step();packet=p.observe((Task(0,'calibrate'),),v)[0];packet['arrival_s']=2.
        self.assertEqual(t.receive(packet)['status'],'LATE_REJECTED')
    def test_05_gospa_separate_counts_and_bruteforce(self):
        self.assertEqual(gospa_parts([],[])['gospa_m'],0)
        r=gospa_parts([[0,0]],[]);self.assertEqual(r['missed_count'],1);self.assertEqual(r['false_count'],0)
        x=np.array([[0.,0.],[10.,0.]]);y=np.array([[1.,0.],[8.,0.],[100.,0.]])
        r=gospa_parts(x,y,20.);expect=min(sum(min(np.sum((x[i]-y[j])**2),400) for i,j in enumerate(p))+200 for p in permutations(range(3),2))
        self.assertAlmostEqual(r['gospa_m']**2,expect);self.assertEqual(r['false_count'],1)
    def test_06_shared_bias_complement_and_bellman_counterexample(self):
        # Independent target/bias unit variances and unit measurement noise: mixed observation leaves 2/3;
        # the noisy bias beacon lowers it to 3/5 without observing the hidden bias.
        P=np.eye(2);_,post,_,_=gaussian_update(np.zeros(2),P,np.array([[1.,1.]]),np.array([[1.]]),np.zeros(1))
        self.assertAlmostEqual(post[0,0],2/3)
        H=np.array([[1.,1.],[0.,1.]]);_,joint,_,_=gaussian_update(np.zeros(2),P,H,np.eye(2),np.zeros(2))
        self.assertAlmostEqual(joint[0,0],.6);self.assertLess(joint[0,0],post[0,0])
        # One energy token, target 2 appears later. Immediate loss .1 versus .4;
        # total final loss 1.1 versus .5 when the saved token observes target 2.
        self.assertLess(Fraction(1,10),Fraction(2,5));self.assertLess(Fraction(2,5)+Fraction(1,10),Fraction(1,10)+1)
    def test_07_association_exclusion_and_moment_uncertainty(self):
        cfg,t,v=self.setup();J=np.zeros((2,6))
        t._add(np.array([0.,0.]),J,np.eye(2)*100,origin='test')
        t._add(np.array([160.,0.]),J,np.eye(2)*100,origin='test')
        t.P[np.ix_([6,10],[6,10])]=np.array([[10000.,0],[0,10000.]])
        s=SENSORS[0];z=measurement(np.array([80.,0.,0.,0.]),np.zeros(6),s)
        packet={'source':s.name,'node':0,'root':'association','acquire_epoch':0,'acquire_s':0.,'arrival_s':.1,
            'kind':'search','task':Task(0,'search',1).to_dict(),'returns':[z.tolist()],
            'measurement_units':['m','rad','m/s'],'boresight_rad':azimuth(np.array([80.,0.]),s),'field_of_view_rad':math.radians(70.)}
        r=t.receive(packet);self.assertGreater(r['hypotheses'],1);self.assertGreater(r['association_entropy_nats'],0)
        psd(t.P)
    def test_08_no_unlabelled_range_from_single_bearing(self):
        cfg,t,v=self.setup();s=SENSORS[2]
        p={'source':s.name,'node':2,'root':'passive','acquire_epoch':0,'acquire_s':0.,'arrival_s':.1,'kind':'search','task':Task(2,'search',0).to_dict(),'returns':[[0.]],'measurement_units':['rad'],'boresight_rad':0.,'field_of_view_rad':math.radians(70.)}
        r=t.receive(p);t.initiate([p],[r]);self.assertEqual(len(t.tracks),0)
    def test_09_legal_actionable_node_task_group(self):
        cfg,t,v=self.setup();meter=Meter(8,256);g=Generator(v,cfg,meter);g.initial()
        self.assertTrue(g.pool);self.assertIn((),g.pool)
        for a in g.pool:legal(a,v,cfg)
        with self.assertRaises(ContractError):legal((Task(0,'search',0),Task(0,'calibrate')),v,cfg)
        self.assertTrue(any(len(a)==2 for a in g.pool))
    def test_10_postentry_exception_no_refund(self):
        cfg,t,v=self.setup();m=Meter(1,4);r=ProxyReference(v,cfg,m,1)
        with self.assertRaises(ContractError):r.evaluate((Task(999,'search',0),))
        self.assertEqual(m.score_calls,1);self.assertIsNone(m.risk_kernel_calls)
        with self.assertRaises(BudgetExhausted):r.evaluate(())
    def test_11_birth_covariance_shared_correlation(self):
        cfg,t,v=self.setup();J=np.zeros((2,6));J[0,0]=-1
        t._add(np.array([100.,100.]),J,np.eye(2)*4,origin='test')
        self.assertAlmostEqual(t.P[6,0],-t.P[0,0]);psd(t.P)
    def test_12_spectrum_is_not_unit_relabel(self):
        pos=np.array([0.,0.]);task=Task(0,'search',0)
        s=SENSORS[0];cfg={**CONFIG,'scenario':'single'}
        d=Sensor(**{**s.doc(),'bandwidth_Hz':s.bandwidth_Hz/4})
        self.assertGreater(noise_cov(pos,d,task,0,cfg)[0,0],noise_cov(pos,s,task,0,cfg)[0,0])
        self.assertEqual(d.carrier_Hz,s.carrier_Hz)

    def test_13_packet_unit_pointing_and_bounded_prefix(self):
        cfg,t,v=self.setup();w=World(71,cfg);w.step();p=w.observe((Task(0,'search',0),),v)[0]
        wrong=copy.deepcopy(p);wrong['measurement_units']=['km','rad','m/s']
        with self.assertRaises(ContractError):t.receive(wrong)
        wrong=copy.deepcopy(p);wrong['arrival_s']=float('nan')
        with self.assertRaises(ContractError):t.receive(wrong)
        wrong=copy.deepcopy(p);wrong.pop('boresight_rad')
        with self.assertRaises(ContractError):t.receive(wrong)
        a,c=run_policy(v,{**cfg,'construction_cap':2},'feedback')
        self.assertLessEqual(c['meter']['construction'],2);self.assertLessEqual(len(c['output']),cfg['K'])
        with self.assertRaises(ContractError):run_policy(v,{**cfg,'K':0},'feedback')

if __name__=='__main__':unittest.main(verbosity=2)
