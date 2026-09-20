"""Decisive and regression checks for Stage2 reference-only receiver revisions."""
from __future__ import annotations
import unittest
import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.special import logsumexp
from .tdoa_iq_estimator import estimate_toa_matched_filter,estimate_tdoa_cross_correlation,tdoa_trials_from_raw_iq,ClockPosterior
from ._validation import NoUsableSignal
from .frequency_likelihood import ReceiverNoiseModel,frequency_likelihood,frequency_posterior,caf_peak_reference,subtract_independent_oscillator
from .credible_grid import credible_region,cell_edges,cell_areas
from .pbr_full_caf_factor_v2 import build_pbr_factor,finite_window_shift
from .passive_geometry import *


def dense_joint_phase_integral(r,y,fs,grid,noise,phases=1024):
    """Independent 2x2 complex Gaussian joint densities, phase quadrature.

    Does not use the conditional Gaussian or Bessel formula under test.
    Integrating a uniform phase by a trapezoidal periodic sum is the numerical
    reference. Absolute likelihood differs by p(r) and pi constants; compare
    frequency likelihood ratios.
    """
    ps,nr,ny,gains,weights=noise.validated()
    z=np.stack([r,y],axis=1)
    phi=2*np.pi*np.arange(phases)/phases
    t=np.arange(len(r))/fs
    results=[]
    for f in grid:
        terms=[]
        theta=phi[:,None]+2*np.pi*f*t[None,:]
        for gain,weight in zip(gains,weights):
            if weight==0:continue
            cov=np.empty((phases,len(r),2,2),dtype=complex)
            cov[...,0,0]=ps+nr;cov[...,1,1]=gain*gain*ps+ny
            cov[...,0,1]=gain*ps*np.exp(-1j*theta);cov[...,1,0]=cov[...,0,1].conj()
            _,logdet=np.linalg.slogdet(cov)
            solved=np.linalg.solve(cov,np.broadcast_to(z,(phases,*z.shape))[...,None])[...,0]
            quadratic=np.real(np.einsum('ni,pni->pn',z.conj(),solved))
            ll=np.sum(-logdet-quadratic,axis=1)
            terms.append(logsumexp(ll)-np.log(phases)+np.log(weight))
        results.append(logsumexp(terms))
    return np.asarray(results)

class DelayTests(unittest.TestCase):
    def test_repeated_peak_inside_window(self):
        z=np.array([0,10,4,0,3,10,9,0],complex)
        self.assertEqual(estimate_toa_matched_filter(z,[1],1,(4,6),True),5.375)
    def test_sample_rate_scaling(self):
        z=np.array([0,10,4,0,3,10,9,0],complex)
        self.assertAlmostEqual(estimate_toa_matched_filter(z,[1],2e6,(4/2e6,6/2e6),True),5.375/2e6)
    def test_edge_does_not_use_outside_neighbours(self):
        z=np.array([0,10,4,0,3,10,9,0],complex)
        self.assertEqual(estimate_toa_matched_filter(z,[1],1,(5,6),True),5.)
        self.assertEqual(estimate_toa_matched_filter(z,[1],1,(5,5),True),5.)
    def test_empty_reversed_fractional_windows_rejected(self):
        for w in [(9,10),(3,2),(1.1,1.2),(np.nan,2)]:
            with self.subTest(w=w),self.assertRaises(ValueError):estimate_toa_matched_filter([0,1,0],[1],1,w)
    def test_bad_signal_or_rate_rejected(self):
        for z in [[],[[1,2]],[np.nan]]:
            with self.subTest(z=z),self.assertRaises(ValueError):estimate_toa_matched_filter(z,[1],1)
        for fs in [0,-1,np.inf]:
            with self.subTest(fs=fs),self.assertRaises(ValueError):estimate_toa_matched_filter([1],[1],fs)
    def test_zero_no_arbitrary_peak(self):
        with self.assertRaises(NoUsableSignal):estimate_toa_matched_filter([0,0],[1],1)
    def test_scipy_same_complex_received_channels(self):
        rng=np.random.default_rng(718);r=rng.normal(size=64)+1j*rng.normal(size=64)
        y=np.r_[np.zeros(7),r[:-7]]
        expected=correlation_lags(len(y),len(r),'full')[np.argmax(abs(correlate(y,r,'full',method='direct')))]/1000
        self.assertEqual(estimate_tdoa_cross_correlation(y,r,1000,(-.01,.01),False),expected)
        self.assertEqual(expected,.007)
        self.assertEqual(estimate_tdoa_cross_correlation(r,y,1000,(-.01,.01),False),-.007)
    def test_paired_trial_length_not_silently_truncated(self):
        with self.assertRaises(ValueError):tdoa_trials_from_raw_iq([[1]],[],[1],1,ClockPosterior(np.zeros(2),np.eye(2)),0,1)
    def test_no_window_parabolic_matches_expected_peak(self):
        self.assertAlmostEqual(estimate_toa_matched_filter([0,3,10,9,0],[1],1,None,True),2.375)

class FrequencyTests(unittest.TestCase):
    def setUp(self):
        self.fs=128.;self.n=32;self.grid=np.arange(-30,31,dtype=float)
        self.r=np.ones(self.n,complex);self.y=np.exp(2j*np.pi*9*np.arange(self.n)/self.fs)
        self.model=ReceiverNoiseModel(1,0,1)
    def test_40_hz_counterexample(self):
        n=1024;r=np.ones(n,complex);y=np.exp(2j*np.pi*40*np.arange(n)/1024)
        p=frequency_posterior(frequency_likelihood(r,y,1024,np.arange(-60,61.),self.model))
        self.assertAlmostEqual(p.mean_hz,40);self.assertGreater(p.probabilities.max(),.99)
        self.assertTrue(np.all(np.isfinite(p.log_probabilities)))
        self.assertLess(np.min(p.log_probabilities),-700)  # underflow identity still retained
    def test_peak_reference_does_not_claim_posterior_covariance(self):
        out=caf_peak_reference(self.r,self.y,self.fs,self.grid)
        self.assertEqual(out['frequency_hz'],9);self.assertIsNone(out['variance_hz2']);self.assertIsNone(out['posterior'])
    def test_dense_joint_gaussian_phase_gain_integral(self):
        rng=np.random.default_rng(83);r=rng.normal(size=5)+1j*rng.normal(size=5);y=rng.normal(size=5)+1j*rng.normal(size=5)
        m=ReceiverNoiseModel(1.,.7,1.1,(.4,1.,2.),(.2,.5,.3));grid=np.array([-13.,0.,9.])
        a=frequency_likelihood(r,y,64,grid,m).log_likelihood
        b=dense_joint_phase_integral(r,y,64,grid,m)
        np.testing.assert_allclose(a-a[0],b-b[0],atol=1e-11,rtol=1e-11)
    def test_unknown_phase_invariance(self):
        a=frequency_likelihood(self.r,self.y,self.fs,self.grid,self.model).log_likelihood
        b=frequency_likelihood(self.r*np.exp(.2j),self.y*np.exp(2.1j),self.fs,self.grid,self.model).log_likelihood
        np.testing.assert_allclose(a-a[0],b-b[0],atol=1e-11)
    def test_gain_marginalization_changes_relative_likelihood(self):
        m=ReceiverNoiseModel(1,.5,2,(.1,1.,3.),(.3,.3,.4))
        a=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,m))
        b=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,ReceiverNoiseModel(1,.5,2)))
        self.assertGreater(np.max(abs(a.probabilities-b.probabilities)),1e-5)
    def test_reference_noise_changes_uncertainty(self):
        a=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,self.model))
        b=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,ReceiverNoiseModel(1,100,1)))
        self.assertGreater(b.variance_hz2,a.variance_hz2)
    def test_signal_scaling_fixed_noise_changes_concentration(self):
        a=frequency_posterior(frequency_likelihood(.1*self.r,.1*self.y,self.fs,self.grid,self.model))
        b=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,self.model))
        self.assertGreater(a.variance_hz2,b.variance_hz2)
    def test_all_voltage_and_variance_units_scaling_invariant(self):
        m=ReceiverNoiseModel(1,.5,2,(.5,1.,2.),(.2,.5,.3))
        a=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,m))
        b=frequency_posterior(frequency_likelihood(self.r*100,self.y*100,self.fs,self.grid,ReceiverNoiseModel(1e4,5000,20000,m.gain_values,m.gain_probabilities)))
        np.testing.assert_allclose(a.probabilities,b.probabilities,atol=2e-14)
    def test_finite_frequency_prior_separate(self):
        ll=frequency_likelihood(self.r,self.y,self.fs,self.grid,self.model)
        prior=np.zeros(len(self.grid));prior[1]=1
        p=frequency_posterior(ll,prior)
        self.assertEqual(p.mean_hz,self.grid[1]);self.assertNotEqual(ll.semantic_role,p.semantic_role)
    def test_no_information_returns_declared_prior_not_a_detected_peak(self):
        ll=frequency_likelihood(np.zeros(self.n),self.y,self.fs,self.grid,self.model)
        p=frequency_posterior(ll)
        np.testing.assert_allclose(p.probabilities,np.ones(len(self.grid))/len(self.grid),atol=1e-14)
        with self.assertRaises(NoUsableSignal):caf_peak_reference(np.zeros(self.n),self.y,self.fs,self.grid)
    def test_alias_has_no_unaliased_covariance_claim(self):
        y=np.exp(2j*np.pi*40*np.arange(128)/64)
        self.assertEqual(caf_peak_reference(np.ones(128),y,64,np.arange(-32,32.))['frequency_hz'],-24.)
        with self.assertRaises(ValueError):caf_peak_reference(np.ones(128),y,64,np.arange(-32,33.))
    def test_independent_oscillator_subtraction(self):
        p=frequency_posterior(frequency_likelihood(self.r,self.y,self.fs,self.grid,self.model))
        out=subtract_independent_oscillator(p,2.,4.)
        self.assertAlmostEqual(out['mean_hz'],p.mean_hz-2);self.assertAlmostEqual(out['variance_hz2'],p.variance_hz2+4)
    def test_noise_model_validation_and_bounded_grid(self):
        for m in [ReceiverNoiseModel(1,0,0),ReceiverNoiseModel(1,-1,1),ReceiverNoiseModel(1,1,1,(1,),(-1,))]:
            with self.subTest(m=m),self.assertRaises(ValueError):frequency_likelihood(self.r,self.y,self.fs,self.grid,m)
        with self.assertRaises(ValueError):frequency_likelihood(self.r,self.y[:-1],self.fs,self.grid,self.model)

class GridTests(unittest.TestCase):
    def test_out_of_grid_never_maps_to_boundary_success(self):
        p=np.zeros((3,3));p[-1,-1]=1
        a=credible_region(p,[0,.001,.002],[-10,0,10],.9,(1,1e6))
        self.assertFalse(a['truth_inside_credible_region']);self.assertFalse(a['truth_in_support'])
        self.assertIsNone(a['nll_cell_mass']);self.assertEqual(a['outside_distance_s'],.998)
        self.assertEqual(a['outside_distance_hz'],999990.)
    def test_exact_boundary_included_but_not_extrapolated(self):
        p=np.ones((2,2))
        self.assertTrue(credible_region(p,[0,1],[0,1],1,(1,1))['truth_inside_credible_region'])
        self.assertFalse(credible_region(p,[0,1],[0,1],1,(1+1e-10,1))['truth_in_support'])
    def test_nonuniform_area_hpd_density_not_mass_ranking(self):
        p=np.array([[.1,.1],[.25,.25],[.15,.15]])
        a=credible_region(p,[0,1,10],[0,1],.2)
        np.testing.assert_equal(np.array(a['selected_cell_mask']),[[True,True],[False,False],[False,False]])
        self.assertAlmostEqual(cell_areas([0,1,10],[0,1]).sum(),10.)
    def test_no_truth_and_zero_probability_nll(self):
        p=np.zeros((2,2));p[-1,-1]=1
        self.assertIsNone(credible_region(p,[0,1],[0,1])['truth_in_support'])
        a=credible_region(p,[0,1],[0,1],.9,(0,0));self.assertIsNone(a['nll_cell_mass']);self.assertEqual(a['nll_reason'],'zero_cell_probability')
    def test_invalid_grids_or_empty_probabilities(self):
        for g in [[1],[1,1],[2,1],[0,np.inf]]:
            with self.subTest(g=g),self.assertRaises(ValueError):cell_edges(g)
        with self.assertRaises(ValueError):credible_region(np.zeros((2,2)),[0,1],[0,1])
    def test_actual_pbr_repaired_coverage_and_prior_role(self):
        rng=np.random.default_rng(548);r=rng.normal(size=32)+1j*rng.normal(size=32);y=rng.normal(size=32)+1j*rng.normal(size=32)
        f,p=build_pbr_factor(r,y,1000,[0,.001,.002],[-10,0,10],.002,1e-6,10,.01,truth=(1,1e6))
        self.assertFalse(f.truth_inside_if_known_for_audit);self.assertFalse(f.calibrated_likelihood)
        self.assertIn('includes_tracking_prior',f.semantic_role);self.assertAlmostEqual(p.sum(),1.)
    def test_windowed_shift_not_circular(self):
        r=np.zeros(64,complex);r[60]=1
        y=finite_window_shift(r,8,64)
        self.assertLess(np.max(abs(y)),1e-12)

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([3100.,1400.,65.,-21.,3e-8,2.,.004])
        self.rx=Station((0.,0.),(3.,-2.),.13);self.ref=Station((2000.,-500.),(-1.,5.))
        self.av=LinkAvailability(True,True,True)
    def test_direct_and_bistatic_derivatives(self):
        for mode in ['direct_emitter','bistatic_scatter']:
            y,J=measurement_and_jacobian(self.x,self.rx,self.ref,1.3e9,mode=mode,availability=self.av)
            step=np.array([.002,.002,.001,.001,1e-9,1e-4,1e-5]);F=np.zeros_like(J)
            for i,h in enumerate(step):
                dx=np.zeros(7);dx[i]=h
                a=measurement_and_jacobian(self.x+dx,self.rx,self.ref,1.3e9,mode=mode,availability=self.av)[0]
                b=measurement_and_jacobian(self.x-dx,self.rx,self.ref,1.3e9,mode=mode,availability=self.av)[0]
                d=a-b;d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi;F[:,i]=d/(2*h)
            np.testing.assert_allclose(J,F,rtol=1e-5,atol=1e-9)
    def test_frequency_sign_against_moving_path_finite_difference(self):
        x=self.x.copy();x[4:]=0;dt=.001;fc=1.3e9
        for mode in ['direct_emitter','bistatic_scatter']:
            delays=[]
            for sign in [-1,1]:
                xx=x.copy();xx[:2]+=sign*dt*x[2:4]
                rx=Station(tuple(np.asarray(self.rx.position_m)+sign*dt*np.asarray(self.rx.velocity_mps)),self.rx.velocity_mps)
                ref=Station(tuple(np.asarray(self.ref.position_m)+sign*dt*np.asarray(self.ref.velocity_mps)),self.ref.velocity_mps)
                delays.append(measurement_and_jacobian(xx,rx,ref,fc,mode=mode,availability=self.av)[0][0])
            fd=measurement_and_jacobian(x,self.rx,self.ref,fc,mode=mode,availability=self.av)[0][1]
            self.assertAlmostEqual(fd,-fc*(delays[1]-delays[0])/(2*dt),places=5)
    def test_carrier_scaling_and_seconds_to_metres(self):
        x=self.x.copy();x[4:]=0
        a,_=measurement_and_jacobian(x,self.rx,self.ref,1e9,mode='direct_emitter',availability=self.av)
        b,_=measurement_and_jacobian(x,self.rx,self.ref,2e9,mode='direct_emitter',availability=self.av)
        self.assertAlmostEqual(b[1],2*a[1]);self.assertAlmostEqual(b[0],a[0])
        self.assertAlmostEqual(C0*a[0],np.linalg.norm(x[:2]-self.rx.position_m)-np.linalg.norm(x[:2]-self.ref.position_m))
    def test_shared_reference_covariance_not_diagonal(self):
        R=np.diag([4.,9.,16.])*1e-18
        A=difference_covariance(R,0)
        np.testing.assert_allclose(A,np.array([[13.,4.],[4.,20.]])*1e-18,atol=1e-31)
    def test_single_aoa_has_no_radial_information(self):
        _,H=measurement_and_jacobian(self.x,self.rx,self.ref,1e9,mode='direct_emitter',availability=self.av)
        self.assertAlmostEqual(H[2,:2]@(self.x[:2]-self.rx.position_m),0.,places=12)
        self.assertLess(np.linalg.matrix_rank(H[:,:4]),4)
    def test_aoa_yaw_period_and_reference_availability(self):
        y,_=measurement_and_jacobian(self.x,self.rx,self.ref,1e9,mode='direct_emitter',availability=self.av)
        xx=self.x.copy();xx[6]+=2*np.pi
        yy,_=measurement_and_jacobian(xx,self.rx,self.ref,1e9,mode='direct_emitter',availability=self.av)
        self.assertAlmostEqual(y[2],yy[2])
        for mode in ['direct_emitter','bistatic_scatter']:
            with self.subTest(mode=mode),self.assertRaises(ValueError):measurement_and_jacobian(self.x,self.rx,self.ref,1e9,mode=mode,availability=LinkAvailability())
    def test_singular_geometry_rejected(self):
        x=self.x.copy();x[:2]=self.rx.position_m
        with self.assertRaises(ValueError):measurement_and_jacobian(x,self.rx,self.ref,1e9,mode='direct_emitter',availability=self.av)
    def test_full_joint_frequency_covariance_changes_grid(self):
        P=np.diag([250000.,250000.,100.,100.,1e-14,1600.,.01])
        out=[]
        for sign in [-1,1]:
            Q=P.copy();Q[0,5]=Q[5,0]=sign*10000
            out.append(causal_frequency_grid(self.x,Q,self.rx,self.ref,1e9,4000,1,mode='direct_emitter',availability=self.av))
        self.assertNotAlmostEqual(out[0]['predicted_variance_hz2'],out[1]['predicted_variance_hz2'])
        self.assertNotEqual(out[0]['grid_hz'],out[1]['grid_hz'])
    def test_alias_and_budget_not_silent_truncation(self):
        P=np.diag([100.,100.,1.,1.,1e-15,100.,.01]);x=self.x.copy();x[5]=1e5
        a=causal_frequency_grid(x,P,self.rx,self.ref,1e9,1000,1,mode='direct_emitter',availability=self.av)
        self.assertEqual(a['status'],'aliased_prediction_requires_unwrapping');self.assertEqual(a['grid_hz'],[])
        x[5]=0
        b=causal_frequency_grid(x,P,self.rx,self.ref,1e9,10000,.001,mode='direct_emitter',availability=self.av,max_points=3)
        self.assertEqual(b['status'],'grid_budget_or_resolution_unavailable')

if __name__=='__main__':unittest.main(verbosity=2)
