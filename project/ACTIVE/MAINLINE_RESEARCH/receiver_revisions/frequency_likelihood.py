"""Noise-modelled FDOA reference; likelihood and posterior are separate objects.

At aligned samples: r=s+er, y=g*exp(j(phi+2*pi*f*t))*s+ey.
s ~ CN(0, source_variance), er and ey independent CN(0, noise_variance).
Samples are independent; the phase is constant with a uniform prior. Gain has
an explicitly supplied finite prior. These are voltage-squared sample units,
NOT a receiver noise figure or a bandwidth-normalised spectral density.

Marginalise the unknown waveform by conditioning s on the *received* r,
then integrate phase analytically and gain by its finite prior. The resulting
likelihood is not an arbitrary temperature-scaled CAF. This reference does not
model coloured samples, time-varying amplitude, CAF delay coupling, or tracking
association. No exact emitted waveform is an input.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import i0e, logsumexp
from ._validation import signal_1d, positive, grid_1d, probabilities, NoUsableSignal

@dataclass(frozen=True)
class ReceiverNoiseModel:
    source_variance: float
    reference_noise_variance: float
    surveillance_noise_variance: float
    gain_values: tuple[float,...] = (1.0,)
    gain_probabilities: tuple[float,...] = (1.0,)

    def validated(self):
        ps=positive(self.source_variance,'source_variance')
        nr=float(self.reference_noise_variance)
        if not np.isfinite(nr) or nr<0: raise ValueError('reference_noise_variance must be finite and nonnegative')
        ny=positive(self.surveillance_noise_variance,'surveillance_noise_variance')
        g=np.asarray(self.gain_values,dtype=float)
        if g.ndim!=1 or not len(g) or not np.all(np.isfinite(g)) or np.any(g<0):
            raise ValueError('gains must be nonnegative finite values')
        w=probabilities(self.gain_probabilities,len(g),'gain_probabilities')
        return ps,nr,ny,g,w

@dataclass(frozen=True)
class FrequencyLikelihood:
    grid_hz: np.ndarray
    log_likelihood: np.ndarray
    fs_hz: float
    sample_count: int
    noise_model: ReceiverNoiseModel
    semantic_role: str = 'phase_gain_marginalized_observation_likelihood_no_frequency_prior'

@dataclass(frozen=True)
class FrequencyPosterior:
    grid_hz: np.ndarray
    probabilities: np.ndarray
    mean_hz: float
    variance_hz2: float
    boundary_mass: float
    log_probabilities: np.ndarray
    semantic_role: str = 'finite_grid_posterior_includes_declared_frequency_prior'

    def credible_indices(self, mass: float=.9) -> np.ndarray:
        if not 0<mass<=1: raise ValueError('mass must be in (0,1]')
        order=np.argsort(-self.probabilities,kind='stable')
        count=min(len(order),int(np.searchsorted(np.cumsum(self.probabilities[order]),mass))+1)
        return order[:count]


def _paired_inputs(reference, surveillance, fs_hz, grid_hz):
    r=signal_1d(reference,'received_reference'); y=signal_1d(surveillance,'received_surveillance')
    if r.shape!=y.shape: raise ValueError('aligned received channels must have equal length')
    fs=positive(fs_hz,'fs_hz'); grid=grid_1d(grid_hz,'frequency_grid_hz')
    # One half-open Nyquist interval. Duplicated aliases are not independent modes.
    if np.any(grid < -fs/2) or np.any(grid >= fs/2):
        raise ValueError('grid must be inside [-fs/2,fs/2); unaliasing needs another model')
    if len(r)*len(grid)>4_000_000:
        raise ValueError('reference evaluation domain exceeds the bounded 4-million sample-grid budget')
    return r,y,fs,grid


def frequency_likelihood(reference, surveillance, fs_hz: float, grid_hz,
                         noise_model: ReceiverNoiseModel) -> FrequencyLikelihood:
    r,y,fs,grid=_paired_inputs(reference,surveillance,fs_hz,grid_hz)
    ps,nr,ny,gains,weights=noise_model.validated()
    a=ps/(ps+nr)
    conditional_source_variance=ps*nr/(ps+nr)
    t=np.arange(len(r))/fs
    cross=np.exp(-2j*np.pi*grid[:,None]*t[None,:]) @ (r.conj()*y)
    er=float(np.vdot(r,r).real); ey=float(np.vdot(y,y).real)
    gain_terms=[]
    for gain,weight in zip(gains,weights):
        if weight==0: continue
        variance=ny+gain*gain*conditional_source_variance
        amplitude=gain*a
        kappa=2*amplitude*np.abs(cross)/variance
        log_bessel=np.log(i0e(kappa))+kappa
        term=(-len(r)*np.log(variance)-(ey+amplitude*amplitude*er)/variance+log_bessel+np.log(weight))
        gain_terms.append(term)
    log_likelihood=logsumexp(np.stack(gain_terms),axis=0)
    if not np.all(np.isfinite(log_likelihood)):
        raise FloatingPointError('likelihood exceeded finite numerical range; do not replace by a uniform posterior')
    return FrequencyLikelihood(grid.copy(),log_likelihood,fs,len(r),noise_model)


def frequency_posterior(likelihood: FrequencyLikelihood, prior_probabilities=None) -> FrequencyPosterior:
    """Discrete grid prior, not an assumed continuous-frequency credible interval.

    Callers inserting likelihood into a tracker should use log_likelihood, not
    this already-prior-conditioned posterior. Off-grid error is not represented
    by the discrete variance; boundary mass is diagnostic, not proof of coverage.
    """
    n=len(likelihood.grid_hz)
    prior=probabilities(np.ones(n) if prior_probabilities is None else prior_probabilities,n,'frequency_prior')
    log_prior=np.full(n,-np.inf); keep=prior>0; log_prior[keep]=np.log(prior[keep])
    logits=likelihood.log_likelihood+log_prior
    logp=logits-logsumexp(logits)
    p=np.exp(logp)
    mean=float(p@likelihood.grid_hz)
    variance=float(p@((likelihood.grid_hz-mean)**2))
    return FrequencyPosterior(likelihood.grid_hz.copy(),p,mean,variance,float(p[0]+p[-1]),logp)


def caf_peak_reference(reference, surveillance, fs_hz: float, grid_hz) -> dict:
    """Point-only reference valid also for noiseless data; no invented covariance."""
    r,y,fs,grid=_paired_inputs(reference,surveillance,fs_hz,grid_hz)
    cross=np.exp(-2j*np.pi*grid[:,None]*(np.arange(len(r))/fs)[None,:]) @ (r.conj()*y)
    power=np.abs(cross)**2
    if not np.any(power>0): raise NoUsableSignal('zero CAF: no point estimate')
    i=int(np.argmax(power))
    return {'frequency_hz':float(grid[i]),'variance_hz2':None,'variance_reason':'point_only_no_noise_calibration',
            'boundary_peak':i in (0,len(grid)-1),'likelihood':None,'posterior':None}


def subtract_independent_oscillator(posterior: FrequencyPosterior, mean_hz: float, variance_hz2: float) -> dict:
    """Scalar independent bias marginalisation; forbidden for a shared joint bias.

    The caller must explicitly opt into independence. A correlated oscillator
    already in the owner's joint state must instead enter that joint likelihood.
    """
    if not np.isfinite(mean_hz) or not np.isfinite(variance_hz2) or variance_hz2<0:
        raise ValueError('oscillator moments must be finite with nonnegative variance')
    return {'mean_hz':posterior.mean_hz-float(mean_hz),'variance_hz2':posterior.variance_hz2+float(variance_hz2),
            'assumption':'independent calibration posterior; not valid for correlated/shared joint bias'}


def continuous_frequency_estimate(reference, surveillance, fs_hz, grid_hz,
                                  noise_model: ReceiverNoiseModel, max_modes=3):
    """Bounded continuous local likelihood modes, NOT a categorical variance.

    First bracket peaks with the caller's causal capture grid, refine each peak
    with the original sample likelihood, and compute observed information by a
    local symmetric derivative. Ambiguous modes and capture-boundary maxima are
    retained as such; they are not collapsed into a zero-variance Gaussian.
    Curvature uncertainty is a local asymptotic approximation, not a guaranteed
    finite-SNR confidence interval. Positive receiver noise is required.
    """
    from scipy.optimize import minimize_scalar
    coarse=frequency_likelihood(reference,surveillance,fs_hz,grid_hz,noise_model)
    f=coarse.grid_hz;ell=coarse.log_likelihood
    peaks=[i for i in range(1,len(f)-1) if ell[i]>=ell[i-1] and ell[i]>=ell[i+1]]
    peaks=sorted(peaks,key=lambda i:-ell[i])[:max_modes]
    modes=[];evaluations=len(f)
    def loglike(x):
        nonlocal evaluations
        evaluations+=1
        # frequency_likelihood requires >=2 grid points; evaluate neighboring
        # samples and use one. The spacing is not the uncertainty estimate.
        eps=max(1e-8,abs(float(x))*1e-10)
        return float(frequency_likelihood(reference,surveillance,fs_hz,[x,x+eps],noise_model).log_likelihood[0])
    for i in peaks:
        fit=minimize_scalar(lambda x:-loglike(x),bounds=(float(f[i-1]),float(f[i+1])),method='bounded',
                            options={'maxiter':40,'xatol':1e-9})
        x=float(fit.x);h=min(float(f[i+1]-f[i-1])/128.,fs_hz/len(reference)/128.)
        a=loglike(x-h);b=loglike(x);c=loglike(x+h);information=-(a-2*b+c)/h**2
        if not fit.success or information<=0 or not np.isfinite(information):continue
        modes.append({'mean_hz':x,'variance_hz2':1/information,'log_likelihood':b,'observed_information_s2':information})
    boundary=max(ell[0],ell[-1])>=max((x['log_likelihood'] for x in modes),default=-np.inf)
    if not modes or boundary:
        return {'status':'CAPTURE_BOUNDARY_OR_NO_IDENTIFIABLE_MODE','mean_hz':None,'variance_hz2':None,
                'modes':modes,'evaluations':evaluations,'semantic_role':'unresolved_observation_likelihood'}
    modes.sort(key=lambda x:-x['log_likelihood'])
    ambiguous=len(modes)>1 and modes[0]['log_likelihood']-modes[1]['log_likelihood']<4.6
    return {'status':'MULTIMODAL_REQUIRES_MIXTURE' if ambiguous else 'LOCAL_CONTINUOUS_LIKELIHOOD',
            'mean_hz':None if ambiguous else modes[0]['mean_hz'],
            'variance_hz2':None if ambiguous else modes[0]['variance_hz2'],
            'modes':modes,'evaluations':evaluations,'capture_hz':[float(f[0]),float(f[-1])],
            'semantic_role':'sample_likelihood_local_curvature_no_tracking_prior'}
