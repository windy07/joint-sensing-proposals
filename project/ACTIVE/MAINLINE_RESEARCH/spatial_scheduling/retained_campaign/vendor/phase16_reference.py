"""Independent finite-Gaussian-mixture Bayes reference; never a live posterior.

All state/measurement coordinates are explicitly metres. A scalar observation
uses fixed affine H,b and independent proper Gaussian noise; existence is one.
Quantization bounds are valid over real arithmetic. Their float evaluation and
QUADPACK error estimates are NOT outward-rounded certificates: numerical
agreement must never authorize a physical action. No Phase15 witness helper,
DFG helper, truth state, sampling, or trained score is imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, logsumexp


class NumericUnavailable(ValueError):
    pass


def _finite(a):
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all():
        raise NumericUnavailable('nonfinite declared model')
    a.flags.writeable=False
    return a


@dataclass(frozen=True)
class Experiment:
    weights: tuple
    means: tuple
    covariances: tuple
    h: tuple | None
    noise_variance: float | None
    cost: float = 0.0
    offset: float = 0.0
    task_index: int = 0
    units: tuple = ('m', 'm')
    observation_unit: str = 'm'
    decision_epoch: int = 0
    terminal_epoch: int = 0
    available_epoch: int = 0
    truth_access: bool = False


class BayesReference:
    """Paid offline calculation from an immutable public mixture snapshot."""
    def __init__(self, experiment: Experiment, meter):
        if not callable(meter):
            raise ValueError('explicit prepaid calculation meter required')
        meter('reference_models', 1)
        self.meter = meter
        self.e = experiment
        self.w, self.mu, self.cov = map(_finite, (
            experiment.weights, experiment.means, experiment.covariances))
        if self.mu.ndim != 2 or self.w.shape != (len(self.mu),):
            raise ValueError('finite complete mixture shape')
        self.j, self.n = self.mu.shape
        if (self.j < 1 or self.cov.shape != (self.j, self.n, self.n)
                or np.any(self.w <= 0) or abs(math.fsum(self.w)-1) > 1e-12):
            raise ValueError('positive normalized complete mixture required')
        self.w = self.w / math.fsum(self.w)
        self.w.flags.writeable=False
        if (experiment.units != ('m',)*self.n or experiment.observation_unit != 'm'
                or experiment.truth_access is not False
                or type(experiment.task_index) is not int
                or not 0 <= experiment.task_index < self.n
                or any(type(t) is not int or t<0 for t in (experiment.available_epoch,experiment.decision_epoch,experiment.terminal_epoch))
                or experiment.available_epoch > experiment.decision_epoch
                or experiment.terminal_epoch < experiment.decision_epoch):
            raise ValueError('unsupported units, truth, task or causal horizon')
        if not math.isfinite(experiment.cost) or experiment.cost < 0:
            raise ValueError('nonnegative finite m^2 cost required')
        for c in self.cov:
            if not np.array_equal(c, c.T):
                raise ValueError('explicitly symmetric covariance required')
            np.linalg.cholesky(c)
        i = experiment.task_index
        self.x = self.mu[:, i]
        self.vx = self.cov[:, i, i]
        self.prior_mean = float(self.w @ self.x)
        self.prior_variance = float(self.w @ (self.vx+(self.x-self.prior_mean)**2))
        self.active = experiment.h is not None
        if not self.active:
            if experiment.noise_variance is not None or experiment.cost != 0 or experiment.offset != 0:
                raise ValueError('fallback is no observation and zero cost')
            return
        h = _finite(experiment.h)
        if h.shape != (self.n,) or not math.isfinite(experiment.offset):
            raise ValueError('affine observation dimension')
        if experiment.noise_variance is None or not math.isfinite(experiment.noise_variance) or experiment.noise_variance <= 0:
            raise ValueError('positive finite noise variance required')
        self.h = h
        self.ch = np.einsum('jik,k->ji', self.cov, h)
        self.s = self.ch @ h + experiment.noise_variance
        self.eta = self.mu @ h + experiment.offset
        self.sd = np.sqrt(self.s)
        self.k = self.ch[:, i] / self.s
        self.v = self.vx - self.ch[:, i]**2 / self.s
        if np.any(self.v <= 0) or not np.isfinite(self.v).all():
            raise NumericUnavailable('proper conditional variances required; no clipping')

    def conditional(self, y):
        self.meter('conditional_evaluations', 1)
        if not self.active or not math.isfinite(y):
            raise ValueError('finite active scalar observation required')
        logq = np.log(self.w)-0.5*(np.log(2*math.pi*self.s)+(y-self.eta)**2/self.s)
        lp = float(logsumexp(logq))
        weights = np.exp(logq-lp)
        if np.any(weights<=0) or not np.isfinite(weights).all():
            raise NumericUnavailable('conditional mode probability underflow/nonfinite; no pruning')
        means = self.mu+self.ch/self.s[:, None]*(y-self.eta)[:, None]
        covariance = self.cov-np.einsum('ji,jk->jik', self.ch, self.ch)/self.s[:, None, None]
        # A reference-only posterior calculation; caller must never publish it.
        return dict(weights=weights, means=means, covariances=covariance,
                    log_predictive_density=lp)

    @staticmethod
    def _phi(z):
        z = np.asarray(z)
        return np.exp(-0.5*z*z)/math.sqrt(2*math.pi)

    def _bin(self, left, right, threshold=None):
        self.meter('quantization_cells', 1)
        a, b = (left-self.eta)/self.sd, (right-self.eta)/self.sd
        # Right-tail subtraction uses survival values, avoiding 1-1 loss.
        p = np.where(a >= 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
        pa, pb = self._phi(a), self._phi(b)
        ap = np.multiply(a, pa, out=np.zeros_like(a), where=np.isfinite(a))
        bp = np.multiply(b, pb, out=np.zeros_like(b), where=np.isfinite(b))
        z1, z2 = pa-pb, p+ap-bp
        if np.any(p < 0) or not np.isfinite(z2).all():
            raise NumericUnavailable('truncated normal arithmetic failure')
        sx = self.x*p+self.k*self.sd*z1
        hx = (self.v+self.x**2)*p+2*self.x*self.k*self.sd*z1+self.k**2*self.s*z2
        probability = float(self.w @ p)
        moment = float(self.w @ sx)
        second = float(self.w @ hx)
        genie = float((self.w*self.v) @ p)
        if probability <= 0:
            # A zero due to floating underflow is not evidence of an impossible bin.
            raise NumericUnavailable('unresolved zero/underflow bin probability')
        risk = second-moment*moment/probability
        if risk < -1e-12 or genie < -1e-12:
            raise NumericUnavailable('negative bin risk')
        if math.isfinite(left) and math.isfinite(right):
            mm = np.stack((self.x+self.k*(left-self.eta), self.x+self.k*(right-self.eta)))
            ss = np.stack((-(left-self.eta)/self.s, -(right-self.eta)/self.s))
            mr, sr = float(np.ptp(mm)), float(np.ptp(ss))
            lipschitz = float(np.max(np.abs(self.k)))+mr*sr/4
            gap = probability*lipschitz*lipschitz*(right-left)**2/4
            # Both bounds are real-arithmetic consequences; no floating certificate.
            gap = min(gap, max(0., risk-genie))
        else:
            lipschitz = None
            gap = max(0., risk-genie)
        tail_lower=tail_upper=None
        if threshold is not None:
            radius=math.sqrt(threshold); conditional_sd=np.sqrt(self.v)
            if math.isfinite(left) and math.isfinite(right):
                mid=(left+right)/2
                self.meter('conditional_evaluations',1)
                logq=np.log(self.w)-.5*(np.log(2*math.pi*self.s)+(mid-self.eta)**2/self.s)
                wm=np.exp(logq-logsumexp(logq))
                mean_mid=float(wm @ (self.x+self.k*(mid-self.eta)))
                mean_radius=lipschitz*(right-left)/2
                local_lo=np.min(mm,axis=0)-mean_mid-mean_radius
                local_hi=np.max(mm,axis=0)-mean_mid+mean_radius
                near=np.where((local_lo<=0)&(local_hi>=0),0.,np.minimum(np.abs(local_lo),np.abs(local_hi)))
                far=np.maximum(np.abs(local_lo),np.abs(local_hi))
                lowp=ndtr((-radius-near)/conditional_sd)+ndtr((near-radius)/conditional_sd)
                highp=ndtr((-radius-far)/conditional_sd)+ndtr((far-radius)/conditional_sd)
                tail_lower=float((self.w*p) @ lowp)
                tail_upper=float((self.w*p) @ highp)
            else:
                tail_lower=float((self.w*p) @ (2*ndtr(-radius/conditional_sd)))
                tail_upper=min(probability,max(0.,risk)/threshold)
        return dict(left=left, right=right, probability=probability, first=moment,
                    second=second, risk=risk, genie=genie, gap=gap,
                    lipschitz=lipschitz,tail_lower=tail_lower,tail_upper=tail_upper)

    def quantization_bounds(self, *, bins=4096, tail_sd=10., threshold=.25):
        if not math.isfinite(threshold) or threshold<=0:
            raise ValueError('finite positive squared-error threshold required')
        if not self.active:
            return dict(lower=self.prior_variance, upper=self.prior_variance,
                        cell_count=0, genie=self.prior_variance,
                        finite_gap=0., tail_gap=0., formal_numeric_certificate=False)
        if type(bins) is not int or not 2 <= bins <= 16384 or not math.isfinite(tail_sd) or not 5 <= tail_sd <= 12:
            raise ValueError('frozen bounded bin configuration required')
        lo, hi = float(np.min(self.eta-tail_sd*self.sd)), float(np.max(self.eta+tail_sd*self.sd))
        edges = np.linspace(lo, hi, bins+1)
        rows = [self._bin(-math.inf, lo, threshold)]
        rows += [self._bin(float(a), float(b), threshold) for a,b in zip(edges[:-1], edges[1:])]
        rows += [self._bin(hi, math.inf, threshold)]
        upper = math.fsum(r['risk'] for r in rows)
        genie = float(self.w @ self.v)
        gap = math.fsum(r['gap'] for r in rows)
        lower = max(genie, upper-gap)
        probability = math.fsum(r['probability'] for r in rows)
        if abs(probability-1) > 1e-10 or lower > upper+1e-10:
            raise NumericUnavailable('quantization probability/risk consistency failed')
        return dict(lower=lower+self.e.cost, upper=upper+self.e.cost,
                    genie=genie+self.e.cost, finite_gap=math.fsum(r['gap'] for r in rows[1:-1]),
                    tail_gap=rows[0]['gap']+rows[-1]['gap'], probability_sum=probability,
                    cell_count=len(rows), domain=[lo,hi], formal_numeric_certificate=False,
                    tail_lower=math.fsum(r['tail_lower'] for r in rows),
                    tail_upper=math.fsum(r['tail_upper'] for r in rows),
                    algorithm='truncated-moment quantization with posterior-mean oscillation')

    def _conditional_metrics(self, y, threshold):
        self.meter('integrand_evaluations', 1)
        logq = np.log(self.w)-.5*(np.log(2*math.pi*self.s)+(y-self.eta)**2/self.s)
        lp = float(logsumexp(logq)); density = math.exp(lp)
        w = np.exp(logq-lp)
        t = self.x+self.k*(y-self.eta)
        m = float(w @ t)
        d = t-m
        variance = float(w @ (self.v+d*d))
        c = math.sqrt(threshold); sd = np.sqrt(self.v)
        a, b = (c-d)/sd, (-c-d)/sd
        p = ndtr(-a)+ndtr(b)
        second_tail = (d*d+self.v)*p+sd*(d+c)*self._phi(a)+sd*(c-d)*self._phi(b)
        excess = second_tail-threshold*p
        if np.any(excess < -1e-12):
            raise NumericUnavailable('negative stop-loss integrand')
        return (density*variance, density*float(w @ p), density*float(w @ excess))

    def integrate(self, *, threshold=.25, tail_mass=.1, tail_sd=10., epsabs=1e-10, epsrel=1e-8):
        """Finite-domain quadrature + analytic omitted-tail upper bounds.

        QUADPACK's abserr is an estimate, not a proved numerical enclosure.
        Tail probability and CVaR are physical squared error, excluding cost.
        The stop-loss formula at the fixed threshold yields a CVaR upper bound,
        not the optimized CVaR. No threshold search or outcome-based tuning.
        """
        if (not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(tail_mass)
                or not 0 < tail_mass <= 1 or not math.isfinite(tail_sd) or not 5<=tail_sd<=12
                or not math.isfinite(epsabs) or not 0<epsabs<=1e-4
                or not math.isfinite(epsrel) or not 0<epsrel<=1e-4):
            raise ValueError('positive squared-error threshold and tail mass required')
        if not self.active:
            d = self.x-self.prior_mean; sd = np.sqrt(self.vx); c = math.sqrt(threshold)
            a,b = (c-d)/sd,(-c-d)/sd
            p = ndtr(-a)+ndtr(b)
            excess = (d*d+self.vx-threshold)*p+sd*(d+c)*self._phi(a)+sd*(c-d)*self._phi(b)
            return dict(mse=self.prior_variance, objective=self.prior_variance,
                        tail_probability=float(self.w @ p), stop_loss=float(self.w @ excess),
                        cvar_upper=threshold+float(self.w @ excess)/tail_mass,
                        omitted_risk_upper=0., omitted_probability=0., error_estimates=[0.,0.,0.],
                        formal_numeric_certificate=False, tail_includes_resource_cost=False,
                        cvar_upper_is_numerical_estimate=True,reported_integral_scope='closed-form no-observation')
        lo, hi = float(np.min(self.eta-tail_sd*self.sd)), float(np.max(self.eta+tail_sd*self.sd))
        tails = (self._bin(-math.inf,lo), self._bin(hi,math.inf))
        omitted_risk = math.fsum(r['second'] for r in tails)
        omitted_p = math.fsum(r['probability'] for r in tails)
        points = sorted(set([lo,hi]+[float(z) for z in self.eta if lo < z < hi]))
        values=[]; errors=[]
        for metric in range(3):
            val=[]; err=[]
            for a,b in zip(points[:-1],points[1:]):
                self.meter('quadrature_calls',1)
                value,error = quad(lambda y:self._conditional_metrics(y,threshold)[metric],
                                   a,b,epsabs=epsabs,epsrel=epsrel,limit=200)
                val.append(value);err.append(error)
            values.append(math.fsum(val));errors.append(math.fsum(err))
        return dict(mse=values[0], objective=values[0]+self.e.cost,
                    tail_probability=values[1], stop_loss=values[2],
                    cvar_upper=threshold+(values[2]+omitted_risk)/tail_mass,
                    omitted_risk_upper=omitted_risk,omitted_probability=omitted_p,
                    error_estimates=errors, domain=[lo,hi],
                    formal_numeric_certificate=False, tail_includes_resource_cost=False,
                    cvar_upper_is_numerical_estimate=True,reported_integral_scope='finite domain; omitted tails bounded separately',
                    algorithm='independent direct posterior-variance and conditional-error quadrature')


def certified_decision(*, numerical_result, risk_limit, budget_available):
    """Numerical development evidence cannot promote itself to exact authority."""
    return {'status':'UNKNOWN', 'committed':False,
            'reason': 'BUDGET_EXHAUSTED' if not budget_available else 'NO_OUTWARD_ROUNDED_OR_PHYSICAL_CERTIFICATE',
            'risk_limit_m2':risk_limit, 'numeric_result_available':numerical_result is not None}
