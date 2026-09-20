"""Causal planar passive measurement references, not a second posterior owner.

State order and units: x[m], y[m], vx[m/s], vy[m/s], effective delay bias[s],
frequency bias[Hz], angle bias[rad]. Biases are observable *combinations*; this
model does not identify separate clocks, cables and array-element phase errors.
All station/source states and availability supplied here are control-view inputs.
The unknown target vector is a belief mean in prediction and world truth only
when an external simulator explicitly generates measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ._validation import positive

C0=299_792_458.0

@dataclass(frozen=True)
class Station:
    position_m: tuple[float,float]
    velocity_mps: tuple[float,float]=(0.0,0.0)
    array_yaw_rad: float=0.0

@dataclass(frozen=True)
class LinkAvailability:
    target_signal_available: bool=False
    illuminator_available: bool=False
    measured_reference_available: bool=False


def _station(s: Station):
    p=np.asarray(s.position_m,float);v=np.asarray(s.velocity_mps,float)
    if p.shape!=(2,) or v.shape!=(2,) or not np.all(np.isfinite(np.r_[p,v,s.array_yaw_rad])):
        raise ValueError('station coordinates, velocity and yaw must be finite planar values')
    return p,v


def _geometry(p,station):
    s,v=_station(station);d=p-s;rho=float(np.linalg.norm(d))
    if rho<=1e-9: raise ValueError('target coincides with a station: singular range/angle geometry')
    u=d/rho;P=(np.eye(2)-np.outer(u,u))/rho
    return rho,u,P,v


def measurement_and_jacobian(state, receiver: Station, reference: Station, carrier_hz: float,
                             *, mode: str, availability: LinkAvailability):
    """[TDOA seconds, FDOA Hz, AOA rad] and analytic joint Jacobian.

    direct_emitter: reference is another receiver measuring the target emission.
    bistatic_scatter: reference is the external illuminator; receiver also has
    its own measured direct-path reference from that source. Delay and Doppler
    are excess relative to that direct path, not absolute monostatic range.
    """
    x=np.asarray(state,float)
    if x.shape!=(7,) or not np.all(np.isfinite(x)): raise ValueError('joint state must have seven finite components')
    fc=positive(carrier_hz,'carrier_hz')
    if mode=='direct_emitter':
        if not availability.target_signal_available or not availability.measured_reference_available:
            raise ValueError('direct-emitter mode unavailable without a measurable target emission in both channels')
    elif mode=='bistatic_scatter':
        if not availability.illuminator_available or not availability.measured_reference_available:
            raise ValueError('bistatic mode unavailable without external illumination and acquired direct reference')
    else: raise ValueError('mode must be direct_emitter or bistatic_scatter')
    p=x[:2];v=x[2:4];rho_i,u_i,P_i,vi=_geometry(p,receiver);rho_r,u_r,P_r,vr=_geometry(p,reference)
    H=np.zeros((3,7));k=fc/C0
    if mode=='direct_emitter':
        delay=(rho_i-rho_r)/C0
        freq=-k*(u_i@(v-vi)-u_r@(v-vr))
        H[0,:2]=(u_i-u_r)/C0
        H[1,:2]=-k*(P_i@(v-vi)-P_r@(v-vr));H[1,2:4]=-k*(u_i-u_r)
    else:
        ri,vi=_station(receiver);rt,vt=_station(reference)
        direct=ri-rt;direct_range=float(np.linalg.norm(direct))
        if direct_range<=1e-9: raise ValueError('bistatic direct-reference path has zero length')
        direct_rate=(direct/direct_range)@(vi-vt)
        delay=(rho_i+rho_r-direct_range)/C0
        freq=-k*(u_i@(v-vi)+u_r@(v-vt)-direct_rate)
        H[0,:2]=(u_i+u_r)/C0
        H[1,:2]=-k*(P_i@(v-vi)+P_r@(v-vt));H[1,2:4]=-k*(u_i+u_r)
    angle=(np.arctan2(u_i[1],u_i[0])-receiver.array_yaw_rad+x[6]+np.pi)%(2*np.pi)-np.pi
    H[2,:2]=np.array([-u_i[1],u_i[0]])/rho_i
    H[:,4:]=np.eye(3)
    return np.array([delay+x[4],freq+x[5],angle]),H


def reference_difference(n: int, reference: int) -> np.ndarray:
    if not isinstance(n,int) or n<2 or not 0<=reference<n: raise ValueError('invalid station count/reference index')
    indices=[i for i in range(n) if i!=reference]
    D=np.zeros((n-1,n));D[:,reference]=-1
    D[np.arange(n-1),indices]=1
    return D


def difference_covariance(station_covariance, reference: int) -> np.ndarray:
    R=np.asarray(station_covariance,float)
    if R.ndim!=2 or R.shape[0]!=R.shape[1] or not np.all(np.isfinite(R)):
        raise ValueError('station covariance must be finite square')
    scale=max(float(np.max(np.abs(R))),np.finfo(float).tiny)
    if np.max(np.abs(R-R.T))>1e-12*scale or np.linalg.eigvalsh(R/scale).min() < -1e-12:
        raise ValueError('station covariance must be symmetric positive semidefinite')
    D=reference_difference(len(R),reference)
    return D@R@D.T


def causal_frequency_grid(state_mean, joint_covariance, receiver: Station, reference: Station,
                          carrier_hz: float, fs_hz: float, step_hz: float, *, mode: str,
                          availability: LinkAvailability, sigma_multiplier: float=4.,
                          observation_variance_hz2: float=0., max_points: int=4096) -> dict:
    """Local Gaussian predictive interval using all cross-covariances.

    This is not a global nonlinear coverage guarantee. No target truth input is
    accepted. A predicted aliased centre is rejected rather than silently used
    as a physical unaliased frequency. Edge truncation is explicitly recorded.
    """
    mean,H=measurement_and_jacobian(state_mean,receiver,reference,carrier_hz,mode=mode,availability=availability)
    P=np.asarray(joint_covariance,float)
    if P.shape!=(7,7) or not np.all(np.isfinite(P)): raise ValueError('joint covariance must be finite 7x7')
    # Scale each physical coordinate before PSD tests; seconds and metres differ.
    units=np.array([1000.,1000.,100.,100.,1e-6,100.,1.])
    Q=P/units[:,None]/units[None,:]
    if not np.allclose(Q,Q.T,atol=1e-12,rtol=1e-12) or np.linalg.eigvalsh(Q).min()<-1e-10:
        raise ValueError('joint covariance is not symmetric positive semidefinite')
    obs=float(observation_variance_hz2)
    if not np.isfinite(obs) or obs<0: raise ValueError('observation variance must be nonnegative')
    fs=positive(fs_hz,'fs_hz');step=positive(step_hz,'step_hz');nsig=positive(sigma_multiplier,'sigma_multiplier')
    if not isinstance(max_points,int) or max_points<2: raise ValueError('max_points must be an integer >=2')
    var=float(H[1]@P@H[1]+obs)
    if var < -1e-10: raise FloatingPointError('negative predictive frequency variance')
    sd=float(np.sqrt(max(var,0)));centre=float(mean[1])
    result={'predicted_mean_hz':centre,'predicted_variance_hz2':max(var,0.),'sigma_multiplier':nsig,
            'carrier_hz':float(carrier_hz),'fs_hz':fs,'grid_hz':[],'status':'not_constructed',
            'semantic_role':'local_joint_belief_prediction_no_truth'}
    if not -fs/2<=centre<fs/2:
        result['status']='aliased_prediction_requires_unwrapping';return result
    width=max(step,nsig*sd);lo=centre-width;hi=centre+width
    imin=max(int(np.floor(lo/step)),int(np.ceil(-fs/2/step)))
    imax=min(int(np.ceil(hi/step)),int(np.ceil(fs/2/step))-1)
    n=imax-imin+1
    if n<2 or n>max_points:
        result['status']='grid_budget_or_resolution_unavailable';return result
    result['grid_hz']=(np.arange(imin,imax+1)*step).tolist()
    result['status']='nyquist_truncated' if lo < -fs/2 or hi >= fs/2 else 'local_predictive_grid'
    return result
