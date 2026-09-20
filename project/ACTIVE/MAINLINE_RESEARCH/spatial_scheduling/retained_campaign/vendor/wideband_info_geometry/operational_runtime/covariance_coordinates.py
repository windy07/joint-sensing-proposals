"""Coordinate-covariant covariance algebra for declared physical models.

No absolute variance floor is imposed on mixed-unit matrices. Positive diagonal
entries define normalization scales which transform along with their coordinates.
Only dimensionless roundoff-level PSD repair is allowed; unsupported rank,
asymmetry or conditioning raises an explicit model/numerical failure.
"""
from __future__ import annotations
import math
import numpy as np

ROUND_OFF_TOLERANCE = 1.0e-10
MAX_NORMALIZED_CONDITION = 1.0e12

class CovarianceModelError(ValueError):
    """A declared covariance cannot be evaluated under this numeric contract."""

def _normalized(matrix: np.ndarray):
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or a.shape[0]==0 or not np.all(np.isfinite(a)):
        raise CovarianceModelError('COVARIANCE_SHAPE_OR_NONFINITE')
    diagonal=np.diag(a)
    if np.any(diagonal<0):raise CovarianceModelError('NEGATIVE_VARIANCE')
    active=np.flatnonzero(diagonal>0)
    inactive=np.flatnonzero(diagonal==0)
    if inactive.size and (np.any(a[inactive,:]!=0) or np.any(a[:,inactive]!=0)):
        raise CovarianceModelError('ZERO_VARIANCE_WITH_NONZERO_CROSS_TERM')
    scales=np.sqrt(diagonal[active])
    correlation=a[np.ix_(active,active)]/scales[:,None]/scales[None,:]
    if not np.all(np.isfinite(correlation)):
        raise CovarianceModelError('NORMALIZATION_OVERFLOW')
    if correlation.size and np.max(np.abs(correlation-correlation.T))>ROUND_OFF_TOLERANCE:
        raise CovarianceModelError('NONSYMMETRIC_NORMALIZED_COVARIANCE')
    return a,active,scales,0.5*(correlation+correlation.T)

def checked_covariance(matrix: np.ndarray, *, require_spd: bool=False) -> np.ndarray:
    a,active,scales,c=_normalized(matrix)
    if not active.size:
        if require_spd:raise CovarianceModelError('SINGULAR_COVARIANCE_DENSITY_UNSUPPORTED')
        return np.zeros_like(a)
    eigenvalues=np.linalg.eigvalsh(c)
    if eigenvalues[0]<-ROUND_OFF_TOLERANCE:
        raise CovarianceModelError('INDEFINITE_NORMALIZED_COVARIANCE')
    if require_spd and (active.size!=a.shape[0] or eigenvalues[0]<=0 or eigenvalues[-1]/eigenvalues[0]>MAX_NORMALIZED_CONDITION):
        raise CovarianceModelError('SINGULAR_OR_ILL_CONDITIONED_NORMALIZED_COVARIANCE')
    if eigenvalues[0]<0:
        values,vectors=np.linalg.eigh(c)
        c=(vectors*np.maximum(values,0))@vectors.T
    out=np.zeros_like(a)
    out[np.ix_(active,active)]=scales[:,None]*c*scales[None,:]
    return out

def covariance_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a=checked_covariance(matrix,require_spd=True)
    d=np.sqrt(np.diag(a));c=a/d[:,None]/d[None,:]
    b=np.asarray(rhs,dtype=float)
    if b.ndim not in (1,2) or b.shape[0]!=a.shape[0] or not np.all(np.isfinite(b)):
        raise CovarianceModelError('SOLVE_RHS_SHAPE_OR_NONFINITE')
    scale=d if b.ndim==1 else d[:,None]
    solution=np.linalg.solve(c,b/scale)/scale
    if not np.all(np.isfinite(solution)):raise CovarianceModelError('SOLVE_OVERFLOW')
    return solution

def covariance_inverse(matrix: np.ndarray) -> np.ndarray:
    a=checked_covariance(matrix,require_spd=True)
    return covariance_solve(a,np.eye(a.shape[0]))

def covariance_logdet(matrix: np.ndarray) -> float:
    a=checked_covariance(matrix,require_spd=True)
    d=np.sqrt(np.diag(a));c=a/d[:,None]/d[None,:]
    root=np.linalg.cholesky(c)
    value=2.0*float(np.log(d).sum()+np.log(np.diag(root)).sum())
    if not math.isfinite(value):raise CovarianceModelError('LOGDET_NONFINITE')
    return value

def covariance_quadratic(residual: np.ndarray,covariance: np.ndarray) -> float:
    r=np.asarray(residual,dtype=float)
    a=checked_covariance(covariance,require_spd=True)
    if r.shape!=(a.shape[0],) or not np.all(np.isfinite(r)):
        raise CovarianceModelError('RESIDUAL_SHAPE_OR_NONFINITE')
    d=np.sqrt(np.diag(a));c=a/d[:,None]/d[None,:]
    whitened=np.linalg.solve(np.linalg.cholesky(c),r/d)
    quadratic=float(whitened@whitened)
    if not math.isfinite(quadratic):raise CovarianceModelError('QUADRATIC_NUMERIC_RANGE')
    return quadratic

def gaussian_log_density(residual: np.ndarray,covariance: np.ndarray) -> float:
    r=np.asarray(residual,dtype=float)
    quadratic=covariance_quadratic(r,covariance)
    value=-.5*(quadratic+covariance_logdet(covariance)+r.size*math.log(2*math.pi))
    if not math.isfinite(value):raise CovarianceModelError('DENSITY_NONFINITE')
    return value

def covariance_root(matrix: np.ndarray, *, require_spd: bool=True) -> np.ndarray:
    a=checked_covariance(matrix,require_spd=require_spd)
    _,active,d,c=_normalized(a)
    out=np.zeros_like(a)
    if not active.size:return out
    if require_spd:
        root=np.linalg.cholesky(c)
    else:
        # PSD supports are allowed for cubature/sampling, but no Lebesgue density
        # or inverse is thereby defined. Eigen-root degeneracy can affect paired
        # floating samples, so strong random coupling is only promised for SPD.
        values,vectors=np.linalg.eigh(c)
        root=vectors@np.diag(np.sqrt(np.maximum(values,0)))
    out[np.ix_(active,active)]=d[:,None]*root
    return out

def gaussian_draw(rng,mean: np.ndarray,covariance: np.ndarray) -> np.ndarray:
    mu=np.asarray(mean,dtype=float)
    if mu.ndim!=1 or not np.all(np.isfinite(mu)):
        raise CovarianceModelError('DRAW_MEAN_SHAPE_OR_NONFINITE')
    root=covariance_root(covariance)
    if root.shape[0]!=mu.size:raise CovarianceModelError('DRAW_SHAPE')
    # Validate before consuming randomness. Same ordered standard-normal stream
    # is coupled across positive diagonal coordinate changes in the new method.
    return mu+root@rng.standard_normal(mu.size)

def linear_gaussian_update(covariance: np.ndarray,h: np.ndarray,noise: np.ndarray):
    p=checked_covariance(covariance);r=checked_covariance(noise,require_spd=True)
    h=np.asarray(h,dtype=float)
    if h.ndim!=2 or h.shape!=(r.shape[0],p.shape[0]) or not np.all(np.isfinite(h)):
        raise CovarianceModelError('LINEAR_OBSERVATION_SHAPE_OR_NONFINITE')
    innovation=checked_covariance(h@p@h.T+r,require_spd=True)
    gain=covariance_solve(innovation,h@p).T
    residual_map=np.eye(p.shape[0])-gain@h
    # Joseph form avoids subtracting nearly equal, mixed-scale covariances.
    posterior=checked_covariance(residual_map@p@residual_map.T+gain@r@gain.T)
    return gain,posterior,innovation

def information_covariance_update(prior: np.ndarray,information: np.ndarray) -> np.ndarray:
    p=checked_covariance(prior,require_spd=True)
    j=checked_covariance(information)
    if j.shape!=p.shape:raise CovarianceModelError('INFORMATION_SHAPE_MISMATCH')
    d=np.sqrt(np.diag(p));c=p/d[:,None]/d[None,:]
    precision=covariance_inverse(c)+d[:,None]*j*d[None,:]
    return checked_covariance(d[:,None]*covariance_inverse(precision)*d[None,:],require_spd=True)
