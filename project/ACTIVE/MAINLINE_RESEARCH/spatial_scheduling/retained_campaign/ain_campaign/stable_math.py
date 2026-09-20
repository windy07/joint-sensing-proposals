"""Local numerical primitives; no posterior owner, policy import or physical commit.

The probability law retains finite source log-weights, even if exp(log_w) is
below binary64's range. Numerical quadrature estimates are never certificates.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp, expit, ndtr
from scipy.integrate import quad
from .common import ContractError


def mixture_weights(doc):
    w=np.asarray(doc['weights'],dtype=float)
    if w.ndim!=1 or len(w) not in (1,2) or not np.all(np.isfinite(w)) or np.any(w<0):
        raise ContractError('INVALID_MIXTURE_WEIGHTS')
    if 'log_weights' in doc:
        lw=np.asarray(doc['log_weights'],dtype=float)
        if lw.shape!=w.shape or not np.all(np.isfinite(lw)):
            raise ContractError('NONFINITE_SOURCE_LOG_WEIGHTS')
        normalizer=float(logsumexp(lw))
        if abs(normalizer)>1e-8:raise ContractError('UNNORMALIZED_SOURCE_LOG_WEIGHTS')
        lw=lw-normalizer
        represented=np.exp(lw)
        if not np.allclose(w,represented,rtol=1e-10,atol=1e-14):
            raise ContractError('WEIGHT_LOG_WEIGHT_DISAGREEMENT')
    else:
        if np.any(w<=0) or abs(math.fsum(w)-1)>1e-10:
            raise ContractError('POSITIVE_MIXTURE_WEIGHT_REQUIRED')
        lw=np.log(w);lw=lw-float(logsumexp(lw));represented=np.exp(lw)
    # No component is removed: row count, log mass, means and covariance persist.
    return represented,lw


def conditional_position_variance(P,H,R,k):
    """Joseph-form scalar covariance using the same solved gain; no jitter."""
    P=np.asarray(P,dtype=float);H=np.asarray(H,dtype=float);R=np.asarray(R,dtype=float)
    if not np.allclose(P,P.T,rtol=1e-10,atol=1e-12*max(1.,float(np.max(np.abs(P))))):
        raise ContractError('ASYMMETRIC_PRIOR_COVARIANCE')
    selector=np.zeros(P.shape[0]);selector[0]=1.
    residual=selector-k@H
    v=float(residual@P@residual+k@R@k)
    if not math.isfinite(v) or v<=0:raise ContractError('NONPOSITIVE_JOSEPH_VARIANCE')
    return v


def overlap_tail_refinement(weights,logweights,alpha,d,v,rho,L2):
    """Bounded one-dimensional numerical fallback after unresolved GH estimates.

    Integrate on each conditional standardized normal coordinate [-10,10].
    Posterior variance <=1/4 and tail probability <=1. Therefore truncation
    errors are <=0.5*Phi(-10) and <=2*Phi(-10), respectively. QUADPACK's error
    estimates and binary64 rounding are not rigorous enclosures.
    """
    if not (math.isfinite(alpha) and alpha>0 and v>0 and L2>0):
        raise ContractError('SCALAR_REFINEMENT_DOMAIN')
    weights=np.asarray(weights,float);logweights=np.asarray(logweights,float)
    logodds=float(logweights[1]-logweights[0]);sd=math.sqrt(v);count=0
    J=tail=ej=et=0.;limit=10.;norm=1/math.sqrt(2*math.pi)
    for mode,prior in enumerate(weights):
        # Keeping a zero represented mass does not remove its source row.
        center=(2*mode-1)*alpha
        transition=-logodds/(2*alpha)-center
        points=sorted(set(x for k in (-16.,-4.,-1.,0.,1.,4.,16.)
                          if -limit<(x:=transition+k/alpha)<limit))
        def values(z,which):
            nonlocal count
            count+=1;p=float(expit(logodds+2*alpha*(center+z)))
            if which==0:value=p*(1-p)
            else:
                m0=-p*d;m1=(1-p)*d
                t0=ndtr((-rho-m0)/sd)+ndtr((m0-rho)/sd)
                t1=ndtr((-rho-m1)/sd)+ndtr((m1-rho)/sd)
                value=(1-p)*t0+p*t1
            return norm*math.exp(-.5*z*z)*value
        if prior==0.:continue
        x,e=quad(values,-limit,limit,args=(0,),epsabs=1e-11,epsrel=1e-9,limit=64,points=points)
        y,f=quad(values,-limit,limit,args=(1,),epsabs=1e-10,epsrel=1e-9,limit=64,points=points)
        J+=prior*x;tail+=prior*y;ej+=prior*e;et+=prior*f
    ej+=.5*float(ndtr(-limit));et+=2*float(ndtr(-limit))
    if not all(math.isfinite(x) for x in (J,tail,ej,et)) or not 0<=J<=.25+1e-12 or not 0<=tail<=1+1e-12:
        raise ContractError('SCALAR_REFINEMENT_FAILED')
    return J,tail,d*d*ej,et,count
