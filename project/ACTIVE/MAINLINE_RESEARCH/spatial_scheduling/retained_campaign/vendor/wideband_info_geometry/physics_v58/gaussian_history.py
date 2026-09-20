"""Joint conditional history for a declared frozen linear Gaussian graph.

This is a conditional kernel, never a second current-state posterior. Its
boundary marginal belongs exclusively to DynamicFactorGraph. All array axes
carry explicit variable dimensions/units. Shared latent variables remain joint;
independent observation noise must be supplied by the likelihood model.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, fields
from typing import Any, Sequence
import numpy as np
from wideband_info_geometry.small_symmetric_linalg import platform_cholesky, platform_solve


def immutable(value):
    a=np.asarray(value,dtype=float)
    return np.frombuffer(a.tobytes(order='C'),dtype=a.dtype).reshape(a.shape)


def sym(value):
    a=np.asarray(value,dtype=float)
    return (a+a.T)*0.5


def solve_spd(matrix,right):
    """Positive diagonal equilibration; no ridge, PSD clipping or new noise."""
    a=sym(matrix); b=np.asarray(right,dtype=float)
    d=np.sqrt(np.diag(a))
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(d<=0):
        raise ValueError('Conditional covariance must be finite positive definite')
    c=a/d[:,None]/d[None,:]
    l=platform_cholesky(c)
    scaled=b/d if b.ndim==1 else b/d[:,None]
    result=platform_solve(l.T,platform_solve(l,scaled))
    return result/d if b.ndim==1 else result/d[:,None]


def slices(variables):
    out={}; start=0
    for var in variables:
        if var.key in out:raise ValueError('Repeated conditional variable key')
        out[var.key]=slice(start,start+var.dimension);start+=var.dimension
    return out,start


def ordered(variables):
    by_key={}
    for var in variables:
        if var.key in by_key:
            other=by_key[var.key]
            if other.dimension!=var.dimension or other.units!=var.units:
                raise ValueError('Conditional variable coordinate schema changed')
        by_key[var.key]=var
    return tuple(by_key[k] for k in sorted(by_key,key=lambda x:x.token()))


@dataclass(frozen=True)
class GaussianHistory:
    """H | Q ~ N(offset + boundary_map Q, covariance).

    Variable specifications retain coordinate identity; their original prior
    fields are descriptive and are never multiplied into a posterior again.
    The covariance is conditional on the entire boundary, not a marginal.
    """
    history_variables: tuple[Any,...]
    boundary_variables: tuple[Any,...]
    offset: np.ndarray
    boundary_map: np.ndarray
    covariance: np.ndarray
    conditioning_acquisition_ids: tuple[str,...]=()
    conditioning_dependency_ids: tuple[str,...]=()
    fidelity: str='frozen_joint_linear_Gaussian_conditional'

    def __post_init__(self):
        hi,nh=slices(self.history_variables);qi,nq=slices(self.boundary_variables)
        if not nh or not nq or set(hi).intersection(qi):
            raise ValueError('History and boundary must be nonempty disjoint scopes')
        m=np.asarray(self.offset,dtype=float).reshape(-1)
        t=np.asarray(self.boundary_map,dtype=float)
        c=sym(self.covariance)
        if m.shape!=(nh,) or t.shape!=(nh,nq) or c.shape!=(nh,nh):
            raise ValueError('Joint history coordinate dimensions disagree')
        if not all(np.isfinite(v).all() for v in (m,t,c)):
            raise ValueError('Joint history must be finite')
        # Proper Gaussian kernels only. Singular support needs its own exact
        # support representation; a pseudoinverse would silently change it.
        solve_spd(c,np.zeros(nh))
        object.__setattr__(self,'offset',immutable(m))
        object.__setattr__(self,'boundary_map',immutable(t))
        object.__setattr__(self,'covariance',immutable(c))

    @property
    def dimension(self):return self.offset.size

    def __deepcopy__(self,memo):
        # NumPy's default deepcopy can turn a read-only array writable. Preserve
        # the identical validated values in immutable, separately owned buffers;
        # detach descriptive variable records as well. No posterior is computed.
        result=object.__new__(type(self))
        memo[id(self)]=result
        for field in fields(self):
            value=getattr(self,field.name)
            object.__setattr__(result,field.name,
                immutable(value) if isinstance(value,np.ndarray) else copy.deepcopy(value,memo))
        return result

    @property
    def array_payload_bytes(self):
        return self.offset.nbytes+self.boundary_map.nbytes+self.covariance.nbytes

    def expand_boundary(self,extra_variables:Sequence[Any]):
        variables=ordered((*self.boundary_variables,*extra_variables))
        old,_=slices(self.boundary_variables);new,nq=slices(variables)
        t=np.zeros((self.dimension,nq))
        for key,sl in old.items():t[:,new[key]]=self.boundary_map[:,sl]
        return GaussianHistory(self.history_variables,variables,self.offset,t,self.covariance,
                               self.conditioning_acquisition_ids,self.conditioning_dependency_ids)


def compose_cut(previous:GaussianHistory|None,old_variables,boundary_variables,
                offset,boundary_map,covariance,*,dimension_cap:int):
    """Compose p(H|O,R) p(O|B) into one joint p(H,O|B,R).

    No boundary posterior is formed. Cross-covariance T_O V is retained, so
    later observations of different historical epochs are not made independent.
    """
    old_variables=tuple(old_variables);boundary_variables=tuple(boundary_variables)
    oi,no=slices(old_variables);bi,nb=slices(boundary_variables)
    if type(dimension_cap) is not int or dimension_cap<=0:
        raise ValueError('History dimension capacity must be an explicit positive integer')
    if no+(0 if previous is None else previous.dimension)>dimension_cap:
        raise ValueError('Retained history capacity exceeded before graph publication')
    local=GaussianHistory(old_variables,boundary_variables,offset,boundary_map,covariance)
    if previous is None:return local
    hi,nh=slices(previous.history_variables)
    if set(hi).intersection(oi) or set(hi).intersection(bi):
        raise ValueError('Previously historical variables cannot reappear as a new cut scope')
    pi,_=slices(previous.boundary_variables)
    remaining=tuple(v for v in previous.boundary_variables if v.key not in oi)
    q=ordered((*boundary_variables,*remaining));qi,nq=slices(q)
    to=np.zeros((nh,no));tr=np.zeros((nh,nq));g=np.zeros((no,nq))
    for key,sl in pi.items():
        if key in oi:to[:,oi[key]]=previous.boundary_map[:,sl]
        else:tr[:,qi[key]]=previous.boundary_map[:,sl]
    for key,sl in bi.items():g[:,qi[key]]=local.boundary_map[:,sl]
    m=np.r_[previous.offset+to@local.offset,local.offset]
    t=np.vstack((tr+to@g,g))
    cross=to@local.covariance
    c=np.block([[previous.covariance+cross@to.T,cross],
                [cross.T,local.covariance]])
    return GaussianHistory((*previous.history_variables,*old_variables),q,m,t,sym(c),
                           previous.conditioning_acquisition_ids,previous.conditioning_dependency_ids)


@dataclass(frozen=True)
class ConditionedHistory:
    history: GaussianHistory
    effective_jacobian: np.ndarray
    effective_measurement: np.ndarray
    effective_covariance: np.ndarray


def condition_history(history:GaussianHistory,factor,live_variables):
    """One new linear observation, with independent noise conditional on scope.

    Returns both the induced boundary likelihood and the UPDATED joint history
    kernel. Keeping only either one is insufficient for subsequent late data.
    The graph caller must atomically validate/admit/publish both exactly once.
    """
    hi,nh=slices(history.history_variables)
    extra=[]
    for key in factor.variables:
        if key not in hi:
            if key not in live_variables:raise KeyError('Historical factor references unavailable coordinates')
            extra.append(live_variables[key])
    expanded=history.expand_boundary(extra)
    qi,nq=slices(expanded.boundary_variables)
    if not set(factor.variables).intersection(hi):
        raise ValueError('Historical assimilation requires at least one retained historical variable')
    rows=factor.measurement.size
    fh=np.zeros((rows,nh));fq=np.zeros((rows,nq))
    if len(set(factor.variables))!=len(factor.variables):
        raise ValueError('Repeated observation coordinates are not supported')
    for key,jac in zip(factor.variables,factor.jacobians):
        dest,index=(fh,hi[key]) if key in hi else (fq,qi[key])
        if jac.shape!=(rows,index.stop-index.start):raise ValueError('Historical observation coordinate mismatch')
        dest[:,index]=jac
    mean=expanded.offset;t=expanded.boundary_map;c=expanded.covariance
    heff=fq+fh@t;yeff=factor.measurement-fh@mean
    reff=sym(factor.covariance+fh@c@fh.T)
    gain=solve_spd(reff,fh@c).T
    mnew=mean+gain@yeff;tnew=t-gain@heff
    residual=np.eye(nh)-gain@fh
    cnew=sym(residual@c@residual.T+gain@factor.covariance@gain.T)
    updated=GaussianHistory(expanded.history_variables,expanded.boundary_variables,mnew,tnew,cnew,
                            (*expanded.conditioning_acquisition_ids,factor.factor_id),
                            expanded.conditioning_dependency_ids)
    return ConditionedHistory(updated,immutable(heff),immutable(yeff),immutable(reff))
