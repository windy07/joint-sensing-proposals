"""Explicit finite-support continuous grid cells and mass/density distinction."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from ._validation import grid_1d


def cell_edges(centres) -> np.ndarray:
    """Closed support [first,last] with midpoint interior cell boundaries.

    Endpoint cells are truncated at their centres. Their widths are half a
    neighbour gap on a uniform grid; no implicit extrapolation beyond support.
    """
    g=grid_1d(centres,'grid_centres')
    return np.r_[g[0],(g[:-1]+g[1:])/2,g[-1]]


def cell_areas(delay_grid_s, frequency_grid_hz) -> np.ndarray:
    return np.diff(cell_edges(delay_grid_s))[:,None]*np.diff(cell_edges(frequency_grid_hz))[None,:]


def credible_region(probability_mass, delay_grid_s, frequency_grid_hz, requested_mass=.9, truth=None) -> dict:
    d=grid_1d(delay_grid_s,'delay_grid_s'); f=grid_1d(frequency_grid_hz,'frequency_grid_hz')
    p=np.asarray(probability_mass,dtype=float)
    if p.shape!=(len(d),len(f)) or not np.all(np.isfinite(p)) or np.any(p<0) or p.sum()<=0:
        raise ValueError('grid probability masses must be finite, nonnegative and have positive total')
    if not 0<requested_mass<=1: raise ValueError('requested_mass must be in (0,1]')
    p=p/p.sum(); areas=cell_areas(d,f)
    # HPD ranks density, then accumulates probability mass (not area).
    order=np.argsort(-(p/areas).ravel(),kind='stable')
    cells=min(p.size,int(np.searchsorted(np.cumsum(p.ravel()[order]),requested_mass,side='left'))+1)
    selected=np.zeros(p.size,dtype=bool); selected[order[:cells]]=True
    out={'requested_mass':float(requested_mass),'achieved_mass':float(p.ravel()[selected].sum()),
         'cell_count':cells,'selected_cell_mask':selected.reshape(p.shape).tolist(),
         'support_delay_s':[float(d[0]),float(d[-1])],'support_frequency_hz':[float(f[0]),float(f[-1])],
         'cell_area_unit':'s*Hz','hpd_ranking':'mass divided by cell area',
         'truth_in_support':None,'truth_inside_credible_region':None,'outside_distance_s':None,
         'outside_distance_hz':None,'nll_cell_mass':None,'nll_reason':'truth_not_supplied'}
    if truth is None: return out
    truth=np.asarray(truth,dtype=float)
    if truth.shape!=(2,) or not np.all(np.isfinite(truth)): raise ValueError('truth must be finite [seconds,hertz]')
    td,tf=map(float,truth)
    out['outside_distance_s']=float(max(d[0]-td,td-d[-1],0))
    out['outside_distance_hz']=float(max(f[0]-tf,tf-f[-1],0))
    support=(d[0]<=td<=d[-1]) and (f[0]<=tf<=f[-1])
    out['truth_in_support']=bool(support)
    if not support:
        out.update(truth_inside_credible_region=False,nll_reason='truth_outside_grid_support')
        return out
    # Shared boundaries assigned to the cell on the right; last endpoint kept.
    i=min(len(d)-1,int(np.searchsorted(cell_edges(d),td,side='right')-1))
    j=min(len(f)-1,int(np.searchsorted(cell_edges(f),tf,side='right')-1))
    out['truth_inside_credible_region']=bool(selected.reshape(p.shape)[i,j])
    prob=float(p[i,j])
    out['nll_cell_mass']=float(-np.log(prob)) if prob>0 else None
    out['nll_reason']='finite_cell_probability' if prob>0 else 'zero_cell_probability'
    return out
