"""Independent task scoring; no network/policy imports and no posterior writes."""
from __future__ import annotations
from itertools import product
import math,time
import numpy as np
from scipy.special import expit,ndtr
from numpy.polynomial.hermite import hermgauss
from v10_finite import model as old
from phase16_reference import Experiment as ScalarExperiment,BayesReference as ScalarReference
from .common import *
from .contexts import Context
from .stable_math import mixture_weights,conditional_position_variance,overlap_tail_refinement

_GH={}
def gh(n):
 if n not in _GH:
  x,w=hermgauss(n);_GH[n]=(x*np.sqrt(2.),w/np.sqrt(np.pi))
 return _GH[n]

class Scorer:
 def __init__(self,ctx:Context,meter:Meter,mode='production',shared=True,nodes=64):
  t=time.perf_counter();self.ctx=ctx;self.d=ctx.doc();self.m=meter;self.mode=mode;self.shared=shared;self.cache={};self.nodes=nodes
  if mode not in ('compatibility','production'):raise ContractError('SCORER_MODE')
  if ctx.track=='finite':
   self.old=old.load_context(self.d['finite_law']);self.p=np.asarray([float(x) for x in self.old.prior]);self.x=np.asarray([float(x) for x in self.old.x]);self.q=np.asarray([[float(x) for x in c.q] for c in self.old.channels]);self.dec=np.asarray([float(x) for x in self.old.decisions])
   self.sq=(self.x[:,None]-self.dec[None,:])**2;self.tail=(np.abs(self.x[:,None]-self.dec[None,:])>float(self.old.rho)).astype(float);self.loss=self.sq+float(self.old.lambda_tail)*self.tail
   meter.descriptor_entries+=self.q.size+self.p.size+self.loss.size
  else:
   self.w,self.logw=mixture_weights(self.d);self.mu=np.asarray(self.d['means']);self.P=np.asarray(self.d['covariances'][0]);self.cx=self.P[:,0];self.L2=self.d['length_m']**2
   self.mixmean=self.w@self.mu;center=self.mu-self.mixmean;self.mixture_cov=self.P+np.einsum('i,ij,ik->jk',self.w,center,center)
   meter.descriptor_entries+=self.mu.size+self.P.size+len(self.d['channels'])*9
  meter.prepare_seconds+=time.perf_counter()-t

 def _finite(self,a):
  if self.mode=='compatibility':
   m=old.Meter(1,1);v=old.finite_score(self.old,a,m)
   self.m.outcome_rows+=v.outcome_count
   return {'risk_m2':v.risk_m2,'mse_m2':v.squared_error_m2,'tail_probability':v.tail_probability,'numeric_error_estimate_m2':0.,'assurance':'NUMERICAL_FINITE_SUM'}
  groups=self.ctx.groups(a);ys=list(product((0,1),repeat=len(groups)));joint=np.ones((len(ys),len(self.p)))
  for j,i in enumerate(groups):joint*=np.where(np.asarray(ys)[:,j,None],self.q[i],1-self.q[i])
  weights=joint*self.p;values=weights@self.loss;chosen=values.argmin(axis=1);cols=np.arange(len(ys));energy=sum(self.d['channels'][i]['energy_J'] for i in a)
  mse=float(np.sum(weights*self.sq[:,chosen].T));tail=float(np.sum(weights*self.tail[:,chosen].T));risk=float(np.sum(values[cols,chosen]))+self.d['lambda_energy']*energy
  self.m.outcome_rows+=len(ys)
  return {'risk_m2':risk,'mse_m2':mse,'tail_probability':tail,'numeric_error_estimate_m2':0.,'assurance':'NUMERICAL_FINITE_SUM'}

 def matrices(self,a):
  cs=[self.d['channels'][i] for i in self.ctx.groups(a)]
  H=np.asarray([c['H'] for c in cs],dtype=float).reshape(len(cs),3)
  sd=np.array([c['noise_sd_m'] for c in cs]);l=np.array([c['common_loading'] for c in cs]);R=np.diag(sd**2)+np.outer(l,l)
  return H,R

 def scalar_reference(self,a):
  """Lawful (x,n) projection: native H has no velocity measurement coefficient.

  Scalar Phase16 remains a read-only reference, not a second operational owner.
  No numerical covariance symmetrization or mode removal is performed.
  """
  H,R=self.matrices(a)
  if H.shape!=(1,3) or H[0,1]!=0:raise ContractError('NOT_A_PHASE16_SCALAR_PROJECTION')
  ix=[0,2];means=self.mu[:,ix];cov=np.asarray(self.d['covariances'])[:,ix,:][:,:,ix]
  ex=ScalarExperiment(tuple(self.w),tuple(tuple(x) for x in means),tuple(tuple(tuple(z) for z in x) for x in cov),tuple(H[0,ix]),float(R[0,0]),cost=0.,units=('m','m'),decision_epoch=self.d['epoch'],terminal_epoch=self.d['epoch'],available_epoch=self.d['epoch'])
  def charge(kind,n):
   if kind=='reference_models':self.m.phase16_models+=n;self.m.matrix_factorizations+=n*len(ex.weights)
   elif kind=='conditional_evaluations':self.m.phase16_conditional_calls+=n
   else:self.m.quadrature_nodes+=n
  return ScalarReference(ex,charge)

 def _dynamic(self,a):
  energy=sum(self.d['channels'][i]['energy_J'] for i in a);cost=self.d['lambda_energy']*energy;rho=self.d['rho_m']
  if not a:
   delta=self.mu[:,0]-self.mixmean[0];mse=float(self.P[0,0]+self.w@(delta**2));sd=math.sqrt(self.P[0,0]);tp=float(self.w@(ndtr((-rho-delta)/sd)+ndtr((delta-rho)/sd)))
   return {'risk_m2':mse+cost+self.d['lambda_tail_m2']*tp,'mse_m2':mse,'tail_probability':tp,'numeric_error_estimate_m2':0.,'assurance':'NUMERICAL_CLOSED_FORM'}
  H,R=self.matrices(a);S=H@self.P@H.T+R;chol=np.linalg.cholesky(S);self.m.matrix_factorizations+=1
  if np.linalg.cond(S)>1e12:raise ContractError('ILL_CONDITIONED_REFERENCE')
  # Position index 0. Full 3-coordinate covariance is retained in the law.
  cross=H@self.cx;k=np.linalg.solve(chol.T,np.linalg.solve(chol,cross));self.m.linear_solves+=2
  v=conditional_position_variance(self.P,H,R,k)
  if len(self.ctx.groups(a))==1 and np.all(self.w>0):
   scalar=self.scalar_reference(a)
   if abs(v-float(scalar.v[0]))>1e-8*self.L2:raise ContractError('PHASE16_PROJECTION_DISAGREEMENT')
   # The scalar comparator uses subtraction; retain the stable Joseph value.
   if float(scalar.v[0])<=0:raise ContractError('PHASE16_NONPOSITIVE_VARIANCE')
  if v<=0 or not math.isfinite(v):raise ContractError('CONDITIONAL_VARIANCE')
  if len(self.w)==1:
   tail=float(2*ndtr(-rho/math.sqrt(v)));return {'risk_m2':v+cost+self.d['lambda_tail_m2']*tail,'mse_m2':v,'tail_probability':tail,'numeric_error_estimate_m2':0.,'assurance':'NUMERICAL_GAUSSIAN'}
  dm=self.mu[1]-self.mu[0];dn=H@dm;solved_dn=np.linalg.solve(chol,dn);self.m.linear_solves+=1;a2=float(solved_dn@solved_dn)/4.;d=float(dm[0]-k@dn);w=float(self.w[1]);w0=float(self.w[0]);alpha=math.sqrt(a2)
  def overlap(n):
   xx,ww=gh(n);j=tp=0.
   for mode,prior in ((0,w0),(1,w)):
    t=(2*mode-1)*alpha+xx;p=expit(float(self.logw[1]-self.logw[0])+2*alpha*t)
    j+=prior*float(ww@(p*(1-p)))
    means0=-p*d;means1=(1-p)*d;sd=math.sqrt(v)
    pt0=ndtr((-rho-means0)/sd)+ndtr((means0-rho)/sd);pt1=ndtr((-rho-means1)/sd)+ndtr((means1-rho)/sd)
    tp+=prior*float(ww@((1-p)*pt0+p*pt1))
   self.m.quadrature_nodes+=2*n
   return j,tp
  if alpha<1e-12:
   j=w*w0;sd=math.sqrt(v);tp=w0*(ndtr((-rho+w*d)/sd)+ndtr((-w*d-rho)/sd))+w*(ndtr((-rho-w0*d)/sd)+ndtr((w0*d-rho)/sd));error=0.
  else:
   # Even when d=0, association may remain unresolved. J is also a training
   # target, so a vanishing contribution to position risk must not set J=0.
   j,tp=overlap(self.nodes);j0,tp0=overlap(max(8,self.nodes//2));error=abs(j-j0)*d*d
   # Escalate only unresolved scalar work, never silently replace it by zero.
   tail_error=abs(tp-tp0)
   if error>1e-5*self.L2 or tail_error>1e-5:
    j,tp,error,tail_error,neval=overlap_tail_refinement(self.w,self.logw,alpha,d,v,rho,self.L2)
    self.m.quadrature_nodes+=neval
  mse=v+d*d*j
  if not math.isfinite(mse):raise ContractError('MIXTURE_NUMERICAL_RANGE')
  return {'risk_m2':mse+cost+self.d['lambda_tail_m2']*tp,'mse_m2':mse,'tail_probability':float(tp),'numeric_error_estimate_m2':float(error),'assurance':'NUMERICAL_1D_GAUSSIAN_MIXTURE','remaining_association_variance':float(j),'tail_error_estimate':float(locals().get('tail_error',0.)),'positive_mode_count':len(self.w),'log_weights_retained':True}

 def evaluate(self,a):
  self.ctx.legal(a);self.m.charge();start=time.perf_counter();energy=sum(self.d['channels'][i]['energy_J'] for i in a)
  # Identical observation atoms may share inference, not independent score budgets.
  key=(self.ctx.digest,tuple(self.d['channels'][i]['root'] for i in self.ctx.groups(a)),self.mode,self.nodes)
  try:
   if self.shared and key in self.cache:
    row=dict(self.cache[key]);row['risk_m2']=row['mse_m2']+self.d['lambda_tail_m2']*row['tail_probability']+self.d['lambda_energy']*energy;self.m.cache_hits+=1
   else:
    self.m.atomic_scores+=1
    if self.m.kernel_calls is not None:self.m.kernel_calls+=1
    row=self._finite(a) if self.ctx.track=='finite' else self._dynamic(a)
    if self.shared:self.cache[key]=dict(row)
   self.validate(row)
   self.m.completed_scores+=1
   return {**row,'action':list(a),'context_digest':self.ctx.digest,'energy_J':energy,'formal_certificate':False}
  except Exception:
   self.m.reference_failures+=1;self.m.kernel_calls=None;raise
  finally:self.m.score_seconds+=time.perf_counter()-start
 def validate(self,row):
  for k in ('risk_m2','mse_m2','tail_probability','numeric_error_estimate_m2'):
   if k not in row or not math.isfinite(float(row[k])):raise ContractError('INVALID_REFERENCE_ROW')
  if row['risk_m2']< -1e-8 or row['mse_m2']< -1e-8 or not -1e-12<=row['tail_probability']<=1+1e-12:raise ContractError('REFERENCE_RANGE')
 def evaluate_many(self,actions,chunk=16):
  # Bounded chunks, identical candidate/RNG order. No outcome generation.
  for i in range(0,len(actions),chunk):
   for a in actions[i:i+chunk]:yield self.evaluate(tuple(a))
