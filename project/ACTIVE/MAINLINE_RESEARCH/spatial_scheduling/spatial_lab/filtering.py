"""Augmented-state Gaussian moment tracker with bounded joint association sums.

This is the only posterior owner in the new spatial experiment. The native
legacy DFG is untouched. Covariance forecasts are detached, read-only copies.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import product
from scipy.special import logsumexp
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
import math,time
import numpy as np
from .core import *
from .physics import *
from .shared_events import moment_match, emission_branches

class MultiTracker:
    def __init__(self,config):
        self.config=config;self.sensors=sensor_list(config);self.mean=np.zeros(NBIAS)
        self.P=np.diag(BIAS_INITIAL_SD**2)
        self.tracks=[];self.next_id=0;self.epoch=-1;self.seen={};self.arrived=[]
        self.initiated_roots=set();self.emission_components=[]
        self.stats={'duplicate_packets':0,'late_packets':0,'unknown_source_packets':0,
            'dropped_packets':0,'reference_missing_packets':0,'birth_shared_capture_deferred':0,'association_hypotheses':0,'association_cap_fallbacks':0,'birth_cap_rejections':0,
            'updates':0,'calibration_updates':0,'full_joint_mixture_updates':0,
            'birth_reused_measurements_suppressed':0,'birth_separation_rejections':0,
            'birth_fusion_updates':0,'birth_fusion_measurements':0,
            'birth_ambiguous_measurements_deferred':0,'duplicate_initiation_packets':0}
        self.last_betas={};self.nis=[];self.births=[];self.last_calibration_epochs=[-1]*4;self.linearization_anchor=None
    def _supported_birth(self,pos,sensor,task,stage):
        reason=covariance_support_reason(pos,sensor,task,self.epoch,self.config)
        if reason is None:return True
        key='birth_'+stage+'_'+reason+'_deferred'
        self.stats[key]=self.stats.get(key,0)+1
        return False
    def fingerprint(self):
        return digest({'mean':self.mean.tolist(),'P':self.P.tolist(),'tracks':self.tracks,
            'epoch':self.epoch,'seen':self.seen,'linearization_anchor':self.linearization_anchor.tolist() if self.linearization_anchor is not None else None,
            'last_calibration_epochs':self.last_calibration_epochs,'initiated_roots':sorted(self.initiated_roots),
            'emission_components':[{**b,'mean':b['mean'].tolist(),'P':b['P'].tolist()} for b in self.emission_components]})
    def predict(self,epoch):
        if epoch!=self.epoch+1:raise ContractError('NONCONSECUTIVE_TIME')
        # The world draws a new target emission independently at each epoch.
        # Collapse only here; within an epoch a miss must update the same event.
        self.emission_components=[]
        self.epoch=epoch;self.nis=[];self.last_betas={}
        n=len(self.mean);F=np.eye(n);Q=np.zeros((n,n))
        Q[:NBIAS,:NBIAS]=np.diag(BIAS_WALK_SD**2)
        f,q=transition(self.config['dt_s'],self.config['acceleration_psd_m2_s3'])
        for t in self.tracks:
            j=t['start'];F[j:j+4,j:j+4]=f;Q[j:j+4,j:j+4]=q
        self.mean=F@self.mean;self.P=F@self.P@F.T+Q
        # Administrative deletion uses only stale arrived evidence, never truth.
        keep=[t for t in self.tracks if epoch-t['last_hit']<=self.config['coast_limit_cycles']]
        if len(keep)!=len(self.tracks):
            ix=list(range(NBIAS))
            for t in keep:ix+=list(range(t['start'],t['start']+4))
            self.mean=self.mean[ix];self.P=self.P[np.ix_(ix,ix)]
            for k,t in enumerate(keep):t['start']=NBIAS+4*k
            self.tracks=keep
        # All observations in this epoch use one coordinate system.
        self.linearization_anchor=self.mean.copy()
    def view(self,energy,tx,unseen):
        d={'schema':'ain.spatial.causal.v1','epoch':self.epoch,'decision_s':self.epoch*self.config['dt_s'],
            'units':{'position':'m','velocity':'m/s','bearing':'rad','time':'s','RF_energy':'J'},
            'mean':self.mean.tolist(),'covariance':self.P.tolist(),'tracks':deepcopy(self.tracks),
            'energy_remaining_exact_J':str(energy),'tx_remaining_exact_s':str(tx),
            'unseen_intensity':list(map(float,unseen)),
            'arrived_packet_ids':list(self.seen),'posterior_kind':'JOINT_ASSOCIATION_GAUSSIAN_MOMENT',
            'known_node_states':[availability(self.epoch,s.id,self.config) for s in self.sensors],
            'emission_event_posterior':self.emission_summary(),'last_calibration_epochs':list(self.last_calibration_epochs)}
        d['digest']=digest(d);return d
    def receive(self,packet,now_s=None):
        if type(packet.get('node')) is not int or not 0<=packet['node']<len(self.sensors):raise ContractError('PACKET_NODE_SCHEMA')
        if packet['source'] != self.sensors[packet['node']].name:
            self.stats['unknown_source_packets']+=1;raise ContractError('UNKNOWN_SOURCE')
        if not all(isinstance(packet.get(k),(int,float)) and math.isfinite(packet[k]) for k in ('acquire_s','arrival_s')):raise ContractError('NONFINITE_TIME')
        if packet['arrival_s']<packet['acquire_s'] or not packet['acquire_epoch']*self.config['dt_s']<=packet['acquire_s']<(packet['acquire_epoch']+1)*self.config['dt_s']:raise ContractError('TIME_CORRESPONDENCE')
        if not packet.get('delivered',False):
            self.stats['dropped_packets']+=1;return {'status':'TRANSMISSION_FAILED','unassigned':[]}
        if not packet.get('reference_acquired',False):
            self.stats['reference_missing_packets']+=1;return {'status':'NO_REFERENCE','unassigned':[]}
        if packet['acquire_epoch']!=self.epoch or packet['arrival_s']>(self.epoch+1)*self.config['dt_s']:
            self.stats['late_packets']+=1;return {'status':'LATE_REJECTED','unassigned':[]}
        if packet['arrival_s'] > ((self.epoch+1)*self.config['dt_s'] if now_s is None else now_s):
            return {'status':'NOT_ARRIVED','unassigned':[]}
        sensor=self.sensors[packet['node']]
        task=Task(**packet['task']);offset=packet['acquire_s']-self.epoch*self.config['dt_s']
        expected=measurement_units(task)
        if packet.get('measurement_units')!=expected:raise ContractError('MEASUREMENT_UNITS')
        if packet['task']['node']!=packet['node']:raise ContractError('PACKET_ACTION_BINDING')
        for z in packet['returns']:
            if len(z)!=len(expected) or not np.isfinite(z).all():raise ContractError('NONFINITE_OR_WRONG_MEASUREMENT_DIMENSION')
        # The declared pointing is frozen at acquisition, not recomputed after another packet updates the posterior.
        if packet['kind']!='calibrate' and ('boresight_rad' not in packet or not 0<packet.get('field_of_view_rad',0)<=2*math.pi):raise ContractError('POINTING_SUPPORT_MISSING')
        root=packet['root'];content=digest({k:v for k,v in packet.items() if k!='transport_copy'})
        if root in self.seen:
            if self.seen[root]!=content:raise ContractError('PROVENANCE_CONTENT_CONFLICT')
            self.stats['duplicate_packets']+=1;return {'status':'DUPLICATE','unassigned':[]}
        self.seen[root]=content;self.arrived.append(root)
        sensor=self.sensors[packet['node']]
        if packet['kind']=='calibrate':
            if not packet['returns']:return {'status':'CALIBRATION_NOT_DETECTED','unassigned':[]}
            components=self.emission_components or [{'log_weight':0.,'bits':{},'mean':self.mean.copy(),'P':self.P.copy()}]
            nis=[]
            for b in components:
                pred,H,R,_=calibration_model(task,b['mean'][:NBIAS],len(b['mean']),self.epoch,self.config)
                innov=np.asarray(packet['returns'][0])-pred;innov[2]=float(wrap(innov[2]))
                b['mean'],b['P'],ll,ni=gaussian_update(b['mean'],b['P'],H,R,innov)
                b['log_weight']+=ll;nis.append(ni)
            weights=self._mix_components(components)
            self.last_calibration_epochs[packet['node']]=self.epoch
            self.stats['calibration_updates']+=1;self.stats['updates']+=1;self.nis.append(float(weights@nis))
            return {'status':'CALIBRATED','unassigned':[],'emission_event_posterior':self.emission_summary(),'last_calibration_epochs':list(self.last_calibration_epochs)}
        if not self.tracks:
            return {'status':'NO_EXISTING_TRACK','unassigned':list(range(len(packet['returns']))),
                'association_used_probability':[0.]*len(packet['returns']),
                'birth_eligibility':'NO_RETAINED_ASSOCIATION_SUPPORT'}
        components=self._prepare_components(task)
        results=[]
        for b in components:
            r=self._association(packet,b['mean'],b['P'],b['bits'])
            b['mean'],b['P']=r['mean'],r['P'];b['log_weight']+=r['log_evidence'];results.append(r)
        weights=self._mix_components(components)
        m=len(packet['returns']);used=sum(w*r['used'] for w,r in zip(weights,results))
        betas=sum(w*r['betas'] for w,r in zip(weights,results))
        for tr,beta in zip(self.tracks,betas):
            self.last_betas[tr['id']]=max(float(beta),self.last_betas.get(tr['id'],0))
            if beta>.5:
                if tr['last_hit']!=self.epoch:tr['hit_epochs'].append(self.epoch)
                tr['last_hit']=self.epoch;tr['confirmed']=len(set(tr['hit_epochs']))>=2
        # Structural support includes all finite-log-mass emission branches,
        # including branches whose floating point *reported* weight underflows.
        supported=set().union(*(r['supported'] for r in results))
        suppressed=[k for k in supported if used[k]<.2]
        overflow=any(r['overflow'] for r in results)
        self.stats['association_hypotheses']+=sum(r['hypotheses'] for r in results)
        self.stats['association_cap_fallbacks']+=sum(r['overflow'] for r in results)
        self.stats['full_joint_mixture_updates']+=int(not overflow);self.stats['updates']+=1
        self.stats['birth_reused_measurements_suppressed']+=len(suppressed)
        self.nis.append(float(sum(w*r['nis'] for w,r in zip(weights,results))))
        return {'status':'GNN_CAP_APPROXIMATION' if overflow else 'GATED_JOINT_MOMENT_UPDATE',
            'hypotheses':sum(r['hypotheses'] for r in results),'unassigned':[k for k in range(m) if k not in supported],
            'shared_capture_used':any(r['shared_capture_used'] for r in results),
            'association_used_probability':used.tolist(),'birth_reuse_suppressed':suppressed,
            'birth_eligibility':'NO_RETAINED_ASSOCIATION_SUPPORT',
            'association_entropy_nats':float(-sum(w*math.log(w) for w in weights if w>0)+sum(w*r['entropy'] for w,r in zip(weights,results))),
            'excluded_gate_mass':'UNQUANTIFIED','discarded_cap_mass':None if overflow else 0.,
            'unavailable_linearization_track_ids':sorted(set().union(*(set(r['unavailable_anchor']) for r in results))),
            'emission_event_posterior':self.emission_summary(),
            'emission_conditioned_components':len(components)}

    def emission_summary(self):
        """Causal arrived-data event probabilities, no future/world samples."""
        if not self.emission_components:return {}
        return {str(i):float(sum(math.exp(b['log_weight'])*b['bits'].get(i,0) for b in self.emission_components))
                for i in sorted(set().union(*(set(b['bits']) for b in self.emission_components)))}

    def _prepare_components(self,task):
        components=self.emission_components or [{'log_weight':0.,'bits':{},'mean':self.mean.copy(),'P':self.P.copy()}]
        if task.role=='direct':
            known=set(components[0]['bits']);ids=[t['id'] for t in self.tracks if t['id'] not in known]
            if len(known)+len(ids)>6:raise ContractError('EMISSION_BRANCH_LIMIT_SIX_TARGETS')
            expanded=[]
            for b in components:
                for w,bits in emission_branches(ids,float(self.config.get('emission_probability',.7))):
                    expanded.append({'log_weight':b['log_weight']+math.log(w),'bits':{**b['bits'],**bits},
                                     'mean':b['mean'].copy(),'P':b['P'].copy()})
            components=expanded
        return components

    def _mix_components(self,components):
        norm=float(logsumexp([b['log_weight'] for b in components]))
        if not math.isfinite(norm):raise ContractError('ZERO_EMISSION_ASSOCIATION_EVIDENCE')
        for b in components:b['log_weight']-=norm
        weights=np.exp([b['log_weight'] for b in components])
        self.mean,self.P=moment_match(weights,[b['mean'] for b in components],[b['P'] for b in components])
        self.emission_components=components
        return weights

    def _association(self,packet,mean,covariance,emissions):
        """One packet likelihood conditional on persistent epoch emissions."""
        sensor=self.sensors[packet['node']];task=Task(**packet['task'])
        offset=packet['acquire_s']-self.epoch*self.config['dt_s'];expected=measurement_units(task)
        Z=[np.asarray(z) for z in packet['returns']]
        n=len(self.tracks);m=len(Z)
        task=Task(**packet['task']);view={'epoch':self.epoch,'mean':mean.tolist(),'tracks':self.tracks}
        if self.linearization_anchor is None:
            self.linearization_anchor=mean.copy()
        anchor=self.linearization_anchor
        if anchor.shape!=mean.shape:raise ContractError('LINEARIZATION_ANCHOR_LAYOUT')
        pred=[];Hs=[];Rs=[];pds=[];innovs={};allowed=[];unavailable_anchor=[]
        for i,tr in enumerate(self.tracks):
            j=tr['start'];state=mean[j:j+4]
            reference=anchor[j:j+4]
            H=jacobian(reference,sensor,len(mean),j,task,offset)
            pred.append(measurement(reference,anchor[:NBIAS],sensor,task,offset)+H@(mean-anchor))
            pos=state[:2]+state[2:]*offset
            in_beam=covers(pos,task,view,sensor)
            pd=conditional_detection_probability(pos,sensor,task,self.epoch,self.config,emission=emissions.get(tr['id'])) if in_beam else 0.
            # An invisible track has only the missed-detection association. No
            # arbitrary noise floor is needed: its H/R will never be consumed.
            reference_pos=reference[:2]+reference[2:]*offset
            if pd>0:
                reason=covariance_support_reason(reference_pos,sensor,task,self.epoch,self.config)
                if reason is not None:
                    # This local likelihood has no valid covariance in its
                    # frozen coordinates. Do not manufacture signal or move
                    # the common linearization anchor after another packet.
                    pd=0.;unavailable_anchor.append(tr['id'])
                    key='association_anchor_'+reason+'_components_deferred'
                    self.stats[key]=self.stats.get(key,0)+1
            R=noise_cov(reference_pos,sensor,task,self.epoch,self.config) if pd>0 else np.eye(len(expected))
            Hs.append(H);Rs.append(R)
            pds.append(min(pd,1.-1e-12));ids=[]
            S=H@covariance@H.T+R
            for k,z in enumerate(Z):
                dz=innovation(z,pred[-1],sensor,task);ni=float(dz@np.linalg.solve(S,dz))
                if ni<=self.config['association_gate_chi2']:
                    ids.append(k);innovs[i,k]=dz
            allowed.append([-1,*ids] if pd>0 else [-1])
        assignments=[];cap=self.config['association_hypothesis_cap'];overflow=False
        def visit(i,used,a):
            nonlocal overflow
            if overflow:return
            if i==n:
                assignments.append(tuple(a));overflow=len(assignments)>cap;return
            for k in allowed[i]:
                if k>=0 and k in used:continue
                visit(i+1,used|({k} if k>=0 else set()),a+[k])
        visit(0,set(),[])
        if overflow:
            # Explicit approximation, not a certificate or a zero omitted mass.
            c=np.full((n,m+n),1e9)
            for i in range(n):
                c[i,m+i]=self.config['association_gate_chi2']
                for k in allowed[i]:
                    if k>=0:
                        S=Hs[i]@covariance@Hs[i].T+Rs[i];c[i,k]=float(innovs[i,k]@np.linalg.solve(S,innovs[i,k]))
            ii,jj=linear_sum_assignment(c);a=[-1]*n
            for i,k in zip(ii,jj):
                if k<m and c[i,k]<c[i,m+i]:a[i]=int(k)
            assignments=[tuple(a)]
        logs=[];means=[];covs=[];nis=[]
        dim=len(expected)
        if task.role=='monostatic':
            volume=(sensor.max_range_m-C0*task.pulse_width_s/2)*task.beam_rad*C0/(2*task.carrier_Hz*task.pri_s)
        elif task.mode=='aoa':volume=task.beam_rad
        else:
            from .actions import device
            volume=2*device(self.config,task.node)['passive_delay_window_s']/task.pri_s*task.beam_rad
            if task.role=='direct':volume*=2*math.pi
        kappa=max(packet.get('clutter_mean',0.),1e-12)/volume
        common=np.zeros((dim,dim))
        for i,tr in enumerate(self.tracks):
            if pds[i]>0:
                j=tr['start'];pos=anchor[j:j+2]+anchor[j+2:j+4]*offset
                common=noise_parts(pos,sensor,task,self.epoch,self.config)[1];break
        for a in assignments:
            selected=[i for i,k in enumerate(a) if k>=0]
            logp=sum(math.log(pds[i]) if k>=0 else math.log1p(-pds[i]) for i,k in enumerate(a))
            if selected:
                H=np.vstack([Hs[i] for i in selected]);R=block_diag(*[Rs[i] for i in selected]);dz=np.concatenate([innovs[i,a[i]] for i in selected])
                # Shared bistatic reference extraction error is one nuisance draw
                # per capture, hence off-diagonal likelihood covariance blocks.
                for ri in range(len(selected)):
                    for rj in range(len(selected)):
                        if ri!=rj:R[ri*dim:(ri+1)*dim,rj*dim:(rj+1)*dim]=common
                mu,P,ll,ni=gaussian_update(mean,covariance,H,R,dz)
                logp+=ll-len(selected)*math.log(kappa)
            else:mu=mean.copy();P=covariance.copy();ni=0.
            logs.append(logp);means.append(mu);covs.append(P);nis.append(ni)
        norm=float(logsumexp(logs));w=np.exp(np.asarray(logs)-norm)
        mix,Pmix=moment_match(w,means,covs)
        used=np.zeros(m);betas=np.zeros(n)
        for i in range(n):
            betas[i]=sum(p for p,a in zip(w,assignments) if a[i]>=0)
            for k in range(m):used[k]+=sum(p for p,a in zip(w,assignments) if a[i]==k)
        supported={k for a in assignments for k in a if k>=0}
        return {'mean':mix,'P':Pmix,'log_evidence':norm,'used':used,'betas':betas,
                'supported':supported,'overflow':overflow,'hypotheses':len(assignments),
                'unavailable_anchor':unavailable_anchor,
                'shared_capture_used':bool(np.any(common) and supported),'nis':float(w@nis),
                'entropy':float(-sum(p*math.log(p) for p in w if p>0))}
    def _add(self,pos,Jb,Rpos,velocity=None,origin='active',*,state_bias=None,state_noise=None,roots=()):
        if len(self.tracks)>=self.config['max_tracks']:
            self.stats['birth_cap_rejections']+=1;return False
        bounds=self.config.get('surveillance_bounds_m',[-3000.,3000.,-2200.,3000.])
        if not (bounds[0]<pos[0]<bounds[1] and bounds[2]<pos[1]<bounds[3]):return False
        for t in self.tracks:
            if np.linalg.norm(pos-self.mean[t['start']:t['start']+2])<self.config['birth_separation_m']:
                self.stats['birth_separation_rejections']+=1;return False
        n=len(self.mean);G=np.zeros((4,n));G[:2,:NBIAS]=Jb
        Q=np.zeros((4,4));Q[:2,:2]=Rpos;Q[2:,2:]=np.eye(2)*30.**2
        if state_bias is not None:G[:,:NBIAS]=state_bias
        if state_noise is not None:Q=np.asarray(state_noise)
        state=np.r_[pos,[0.,0.] if velocity is None else velocity]
        prior_mean=self.mean.copy()
        if self.linearization_anchor is None:self.linearization_anchor=prior_mean.copy()
        # Extend the inverse-observation birth map in the same bias coordinates.
        self.linearization_anchor=np.r_[self.linearization_anchor,state+G@(self.linearization_anchor-prior_mean)]
        # Extend every retained conditional Gaussian with the same inverse
        # observation map; do not destroy its existing cross-covariances.
        for b in self.emission_components:
            cross=G@b['P'];newP=G@b['P']@G.T+Q
            b['P']=np.block([[b['P'],cross.T],[cross,newP]])
            b['mean']=np.r_[b['mean'],state+G@(b['mean']-prior_mean)]
            if origin=='direct':b['bits'][self.next_id]=1
        cross=G@self.P;newP=G@self.P@G.T+Q
        self.P=np.block([[self.P,cross.T],[cross,newP]])
        self.mean=np.r_[self.mean,state]
        if not self.emission_components and origin=='direct':
            self.emission_components=[{'log_weight':0.,'bits':{self.next_id:1},'mean':self.mean.copy(),'P':self.P.copy()}]
        tr={'id':self.next_id,'start':n,'confirmed':False,'hit_epochs':[self.epoch],
            'last_hit':self.epoch,'birth_epoch':self.epoch,'origin':origin,'birth_roots':list(roots)}
        self.tracks.append(tr);self.births.append(dict(tr));self.next_id+=1;return True
    def _active_birth_moments(self,z,s,task):
        r=float(z[0]-self.mean[s.range_bias]);theta=float(wrap(z[1]-self.mean[s.angle_bias]))
        if r<=20:return None
        u=np.array([math.cos(theta),math.sin(theta)]);v=np.array([-u[1],u[0]]);vr=float(z[2]+C0/s.carrier_Hz/2*self.mean[s.frequency_bias])
        pos=np.asarray(s.position)+r*u;state=np.r_[pos,vr*u]
        if not self._supported_birth(pos,s,task,'active_initial'):return None
        # Inverse measurement map: p=s+r*u(theta), velocity=vr*u(theta).
        # The unobserved transverse velocity has an independent zero-mean prior.
        Jz=np.zeros((4,3));Jz[:2,0]=u;Jz[:2,1]=r*v
        Jz[2:,1]=vr*v;Jz[2:,2]=u
        Jb=np.zeros((4,NBIAS));Jb[:,s.range_bias]=-Jz[:,0];Jb[:,s.angle_bias]=-Jz[:,1];Jb[:,s.frequency_bias]=Jz[:,2]*C0/s.carrier_Hz/2
        R=noise_cov(pos,s,task,self.epoch,self.config);Q=Jz@R@Jz.T
        sigma=float(self.config.get('birth_tangential_velocity_std_mps',30.))
        if not math.isfinite(sigma) or sigma<=0:raise ContractError('BIRTH_TRANSVERSE_VELOCITY_PRIOR')
        Q[2:,2:]+=sigma**2*np.outer(v,v)
        return state,Jb,(Q+Q.T)/2
    def _passive_birth_moments(self,z,s,task,offset):
        if task.mode=='aoa':return None  # single bearing has no range identity
        from scipy.optimize import least_squares
        theta=float(wrap(z[2]-self.mean[s.angle_bias]));u=np.array([math.cos(theta),math.sin(theta)])
        if task.role=='bistatic':
            a=np.asarray(s.position)-ILLUMINATOR
            total=C0*z[0]-self.mean[s.range_bias]+np.linalg.norm(a)
            den=2*(total+a@u)
            if abs(den)<1e-6:return None
            r=(total**2-a@a)/den
        else:
            ref=self.sensors[task.reference_node];a=np.asarray(s.position)-ref.position
            dr=C0*z[0]-self.mean[s.range_bias]+self.mean[ref.range_bias]
            den=2*(a@u+dr)
            if abs(den)<1e-6:return None
            r=(dr**2-a@a)/den
        if not 20<r<30000:return None
        x0=np.r_[np.asarray(s.position)+r*u,0.,0.]
        if not self._supported_birth(x0[:2],s,task,'passive_initial'):return None
        R=noise_cov(x0[:2],s,task,self.epoch,self.config)
        # Redundant TDOA + two bearings cannot be gated against thermal noise
        # alone while common timing/array biases are still uncertain. Freeze
        # the nuisance linearization at the observation-derived initialization.
        H0=jacobian(x0,s,NBIAS+4,NBIAS,task,offset);Hb0=H0[:,:NBIAS]
        S=R+Hb0@self.P[:NBIAS,:NBIAS]@Hb0.T;L=np.linalg.cholesky(S)
        sigma=float(self.config.get('birth_tangential_velocity_std_mps',30.))
        if not math.isfinite(sigma) or sigma<=0:raise ContractError('BIRTH_VELOCITY_PRIOR')
        precision=np.diag([0.,0.,1/sigma**2,1/sigma**2])
        def fun(x):
            dz=innovation(z,measurement(x,self.mean[:NBIAS],s,task,offset),s,task)
            return np.r_[np.linalg.solve(L,dz),x[2:]/sigma]
        def derivative(x):
            Hx=jacobian(x,s,NBIAS+4,NBIAS,task,offset)[:,NBIAS:]
            return np.vstack([-np.linalg.solve(L,Hx),np.array([[0.,0.,1/sigma,0.],[0.,0.,0.,1/sigma]])])
        fit=least_squares(fun,x0,jac=derivative,max_nfev=24,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        if not fit.success or np.sum(fun(fit.x)**2)>self.config['association_gate_chi2']:
            self.stats['birth_nuisance_fit_rejected']=self.stats.get('birth_nuisance_fit_rejected',0)+1;return None
        if not self._supported_birth(fit.x[:2]+offset*fit.x[2:],s,task,'passive_fitted'):return None
        H=jacobian(fit.x,s,NBIAS+4,NBIAS,task,offset);Hx=H[:,NBIAS:];Hb=H[:,:NBIAS]
        A=np.linalg.solve(L,Hx);info=A.T@A+precision
        if np.linalg.cond(info)>1e14:return None
        Qlocal=np.linalg.inv(info);K=Qlocal@np.linalg.solve(L.T,A).T
        Jb=-K@Hb
        # Stable independent-error decomposition. _add already adds Jb Pbb Jb^T;
        # adding the marginal covariance here would count the bias twice.
        Qnoise=K@R@K.T+Qlocal@precision@Qlocal
        # Conservative inverse initialization leaves the old bias marginal
        # unchanged; it is not a full joint birth-count Bayes posterior.
        return fit.x,Jb,(Qnoise+Qnoise.T)/2

    def initiate(self,packets,results):
        """Called immediately after each packet's update in arrival order.

        A shared-noise capture used by existing tracks cannot separately seed a
        new independent track. At most one new track is initialized from a
        shared-reference capture. This conservative loss of birth evidence is
        explicit; it never invents zero cross-target reference correlation.
        """
        if len(packets)!=len(results):raise ContractError('BIRTH_PACKET_RESULT_BINDING')
        allowed={'NO_EXISTING_TRACK','GATED_JOINT_MOMENT_UPDATE','GNN_CAP_APPROXIMATION'}
        for p,res in zip(packets,results):
            if res['status'] not in allowed or p['root'] not in self.seen or p['kind']=='calibrate':continue
            if p['root'] in self.initiated_roots:
                self.stats['duplicate_initiation_packets']+=1;continue
            self.initiated_roots.add(p['root']);s=self.sensors[p['node']];task=Task(**p['task'])
            eligible=res.get('unassigned',[])
            if res.get('shared_capture_used',False):
                self.stats['birth_shared_capture_deferred']+=len(eligible);continue
            offset=p['acquire_s']-self.epoch*self.config['dt_s']
            initialized=False
            for k in eligible:
                if task.role=='bistatic' and initialized:
                    self.stats['birth_shared_capture_deferred']+=1;continue
                if task.role=='monostatic':
                    moments=self._active_birth_moments(np.asarray(p['returns'][k]),s,task)
                    if moments is not None:
                        state,Jb,Q=moments;F,_=transition(-offset,0.)
                        state=F@state;Jb=F@Jb;Q=F@Q@F.T;moments=state,Jb,Q
                else:moments=self._passive_birth_moments(np.asarray(p['returns'][k]),s,task,offset)
                if moments is None:continue
                state,Jb,Q=moments
                initialized=self._add(state[:2],Jb[:2],Q[:2,:2],velocity=state[2:],origin=task.role,
                    state_bias=Jb,state_noise=Q,roots=(p['root'],)) or initialized
