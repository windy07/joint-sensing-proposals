"""Fixed-input comparison with installed Stone Soup 1.8, no training.

Pass --library only for a separately installed reference package directory.
This checks geometry, prediction, single-target filtering and set metrics.
It does not equate the custom multi-target association model to Stone Soup.
"""
import sys, argparse, datetime, hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--library');ap.add_argument('--output',required=True)
args=ap.parse_args()
if args.library:sys.path.insert(0,args.library)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import stonesoup
from stonesoup.models.transition.linear import ConstantVelocity,CombinedLinearGaussianTransitionModel
from stonesoup.models.measurement.nonlinear import CartesianToBearingRangeRate
from stonesoup.types.state import State,GaussianState
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.metricgenerator.ospametric import GOSPAMetric
from spatial_lab.core import transition,gaussian_update,read_json,write_json
from spatial_lab.physics import SENSORS,Task,measurement,jacobian,noise_cov,innovation
from spatial_lab.metrics import gospa_parts

c=read_json(ROOT/'spatial_configs/local.json')['model']
ix=[0,2,1,3];embed=[0,2,1,3];order=[1,0,2]
tm=CombinedLinearGaussianTransitionModel([ConstantVelocity(.8),ConstantVelocity(.8),ConstantVelocity(0.)])
dt=datetime.timedelta(seconds=1);f,q=transition(1.,.8)
fs=np.asarray(tm.matrix(time_interval=dt));qs=np.asarray(tm.covar(time_interval=dt))
np.testing.assert_allclose(fs[:4,:4],f[np.ix_(ix,ix)],atol=1e-13)
np.testing.assert_allclose(qs[:4,:4],q[np.ix_(ix,ix)],atol=1e-13)
checks={'transition_matrix_max_error':float(np.max(abs(fs[:4,:4]-f[np.ix_(ix,ix)]))),
        'process_covariance_max_error':float(np.max(abs(qs[:4,:4]-q[np.ix_(ix,ix)])))}
state=np.array([320.,450.,18.,-8.]);geometry=[]
for s in SENSORS[:2]:
    r=noise_cov(state[:2],s,Task(s.id,'track',0),0,c)
    model=CartesianToBearingRangeRate(6,(0,2,4),r[np.ix_(order,order)],
                                     translation_offset=np.array([*s.position,0.]).reshape(3,1))
    st=State(np.r_[state[ix],0.,0.])
    expected=measurement(state,np.zeros(6),s)[order]
    actual=np.asarray(model.function(st,noise=False),float).ravel()
    np.testing.assert_allclose(actual,expected,atol=1e-9)
    H=jacobian(state,s,10,6)[order,6:10][:,ix]
    HH=np.asarray(model.jacobian(st))[:,:4]
    np.testing.assert_allclose(HH,H,atol=2e-5,rtol=2e-5)
    geometry.append({'node':s.name,'measurement_max_error':float(max(abs(actual-expected))),
                     'jacobian_max_error':float(np.max(abs(HH-H)))})
checks['geometry']=geometry

# Twelve successive observations; both filters receive the identical measurement.
m=np.array([330.,442.,15.,-6.]);P=np.diag([120.,100.,16.,20.]);truth=state.copy()
Pss=np.zeros((6,6));Pss[:4,:4]=P[np.ix_(ix,ix)]
now=datetime.datetime(2026,1,1);ss=GaussianState(np.r_[m[ix],0.,0.],Pss,timestamp=now)
predictor=KalmanPredictor(tm);record=[]
for k in range(12):
    now+=dt;truth=f@truth;m=f@m;P=f@P@f.T+q
    ss=predictor.predict(ss,timestamp=now);s=SENSORS[k%2];task=Task(s.id,'track',0)
    R=noise_cov(m[:2],s,task,k,c)
    z=measurement(truth,np.zeros(6),s)+np.linalg.cholesky(R)@np.array([.2*np.sin(k),.15*np.cos(k),.1])
    H=jacobian(m,s,10,6)[:,6:10];dz=innovation(z,measurement(m,np.zeros(6),s),s)
    m,P,_,_=gaussian_update(m,P,H,R,dz)
    model=CartesianToBearingRangeRate(6,(0,2,4),R[np.ix_(order,order)],
                                     translation_offset=np.array([*s.position,0.]).reshape(3,1))
    detection=Detection(z[order],timestamp=now,measurement_model=model)
    ss=ExtendedKalmanUpdater(model).update(SingleHypothesis(ss,detection))
    sm=np.asarray(ss.state_vector,float).ravel()[:4][ix];SP=np.asarray(ss.covar)[:4,:4][np.ix_(ix,ix)]
    record.append({'epoch':k,'mean_max_error':float(np.max(abs(sm-m))),
                   'covariance_max_error':float(np.max(abs(SP-P)))})
    np.testing.assert_allclose(sm,m,atol=2e-3,rtol=1e-6)
    np.testing.assert_allclose(SP,P,atol=2e-3,rtol=1e-5)
checks['sequential_filter']=record
checks['sequential_mean_max_error']=max(x['mean_max_error'] for x in record)
checks['sequential_covariance_max_error']=max(x['covariance_max_error'] for x in record)

g=GOSPAMetric(p=2,c=20.)
cases=[([],[]),([[0,0]],[]),([[0,0]],[[100,0]]),([[0,0],[10,0]],[[1,0],[8,0],[100,0]])]
gs=[]
for x,y in cases:
    ours=gospa_parts(x,y,20.)
    if not x and not y:
        continue
    ref=g.compute_gospa_metric([State(p,timestamp=now) for p in y],[State(p,timestamp=now) for p in x])
    metric=ref[0] if isinstance(ref,tuple) else ref
    value=metric.value
    val=float(value['distance'])
    np.testing.assert_allclose(val,ours['gospa_m'],atol=1e-10)
    gs.append({'truth_count':len(x),'estimate_count':len(y),'ours_m':ours['gospa_m'],'reference_m':val,
               'missed_count':ours['missed_count'],'false_count':ours['false_count']})
checks['gospa']=gs
write_json(args.output,{'status':'PASS','reference':'Stone Soup','version':stonesoup.__version__,
    'checks':checks,'simulation_worlds':0,'known_association_filter_steps':12,
    'full_multitarget_tracker_equivalence':False,'measurement_hardware_validation':False})
print('PASS',args.output)
