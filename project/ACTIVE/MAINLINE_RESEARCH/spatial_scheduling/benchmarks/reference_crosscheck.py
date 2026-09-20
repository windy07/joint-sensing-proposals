"""Optional mature-library coordinate/transition check. Never installs packages.
Run after reviewing and installing Stone Soup in a separate environment, if desired.
This is a formula crosscheck, not an end-to-end tracker performance comparison.
"""
import sys,json,datetime,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from spatial_lab.core import transition,write_json
p=ROOT/'benchmarks'/'mature_reference_status.json'
if importlib.util.find_spec('stonesoup') is None:
    write_json(p,{'status':'NOT_EXECUTED_MISSING_DEPENDENCY','dependency':'stonesoup','no_install_performed':True})
    print(p);sys.exit(3)
from stonesoup.models.transition.linear import ConstantVelocity,CombinedLinearGaussianTransitionModel
model=CombinedLinearGaussianTransitionModel([ConstantVelocity(.8),ConstantVelocity(.8)])
F,Q=transition(1.,.8);perm=[0,2,1,3]
f=np.asarray(model.matrix(time_interval=datetime.timedelta(seconds=1)))
q=np.asarray(model.covar(time_interval=datetime.timedelta(seconds=1)))
np.testing.assert_allclose(f,F[np.ix_(perm,perm)],atol=1e-12)
np.testing.assert_allclose(q,Q[np.ix_(perm,perm)],atol=1e-12)
write_json(p,{'status':'PASSED_TRANSITION_ONLY','coordinate_map':'ours [x,y,vx,vy]; Stone Soup [x,vx,y,vy]',
'F_max_error':float(np.max(np.abs(f-F[np.ix_(perm,perm)]))),'Q_max_error':float(np.max(np.abs(q-Q[np.ix_(perm,perm)]))),
'full_tracker_comparison':False})
print(p)
