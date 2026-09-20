"""Four optimizer steps and four unseen closed-loop cycles; no efficacy claim."""
from pathlib import Path
import os, sys, time
# Keep the local NumPy MKL and PyTorch runtime from initializing two OpenMP DLLs.
# This selects sequential BLAS, rather than suppressing duplicate-runtime errors.
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial_lab.core import read_json,write_json,ContractError
from spatial_lab.run import run_case,source_digest
from spatial_lab.spatial_graph import train,Priority

def main(output):
    out=Path(output);start=time.perf_counter()
    plan=read_json(ROOT/'spatial_configs/local_review.json')
    cfg={**plan['model'],'scenario':'single','cycles':4}
    teacher=out/'teacher.json'
    write_json(teacher,run_case(9401881,cfg,'lookahead'))
    receipt=train([teacher],out/'spatial_graph.pt',updates=4)
    priority=Priority(out/'spatial_graph.pt')
    inference=run_case(9401882,cfg,'learned',priority)
    write_json(out/'inference.json',inference)
    try:run_case(9401881,cfg,'learned',Priority(out/'spatial_graph.pt'))
    except ContractError as exc:
        assert str(exc)=='TRAIN_EVALUATION_WORLD_OVERLAP'
        overlap=True
    else:raise AssertionError('training overlap accepted')
    assert receipt['updates']==4 and receipt['parameter_l1_change']>0
    assert receipt['checkpoint_reload_max_difference']==0
    assert inference['status']=='COMPLETE'
    result={'status':'PASS','source_digest':source_digest(),'training':receipt,
      'new_worlds':2,'teacher_cycles':4,'inference_cycles':4,'optimizer_updates':4,
      'evaluation_world_overlap_rejected':overlap,'quality_claim':False,
      'wall_seconds':time.perf_counter()-start}
    write_json(out/'GRAPH_WIRING.json',result);print(result)

if __name__=='__main__':main(sys.argv[1])
