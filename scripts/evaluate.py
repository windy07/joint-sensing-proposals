"""Run a selected confirmation episode with the released best checkpoint."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job', help='Exact job ID from the frozen confirmation design')
    ap.add_argument('--world-seed', type=int, default=623000001)
    ap.add_argument('--arm', choices=('integrated_agent','deep_sets','set_transformer'), default='integrated_agent')
    ap.add_argument('--model-seed', type=int, choices=(633431,633432,633433), default=633431)
    ap.add_argument('--variant', choices=('n0f0','n1f0','n0f1','n1f1','fixed_analytic','numeric_optimizer'), default='n1f1')
    ap.add_argument('--models-root', type=Path, default=ROOT/'run/models')
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/confirmation')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save-trace', action='store_true')
    ap.add_argument('--execute', action='store_true', help='Execute on Linux; without this flag only inspect the job')
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT/'posttrain_confirmation'))
    from common import read, write, sha, load_science
    load_science(ROOT/'project')
    from runner import validate_design, intervention
    from frozen_contract import validate_frozen, validate_checkpoint
    design=read(ROOT/'posttrain_confirmation/confirmation_design.json')
    original=read(ROOT/'project/STAGE6_SERVER/plan_12800.json')
    validate_design(design,original)
    baseline=args.variant in ('fixed_analytic','numeric_optimizer')
    jobs=[j for j in design['jobs'] if (j['id']==args.job if args.job else
        j['world_seed']==args.world_seed and j['variant']==args.variant and
        j['arm']==(None if baseline else args.arm) and
        j['model_seed']==(None if baseline else args.model_seed))]
    if len(jobs)!=1:
        raise ValueError('Select exactly one job present in the frozen design')
    job=jobs[0]
    if not args.execute:
        print(json.dumps({'status':'PREVIEW','job':job,'optimizer_updates':0,'world_advances':0},indent=2))
        return
    if sys.platform!='linux':
        raise RuntimeError('Run simulation evaluations on Linux')
    receipt=args.output/'receipts'/(job['id']+'.json')
    if receipt.exists():
        raise FileExistsError('Choose a fresh output directory; existing receipt: '+str(receipt))
    import torch
    from spatial_lab import work_budget
    from spatial_lab.run import run_case
    from spatial_lab.spatial_graph import Priority
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
        torch.backends.cuda.matmul.allow_tf32=False
    learned=None; checkpoint_sha=None
    if job['arm'] is not None:
        path=args.models_root/job['arm']/str(job['model_seed'])/'best.pt'
        checkpoint_sha=sha(path)
        validate_checkpoint(validate_frozen(design),job['arm'],job['model_seed'],checkpoint_sha)
        learned=Priority(path,device=args.device,evaluation_split='confirmation')
    method=job['variant'] if job['arm'] is None else ('numeric_network' if job['variant'][1]=='1' else 'fixed_network')
    result={'status':'RUNNING','job':job,'design_digest':design['design_digest'],
            'contract_digest':design['contract_digest'],'checkpoint_sha256':checkpoint_sha,
            'device':args.device,'actual_optimizer_updates':0}
    write(receipt,result)
    token=work_budget.enable_server(design['design_digest'])
    try:
        with work_budget.prohibit_optimizers(), intervention(job):
            case=run_case(job['world_seed'],job['configuration'],method,learned)
        if len(case['trace'])!=job['configuration']['cycles'] or case['world_advances']!=len(case['trace']):
            raise ValueError('Incomplete episode')
        result.update(status='COMPLETE',summary=case['summary'],actual_world_advances=case['world_advances'])
        if args.save_trace:
            write(args.output/'cases'/(job['id']+'.json.gz'),case)
        write(receipt,result)
    except Exception as exc:
        write(receipt,{**result,'status':'FAILED','error':repr(exc)})
        raise
    finally:
        work_budget.disable_server(token)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
