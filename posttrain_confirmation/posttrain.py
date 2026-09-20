"""Prepare, inspect, run a bounded development job, or summarize results."""
from pathlib import Path
import argparse
import json
from common import read, write, digest, load_science


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--server-plan', required=True)
    plan.add_argument('--output', required=True)
    check = sub.add_parser('checkpoints')
    check.add_argument('--project', required=True)
    check.add_argument('--server-plan', required=True)
    check.add_argument('--models-root', required=True)
    check.add_argument('--output', required=True)
    run = sub.add_parser('run-job')
    for arg in ('project', 'server-plan', 'design', 'job', 'models-root', 'output'):
        run.add_argument('--' + arg, required=True)
    run.add_argument('--device', default='cpu')
    run.add_argument('--execute', action='store_true')
    report = sub.add_parser('summarize')
    for arg in ('design', 'results', 'output'):
        report.add_argument('--' + arg, required=True)
    report.add_argument('--draws', type=int, default=10000)
    args = ap.parse_args()
    if args.command == 'plan':
        from design import make_design
        doc = make_design(read(args.server_plan))
        doc['design_digest'] = digest(doc)
        write(args.output, doc)
        result = {'status': doc['status'], 'episodes': {s: sum(j['split'] == s for j in doc['jobs'])
                   for s in ('development', 'confirmation')}, 'actual_world_advances': 0, 'actual_optimizer_updates': 0}
    elif args.command == 'checkpoints':
        load_science(args.project)
        from design import audit_all
        result = audit_all(args.models_root, read(args.server_plan))
        write(args.output, result)
    elif args.command == 'run-job':
        from runner import run_job
        result = run_job(args.project, args.server_plan, args.design, args.job, args.models_root, args.output,
                         device=args.device, execute=args.execute)
        result = {k: v for k, v in result.items() if k not in ('summary', 'job')}
    else:
        from paired_statistics import analyze
        design = read(args.design)
        if design['design_digest'] != digest({k: v for k, v in design.items() if k != 'design_digest'}):
            raise ValueError('DESIGN_CHANGED')
        from frozen_contract import validate_frozen
        validate_frozen(design)
        design = {**design, 'jobs': [j for j in design['jobs'] if j['split'] == 'confirmation']}
        records = [read(p) for p in sorted((Path(args.results) / 'receipts').glob('*.json'))]
        result = analyze(design, records, draws=args.draws)
        result['scope'] = 'confirmation'
        result['contract_digest'] = design['contract_digest']
        result['primary_comparison'] = 'integrated_agent/n1f1 minus numeric_optimizer'
        result['secondary_inference'] = 'descriptive intervals; no multiplicity-adjusted superiority claims'
        write(args.output, result)
        result = {k: v for k, v in result.items() if k != 'metrics'}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
