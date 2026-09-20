"""Recompute paired statistics from the released episode summaries, without simulation."""
from pathlib import Path
import argparse
import gzip
import json
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/statistics.json')
    ap.add_argument('--draws',type=int,default=10000)
    ap.add_argument('--receipts',type=Path,help='Use newly evaluated receipts instead of the released summaries')
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT/'posttrain_confirmation'))
    from common import read, write
    from frozen_contract import validate_frozen
    from paired_statistics import analyze
    design=read(ROOT/'posttrain_confirmation/confirmation_design.json')
    validate_frozen(design)
    if args.receipts:
        records=[read(p) for p in sorted(args.receipts.glob('*.json'))]
    else:
        with gzip.open(ROOT/'results/confirmation/episode_summaries.json.gz','rt',encoding='utf-8') as f:
            records=json.load(f)
    report=analyze(design,records,draws=args.draws)
    report.update(scope='confirmation',contract_digest=design['contract_digest'],
                  primary_comparison='integrated_agent/n1f1 minus numeric_optimizer')
    write(args.output,report)
    print(json.dumps({'status':report['status'],'receipts':len(records),'output':str(args.output)}))


if __name__=='__main__':
    main()
