"""Finite confirmation queue followed by fixed summaries; no recurring monitor."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, read, write, sha, load_science
from runner import validate_design


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    args=ap.parse_args();root=Path(args.root).resolve()
    out=root/'run/POSTTRAIN_CONFIRMATION';out.mkdir(exist_ok=True)
    design=ROOT/'confirmation_design.json'
    load_science(root/'project')
    validate_design(read(design),read(root/'project/STAGE6_SERVER/plan_12800.json'))
    commands=[
        ('evaluation',[sys.executable,'-B','-u',str(ROOT/'resource_queue.py'),'--root',str(root),
          '--design',str(design),'--output',str(out),'--max-workers','64','--gpu-workers','24',
          '--required-free-gpus','3','--poll-seconds','3','--pilot-evidence',str(root/'run/POSTTRAIN_DEVELOPMENT')]),
        ('integrity',[sys.executable,'-B','-u',str(ROOT/'verify_confirmation_results.py'),
          '--root',str(root),'--design',str(design),'--output',str(out)]),
        ('statistics',[sys.executable,'-B','-u',str(ROOT/'posttrain.py'),'summarize','--design',str(design),
          '--results',str(out),'--output',str(out/'CONFIRMATION_STATISTICS.json'),'--draws','10000']),
        ('tables',[sys.executable,'-B','-u',str(ROOT/'paper_tables.py'),'--report',str(out/'CONFIRMATION_STATISTICS.json'),
          '--output',str(out/'PAPER_TABLES')])]
    status={'started_at':datetime.datetime.now().astimezone().isoformat(),'status':'RUNNING',
            'contract_digest':read(design)['contract_digest'],'completed_steps':[]}
    for name,cmd in commands:
        status['phase']=name;status['updated_ns']=time.time_ns();write(out/'PIPELINE_STATUS.json',status)
        print({'phase':name,'status':'STARTING'},flush=True)
        code=subprocess.call(cmd)
        if code:
            status.update(status='FAILED',returncode=code,updated_ns=time.time_ns())
            write(out/'PIPELINE_STATUS.json',status);return code
        if name=='statistics' and read(out/'CONFIRMATION_STATISTICS.json')['status']!='COMPLETE':
            status.update(status='FAILED',reason='INCOMPLETE_PRIMARY_STATISTICS',updated_ns=time.time_ns())
            write(out/'PIPELINE_STATUS.json',status);return 2
        status['completed_steps'].append(name)
    status.update(status='COMPLETE',phase='complete',finished_at=datetime.datetime.now().astimezone().isoformat(),updated_ns=time.time_ns())
    write(out/'PIPELINE_STATUS.json',status);print(status,flush=True);return 0


if __name__=='__main__':sys.exit(main())
