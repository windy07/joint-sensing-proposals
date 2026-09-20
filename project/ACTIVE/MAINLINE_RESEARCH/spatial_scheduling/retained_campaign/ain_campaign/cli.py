from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
from .common import *

def main():
 p=argparse.ArgumentParser(description='AIN integrated graph-agent campaign, offline only')
 p.add_argument('stage',choices=['preflight','calibrate','labels','train','freeze','evaluate','commit','audit','analyze','campaign','smoke','export','checks','final-checks'])
 p.add_argument('--config',type=Path,default=ROOT/'configs'/'campaign.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--tag');p.add_argument('--workers',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--require-cuda',action='store_true');p.add_argument('--tier',choices=['default','reduced']);p.add_argument('--archive',type=Path)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 try:
  from .planning import validate_config
  c=validate_config(read_json(a.config))
  if a.stage=='campaign':
   from .campaign import run_campaign
   r=run_campaign(a.config,a.output,a.resume,a.require_cuda,a.tier)
  elif a.stage=='smoke':
   from .campaign import smoke
   r=smoke(c,a.output,a.device)
  elif a.stage=='preflight':
   from .planning import preflight
   r=preflight(c,a.output,a.require_cuda)
  elif a.stage=='calibrate':
   from .planning import calibrate
   r=calibrate(c,a.output,a.tier)
  elif a.stage=='labels':
   from .dataset import generate_labels
   r=generate_labels(c,a.output,a.workers)
  elif a.stage=='train':
   from .train import train_job
   if a.arm is None or a.seed is None:raise ContractError('ARM_AND_SEED_REQUIRED')
   r=train_job(c,a.output,a.arm,a.seed,a.device)
  elif a.stage=='freeze':
   from .evaluate import freeze_plan
   r=freeze_plan(c,a.output)
  elif a.stage=='evaluate':
   from .evaluate import evaluate_job
   if not a.tag:raise ContractError('EVALUATION_TAG_REQUIRED')
   r=evaluate_job(c,a.output,a.tag,a.device)
  elif a.stage=='commit':
   from .evaluate import commit_evaluation
   r=commit_evaluation(c,a.output)
  elif a.stage=='audit':
   from .evaluate import audit_all
   r=audit_all(c,a.output,a.workers)
  elif a.stage=='analyze':
   from .analysis import analyze
   r=analyze(c,a.output)
  elif a.stage=='export':
   from .analysis import export_results
   r=export_results(a.output,a.archive or a.output.parent/(a.output.name+'_export.zip'))
  elif a.stage=='final-checks':
   from .final_checks import checks
   r=checks(c,a.output)
  elif a.stage=='checks':
   from .checks import checks
   r=checks(c,a.output)
  print(json.dumps(plain(r),indent=2,allow_nan=False));return 0
 except BaseException as e:
  atomic_json(a.output/(a.stage+'_FAILURE.json'),{'status':'FAILED','stage':a.stage,'error':repr(e),'traceback':traceback.format_exc(),'source_digest':source_digest()});traceback.print_exc();return 1
