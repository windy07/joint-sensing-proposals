"""Finite local integration: real teacher data, short training, restart, execution.

These runs test software behavior. Their seeds, weights and results are separate
from the registered server campaign, and cannot establish a population effect.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path
from copy import deepcopy
import torch
from .core import ContractError,read_json,write_json,digest,Task
from .binding import ROOT,source_manifest,source_digest,policy_configuration,state_digest
from .preflight import preflight
from .diagnostic_work import diagnostic_authority

RELEASE=ROOT.parents[2]
CONTRACT=RELEASE/'STAGE5_BINDINGS/LOCAL_DIAGNOSTIC_CONTRACT.json'

def run_local(output,phase='all',*,execute=False,prefix='final'):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    contract=read_json(CONTRACT)
    if not execute:
        return {'status':'DRY_RUN','contract':contract,'actual_world_advances':0,'actual_optimizer_updates':0}
    torch.set_num_threads(1)
    for c in contract['allowed_cases']:preflight(c['configuration'])
    def scope(job):return diagnostic_authority(CONTRACT,out,prefix+'/'+job,execute=True)
    from .case_identity import case_identity,checkpoint_identity,seal_result,validate_cached
    from .run import run_case
    from .sequence_training import train,load_sequences
    from .spatial_graph import Priority
    phases=['data','train','resume','evaluate','state_resume'] if phase=='all' else [phase]
    report={'status':'RUNNING','source_manifest':source_manifest(),'contract_digest':digest(contract),'phases':[]}
    allowed=[policy_configuration(x['configuration'],'numeric_network') for x in contract['allowed_cases']]
    train_cases=[x for x in contract['allowed_cases'] if x['split']=='training']
    eval_cases=[x for x in contract['allowed_cases'] if x['split']=='evaluation']
    paths=[out/'data'/f'{x["seed"]}.json' for x in train_cases]
    model_seed=contract['model_seeds'][0]
    def case(spec,method,model_path=None,progress=None,resume_state=None):
        ck=checkpoint_identity(model_path);ident=case_identity(spec['seed'],method,spec['configuration'],ck)
        priority=Priority(model_path,evaluation_split='interface_check') if model_path else None
        result=run_case(spec['seed'],spec['configuration'],method,priority,progress_path=progress,resume_state=resume_state)
        result.update(diagnostic_only=True,evaluation_split='development-diagnostic',diagnostic_contract=digest(contract))
        return seal_result(result,ident)
    try:
        for part in phases:
            started=time.perf_counter();rows=[]
            if part=='data':
                for spec,path in zip(train_cases,paths):
                    if path.exists():raise ContractError('DIAGNOSTIC_DATA_ALREADY_EXISTS')
                    with scope('data/'+str(spec['seed'])):
                        result=case(spec,'numeric_optimizer',progress=out/'partial'/f'data_{spec["seed"]}.json')
                        write_json(path,result)
                        sequences,manifest=load_sequences([path])
                        rows.append({'seed':spec['seed'],'contexts':len(sequences[0]['contexts']),
                                     'source_digest':result['source_digest'],'summary':result['summary']})
            elif part=='train':
                for arm in contract['backbones']:
                    with scope('train/'+arm):
                        rows.append(train(paths,out/'models'/arm,updates=8,arm=arm,seed=model_seed,device='cpu',execute=True,
                            development_seeds=[x['seed'] for x in eval_cases],allowed_configurations=allowed))
            elif part=='resume':
                for arm in contract['backbones']:
                    dest=out/'resume_models'/arm
                    with scope('restart/'+arm):
                        train(paths,dest,updates=4,arm=arm,seed=model_seed,device='cpu',execute=True,
                            development_seeds=[x['seed'] for x in eval_cases],allowed_configurations=allowed)
                        result=train(paths,dest,updates=8,arm=arm,seed=model_seed,device='cpu',execute=True,
                            development_seeds=[x['seed'] for x in eval_cases],allowed_configurations=allowed,resume=dest/'latest.pt')
                    a=torch.load(out/'models'/arm/'latest.pt',weights_only=True,map_location='cpu')
                    b=torch.load(dest/'latest.pt',weights_only=True,map_location='cpu')
                    if a['state_digest']!=b['state_digest']:raise ContractError('TRAIN_RESTART_PARAMETERS_DIFFER')
                    rows.append({'arm':arm,'uninterrupted_updates':8,'split_updates':[4,4],
                        'state_digest':a['state_digest'],'parameters_bit_identical':True,'cursor_world':a['cursor']['world_cursor'],
                        'cursor_context':a['cursor']['context_cursor']})
                    try:Priority(dest/'latest.pt',evaluation_split='confirmation')
                    except ContractError as exc:
                        if 'DIAGNOSTIC_CHECKPOINT' not in str(exc):raise
                    else:raise ContractError('DIAGNOSTIC_MODEL_WAS_ALLOWED_IN_CONFIRMATION')
            elif part=='evaluate':
                for spec in eval_cases:
                    methods=[('fixed_analytic',None,None),('numeric_optimizer',None,None)]
                    methods += [('numeric_network',arm,out/'models'/arm/'latest.pt') for arm in contract['backbones']]
                    for method,arm,path in methods:
                        tag=f'{spec["seed"]}_{method}_{arm or "analytic"}'
                        with scope('evaluate/'+tag):
                            result=case(spec,method,path,progress=out/'partial'/f'{tag}.json')
                            for frame in result['trace']:
                                if frame['commit']['failures']:raise ContractError('DIAGNOSTIC_UNEXPECTED_SCORE_FAILURE')
                            write_json(out/'evaluation/cases'/f'{tag}.json',result)
                            rows.append({'seed':spec['seed'],'method':method,'arm':arm,'summary':result['summary']})
            elif part=='state_resume':
                # Exercise a real interrupted world, including queues, joint
                # posterior and neural semantic memory. Reuse the full run as
                # the uninterrupted reference; no extra reference world.
                from unittest.mock import patch
                import spatial_lab.state_io as state_io
                spec=eval_cases[0];arm=contract['backbones'][0];model_path=out/'models'/arm/'latest.pt'
                reference=read_json(out/'evaluation/cases'/f'{spec["seed"]}_numeric_network_{arm}.json')
                progress=out/'state_restart/partial.json';original=state_io.save_cycle
                class ExpectedInterruption(RuntimeError):pass
                def stop_after_saved(*args,**kwargs):
                    result=original(*args,**kwargs)
                    if read_json(args[0])['completed_cycles']==3:raise ExpectedInterruption('registered interruption after three durable cycles')
                    return result
                with scope('state_restart/interrupted'):
                    try:
                        with patch.object(state_io,'save_cycle',side_effect=stop_after_saved):case(spec,'numeric_network',model_path,progress=progress)
                    except ExpectedInterruption as exc:
                        write_json(out/'state_restart/EXPECTED_INTERRUPTION.json',{'status':'EXPECTED_FAILURE_RETAINED','error':repr(exc)})
                    else:raise ContractError('INTERRUPTION_NOT_REACHED')
                with scope('state_restart/continued'):
                    resumed=case(spec,'numeric_network',model_path,progress=progress,resume_state=str(progress)+'.state.json')
                fields=('causal_view','packets','predecision_packets','update_results','posterior_mean','posterior_covariance',
                        'tracks','remaining_RF_energy_J','remaining_transmit_time_s','post_search_age_cycles','post_unseen_intensity')
                for a,b in zip(reference['trace'],resumed['trace']):
                    for key in fields:
                        if digest(a[key])!=digest(b[key]):raise ContractError('WORLD_RESTART_DIFFERENCE:'+key)
                    if digest(a['commit']['selected'])!=digest(b['commit']['selected']):raise ContractError('WORLD_RESTART_ACTION_DIFFERENCE')
                write_json(out/'state_restart/RESUMED_CASE.json',resumed)
                rows=[{'completed_before_resume':3,'cycles':len(resumed['trace']),'physical_state_and_actions_bit_identical':True,
                       'excluded_from_exact_comparison':'wall-clock timing only','extra_world_advances':spec['configuration']['cycles']}]
            else:raise ContractError('UNKNOWN_DIAGNOSTIC_PHASE')
            report['phases'].append({'phase':part,'status':'PASSED','seconds':time.perf_counter()-started,'rows':rows})
            write_json(out/f'{prefix}_{part}_REPORT.json',report['phases'][-1])
            write_json(out/f'{prefix}_REPORT.json',report)
        report['status']='PASSED'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        write_json(out/'failures'/f'{prefix}_{phase}_{time.time_ns()}.json',report)
        raise
    finally:
        report['ledger']=read_json(out/'LOCAL_COMPUTE_LEDGER.json') if (out/'LOCAL_COMPUTE_LEDGER.json').exists() else None
        write_json(out/f'{prefix}_REPORT.json',report)
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['all','data','train','resume','evaluate','state_resume'],default='all')
    p.add_argument('--execute',action='store_true');p.add_argument('--prefix',default='final')
    args=p.parse_args(argv)
    try:
        r=run_local(args.output,args.phase,execute=args.execute,prefix=args.prefix)
        print(json.dumps({'status':r['status'],'phases':[(x['phase'],x['status']) for x in r.get('phases',[])]}));return 0
    except BaseException as exc:print(repr(exc));return 2

if __name__=='__main__':raise SystemExit(main())
