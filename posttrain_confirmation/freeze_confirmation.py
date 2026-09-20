"""Freeze the already planned confirmation matrix before any confirmation outcome."""
import argparse
import datetime
import math
from pathlib import Path
from common import ROOT, read, write, digest, sha, adapter_digest, load_science


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    args=ap.parse_args()
    root=Path(args.root).resolve()
    load_science(root/'project')
    from spatial_lab.binding import source_digest
    from spatial_lab.metrics import gospa_parts
    parent_path=root/'posttrain_eval/development_design.json'
    parent=read(parent_path)
    original=read(root/'project/STAGE6_SERVER/plan_12800.json')
    readiness_path=root/'operations/confirmation_launch_20260916/DEVELOPMENT_READINESS.json'
    readiness=read(readiness_path)
    if (ROOT/'EXPERIMENT_CONTRACT.json').exists() or (root/'run/POSTTRAIN_CONFIRMATION').exists():
        raise ValueError('ALREADY_FROZEN_OR_CONFIRMATION_OUTPUT_EXISTS')
    assert parent['design_digest']==digest({k:v for k,v in parent.items() if k!='design_digest'})
    assert parent['base_plan_digest']==digest(original)
    assert parent['base_source_digest']==source_digest()
    for name in ('policy.py','trace_export.py','paired_statistics.py'):
        assert sha(ROOT/name)==sha(root/'posttrain_eval'/name)
    partitions=[set(v) for v in original['world_splits'].values()]
    assert not any(a&b for i,a in enumerate(partitions) for b in partitions[i+1:])
    jobs=[j for j in parent['jobs'] if j['split']=='confirmation']
    worlds=sorted({j['world_seed'] for j in jobs})
    assert len(jobs)==4992 and len(worlds)==192
    assert all(sum(j['scenario']==s for j in jobs)==24*26 for s in parent['strata'])
    assert readiness['complete']==1664 and readiness['worlds']==64 and not readiness['primary_missing']
    assert readiness['models']['status']=='READY' and len(readiness['models']['models'])==9
    checkpoints={}
    for row in readiness['models']['models']:
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        assert sha(Path(row['checkpoint']).with_name('epoch_010.pt'))==row['epoch10_sha256']
        checkpoints[row['arm']+'/'+str(row['model_seed'])]=row['checkpoint_sha256']
    cutoff=250.
    controls=[]
    for name,truth,estimate,expected in (
        ('perfect_localization',[[0.,0.]],[[0.,0.]],0.),
        ('ten_meter_error',[[0.,0.]],[[10.,0.]],10.),
        ('missed_target',[[0.,0.]],[],cutoff/math.sqrt(2)),
        ('false_track',[],[[0.,0.]],cutoff/math.sqrt(2)),
        ('out_of_gate',[[0.,0.]],[[1000.,0.]],cutoff),
        ('empty_world',[],[],0.)):
        value=gospa_parts(truth,estimate,cutoff)['gospa_m']
        assert math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-12)
        controls.append({'name':name,'expected_m':expected,'observed_m':value,'status':'PASS'})
    replay=read(root/'operations/resource_tuning_20260916/BENCHMARK_REPORT.json')
    assert replay['status']=='COMPLETE' and replay['complete']==80 and not replay['metric_mismatches']
    assert replay['completed_metric_comparisons']==1520
    low,high=readiness['primary_range_m']
    assert 0<=low<high<cutoff*math.sqrt(3)
    assert any(abs(s['delta_mean_m'])>=5 for s in readiness['primary_by_scenario'])
    se=readiness['standard_error_at_24_worlds_per_scenario']
    bound=2*cutoff*math.sqrt(3)*math.sqrt(math.log(40)/(2*192))
    contract={
        'schema_version':'experiment-contract.v2','mode':'confirmatory',
        'experiment_id':'stage6_confirmation_192_worlds_20260916',
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'objective':'Estimate the paired-world change in closed-loop multi-target estimation and resource use for the frozen Stage6 policies on the eight prespecified scenarios.',
        'authorization':{'source':'User instruction: 现在开始','target':str(root),
                         'scope':'4992 confirmation episodes and final integrity/statistical summaries; no optimizer updates'},
        'inputs':[{'development_design':str(parent_path),'digest':parent['design_digest']},
                  {'readiness':str(readiness_path),'sha256':sha(readiness_path)},
                  {'resource_validation':str(root/'operations/resource_tuning_20260916/BENCHMARK_REPORT.json'),
                   'sha256':sha(root/'operations/resource_tuning_20260916/BENCHMARK_REPORT.json')}],
        'resource_budget':{'episodes':4992,'worlds':192,'decision_cycles_per_episode':64,
                           'world_advances':319488,'optimizer_updates':0,'max_workers':64,
                           'gpu_count':3,'gpu_workers_per_device':24,'cpu_threads_per_worker':1,
                           'root_evaluations_per_cycle':12,'construction_attempts_per_cycle':192},
        'failure_handling':{'execution':'Any failed episode stops new dispatch; running episodes finish. Preserve errors and completed results.',
                            'restart':'Only unchanged frozen jobs may resume after an infrastructure failure. Deterministic scientific failure requires review, not silent exclusion.',
                            'missing_primary':'Do not issue a complete primary estimate when an expected primary outcome is missing or nonfinite.',
                            'conditional_secondary':'Retain null reasons and denominators; compare on complete paired worlds for that conditional metric, never average away a missing fitted seed.'},
        'claim_ceiling':'Paired simulation performance for the registered measurement-level model and the nine retained trained fits; the study does not estimate variability from an unlimited population of future training seeds.',
        'estimands':[{'name':'primary','metric':'mean_gospa_m','unit':'m',
                       'contrast':{'integrated_agent/n1f1':1.,'numeric_optimizer':-1.},
                       'direction':'negative is improved estimation','aggregation':'mean of three fixed fitted seeds within world; mean worlds within scenario; equal mean across eight scenarios'},
                      {'name':'secondary','metrics':'the 19 previously defined physical and accounting quantities',
                       'contrasts':'all prespecified method-minus-numerical-search and integrated-agent factorial contrasts; descriptive intervals'}],
        'independent_unit':'world',
        'independent_unit_definition':'One simulator seed generates the shared exogenous target, nuisance, emission and transport law; all policies and retained fitted seeds remain paired within that world.',
        'treatments':[{'architectures':list(original['backbones']),'model_seeds':list(original['model_seeds']),
                       'matrix':'integrated: four numeric/feedback cells; Deep Sets and Set Transformer: fixed/no-feedback and generated/feedback'}],
        'controls':{'positive':{'metric_controls':controls,'resource_replay':'80 unchanged episodes, 1520 metrics reproduced'},
                     'negative':{'null_and_miss_controls':[x for x in controls if x['name'] in ('empty_world','missed_target','false_track')],
                                 'mechanism_controls':'fixed numeric parameters and feedback disabled cells, already executed on development worlds'},
                     'comparators':['fixed_analytic','numeric_optimizer']},
        'analysis_plan':{'implementation_sha256':sha(ROOT/'paired_statistics.py'),'bootstrap_draws':10000,
                          'bootstrap_seed':90231,'confidence':.95,'bootstrap':'stratified paired-world percentile bootstrap; fitted models held fixed',
                          'reporting':'Report all scenarios, all three fitted seeds, metric units and conditional denominators. Do not select scenarios using confirmation outcomes.',
                          'primary_decision':'Upper primary 95% interval below zero supports lower mean GOSPA; upper bound at or below -5 m supports the prespecified practically useful improvement.',
                          'stopping':'Fixed 192 worlds; no outcome-based early stopping or sample-size extension.'},
        'minimum_effects':[{'metric':'mean_gospa_m','improvement_m':5.,
                            'rationale':'Prespecified practical reporting threshold equal to 2 percent of the 250 m GOSPA cutoff; not a launch or sample-selection criterion.'}],
        'sample_size_plan':{'size_rule':'Retain the previously planned 192 new worlds, 24 in each scenario, all 26 configurations per world.',
                             'world_seeds':worlds,
                             'power_or_sensitivity':{'development_plugin_standard_error_m':se,
                               'normal_approx_95_halfwidth_m':1.95996398454*se,
                               'normal_approx_80pct_detectable_effect_m':(1.95996398454+.84162123357)*se,
                               'bounded_95_halfwidth_m':bound,
                               'bound_basis':'Independent bounded world differences in [-250 sqrt(3),250 sqrt(3)], from at most three truths and three retained tracks; Hoeffding halfwidth. Empirical sensitivity is a planning approximation, not guaranteed power.'}},
        'multiplicity':{'primary_family':'One primary comparison on one primary metric at two-sided 95 percent confidence.',
                         'secondary_family':'Fixed descriptive interval family; no familywise superiority claim from unadjusted secondary intervals.'},
        'fidelity':'R1',
        'fidelity_description':'Measurement-level, link-budget and thermal-noise model with scenario-specific occlusion, availability and transport processes.',
        'embargo':{'confirmation_unopened':True,'separation':'Training, validation, development and confirmation world seeds are pairwise disjoint; confirmation output directory absent at freeze.',
                   'monitoring':'Operational status and fixed outcome collection are permitted after freeze. No adaptation of models, job matrix or analysis from partial outcomes.'},
        'admissibility':{
            'A0':{'status':'PASS','reason':'1664 complete development episodes, 64 independent worlds, finite complete primary outcomes; exact metric control values passed.'},
            'A1':{'status':'PASS','reason':'Eight executed scenarios; development GOSPA range is nonconstant.', 'observed_range_m':[low,high]},
            'A2':{'status':'PASS','reason':'Primary outcomes vary below the mathematical upper bound; matched-only metric nulls have explicit reasons and paired denominators. Missed and false target penalties checked separately.'},
            'A3':{'status':'PASS','reason':'Distinct fixed/generated and feedback treatment cells are executable and evaluated; observed scenario-level paired changes exceed the reporting threshold in magnitude.'},
            'A4':{'status':'PASS','reason':'Executed near/transition/far range, emitter/illuminator attenuation, intermittency, outage and transport/occlusion ladder. Claim is confined to the same declared measurement model.'},
            'A5':{'status':'PASS','reason':'Six exact primary-metric controls and 80 resource replays passed; analytic/numerical baselines and all four mechanism cells already executed on development.'}},
        'launch_gate':{'target':'4992 frozen Stage6 confirmation episodes','status':'GO',
                        'reason':'Finite checks, source and checkpoint bindings, disjoint unopened worlds and fixed analysis established before generation.'},
        'bindings':{'jobs_digest':digest(jobs),'adapter_digest':adapter_digest(),'science_digest':source_digest(),
                     'checkpoints':checkpoints,'parent_development_design_digest':parent['design_digest']},
        'digest_convention':'SHA-256 of canonical JSON with sorted keys, compact separators, UTF-8, no NaN; exclude only top-level freeze_digest.'}
    contract['freeze_digest']='sha256:'+digest(contract)
    doc={**parent,'status':'FROZEN_CONFIRMATION_DESIGN','confirmation_status':'FROZEN',
         'jobs':jobs,'adapter_digest':adapter_digest(),'contract_digest':contract['freeze_digest'],
         'parent_development_design_digest':parent['design_digest']}
    doc.pop('design_digest',None);doc['design_digest']=digest(doc)
    from frozen_contract import validate_frozen
    validate_frozen(doc,contract)
    write(ROOT/'EXPERIMENT_CONTRACT.json',contract)
    write(ROOT/'confirmation_design.json',doc)
    write(ROOT/'CHECKPOINT_SELECTION.json',readiness['models'])
    print({'status':'FROZEN','worlds':192,'episodes':4992,'models':9,'metric_controls':len(controls),
           'standard_error_planning_m':se,'optimizer_updates':0,'world_advances':0})


if __name__=='__main__':main()
