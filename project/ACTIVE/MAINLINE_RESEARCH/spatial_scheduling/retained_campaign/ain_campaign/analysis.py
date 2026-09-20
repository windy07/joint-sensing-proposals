"""World-paired scientific analysis. Never treats cycles as independent worlds."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import csv,math,statistics,json,time,zipfile
import numpy as np
from scipy.stats import norm,t as student_t
from .common import *
from .evaluate import get_plan

LABELS={'original_agent':'Original graph Agent','integrated_agent':'Integrated graph Agent','deep_sets':'Deep Sets','set_transformer':'Set Transformer','analytic_constructive':'Analytic construction','beam_block':'Budgeted beam / block','random':'Random construction','no_assoc_supervision':'Without association-risk supervision','integrated_no_protection':'Without protected construction','original_with_protection':'Original model + protected construction','no_assoc_aux':'Without association auxiliary','no_prov_aux':'Without provenance auxiliary','no_residual_priority':'Without residual-improvement priority'}

def csvfile(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:raise ContractError('EMPTY_SCIENTIFIC_TABLE:'+str(path))
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def cvar(values,alpha=.95):
 if not values:return None
 a=np.sort(np.asarray(values,dtype=float));take=(1-alpha)*len(a);whole=int(math.floor(take));frac=take-whole
 return float((a[-whole:].sum() if whole else 0.)+(frac*a[-whole-1] if frac else 0.))/take

def paired_interval(values,alpha=.025):
 a=np.asarray(values,float);n=len(a)
 if n==0:return {'n_worlds':0,'mean':None,'sd':None,'ci_low':None,'ci_high':None,'p_two_sided':None,'mde80_approx':None}
 mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else None
 if n<2:return {'n_worlds':n,'mean':mean,'sd':sd,'ci_low':None,'ci_high':None,'p_two_sided':None,'mde80_approx':None}
 se=sd/math.sqrt(n);q=float(student_t.ppf(1-alpha/2,n-1));p=float(2*student_t.sf(abs(mean/se),n-1)) if se>0 else (0. if mean else 1.)
 return {'n_worlds':n,'mean':mean,'sd':sd,'ci_low':mean-q*se,'ci_high':mean+q*se,'p_two_sided':p,'mde80_approx':float((norm.ppf(1-alpha/2)+norm.ppf(.8))*se)}

def paired_tail_upper(values,alpha=.025):
 """Upper bound for signed seed-averaged excess across independent worlds.

 Fixed family strata need not have identical event probabilities. For zero
 positive events use product/AM-GM inversion; otherwise Hoeffding bounds their
 mean. In both cases E[X] <= mean P(X>0), X in [-1,1].
 """
 a=np.asarray(values,float);n=len(a)
 if not 0<alpha<1:raise ContractError('TAIL_ALPHA')
 if not n:return {'n_worlds':0,'mean':None,'ci_low':None,'ci_high':None}
 if a.ndim!=1 or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ContractError('TAIL_EXCESS_DOMAIN')
 k=int(np.count_nonzero(a>0))
 if k==0:
  upper=-math.expm1(math.log(alpha)/n);method='zero-event independent product / AM-GM, valid for unequal family probabilities'
 else:
  upper=min(1.,k/n+math.sqrt(math.log(1/alpha)/(2*n)));method='independent bounded-indicator Hoeffding, valid for unequal family probabilities'
 return {'n_worlds':n,'mean':float(a.mean()),'positive_excess_worlds':k,'ci_low':None,'ci_high':upper,'confidence':1-alpha,'bound_method':method,'sample_zero_is_not_population_zero':True}

def analyze(config,out):
 out=Path(out);plan=get_plan(config,out);em=read_json(out/'EVALUATION_COMMIT.json');unseal(em);am=read_json(out/'AUDIT_MANIFEST.json')
 if em['plan_digest']!=plan['digest']:raise ContractError('ANALYSIS_PLAN_BINDING')
 if am.get('status')!='COMPLETE' or am['committed_manifest']!=em['digest'] or len(am['units'])!=am['expected_units'] or not am['units']:raise ContractError('INCOMPLETE_ANALYSIS_INPUT')
 expected=2*len(config['families'])*config['size']['final_per_family']
 if am['expected_units']!=expected:raise ContractError('MISSING_FINAL_WORLD_UNITS')
 dest=out/'analysis';dest.mkdir(exist_ok=True);world_rows=[];context_rows=[];shared_rows=[];numerical_audit_failures=0
 for unitrow in am['units']:
  path=out/unitrow['path']
  if not path.exists() or filehash(path)!=unitrow['sha256']:raise ContractError('AUDIT_SHARD_HASH')
  x=read_json(path);unit=x['unit'];numerical_audit_failures+=x['status']!='COMPLETE'
  for wr in x['worlds']:
   path=out/wr['path'];v=read_json(path)
   world_rows.append({'track':unit['track'],'family':unit['family'],'world_index':unit['index'],'world_seed':v['spec']['seed'],'method':v['method'],'model_seed':v['model_seed'],'setting':v['setting']['tag'],'K':v['setting']['K'],'B':v['setting']['B'],'length_m':v['length_m'],'raw_terminal_squared_error_m2':None if v['raw_terminal_normalized_squared_error'] is None else v['raw_terminal_normalized_squared_error']*v['length_m']**2,'normalized_loss':v['primary_normalized_loss'],'normalized_trajectory_loss':v['trajectory_normalized_loss'],'raw_terminal_squared_error_over_L2':v['raw_terminal_normalized_squared_error'],'large_error':v['large_error'],'execution_status':v['execution_status'],'world_seconds':v['seconds'],'policy_seconds':sum(r['commit']['policy_seconds'] for r in v['records']),'paid_scores':sum(r['commit']['meter']['scores'] for r in v['records']),'atomic_scores':sum(r['commit']['meter']['atomic_scores'] for r in v['records']),'construction_attempts':sum(r['commit']['meter']['construction'] for r in v['records']),'priority_candidates':sum(r['commit']['meter']['priority_candidates'] for r in v['records']),'native_operations':json.dumps(v['native_operations']) if v['native_operations'] else ''})
  for cr in x['contexts']:
   a=cr['audit'];rowby={r['commit_digest']:r for r in a['reports']}
   for member,r in zip(cr['members'],a['reports']):
    context_rows.append({'track':unit['track'],'family':unit['family'],'world_index':unit['index'],'context_digest':cr['context_digest'],'method':r['method'],'tag':member['tag'],'setting':member['setting'],'cycle':member['cycle'],'model_seed':int(member['tag'].rsplit('_',1)[-1]) if member['tag'].rsplit('_',1)[-1].isdigit() else config['control_seed'],'length_m':cr.get('length_m',v['length_m']),'Q_state':r['Q_state'],'coverage':r['coverage'],'unconditional_miss':r['unconditional_miss'],'regret_m2':r['finite_reference_regret_m2'],'risk_m2':r['risk_m2'],'posterior_tail_probability':r['tail_probability'],'numerical_ambiguity':r['numerical_ambiguity'],'abstained':r['abstained'],'execution_status':r['execution_status'],'reference_complete':a['reference_complete'],'reference_score_attempts':a['meter']['scores']})
  for r in x['shared_evaluation']:shared_rows.append({**unit,'context_digest':r['context_digest'],'shared':r['shared'],'score_attempts':r['meter']['scores'],'atomic_scores':r['meter']['atomic_scores'],'cache_hits':r['meter']['cache_hits'],'score_seconds':r['meter']['score_seconds'],'maximum_difference_m2':r['maximum_absolute_difference_from_audit'],'agreement':r.get('agreement',True)})
 if not world_rows or not context_rows:raise ContractError('NO_SCIENTIFIC_RECORDS')
 csvfile(dest/'world_results.csv',world_rows);csvfile(dest/'context_results.csv',context_rows)
 if shared_rows:csvfile(dest/'shared_evaluation.csv',shared_rows)
 grouped=defaultdict(list)
 for r in world_rows:grouped[(r['track'],r['family'],r['method'],r['model_seed'],r['setting'])].append(r)
 context_grouped=defaultdict(list)
 for r in context_rows:context_grouped[(r['track'],r['family'],r['method'],r['model_seed'],r['setting'])].append(r)
 metrics=[]
 for key,rs in grouped.items():
  vals=[r['normalized_loss'] for r in rs if r['normalized_loss'] is not None];cr=context_grouped[key]
  non=[r for r in cr if r['Q_state']=='KNOWN_NONEMPTY_NUMERICAL'];empty=[r for r in cr if r['Q_state']=='KNOWN_EMPTY_NUMERICAL'];unknown=[r for r in cr if r['Q_state']=='UNKNOWN'];knowncovered=[r for r in non if r['coverage'] is not None]
  metrics.append({'track':key[0],'family':key[1],'method':key[2],'model_seed':key[3],'setting':key[4],'world_count':len(rs),'risk_unknown_worlds':len(rs)-len(vals),'mean_normalized_loss':None if not vals else float(np.mean(vals)),'world_CVaR95':cvar(vals),'large_error_probability':None if key[0]=='finite' else float(np.mean([r['large_error'] for r in rs])),'raw_terminal_MSE_over_L2_mean':None if key[0]!='dynamic' or not any(r['raw_terminal_squared_error_over_L2'] is not None for r in rs) else float(np.mean([r['raw_terminal_squared_error_over_L2'] for r in rs if r['raw_terminal_squared_error_over_L2'] is not None])),'raw_terminal_MSE_m2_mean':None if not any(r['raw_terminal_squared_error_m2'] is not None for r in rs) else float(np.mean([r['raw_terminal_squared_error_m2'] for r in rs if r['raw_terminal_squared_error_m2'] is not None])),'raw_terminal_CVaR95_m2':cvar([r['raw_terminal_squared_error_m2'] for r in rs if r['raw_terminal_squared_error_m2'] is not None]),'raw_terminal_CVaR95':None if key[0]!='dynamic' else cvar([r['raw_terminal_squared_error_over_L2'] for r in rs if r['raw_terminal_squared_error_over_L2'] is not None]),'raw_error_unknown_worlds':sum(r['raw_terminal_squared_error_over_L2'] is None for r in rs),'mean_posterior_tail_probability':None if not any(r['posterior_tail_probability'] is not None for r in cr) else float(np.mean([r['posterior_tail_probability'] for r in cr if r['posterior_tail_probability'] is not None])),'mean_normalized_finite_reference_regret':None if not any(r['regret_m2'] is not None for r in cr) else float(np.mean([r['regret_m2']/r['length_m']**2 for r in cr if r['regret_m2'] is not None])),'failed_worlds':sum(r['execution_status']!='COMPLETE' for r in rs),'known_nonempty_contexts':len(non),'known_empty_contexts':len(empty),'unknown_Q_contexts':len(unknown),'coverage_known_nonempty':None if not knowncovered else float(np.mean([r['coverage'] for r in knowncovered])),'unknown_coverage_on_nonempty':len(non)-len(knowncovered),'unconditional_missed_contexts':sum(r['unconditional_miss'] is True for r in cr),'unconditional_unknown_contexts':sum(r['unconditional_miss'] is None for r in cr),'abstained_contexts':sum(r['abstained'] for r in cr),'mean_policy_seconds_per_world':float(np.mean([r['policy_seconds'] for r in rs])),'total_paid_scores':sum(r['paid_scores'] for r in rs),'total_atomic_scores':sum(r['atomic_scores'] for r in rs)})
 csvfile(dest/'scientific_metrics.csv',metrics)
 # Five training seeds are paired, not 5 independent copies of every world.
 lookup={(r['track'],r['family'],r['world_index'],r['model_seed'],r['method']):r for r in world_rows if r['setting']=='main'}
 comparisons=[];primary=[];tailprimary=[]
 for track in ('finite','dynamic'):
  pooled=[];pooledtail=[]
  for family in config['families']:
   diffs=[];tails=[];model_values=defaultdict(list)
   for idx in range(config['size']['final_per_family']):
    seedvalues=[];seedtails=[]
    for seed in config['model_seeds']:
     old=lookup[(track,family,idx,seed,'original_agent')];new=lookup[(track,family,idx,seed,'integrated_agent')]
     if old['normalized_loss'] is not None and new['normalized_loss'] is not None:
      val=old['normalized_loss']-new['normalized_loss'];seedvalues.append(val);model_values[seed].append(val)
     if track=='dynamic':seedtails.append(float(new['large_error'])-float(old['large_error']))
    # Require all paired seeds; never silently replace a missing policy output.
    if len(seedvalues)==len(config['model_seeds']):diffs.append(float(np.mean(seedvalues)))
    if seedtails:tails.append(float(np.mean(seedtails)))
   q=paired_interval(diffs);comparisons.append({'track':track,'family':family,'comparison':'Original minus integrated','endpoint':'normalized task loss',**q,'model_seed_mean_effects':json.dumps({str(k):float(np.mean(v)) for k,v in model_values.items()})});pooled+=diffs;pooledtail+=tails
  q=paired_interval(pooled);comparisons.append({'track':track,'family':'pooled_equal_family','comparison':'Original minus integrated','endpoint':'normalized task loss',**q})
  if track=='dynamic':primary=q;tailprimary=paired_tail_upper(pooledtail)
 # Holm-adjust only prespecified six family mean contrasts (exploratory).
 relevant=[r for r in comparisons if r['track']=='dynamic' and r['family'] in config['families'] and r['p_two_sided'] is not None]
 running=0.
 for i,r in enumerate(sorted(relevant,key=lambda r:r['p_two_sided'])):
  running=max(running,(len(relevant)-i)*r['p_two_sided']);r['holm_adjusted_p']=min(1.,running)
 csvfile(dest/'paired_comparisons.csv',comparisons)
 meaningful=primary.get('ci_low') is not None and primary['ci_low']>config['statistics']['effect_margin_normalized'];tailok=tailprimary.get('ci_high') is not None and tailprimary['ci_high']<config['statistics']['tail_noninferiority_absolute']
 summary={'schema':'ain.v11.analysis.v1','status':'COMPLETE_WITH_NUMERICAL_FAILURES' if numerical_audit_failures or any(r['execution_status']!='COMPLETE' for r in world_rows) else 'COMPLETE','plan_digest':plan['digest'],'committed_evaluation':em['digest'],'independent_world_units_per_track':expected//2,'not_independent_units':'cycles, model-seed replicates, candidate rows, and curve settings','primary_dynamic_paired_improvement':primary,'paired_large_error_probability_increase':tailprimary,'practical_margin_met':meaningful,'tail_noninferiority_margin_met':tailok,'claim_decision':'positive primary evidence in declared campaign' if meaningful and tailok else 'mixed or negative results; retain and analyze, no new automatic training round','audit_numerical_partial_units':numerical_audit_failures,'shared_evaluation_mismatches':sum(not r['agreement'] for r in shared_rows),'world_records':len(world_rows),'context_records':len(context_rows),'formal_certificate':False,'confirmation_claim':False,'physical_validation':False,'all_results_retained':True}
 atomic_json(dest/'SUMMARY.json',summary)
 text=['# Campaign results','',f"Status: {summary['status']}",'',f"Primary dynamic improvement (old minus integrated): {primary.get('mean')}. Bonferroni-adjusted interval: [{primary.get('ci_low')}, {primary.get('ci_high')}].",f"Large-error probability increase: {tailprimary.get('mean')}.",'','Finite-law results support mechanisms; they do not replace the dynamic endpoint. World-level CVaR and within-world posterior tail probabilities are separate tables. Unknown qualification is not empty qualification.','',f"Decision: {summary['claim_decision']}."]
 (dest/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf8');plots(config,dest,world_rows,metrics,shared_rows)
 return summary

def plots(config,dest,world_rows,metrics,shared_rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for track in ('finite','dynamic'):
  fig,ax=plt.subplots(figsize=(8,4.5))
  for method in ('original_agent','integrated_agent','deep_sets','set_transformer','analytic_constructive','beam_block'):
   yy=[]
   for fam in config['families']:
    z=[r['mean_normalized_loss'] for r in metrics if r['track']==track and r['family']==fam and r['method']==method and r['setting']=='main' and r['mean_normalized_loss'] is not None];yy.append(float(np.mean(z)) if z else np.nan)
   ax.plot(config['families'],yy,marker='o',label=LABELS[method])
  ax.set_ylabel('Normalized task loss');ax.set_xlabel('World family');ax.set_title('Dynamic trajectory endpoint' if track=='dynamic' else 'Complete-law mechanism risk');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(dest/f'{track}_family_risk.png',dpi=160);plt.close(fig)
  if config['diagnostics']['curve_worlds_per_family']:
   fig,ax=plt.subplots(figsize=(7,4.5))
   for method in ('original_agent','integrated_agent','analytic_constructive','beam_block'):
    xx=[];yy=[]
    for B,tag in [(4,'B4'),(8,'B8'),(16,'main'),(32,'B32')]:
     vals=[r for r in world_rows if r['track']==track and r['method']==method and r['setting']==tag and r['world_index']<config['diagnostics']['curve_worlds_per_family'] and r['normalized_loss'] is not None]
     if vals:xx.append(np.mean([r['paid_scores'] for r in vals]));yy.append(np.mean([r['normalized_loss'] for r in vals]))
    ax.plot(xx,yy,marker='o',label=LABELS[method])
   ax.set_xlabel('Paid task-score calls per world');ax.set_ylabel('Normalized task loss');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dest/f'{track}_risk_cost_curve.png',dpi=160);plt.close(fig)
 # Same training seeds for the two targeted mechanism ablations.
 fig,ax=plt.subplots(figsize=(7,4.5));methods=['integrated_agent',*config['ablations'].get('retrain_arms',['no_assoc_supervision']),'integrated_no_protection','original_with_protection'];yy=[]
 for method in methods:
  v=[r['normalized_loss'] for r in world_rows if r['track']=='dynamic' and r['setting']=='main' and r['method']==method and r['model_seed'] in config['model_seeds'][:config['ablations']['retrain_seeds']] and r['normalized_loss'] is not None];yy.append(np.mean(v) if v else np.nan)
 ax.bar([LABELS[m] for m in methods],yy);ax.set_ylabel('Normalized dynamic terminal loss');ax.tick_params(axis='x',rotation=12);fig.tight_layout();fig.savefig(dest/'mechanism_ablations.png',dpi=160);plt.close(fig)
 if shared_rows:
  fig,ax=plt.subplots(figsize=(5,4));ax.bar(['Independent evaluation','Shared evaluation'],[sum(r['atomic_scores'] for r in shared_rows if r['shared']==flag) for flag in (False,True)]);ax.set_ylabel('Atomic task-score computations');fig.tight_layout();fig.savefig(dest/'shared_evaluation_ablation.png',dpi=160);plt.close(fig)

def export_results(out,destination):
 out=Path(out);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 state=read_json(out/'CAMPAIGN_STATE.json') if (out/'CAMPAIGN_STATE.json').exists() else {'status':'PARTIAL'}
 paths=[]
 for p in sorted(out.rglob('*')):
  if not p.is_file() or '.tmp' in p.name:continue
  rel=p.relative_to(out)
  # Reproducibility bundle keeps raw final/audit rows, selected checkpoints and
  # manifests, not the large training-label cache or redundant optimizer files.
  if rel.parts[0]=='data' or p.name=='last.pt' or p.suffix=='.pt' and 'SELECTED' not in p.name:continue
  paths.append(p)
 selected=[]
 failures=[]
 for f in (out/'final').rglob('*.json') if (out/'final').exists() else []:
  if f.name.endswith('.receipt.json') or f.name in ('INDEX.json','EVAL_RECEIPT.json'):continue
  try:
   obj=read_json(f)
   if isinstance(obj,dict) and (obj.get('failure') or any(x.get('commit',{}).get('candidate_failures') for x in obj.get('records',[]))):failures.append({'path':f.relative_to(out).as_posix(),'sha256':filehash(f),'failure':obj.get('failure'),'status':obj.get('execution_status')})
  except (ValueError,KeyError):continue
 atomic_json(out/'FAILURE_REPLAY_INDEX.json',{'count':len(failures),'rows':failures,'requires_full_context_and_traceback':True});paths.append(out/'FAILURE_REPLAY_INDEX.json')
 if (out/'FROZEN_PLAN.json').exists():
  for r in read_json(out/'FROZEN_PLAN.json')['checkpoints']:
   p=out/r['path'];selected.extend([p,p.with_suffix('.pt.sha256')])
 paths=list(dict.fromkeys(paths+selected))
 with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in paths:
   if p.exists():z.write(p,p.relative_to(out).as_posix())
  z.writestr('EXPORT_STATUS.json',canonical({'status':state.get('status','PARTIAL'),'source_digest':source_digest(),'files':{p.relative_to(out).as_posix():filehash(p) for p in paths if p.exists()},'partial_results_are_not_complete':True}))
 return {'path':str(destination),'bytes':destination.stat().st_size,'sha256':filehash(destination),'campaign_status':state.get('status','PARTIAL')}
