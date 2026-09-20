from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
from ain_campaign.analysis import paired_tail_upper
from ain_campaign.common import atomic_json,source_digest
z=paired_tail_upper([0.]*768)
assert abs(z['ci_high']-(1-.025**(1/768)))<1e-15
p=paired_tail_upper([1.]*24+[0.]*744)
expected=24/768+math.sqrt(math.log(40)/(2*768))
assert abs(p['ci_high']-expected)<1e-15 and 0<p['ci_high']<1
for bad in [[float('nan')],[1.1],[-1.1]]:
 try:paired_tail_upper(bad)
 except ValueError:pass
 else:raise AssertionError('invalid tail excess admitted')
assert paired_tail_upper([])['ci_high'] is None
r={'status':'COMPLETE','checks':3,'source_digest':source_digest(),'zero_events':z,'unequal_strata_nonzero_case':p,'domain_rejections':3,'optimizer_updates':0,'paid_score_attempts':0}
atomic_json(ROOT/'evidence/STRATIFIED_TAIL_CHECK.json',r);print(r)
