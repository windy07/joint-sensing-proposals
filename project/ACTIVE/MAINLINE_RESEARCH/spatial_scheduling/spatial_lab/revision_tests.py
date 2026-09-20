"""Small checks of repaired failure modes, separate from outcome experiments."""
import unittest
from copy import deepcopy
from fractions import Fraction
from unittest.mock import patch
from .tests import CONFIG
from .core import *
from .physics import *
from .filtering import MultiTracker
from .world import World
from .scheduling import *

class Revisions(unittest.TestCase):
    def setup(self):
        cfg={**CONFIG,'scenario':'late_multi'};tracker=MultiTracker(cfg);tracker.predict(0)
        view=bind({**tracker.view(Fraction('9.6'),Fraction('.096'),[.4]*4),'search_age_cycles':[6]*4})
        return cfg,tracker,view

    def test_rejected_and_copied_captures_have_no_search_credit(self):
        c,t,v=self.setup();w=World(44,c);w.step();pack=w.observe((Task(0,'search',0),),v)[0]
        late=deepcopy(pack);late['arrival_s']=2.
        r=t.receive(late)
        self.assertEqual(admitted_schedule([late],[r]),())
        accepted=t.receive(pack);copy=deepcopy(pack);copy['transport_copy']='relay';dup=t.receive(copy)
        self.assertEqual(admitted_schedule([pack,copy],[accepted,dup]),(Task(0,'search',0),))

    def test_empty_or_failed_archive_returns_no_unverified_fallback(self):
        c,t,v=self.setup();a,r=run_policy(v,{**c,'B':0},'feedback')
        self.assertEqual(r['output'],[]);self.assertEqual(a,())
        with patch('spatial_lab.scheduling.forecast_once',side_effect=ContractError('injected')):
            a,r=run_policy(v,c,'feedback')
        self.assertEqual(r['output'],[]);self.assertEqual(r['meter']['score_calls'],c['B'])

    def test_no_calibration_in_any_forecast_depth(self):
        c,t,v=self.setup();called=[];original=atoms
        def observed(view,config,allow_calibration=True):
            called.append(allow_calibration);return original(view,config,allow_calibration)
        with patch('spatial_lab.scheduling.atoms',side_effect=observed):run_policy(v,c,'no_calibration')
        self.assertTrue(called);self.assertFalse(any(called))

    def test_complements_survive_scalar_pool_fill(self):
        c,t,v=self.setup();g=Generator(v,{**c,'candidate_pool_cap':6},Meter(8,256));g.initial()
        for cell in range(4):self.assertIn((Task(2,'search',cell),Task(3,'search',cell)),g.pool)

    def test_coverage_guard_is_paid_and_executable(self):
        c,t,v=self.setup();a,r=run_policy(v,c,'coverage_feedback')
        self.assertTrue(r['coverage_guard_satisfied']);self.assertTrue(any(x.kind=='search' for x in a))
        self.assertLessEqual(r['meter']['score_calls'],c['B']);self.assertLessEqual(len(r['output']),c['K'])
        self.assertEqual(r['selection_scope'],'VERIFIED_REVISIT_FEASIBLE_SET')
        self.assertTrue(all(any(row['schedule']==s for row in r['archive']) for s in r['output']))

    def test_resource_envelope_keeps_future_budget(self):
        c,t,v=self.setup();cc={**c,'pace_resources':True}
        spent=Fraction('0.8');late=bind({**v,'energy_remaining_exact_J':str(Fraction('9.6')-spent),'tx_remaining_exact_s':str(Fraction('.096')-Fraction('.008'))})
        with self.assertRaises(ContractError):legal((Task(0,'search',0),Task(1,'search',1)),late,cc)
        legal((),late,cc)
        future=bind({**late,'epoch':20,'decision_s':20.})
        legal((Task(0,'search',0),Task(1,'search',1)),future,cc)

    def test_unknown_method_rejected(self):
        c,t,v=self.setup()
        with self.assertRaises(ContractError):run_policy(v,c,'misspelled_feedback')

if __name__=='__main__':unittest.main(verbosity=2)
