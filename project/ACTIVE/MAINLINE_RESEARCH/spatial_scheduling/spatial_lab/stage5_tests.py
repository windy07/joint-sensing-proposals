"""Finite regressions for completion fixes, without a world or training run."""
import unittest,tempfile
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import torch
from .core import ContractError,Meter,Task,digest,write_json
from .stage4_tests import roomy
from .scheduling import atoms,ProxyReference
from .feedback import create_feedback,targeted_proposals

class CompletionContracts(unittest.TestCase):
    def test_mixed_feedback_never_exceeds_six_groups(self):
        c,_,v=roomy();c={**c,'forecast_horizon':1};domain=atoms(v,c)
        group=tuple(a for a in domain if a.node==0 and a.kind=='track')[:1]
        paid=ProxyReference(v,c,Meter(1),1).evaluate(group)
        f=create_feedback(v,c,[],[paid],[],Meter(6,96));r=f['records'][0]
        f['records']=[{**r,'reason':'RISK'},r,r,r]
        f['digest']=digest({k:v for k,v in f.items() if k!='digest'})
        self.assertEqual(len(list(targeted_proposals(f,group,v,c,domain))),6)

    def test_new_diagnostic_authority_keeps_default_closed_and_caps_attempts(self):
        from .diagnostic_work import diagnostic_authority,charge
        from .work_budget import require_optimizer_authority
        self.assertRaises(ContractError,require_optimizer_authority)
        c,_,_=roomy();c={**c,'cycles':1,'forecast_horizon':1,'root_score_budget':1}
        contract={'mode':'development-diagnostic','resource_budget':{'world_advance_attempts':0,'optimizer_attempts':0,'device':'cpu','threads':1},
            'allowed_cases':[{'seed':77,'configuration':c}],'backbones':['integrated_agent'],'model_seeds':[99]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/'CONTRACT.json',contract)
            with diagnostic_authority(p/'CONTRACT.json',p,'zero_limit',execute=True):
                self.assertRaisesRegex(ContractError,'CUMULATIVE_LIMIT',charge,'world',seed=77,epoch=0,configuration_digest=digest(c))
                self.assertRaisesRegex(ContractError,'CUMULATIVE_LIMIT',charge,'optimizer',arm='integrated_agent',seed=99,update=1)
            self.assertRaises(ContractError,require_optimizer_authority)

    def test_freeze_rejects_partial_training_even_with_trained_flag(self):
        from .server_pipeline import validate_training_completion,task_list
        p={'backbones':['integrated_agent'],'model_seeds':[19],'optimizer_updates':8,
           'world_splits':{'training':[1],'development':[2],'confirmation':[3]},
           'model':{'cycles':1},'scenarios':[{'name':'fixture','overrides':{}}],'teacher_policies':['numeric_optimizer']}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'data/STAGE_RECEIPT.json',{'status':'COMPLETE'})
            write_json(root/'TRAINING_SUMMARY.json',{'status':'COMPLETE','runs':[{}]})
            jobs=[x for x in task_list(p,root) if x['stage']=='development']
            write_json(root/'development/STAGE_RECEIPT.json',{'status':'COMPLETE','jobs':jobs,'results':[{'status':'COMPLETE'} for _ in jobs]})
            ck={'arm':'integrated_agent','model_seed':19,'trained':True,'optimizer_updates':4,
                'training_purpose':'registered_server_training','training_law_seeds':[1],'development_law_seeds':[2],
                'state':{'finite':torch.ones(1)}}
            # Isolate the newly added stage-completion contract; actual tensor
            # identity is independently checked by verify_checkpoint and G06.
            with patch('spatial_lab.binding.verify_checkpoint'),patch('torch.load',return_value=ck):
                self.assertRaisesRegex(ContractError,'NOT_REGISTERED_COMPLETION',validate_training_completion,p,root)
                ck['optimizer_updates']=8
                self.assertTrue(validate_training_completion(p,root))
                ck['training_purpose']='development_diagnostic'
                self.assertRaisesRegex(ContractError,'NOT_REGISTERED_COMPLETION',validate_training_completion,p,root)

def suite():return unittest.defaultTestLoader.loadTestsFromTestCase(CompletionContracts)
