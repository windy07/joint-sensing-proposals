import copy
import unittest
from common import digest
from frozen_contract import validate_frozen, validate_checkpoint


def fixture():
    jobs = [{'split':'confirmation', 'world_seed':123}]
    contract = {'mode':'confirmatory', 'launch_gate':{'status':'GO'},
                'embargo':{'confirmation_unopened':True},
                'admissibility':{f'A{i}':{'status':'PASS'} for i in range(6)},
                'sample_size_plan':{'world_seeds':[123]},
                'bindings':{'jobs_digest':digest(jobs),'adapter_digest':'a','science_digest':'s',
                            'checkpoints':{'integrated_agent/7':'weight'}}}
    contract['freeze_digest']='sha256:'+digest(contract)
    design={'contract_digest':contract['freeze_digest'],'confirmation_status':'FROZEN',
            'jobs':jobs,'adapter_digest':'a','base_source_digest':'s'}
    return design,contract


class ConfirmationTests(unittest.TestCase):
    def test_frozen_valid(self):
        d,c=fixture();self.assertEqual(validate_frozen(d,c),c)

    def test_unfrozen_rejected(self):
        d,c=fixture();d['confirmation_status']='PLANNED'
        with self.assertRaisesRegex(ValueError,'NOT_FROZEN'):validate_frozen(d,c)

    def test_contract_edit_rejected(self):
        d,c=fixture();c['sample_size_plan']['world_seeds'].append(124)
        with self.assertRaisesRegex(ValueError,'CONTRACT_CHANGED'):validate_frozen(d,c)

    def test_new_world_rejected(self):
        d,c=fixture();d['jobs'][0]['world_seed']=124
        with self.assertRaisesRegex(ValueError,'BINDING_CHANGED'):validate_frozen(d,c)

    def test_replaced_checkpoint_rejected(self):
        d,c=fixture();validate_checkpoint(c,'integrated_agent',7,'weight')
        with self.assertRaisesRegex(ValueError,'CHECKPOINT_CHANGED'):
            validate_checkpoint(c,'integrated_agent',7,'new_weight')

    def test_adapter_change_rejected(self):
        d,c=fixture();d['adapter_digest']='new'
        with self.assertRaisesRegex(ValueError,'BINDING_CHANGED'):validate_frozen(d,c)


if __name__=='__main__':unittest.main(verbosity=2)
