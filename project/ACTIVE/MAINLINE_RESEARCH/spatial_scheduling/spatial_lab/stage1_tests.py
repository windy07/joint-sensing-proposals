"""Decisive regression contracts for revision stage 1, not quality claims."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .core import ContractError,Task,Meter,digest,write_json,read_json
from .tests import CONFIG
from .filtering import MultiTracker
from .scheduling import atoms,bind,run_policy,ProxyReference
from .spatial_graph import graph,graph_digest,recover_joint_covariance,make,scores,Priority
from .sequence_memory import SequenceMemory
from .sequence_training import train,load_sequences,recurrent_parameters,objective
from .binding import ROOT,FEATURE_BINDING,LABEL_BINDING,source_manifest,verify_checkpoint,import_origins
from .run import run_case

torch.set_num_threads(1)

def context(ntracks=0,epoch=0):
    cfg={**CONFIG,'scenario':'single','cycles':4,'B':4,'K':2,'lookahead_cycles':2,
         'tail_width':1,'candidate_pool_cap':32,'pace_resources':False}
    t=MultiTracker(cfg);t.predict(0)
    for i in range(ntracks):
        t._add(np.array([100.+350*i,300.-70*i]),np.zeros((2,6)),np.eye(2)*90000,origin='unit')
    for e in range(1,epoch+1):t.predict(e)
    v=bind({**t.view(Fraction('9.6'),Fraction('.096'),[.4]*4),'search_age_cycles':[1]*4})
    return cfg,t,v


def advance_view(v,epoch):
    return bind({**v,'epoch':epoch,'decision_s':float(epoch)})


class CandidateCapacity(unittest.TestCase):
    def test_G01_capacity_budget_grid_selects_best_success(self):
        c,t,v=context()
        for K in (1,2,4):
            for B in (0,1,2,8):
                with self.subTest(K=K,B=B):
                    a,r=run_policy(v,{**c,'K':K,'B':B},'myopic')
                    self.assertLessEqual(len(r['output']),K)
                    self.assertLessEqual(r['meter']['score_calls'],B)
                    if B:
                        best=min(r['archive'],key=lambda x:(x['proxy_m2'],tuple(Task(**q) for q in x['schedule'])))
                        self.assertEqual(r['selected'],best['schedule'])
                        self.assertTrue(a)
                        self.assertFalse(r['capacity_audit']['emergency_idle_used'])
                    else:
                        self.assertEqual(r['archive'],[]);self.assertEqual(r['output'],[])
                        self.assertTrue(r['capacity_audit']['emergency_idle_used'])
    def test_G01_return_capacity_does_not_change_execution(self):
        c,t,v=context();results=[run_policy(v,{**c,'K':k,'B':8},'feedback') for k in (1,2,4)]
        self.assertEqual(results[0][0],results[1][0]);self.assertEqual(results[1][0],results[2][0])
        self.assertTrue(all(r[1]['capacity_audit']['mandatory_protected_return_slots']==0 for r in results))
    def test_G01_all_failures_are_charged_without_false_verified_idle(self):
        c,t,v=context()
        for K in (1,2,4):
            for B in (0,1,2,8):
                with self.subTest(K=K,B=B),patch('spatial_lab.scheduling.forecast_once',side_effect=ValueError('forced')):
                    a,r=run_policy(v,{**c,'K':K,'B':B},'myopic')
                    self.assertEqual(r['output'],[]);self.assertEqual(a,())
                    self.assertEqual(r['meter']['score_calls'],B)
                    self.assertEqual(len(r['failures']),B)
    def test_G01_partial_failure_retains_later_success(self):
        from .scheduling import forecast_once
        c,t,v=context();counter=[0]
        def faulty(*args,**kwargs):
            counter[0]+=1
            if counter[0]==1:raise ValueError('first candidate fails')
            return forecast_once(*args,**kwargs)
        with patch('spatial_lab.scheduling.forecast_once',side_effect=faulty):
            a,r=run_policy(v,{**c,'B':8,'K':1},'myopic')
        self.assertEqual(r['status'],'PARTIAL_VERIFIED');self.assertTrue(a)
        self.assertEqual(r['meter']['score_calls'],8);self.assertEqual(r['meter']['score_completed'],7)
    def test_G01_revisit_restriction_works_at_K1_B1(self):
        c,t,v=context();v=bind({**v,'search_age_cycles':[6]*4})
        a,r=run_policy(v,{**c,'K':1,'B':1},'coverage_feedback')
        self.assertTrue(r['coverage_guard_satisfied']);self.assertEqual(len(r['output']),1)
        self.assertEqual(r['selection_scope'],'VERIFIED_REVISIT_FEASIBLE_SET')
    def test_G01_network_order_can_change_selected_action_at_K2(self):
        c,t,v=context();c={**c,'B':4,'K':2}
        target=(Task(0,'calibrate'),)
        class ToyPriority:
            def prepare(self,*args):pass
            def priority(self,a):return -100. if a==target else 0.
        original=ProxyReference.evaluate
        def evaluate(ref,a):
            row=original(ref,a)
            if a==target:row['proxy_m2']=0. # controlled selector witness, not a physical/learning result
            return row
        with patch.object(ProxyReference,'evaluate',evaluate):
            a,r=run_policy(v,c,'learned',learned=ToyPriority())
        self.assertEqual(a,target)
        self.assertEqual(r['capacity_audit']['competitive_return_slots'],2)
        self.assertEqual(r['capacity_audit']['network_exclusive_constructed_candidates'],0)
    def test_missing_learned_model_rejected(self):
        c,t,v=context()
        with self.assertRaisesRegex(ContractError,'REQUIRES_MODEL'):run_policy(v,c,'learned')


class JointRepresentation(unittest.TestCase):
    def test_G02_bias_target_sign_changes_features_and_forecast(self):
        c,t,v=context(1);aa=atoms(v,c);gs=[];risks=[]
        for sign in (-1,1):
            p=np.array(v['covariance']);p[0,6]=p[6,0]=1500*sign
            w=bind({**v,'covariance':p.tolist()})
            gs.append(graph(w,aa,c))
            risks.append(ProxyReference(w,c,Meter(1),1).evaluate((Task(0,'track',0),))['proxy_m2'])
        self.assertFalse(torch.equal(gs[0].node_features,gs[1].node_features))
        self.assertNotAlmostEqual(risks[0],risks[1],places=3)
    def test_G02_entire_joint_covariance_reconstructs(self):
        c,t,v=context(3);n=len(v['mean']);random=np.random.default_rng(738)
        A=random.normal(size=(n,n));P=A@A.T+np.eye(n)
        scales=np.array([25.,25.,*.005*np.ones(4),*([300.,300.,20.,20.]*3)])
        P=P*np.outer(scales,scales);v=bind({**v,'covariance':P.tolist()})
        G=graph(v,atoms(v,c),c);got=recover_joint_covariance(G,v)
        np.testing.assert_allclose(got/np.outer(scales,scales),P/np.outer(scales,scales),rtol=2e-6,atol=2e-6)
    def test_G02_cross_target_and_bias_bias_changes_are_visible(self):
        c,t,v=context(2);aa=atoms(v,c);base=graph(v,aa,c)
        for i,j,value in ((6,10,1000.),(0,1,100.),(2,3,1e-5)):
            p=np.array(v['covariance']);p[i,j]=p[j,i]=value
            changed=graph(bind({**v,'covariance':p.tolist()}),aa,c)
            self.assertFalse(torch.equal(base.node_features,changed.node_features))
    def test_G02_covariance_partition_and_positive_definiteness_checked(self):
        c,t,v=context(1);bad=deepcopy(v);bad['tracks'][0]['start']=5
        with self.assertRaisesRegex(ContractError,'PARTITION'):graph(bind(bad),atoms(v,c),c)
        p=np.array(v['covariance']);p[0,0]=-1
        with self.assertRaises(np.linalg.LinAlgError):graph(bind({**v,'covariance':p.tolist()}),atoms(v,c),c)
    def test_same_numeric_graph_reaches_all_three_backbones(self):
        c,t,v=context(1);aa=atoms(v,c);G=graph(v,aa,c);frozen=graph_digest(G)
        for arm in ('integrated_agent','deep_sets','set_transformer'):
            model,cfg=make(arm);model.eval()
            with torch.no_grad():z,out=scores(model,G,aa,[(),(aa[0],)],None)
            self.assertTrue(torch.isfinite(z).all());self.assertEqual(frozen,graph_digest(G))


class SemanticSequenceMemory(unittest.TestCase):
    def test_G03_birth_does_not_reassign_action_identity(self):
        c,t,v=context();old={a.semantic_id():a for a in atoms(v,c)}
        t._add(np.array([100.,300.]),np.zeros((2,6)),np.eye(2)*90000,origin='unit')
        w=t.view(Fraction('9.6'),Fraction('.096'),[.4]*4)
        new={a.semantic_id():a for a in atoms(w,c)}
        self.assertTrue(set(old)<=set(new))
        self.assertTrue(all(old[k]==new[k] for k in old))
    def test_G03_action_output_indices_and_permutation(self):
        c,t,v=context(1);aa=atoms(v,c);bb=list(reversed(aa))
        g,h=graph(v,aa,c),graph(v,bb,c)
        self.assertEqual([g.node_ids[i] for i in g.action_node_index],[a.semantic_id() for a in aa])
        for arm in ('integrated_agent','deep_sets','set_transformer'):
            model,cfg=make(arm);model.eval();pool=[(aa[0],),(aa[-1],)]
            with torch.no_grad():z,_=scores(model,g,aa,pool);w,_=scores(model,h,bb,pool)
            torch.testing.assert_close(z,w,atol=2e-6,rtol=2e-6)
    def test_G03_temporary_absence_retains_semantic_memory(self):
        c,t,v=context(1);aa=atoms(v,c);g=graph(v,aa,c);store=SequenceMemory(4);store.begin('world')
        self.assertEqual(int(store.before(v,g,'cpu').count_nonzero()),0)
        values=torch.arange(len(g.node_ids)*4,dtype=torch.float32).reshape(-1,4)+1
        store.after(v,g,values)
        w=advance_view(v,1);subset=[a for a in aa if a.node!=0];h=graph(w,subset,c)
        store.before(w,h,'cpu');store.after(w,h,torch.ones((len(h.node_ids),4))*3)
        x=advance_view(v,2);gx=graph(x,aa,c);mem=store.before(x,gx,'cpu')
        ident=next(a for a in aa if a.node==0).semantic_id()
        torch.testing.assert_close(mem[gx.node_ids.index(ident)],values[g.node_ids.index(ident)])
    def test_G03_track_retirement_purges_entries_and_reuse_is_rejected(self):
        c,t,v=context(1);g=graph(v,atoms(v,c),c);store=SequenceMemory(4);store.begin('world')
        store.before(v,g,'cpu');store.after(v,g,torch.ones((len(g.node_ids),4)))
        w=bind({**advance_view(v,1),'tracks':[],'mean':v['mean'][:6],
                'covariance':np.array(v['covariance'])[:6,:6].tolist()})
        h=graph(w,atoms(w,c),c);store.before(w,h,'cpu');store.after(w,h,torch.ones((len(h.node_ids),4)))
        self.assertTrue(store.audit[-1]['removed_nodes'])
        self.assertFalse(any('track/0' in k[1] or k[1].endswith('/task=track/object=0') for k in store.store._entries))
        x=advance_view(v,2)
        with self.assertRaisesRegex(ContractError,'REUSED_RETIRED'):store.before(x,graph(x,atoms(x,c),c),'cpu')
    def test_G03_world_reset_and_time_order(self):
        c,t,v=context();g=graph(v,atoms(v,c),c);s=SequenceMemory(4)
        with self.assertRaisesRegex(ContractError,'WORLD_NOT_STARTED'):s.before(v,g,'cpu')
        s.begin('a');s.before(v,g,'cpu');s.after(v,g,torch.ones((len(g.node_ids),4)))
        with self.assertRaisesRegex(ContractError,'NONCONSECUTIVE'):s.before(v,g,'cpu')
        s.begin('b');self.assertEqual(int(s.before(v,g,'cpu').count_nonzero()),0)
    def test_G04_history_weights_have_data_gradient_with_real_sequence(self):
        c,t,v=context(1);aa=atoms(v,c);pool=[(),(aa[0],),(aa[1],)];w=advance_view(v,1)
        for arm in ('integrated_agent','deep_sets','set_transformer'):
            with self.subTest(arm=arm):
                model,cfg=make(arm);model.train();s=SequenceMemory(cfg.memory_dim);s.begin('g')
                G=graph(v,aa,c);mem=s.before(v,G,'cpu')
                z,out=scores(model,G,aa,pool,mem);z.sum().backward()
                self.assertEqual(sum(float(p.grad.abs().sum()) for p in recurrent_parameters(model).values()),0.)
                s.after(v,G,out.next_memory)
                G=graph(w,aa,c);mem=s.before(w,G,'cpu');self.assertFalse(mem.requires_grad)
                self.assertGreater(int(mem.count_nonzero()),0)
                model.zero_grad(set_to_none=True);z,out=scores(model,G,aa,pool,mem);z.sum().backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in recurrent_parameters(model).values()),1e-8)
    def test_N01_set_transformer_obeys_dropout_configuration(self):
        model,cfg=make('set_transformer')
        probabilities=[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)]
        self.assertTrue(probabilities);self.assertEqual(set(probabilities),{cfg.dropout})


class DatasetAndCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.out=Path(cls.temp.name)
        cls.config={**CONFIG,'scenario':'single','cycles':4,'B':4,'K':2,'lookahead_cycles':2,
                    'tail_width':1,'candidate_pool_cap':32}
        cls.records=[]
        for seed in (9780101,9780102):
            p=cls.out/f'teacher_{seed}.json';write_json(p,run_case(seed,cls.config,'lookahead'));cls.records.append(p)
        cls.path=cls.out/'priority.pt'
        cls.receipt=train(cls.records,cls.path,updates=8,seed=1151)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_G04_real_training_recurrent_change_and_reload(self):
        r=self.receipt
        self.assertEqual(r['updates'],8);self.assertEqual(r['actual_visited_world_seeds'],[9780101,9780102])
        self.assertTrue(all(x>0 for x in r['recurrent_weight_l1_change'].values()))
        self.assertEqual(r['optimizer']['weight_decay'],0.)
        self.assertEqual(r['sequence_reload']['logit_max_difference'],0.)
        self.assertEqual(r['sequence_reload']['memory_max_difference'],0.)
    def test_G05_full_graph_is_not_only_paid_atoms(self):
        seq,manifest,cfgs,audit=load_sequences(self.records)
        self.assertEqual(audit['recomputed_paid_labels'],32)
        c=seq[0]['contexts'][0]
        used={a for schedule in c['pool'] for a in schedule}
        self.assertGreater(len(c['actions']),len(used))
        saved=read_json(self.out/'dataset/world_9780101.json')
        self.assertEqual(saved['contexts'][0]['full_graph_digest'],c['graph_digest'])
        self.assertEqual(len(saved['contexts']),4)
    def test_G05_complete_graph_matches_deployment(self):
        d=read_json(self.records[0]);t=d['trace'][0];c=t['commit']['policy_config']
        p=Priority(self.path);p.begin_world(9880201,self.config)
        aa=[Task(**a) for a in t['commit']['input_actions']]
        p.prepare(t['causal_view'],aa,c)
        saved=read_json(self.out/'dataset/world_9780101.json')['contexts'][0]
        self.assertEqual(graph_digest(p.G),saved['full_graph_digest'])
        p.prepare(t['causal_view'],list(reversed(aa)),c)
        self.assertEqual(len(p.memory.audit),1)
    def test_G05_incomplete_teacher_action_graph_is_rejected(self):
        d=read_json(self.records[0]);cm=d['trace'][0]['commit'];cm['input_actions'].pop()
        cm.pop('digest');cm['digest']=digest(cm)
        p=self.out/'incomplete.json';write_json(p,d)
        with self.assertRaisesRegex(ContractError,'INCOMPLETE_TRAINING_ACTION_GRAPH'):load_sequences([p])
    def test_G05_wrong_paid_label_is_recomputed_and_rejected(self):
        d=read_json(self.records[0]);cm=d['trace'][0]['commit'];cm['archive'][0]['proxy_m2']+=100.
        cm.pop('digest');cm['digest']=digest(cm)
        p=self.out/'label.json';write_json(p,d)
        with self.assertRaisesRegex(ContractError,'PAID_LABEL_RECOMPUTATION'):load_sequences([p])
    def test_G05_nonconsecutive_and_duplicate_worlds_rejected(self):
        with self.assertRaisesRegex(ContractError,'DUPLICATE_TRAIN_WORLD'):load_sequences([self.records[0]]*2)
        d=read_json(self.records[0]);d['trace']=d['trace'][::2];p=self.out/'sparse.json';write_json(p,d)
        with self.assertRaisesRegex(ContractError,'NONCONSECUTIVE'):load_sequences([p])
    def test_G06_checkpoint_contract_tamper_rejected(self):
        original=torch.load(self.path,weights_only=True)
        mutations=[('features',lambda c:c.update(feature_binding='old')),
            ('labels',lambda c:c.update(label_binding='old')),
            ('source',lambda c:c['source_manifest']['files'].update({'retained_campaign/vendor/x.py':'bad'})),
            ('dependency',lambda c:c['dependency_contract'].update(torch='0.0')),
            ('cfg',lambda c:c['cfg'].update(hidden_dim=64)),
            ('state',lambda c:next(iter(c['state'].values())).add_(1)),
            ('dataset',lambda c:c['dataset_manifest'][0].update(world_seed=123)),
            ('config',lambda c:c['allowed_policy_configurations'].update(bad={})),
            ('memory',lambda c:c.update(memory_semantics='empty')),
            ('partition',lambda c:c.update(development_law_seeds=c['training_law_seeds'][:1])),
            ('target',lambda c:c['target'].update(future_return_learning=True))]
        for name,modify in mutations:
            with self.subTest(name=name):
                ck=deepcopy(original);modify(ck)
                with self.assertRaises(ContractError):verify_checkpoint(ck)
    def test_G06_old_result_and_checkpoint_rejected(self):
        with self.assertRaises(ContractError):verify_checkpoint({'schema':'ain.spatial.priority.v1'})
        d=read_json(self.records[0]);d.pop('source_manifest');p=self.out/'old.json';write_json(p,d)
        with self.assertRaisesRegex(ContractError,'SOURCE_CLOSURE'):load_sequences([p])
    def test_G06_train_eval_disjoint_and_online_reset(self):
        p=Priority(self.path)
        for seed in (9780101,9780102):
            with self.assertRaisesRegex(ContractError,'WORLD_OVERLAP'):p.begin_world(seed,self.config)
        a=run_case(9880201,self.config,'learned',p)
        b=run_case(9880201,self.config,'learned',p)
        self.assertEqual([t['selected'] if 'selected' in t else t['commit']['selected'] for t in a['trace']],
                         [t['selected'] if 'selected' in t else t['commit']['selected'] for t in b['trace']])
        self.assertEqual(a['trace'][0]['causal_view'],b['trace'][0]['causal_view'])
        self.assertEqual(p.memory.audit[0]['history_nonzero_entries'],0)
    def test_G06_runtime_configuration_and_action_domain_rejected(self):
        d=read_json(self.records[0]);t=d['trace'][0];cfg=t['commit']['policy_config'];v=t['causal_view']
        aa=[Task(**a) for a in t['commit']['input_actions']];p=Priority(self.path);p.begin_world(9880201,self.config)
        with self.assertRaisesRegex(ContractError,'CONFIGURATION_DOMAIN'):p.prepare(v,aa,{**cfg,'bandwidth_scale':2.})
        with self.assertRaisesRegex(ContractError,'INCOMPLETE_RUNTIME_ACTION_GRAPH'):p.prepare(v,aa[:-1],cfg)
    def test_G06_actual_import_origins_and_retained_closure(self):
        origins=import_origins();closure=source_manifest()['files']
        self.assertTrue(any(k.startswith('retained_campaign/vendor/iclr2027_graph_agent') for k in closure))
        for name,entry in origins['internal'].items():
            self.assertTrue(Path(entry['path']).is_relative_to(ROOT))
            self.assertEqual(closure[entry['relative_path']],entry['sha256'])
    def test_G06_real_cli_checkpoint_and_resume(self):
        plan={'model':self.config,'methods':['learned'],
              'cases':[{'scenario':'single','seeds':[9880202]}]}
        path=self.out/'cli_plan.json';write_json(path,plan)
        out=self.out/'cli_results'
        cmd=[sys.executable,'-m','spatial_lab.run','--config',str(path),
             '--output',str(out),'--checkpoint',str(self.path)]
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run(cmd+['--resume'],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        receipt=read_json(out/'RUN_RECEIPT.json')
        self.assertTrue(any(row['status']=='REUSED' for row in receipt['cases']))
    def test_G06_foreign_already_loaded_vendor_is_rejected(self):
        code="import sys,types; p=types.ModuleType('iclr2027_graph_agent'); p.__file__='/foreign/__init__.py'; sys.modules['iclr2027_graph_agent']=p; from spatial_lab.spatial_graph import graph"
        r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(r.returncode,0);self.assertIn('FOREIGN_GRAPH_AGENT_IMPORT',r.stderr)
    def test_G06_loaded_process_cannot_claim_new_disk_revision(self):
        from . import binding
        fake=deepcopy(source_manifest());fake['files']['retained_campaign/vendor/iclr2027_graph_agent/model.py']='changed'
        with patch.object(binding,'source_manifest',return_value=fake):
            with self.assertRaisesRegex(ContractError,'SOURCE_CHANGED_AFTER'):binding.source_digest()

if __name__=='__main__':unittest.main(verbosity=2)
