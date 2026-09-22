#!/usr/bin/env python3
"""DO NOT touch other people's files. DO NOT cancel other people's jobs.

CPU-only tests using new temporary fixtures; no sbatch or model loading.
"""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze as a
from generate import generation_args, replace_arg
from blind_review import select_pairs


class AnalysisTests(unittest.TestCase):
    def test_eos_boundary(self):
        self.assertEqual(a.content_tokens([a.EOS,1,2,a.EOS,3]),([1,2],3))
        self.assertEqual(a.content_tokens([1,a.EOS,3]),([1],1))
        self.assertEqual(a.content_tokens([a.EOS,a.EOS]),([],1))
        self.assertEqual(a.content_tokens([1,2]),([1,2],None))
        self.assertEqual(a.content_tokens([]),([],None))

    def test_ngram_definition_and_short_text(self):
        self.assertIsNone(a.repetition([1,2]))
        self.assertAlmostEqual(a.repetition([1]*6),2/3)
        self.assertAlmostEqual(a.distinct([[1]*5,[1]*5]),1/4)

    def test_prefix_requires_256_predictions_before_eos(self):
        self.assertIsNone(a.prefix_ids([1]*256))
        self.assertIsNone(a.prefix_ids([1]*256+[a.EOS]))
        self.assertEqual(len(a.prefix_ids([1]*257+[a.EOS])),257)
        self.assertEqual(len(a.prefix_ids([a.EOS]+[1]*256+[a.EOS])),257)
        self.assertIsNone(a.prefix_ids([a.EOS,a.EOS]+[1]*300))

    def test_ppl_is_token_weighted_not_mean_ppl(self):
        scores=[dict(token_count=1,mean_nll_nats=0),dict(token_count=3,mean_nll_nats=2)]
        self.assertAlmostEqual(a.pooled_ppl(scores),math.exp(1.5))
        self.assertIsNone(a.pooled_ppl([dict(token_count=0,mean_nll_nats=None)]))

    def test_paths_refuse_escape_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            with patch.object(a,'HOME',Path(tmp)):
                p=a.fresh(Path(tmp)/'new');a.save(p/'one.json',dict(ok=True))
                self.assertEqual(a.read(p/'one.json'),dict(ok=True))
                with self.assertRaises(FileExistsError):a.save(p/'one.json',{})
                with self.assertRaises(FileExistsError):a.fresh(p)
                with self.assertRaises(PermissionError):a.owned(other)
                (p/'escape').symlink_to(other,target_is_directory=True)
                with self.assertRaises(PermissionError):a.owned(p/'escape'/'file')

    def test_wrong_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(a,'HOME',Path(tmp)):
            with patch.object(a.os,'getuid',return_value=999999):
                with self.assertRaises(PermissionError):a.owned(tmp)

    def test_generation_changes_only_output_count_seed_and_mode(self):
        original=['--output-dir','/old','--num-samples','100','--base-seed','100001',
                  '--modes','structured_joint','--nfe-budgets','17','--batch-size','1',
                  '--reference-lm-dtype','float32','--override','checkpointing.save_dir=/old']
        args=generation_args(dict(args=original),Path('/new'),dict(mode='factorized'),300001)
        self.assertEqual(original[1],'/old')
        expected=['--output-dir','/new/generation','--num-samples','500','--base-seed','300001',
                  '--modes','factorized','--nfe-budgets','17','--batch-size','1',
                  '--reference-lm-dtype','float32','--override','checkpointing.save_dir=/new']
        self.assertEqual(args,expected)
        with self.assertRaises(ValueError):replace_arg(['--x','1','--x','2'],'--x','3')

    def test_blind_pairs_balanced_and_no_labels(self):
        groups={(m,c,s):[dict(pair_key=f'p{i}',pair_seed=i,review_text=f'text {i}') for i in range(20)]
                for m in ('FD','DD') for c in ('bf16','fp32') for s in (8,16,32)}
        public,key=select_pairs(groups)
        self.assertEqual(len(public),60)
        self.assertTrue(all(set(p)=={'review_id','A','B'} for p in public))
        for m in ('FD','DD'):
            for s in (8,16,32):
                self.assertEqual(sum(k['A']=='bf16' for k in key if k['model']==m and k['steps']==s),5)
        self.assertEqual((public,key),select_pairs(groups))
        bad=copy.deepcopy(groups);bad['FD','fp32',8][0]['pair_seed']=9999
        # All 20 selected ensures the mismatched seed is checked.
        with self.assertRaises(ValueError):select_pairs(bad,count=20)

    def test_artifact_hash_and_sample_count_checked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(a,'HOME',Path(tmp)):
            d=a.fresh(Path(tmp)/'cells'/'x');a.fresh(d/'generation')
            sample=dict(pair_key='one',reference_lm=dict(revision=a.REVISION,sequence_policy=a.POLICY),
                        sampling_mode='structured_joint',requested_nfe_budget=9)
            path=d/'generation/samples.jsonl';a.write_text(path,json.dumps(sample)+'\n')
            a.save(d/'completed.json',dict(samples_sha256=a.sha(path)))
            with self.assertRaisesRegex(ValueError,'500 unique'):
                a.records(tmp,dict(id='x',mode='structured_joint',steps=8))
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                a.records(tmp,dict(id='x',mode='structured_joint',steps=8))


if __name__=='__main__':unittest.main()
