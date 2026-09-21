from pathlib import Path
import tempfile
import json
import re
import subprocess
import sys
import unittest

import torch

from scripts.ccf_separate_resume import select_checkpoint


def checkpoint(step, topology='dynamic'):
  return {
    'global_step': step,
    'hyper_parameters': {'config': {
      'seed': 1,
      'model': {'length': 1024, 'structured_decoder': {
        'rank': 8, 'top_k': 128, 'factor_embedding_mode': 'separate',
        'factor_conditioner_hidden_dim': 0, 'topology_mode': topology,
        'factor_mode': 'dynamic', 'independent_mode': False,
        'training': {'backbone_mode': 'frozen', 'head_lr': 0.0003,
                     'topology_weight': 0.0 if topology == 'fixed' else 0.1},
      }},
      'lr_scheduler': {'_target_': 'transformers.get_constant_schedule_with_warmup',
                       'num_warmup_steps': 50},
    }},
    'state_dict': {'test': torch.tensor(1)},
    'loops': {'fit_loop': {}},
    'optimizer_states': [{'state': {'sentinel': 1}}],
    'lr_schedulers': [{'last_epoch': step}],
  }


class SeparateResumeTest(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name).resolve()
    self.directory = self.root / 'checkpoints'
    self.directory.mkdir()

  def save(self, name, step, topology='dynamic'):
    path = self.directory / name
    torch.save(checkpoint(step, topology), path)
    return path

  def test_no_checkpoint_restarts_fresh(self):
    self.assertEqual(select_checkpoint(self.root, 'dynamic'),
                     {'checkpoint': None, 'global_step': 0})

  def test_shell_json_extraction_has_no_jq_dependency(self):
    script = (Path(__file__).resolve().parents[1]
              / 'scripts/run_ccf_separate_7k.sh').read_text()
    self.assertNotIn('jq ', script)
    snippets = re.findall(r"python -c '([^']+)'", script)
    self.assertEqual(len(snippets), 4)
    for payload in ({'checkpoint': None, 'global_step': 0},
                    {'checkpoint': '/some path/model.ckpt', 'global_step': 7000}):
      path = self.root / 'resume.json'
      path.write_text(json.dumps(payload))
      results = [subprocess.check_output([sys.executable, '-c', code, str(path)],
                                         text=True).strip() for code in snippets]
      self.assertEqual(results[:2], [payload['checkpoint'] or '', str(payload['global_step'])])
      self.assertEqual(results[2], str(payload['global_step']))

  def test_valid_last_resume_and_source_unmodified(self):
    path = self.save('last.ckpt', 1500)
    before = path.read_bytes()
    self.assertEqual(select_checkpoint(self.root, 'dynamic'),
                     {'checkpoint': str(path), 'global_step': 1500})
    self.assertEqual(path.read_bytes(), before)

  def test_newer_numbered_checkpoint_beats_stale_last(self):
    self.save('last.ckpt', 500)
    path = self.save('0-1000.ckpt', 1000)
    self.assertEqual(select_checkpoint(self.root, 'dynamic')['checkpoint'], str(path))

  def test_corrupt_last_falls_back_without_overwriting(self):
    bad = self.directory / 'last.ckpt'
    bad.write_bytes(b'partial checkpoint')
    path = self.save('0-500.ckpt', 500)
    self.assertEqual(select_checkpoint(self.root, 'dynamic')['checkpoint'], str(path))
    self.assertEqual(bad.read_bytes(), b'partial checkpoint')

  def test_all_corrupt_fails_closed(self):
    (self.directory / 'last.ckpt').write_bytes(b'partial')
    with self.assertRaisesRegex(ValueError, 'none is readable'):
      select_checkpoint(self.root, 'dynamic')

  def test_wrong_arm_fails_closed(self):
    self.save('last.ckpt', 500, topology='fixed')
    with self.assertRaisesRegex(ValueError, 'topology_mode'):
      select_checkpoint(self.root, 'dynamic')

  def test_completed_training_does_not_need_more_updates(self):
    self.save('last.ckpt', 7000)
    self.assertEqual(select_checkpoint(self.root, 'dynamic')['global_step'], 7000)

  def test_best_checkpoint_can_be_used_as_fallback(self):
    path = self.save('best.ckpt', 500)
    self.assertEqual(select_checkpoint(self.root, 'dynamic')['checkpoint'], str(path))

  def test_mismatched_scheduler_or_filename_rejected(self):
    self.save('0-1000.ckpt', 500)
    with self.assertRaisesRegex(ValueError, 'Filename/global_step'):
      select_checkpoint(self.root, 'dynamic')
    payload = checkpoint(500)
    payload['lr_schedulers'][0]['last_epoch'] = 499
    torch.save(payload, self.directory / 'last.ckpt')
    with self.assertRaisesRegex(ValueError, 'scheduler step'):
      select_checkpoint(self.root, 'dynamic')


if __name__ == '__main__':
  unittest.main()
