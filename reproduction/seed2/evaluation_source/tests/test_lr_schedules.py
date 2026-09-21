import copy
from pathlib import Path
import shlex
import tempfile
import unittest

import torch
from transformers import get_constant_schedule_with_warmup

from lr_schedules import linear_decay_continuation


class LinearDecayContinuationTest(unittest.TestCase):

  def optimizer(self):
    return torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=3e-4)

  def test_hydra_instantiates_continuation_schedule(self):
    import hydra

    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
      config = hydra.compose(config_name='config', overrides=[
        'model=contextual-forest-small', 'lr_scheduler=linear_decay_continuation',
        '++model.structured_decoder.factor_embedding_mode=shared',
        'model.structured_decoder.rank=16', 'trainer.max_steps=10000'])
    schedule = hydra.utils.instantiate(config.lr_scheduler, optimizer=self.optimizer())
    self.assertAlmostEqual(schedule.lr_lambdas[0](6000), 1)
    self.assertAlmostEqual(schedule.lr_lambdas[0](10000), 0.1)

  def test_batch_script_keeps_baseline_training_overrides(self):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    def overrides(filename):
      text = (scripts / filename).read_text()
      command = text.split('srun --ntasks=1 python -u main.py', 1)[1].split('\n\n', 1)[0]
      return dict(word.split('=', 1)
                  for word in shlex.split(command.replace('\\\n', ' ')))
    baseline = overrides('continue_four_ccf_matched_6k_to_10k.sh')
    decay = overrides('continue_four_ccf_shared_6k_to_10k_lr_decay.sh')
    self.assertEqual(decay.pop('model.structured_decoder.rank'), '16')
    self.assertEqual(decay.pop('++model.structured_decoder.factor_embedding_mode'),
                     'shared')
    without_schedule = lambda values: {
      key: value for key, value in values.items() if not key.startswith('lr_scheduler')}
    self.assertEqual(without_schedule(baseline), without_schedule(decay))

  def test_absolute_step_rates_and_no_second_warmup(self):
    schedule = linear_decay_continuation(self.optimizer())
    for step, expected in ((0, 3e-4), (6000, 3e-4), (7000, 2.325e-4),
                           (8000, 1.65e-4), (9000, 9.75e-5),
                           (10000, 3e-5), (12000, 3e-5)):
      with self.subTest(step=step):
        self.assertAlmostEqual(3e-4 * schedule.lr_lambdas[0](step), expected)

  def test_invalid_settings(self):
    for settings in ({'decay_start_step': -1}, {'decay_start_step': True},
                     {'decay_end_step': 6000}, {'decay_end_step': 10000.5},
                     {'final_lr_ratio': -0.1}, {'final_lr_ratio': 1.1},
                     {'final_lr_ratio': float('nan')}):
      with self.subTest(settings=settings), self.assertRaises(ValueError):
        linear_decay_continuation(self.optimizer(), **settings)

  def test_old_scheduler_restore_keeps_moments_and_uses_new_decay(self):
    optimizer = self.optimizer()
    old = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=50)
    parameter = optimizer.param_groups[0]['params'][0]
    for _ in range(6000):
      parameter.grad = torch.ones_like(parameter)
      optimizer.step()
      old.step()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(old.state_dict())
    resumed_optimizer = self.optimizer()
    new = linear_decay_continuation(resumed_optimizer)
    resumed_optimizer.load_state_dict(optimizer_state)
    new.load_state_dict(scheduler_state)
    self.assertEqual(new.last_epoch, 6000)
    self.assertEqual(resumed_optimizer.param_groups[0]['lr'], 3e-4)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
      torch.testing.assert_close(resumed_optimizer.state_dict()['state'][0][key],
                                 optimizer_state['state'][0][key], atol=0, rtol=0)
    for step in range(6001, 10001):
      resumed_optimizer.step()
      new.step()
      if step in (7000, 8000, 9000, 10000):
        expected = 3e-4 * (1 - 0.9 * (step - 6000) / 4000)
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]['lr'], expected)

  def test_lightning_full_checkpoint_resume_and_interruption(self):
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    class TinyModel(L.LightningModule):
      def __init__(self, decay):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.decay = decay
        self.rates = []

      def training_step(self, batch, batch_idx):
        self.rates.append((self.global_step, self.optimizers().param_groups[0]['lr']))
        return self.weight.square()

      def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = (linear_decay_continuation(optimizer, 6, 10)
                     if self.decay else get_constant_schedule_with_warmup(optimizer, 2))
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    loader = DataLoader(TensorDataset(torch.zeros(20, 1)), batch_size=1)
    with tempfile.TemporaryDirectory() as directory:
      def run(model, end, resume=None):
        trainer = L.Trainer(
          default_root_dir=directory, accelerator='cpu', devices=1,
          max_steps=end, logger=False, enable_checkpointing=False,
          enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model, loader, ckpt_path=resume)
        return trainer

      source = TinyModel(False)
      trainer = run(source, 6)
      initial_path = str(Path(directory) / 'constant-6.ckpt')
      trainer.save_checkpoint(initial_path)
      uninterrupted = TinyModel(True)
      uninterrupted_trainer = run(uninterrupted, 10, initial_path)
      self.assertEqual(uninterrupted.rates[0], (6, 3e-4))
      self.assertAlmostEqual(uninterrupted.rates[1][1], 2.325e-4)
      self.assertAlmostEqual(uninterrupted_trainer.optimizers[0].param_groups[0]['lr'], 3e-5)

      interrupted = TinyModel(True)
      trainer = run(interrupted, 8, initial_path)
      restart_path = str(Path(directory) / 'decay-8.ckpt')
      trainer.save_checkpoint(restart_path)
      restarted = TinyModel(True)
      trainer = run(restarted, 10, restart_path)
      self.assertEqual(interrupted.rates + restarted.rates, uninterrupted.rates)
      torch.testing.assert_close(restarted.weight, uninterrupted.weight, atol=0, rtol=0)
      self.assertEqual(trainer.global_step, 10)


if __name__ == '__main__':
  unittest.main()
