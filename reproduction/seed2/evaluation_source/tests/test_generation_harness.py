from types import SimpleNamespace
import unittest

import torch

from evaluation.generation_harness import (
  PromptSpec,
  batch_seed,
  expand_paired_samples,
  pairing_digest,
  prompt_from_record,
  run_sampling_group,
  sample_from_initial_state,
  summarize_group,
  unconditional_prompt,
)


class ToyTokenizer:
  eos_token_id = 0
  pad_token_id = 0

  def __call__(self, text, **unused_kwargs):
    return {'input_ids': [ord(character) % 31 + 1 for character in text]}

  def batch_decode(self, rows):
    return [' '.join(str(value) for value in row) for row in rows]


class CountingBackbone(torch.nn.Module):

  def forward(self, tokens):
    batch, length = tokens.shape
    logits = torch.zeros(batch, length, 128, device=tokens.device)
    logits[..., 7] = 1.0
    return logits

  def encode(self, tokens):
    """Deliberately bypass Module.__call__ to mirror the real DIT path."""
    batch, length = tokens.shape
    return torch.ones(batch, length, 3, device=tokens.device)


class FakeDiffusionModel:

  def __init__(self):
    self.device = torch.device('cpu')
    self.mask_index = 99
    self.sampler = 'ddpm'
    self.structured_enabled = True
    self.structured_sampling_mode = 'factorized'
    self.backbone = CountingBackbone()
    self.config = SimpleNamespace(
      backbone='dit',
      sampling=SimpleNamespace(noise_removal=True))

  def _ddpm_update(self, x, unused_t, unused_dt):
    if self.structured_sampling_mode == 'factorized':
      self.backbone(x)
    else:
      self._structured_backbone_output(x)
    # Leave evidence unchanged and resolve one masked position per row.
    result = x.clone()
    for row in range(x.shape[0]):
      positions = torch.nonzero(x[row].eq(self.mask_index)).flatten()
      if positions.numel():
        result[row, positions[0]] = 7
    return result

  def _structured_clean_sample(self, x, unused_conditioning):
    self._structured_backbone_output(x)
    return torch.where(x.eq(self.mask_index), torch.full_like(x, 7), x)

  def _structured_backbone_output(self, x):
    return self.backbone.encode(x)

  def noise(self, t):
    return t, None

  def forward(self, x, unused_conditioning):
    return self.backbone(x)


class GenerationHarnessTest(unittest.TestCase):

  def setUp(self):
    self.tokenizer = ToyTokenizer()

  def test_explicit_token_prompt_is_masked_and_validated(self):
    prompt = prompt_from_record(
      {
        'id': 'explicit',
        'input_ids': [1, 2, 3, 4],
        'active_mask': [False, True, True, False],
        'reference_token_ids': [1, 2, 3, 4],
      },
      tokenizer=self.tokenizer,
      mask_token_id=99,
      sequence_length=4,
      line_number=1)
    self.assertEqual(prompt.initial_token_ids, (1, 99, 99, 4))
    self.assertEqual(prompt.reference_token_ids, (1, 2, 3, 4))

  def test_text_infilling_and_prefix_generation_schemas(self):
    infill = prompt_from_record(
      {'id': 'infill', 'text': 'abcd', 'mask_token_indices': [1, 2]},
      tokenizer=self.tokenizer,
      mask_token_id=99,
      sequence_length=6,
      line_number=1)
    self.assertEqual(infill.active_mask, (False, True, True, False, False, False))
    self.assertEqual(infill.initial_token_ids[1:3], (99, 99))
    self.assertEqual(infill.reference_token_ids[-2:], (0, 0))

    prefix = prompt_from_record(
      {'id': 'prefix', 'prompt': 'ab'},
      tokenizer=self.tokenizer,
      mask_token_id=99,
      sequence_length=5,
      line_number=2)
    self.assertEqual(prefix.active_mask, (False, False, True, True, True))
    self.assertEqual(prefix.initial_token_ids[-3:], (99, 99, 99))

  def test_invalid_prompt_indices_fail_closed(self):
    with self.assertRaisesRegex(ValueError, 'duplicate'):
      prompt_from_record(
        {'text': 'abc', 'mask_token_indices': [1, 1]},
        tokenizer=self.tokenizer,
        mask_token_id=99,
        sequence_length=3,
        line_number=1)
    with self.assertRaisesRegex(ValueError, 'at least one'):
      PromptSpec('bad', (1, 2), (False, False))
    with self.assertRaisesRegex(ValueError, 'not selected'):
      prompt_from_record(
        {'input_ids': [99, 1], 'active_mask': [False, True]},
        tokenizer=self.tokenizer,
        mask_token_id=99,
        sequence_length=2,
        line_number=1)

  def test_pairing_commitment_and_batch_seed_are_deterministic(self):
    prompt = unconditional_prompt(mask_token_id=99, sequence_length=4)
    first = expand_paired_samples([prompt], num_samples=3, base_seed=10)
    second = expand_paired_samples([prompt], num_samples=3, base_seed=10)
    self.assertEqual(pairing_digest(first), pairing_digest(second))
    self.assertEqual(batch_seed(first), batch_seed(second))
    self.assertEqual([item.pair_seed for item in first], [10, 11, 12])

  def test_sampler_preserves_evidence_and_measures_backbone_calls(self):
    model = FakeDiffusionModel()
    initial = torch.tensor([[5, 99, 99, 6]])
    generated, measured_nfe = sample_from_initial_state(
      model, initial, nfe_budget=4)
    self.assertEqual(generated.tolist(), [[5, 7, 7, 6]])
    self.assertEqual(measured_nfe, 4)

  def test_group_emits_every_sample_tokens_metadata_and_summary(self):
    prompt = PromptSpec(
      prompt_id='paired-infill',
      initial_token_ids=(5, 99, 99, 6),
      active_mask=(False, True, True, False),
      reference_token_ids=(5, 7, 8, 6))
    samples = expand_paired_samples([prompt], num_samples=2, base_seed=30)
    model = FakeDiffusionModel()
    records, batch = run_sampling_group(
      model,
      samples,
      sampling_mode='structured_joint',
      nfe_budget=4,
      tokenizer=self.tokenizer,
      device=torch.device('cpu'))
    self.assertEqual(len(records), 2)
    self.assertEqual(records[0]['sample_token_ids'], [5, 7, 7, 6])
    self.assertEqual(records[0]['sample_active_token_ids'], [7, 7])
    self.assertEqual(records[0]['metrics']['reference_token_accuracy'], 0.5)
    self.assertEqual(records[0]['pair_seed'], 30)
    self.assertEqual(batch['active_tokens'], 4)
    self.assertEqual(batch['measured_nfe'], 4)
    self.assertEqual(model.structured_sampling_mode, 'factorized')
    summary = summarize_group(records)
    self.assertEqual(summary['num_sequences'], 2)
    self.assertEqual(summary['reference']['exact_match_rate'], 0.0)
    self.assertEqual(summary['measured_nfe_values'], [4])
    self.assertEqual(summary['pairing_digest'], pairing_digest(samples))

  def test_sampling_mode_and_structured_counter_wrapper_restore_on_failure(self):
    prompt = unconditional_prompt(mask_token_id=99, sequence_length=3)
    samples = expand_paired_samples([prompt], num_samples=1, base_seed=50)
    model = FakeDiffusionModel()
    original_method = model._structured_backbone_output

    def fail_update(unused_x, unused_t, unused_dt):
      raise RuntimeError('synthetic interruption')

    model._ddpm_update = fail_update
    with self.assertRaisesRegex(RuntimeError, 'interruption'):
      run_sampling_group(
        model,
        samples,
        sampling_mode='structured_joint',
        nfe_budget=4,
        tokenizer=self.tokenizer,
        device=torch.device('cpu'))
    self.assertEqual(model.structured_sampling_mode, 'factorized')
    self.assertEqual(
      model._structured_backbone_output.__func__, original_method.__func__)


if __name__ == '__main__':
  unittest.main()
