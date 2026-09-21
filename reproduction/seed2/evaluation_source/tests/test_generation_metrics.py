import math
import unittest
from unittest import mock

from types import SimpleNamespace
import torch

from evaluation.generation_metrics import (
  distinct_n,
  jensen_shannon_divergence,
  ngram_distribution,
  ngram_js_divergence,
  ngrams,
  paired_token_metrics,
  repetition_rate,
  summarize_token_metrics,
  TransformersReferenceLMScorer,
  validate_reference_lm_spec,
)


class GenerationMetricsTest(unittest.TestCase):

  def test_ngrams_distinct_and_repetition_have_explicit_semantics(self):
    self.assertEqual(ngrams([1, 2, 1, 2], 2), [(1, 2), (2, 1), (1, 2)])
    self.assertAlmostEqual(repetition_rate([1, 2, 1, 2], n=2), 1 / 3)
    self.assertAlmostEqual(
      distinct_n([[1, 2, 1, 2], [1, 2]], n=2), 2 / 4)
    self.assertEqual(repetition_rate([1, 2], n=4), 0.0)
    self.assertEqual(distinct_n([[1], []], n=2), 0.0)

  def test_sparse_js_is_symmetric_bounded_and_uses_natural_log(self):
    first = ngram_distribution([[1, 1]], n=1)
    second = ngram_distribution([[2, 2]], n=1)
    forward = jensen_shannon_divergence(first, second)
    backward = jensen_shannon_divergence(second, first)
    self.assertAlmostEqual(forward, math.log(2))
    self.assertAlmostEqual(backward, forward)
    self.assertAlmostEqual(
      ngram_js_divergence([[1, 2]], [[1, 2]], n=1), 0.0)
    self.assertIsNone(ngram_js_divergence([[1]], [[1]], n=2))

  def test_paired_token_metrics_validate_alignment(self):
    metrics = paired_token_metrics([1, 2, 3], [1, 9, 3])
    self.assertEqual(metrics['reference_token_count'], 3)
    self.assertAlmostEqual(metrics['reference_token_accuracy'], 2 / 3)
    self.assertFalse(metrics['reference_exact_match'])
    with self.assertRaisesRegex(ValueError, 'lengths differ'):
      paired_token_metrics([1], [1, 2])

  def test_summary_uses_corpus_level_ngram_js_and_token_weighting(self):
    summary = summarize_token_metrics(
      [[1, 2], [3]],
      reference=[[1, 9], [3]],
      n_values=(1, 2))
    self.assertEqual(summary['num_sequences'], 2)
    self.assertEqual(summary['num_tokens'], 3)
    self.assertAlmostEqual(summary['reference']['token_accuracy'], 2 / 3)
    self.assertAlmostEqual(summary['reference']['exact_match_rate'], 0.5)
    self.assertIn('1', summary['reference']['ngram_js_divergence_nats'])

  def test_invalid_n_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'positive'):
      ngrams([1, 2], 0)

  def test_reference_lm_requires_paired_model_and_revision(self):
    self.assertEqual(
      validate_reference_lm_spec('org/model', 'a' * 40),
      ('org/model', 'a' * 40))
    self.assertEqual(validate_reference_lm_spec(None, None), (None, None))
    with self.assertRaisesRegex(ValueError, 'requires.*revision'):
      validate_reference_lm_spec('org/model', None)
    with self.assertRaisesRegex(ValueError, 'requires --reference-lm'):
      validate_reference_lm_spec(None, 'a' * 40)
    with self.assertRaisesRegex(ValueError, '40-character'):
      validate_reference_lm_spec('org/model', 'main')

  @mock.patch('transformers.AutoModelForCausalLM.from_pretrained')
  @mock.patch('transformers.AutoTokenizer.from_pretrained')
  def test_reference_lm_revision_is_passed_to_both_loaders(
      self, tokenizer_loader, model_loader):
    class FakeTokenizer:
      pad_token_id = 0
      bos_token_id = 0
      eos_token_id = 0
      padding_side = 'left'
      truncation_side = 'left'

      def __len__(self):
        return 17

    tokenizer = FakeTokenizer()
    tokenizer_loader.return_value = tokenizer
    model = mock.Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config = SimpleNamespace(max_position_embeddings=128)
    model.parameters.return_value = [
      torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))]
    model_loader.return_value = model

    scorer = TransformersReferenceLMScorer(
      'org/model', revision='b' * 40, device='cpu')

    tokenizer_loader.assert_called_once_with(
      'org/model', revision='b' * 40, use_fast=True,
      trust_remote_code=False)
    model_loader.assert_called_once_with(
      'org/model', revision='b' * 40, torch_dtype=torch.float32,
      trust_remote_code=False)
    self.assertEqual(scorer.revision, 'b' * 40)
    self.assertEqual(scorer.max_length, 128)
    self.assertEqual(scorer.parameter_dtypes, ['torch.float32'])
    runtime = scorer.runtime_identity()
    self.assertEqual(runtime['tokenizer_vocab_size'], 17)
    self.assertEqual(runtime['batch_size'], 8)
    self.assertEqual(runtime['max_length'], 128)
    self.assertEqual(runtime['requested_dtype'], 'float32')
    self.assertEqual(runtime['parameter_dtypes'], ['torch.float32'])

  def test_reference_lm_scores_only_through_first_nonpadding_eos(self):
    class FakeTokenizer:
      pad_token_id = 2
      eos_token_id = 2
      bos_token_id = 2

      def __call__(self, texts, **unused_kwargs):
        self.texts = texts
        return {
          'input_ids': torch.tensor([
            [2, 10, 11, 2, 20],
            [2, 30, 31, 2, 2],
          ]),
          'attention_mask': torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
          ]),
        }

    class FakeModel:
      def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=torch.zeros(
          *input_ids.shape, 64, device=input_ids.device))

    scorer = TransformersReferenceLMScorer.__new__(
      TransformersReferenceLMScorer)
    scorer.batch_size = 2
    scorer.max_length = 5
    scorer.device = torch.device('cpu')
    scorer.tokenizer = FakeTokenizer()
    scorer.model = FakeModel()

    scores = scorer.score(['has eos then tail', 'padding uses eos id'])

    # The leading GPT-2-style BOS/EOS boundary is ignored. Row one includes
    # the later EOS target but excludes its real tail token. Row two has no
    # later valid EOS: padding EOS ids are ignored by attention_mask.
    self.assertEqual([score.token_count for score in scores], [3, 2])
    self.assertAlmostEqual(scores[0].mean_nll_nats, math.log(64))
    self.assertAlmostEqual(scores[1].mean_nll_nats, math.log(64))


if __name__ == '__main__':
  unittest.main()
