import math
import unittest
from unittest import mock

from evaluation.generation_metrics import ReferenceLMScore
from evaluation.generation_shard_aggregation import _summarize_reference_lm
from scripts.run_generation_pilot import (
  _attach_reference_lm_scores,
  _critical_runtime_package_versions,
  _summarize_attached_reference_lm,
)


class GenerationReferenceLMPlumbingTest(unittest.TestCase):

  @mock.patch('scripts.run_generation_pilot.TransformersReferenceLMScorer')
  def test_revision_is_recorded_in_rows_and_summaries(self, scorer_class):
    scorer = scorer_class.return_value
    scorer.score.return_value = [
      ReferenceLMScore(token_count=4, mean_nll_nats=1.0, perplexity=2.718),
      ReferenceLMScore(token_count=2, mean_nll_nats=2.0, perplexity=7.389),
    ]
    rows = [{'text': 'one'}, {'text': 'two'}]
    revision = 'c' * 40
    scorer.runtime_identity.return_value = {
      'model_name_or_path': 'org/model',
      'model_revision': revision,
      'batch_size': 2,
      'max_length': 256,
      'requested_dtype': 'float32',
      'device': 'cpu',
      'sequence_policy': (
        'retokenize_decoded_text_score_through_first_nonleading_eos_v1'),
    }

    overall = _attach_reference_lm_scores(
      rows,
      model_name_or_path='org/model',
      revision=revision,
      device='cpu',
      batch_size=2,
      max_length=256,
      dtype='float32')
    group = _summarize_attached_reference_lm(rows)
    independently_recomputed = _summarize_reference_lm(rows)

    scorer_class.assert_called_once_with(
      'org/model', revision=revision, device='cpu', batch_size=2,
      max_length=256, dtype='float32')
    self.assertTrue(all(
      row['reference_lm']['revision'] == revision for row in rows))
    self.assertEqual(overall['revision'], revision)
    self.assertEqual(
      overall['runtime_identity'], scorer.runtime_identity.return_value)
    self.assertEqual(
      overall['sequence_policy'],
      'retokenize_decoded_text_score_through_first_nonleading_eos_v1')
    self.assertEqual(group['revision'], revision)
    self.assertEqual(group['sequence_policy'], overall['sequence_policy'])
    self.assertEqual(group['model_name_or_path'], 'org/model')
    self.assertEqual(group['mean_nll_nats'], 4 / 3)
    self.assertEqual(group['perplexity'], math.exp(4 / 3))
    for field in (
        'model_name_or_path', 'revision', 'sequence_policy',
        'num_scored_sequences', 'num_scored_tokens', 'mean_nll_nats',
        'perplexity'):
      self.assertEqual(group[field], overall[field])
      self.assertEqual(group[field], independently_recomputed[field])

  @mock.patch('scripts.run_generation_pilot.importlib.metadata.version')
  def test_critical_runtime_package_versions_are_recorded_exactly(
      self, version):
    version.side_effect = lambda distribution: f'{distribution}-version'

    result = _critical_runtime_package_versions()

    self.assertEqual(result, {
      'numpy': 'numpy-version',
      'safetensors': 'safetensors-version',
      'tokenizers': 'tokenizers-version',
      'transformers': 'transformers-version',
    })


if __name__ == '__main__':
  unittest.main()
