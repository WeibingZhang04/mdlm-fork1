import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.conditional_denoising_records import (
  ConditionalDenoisingRecordWriter,
  write_record_manifest,
)
from scripts.aggregate_hierarchical_document_eval import validate_record


class ConditionalDenoisingRecordsTest(unittest.TestCase):

  def _metadata(self):
    return {
      'protocol_id': 'multicorpus-v1',
      'job_id': 'wiki-seed1-mask50-contextual',
      'arm': 'contextual',
      'train_seed': 1,
      'corruption_seed': 1003,
      'dataset': 'wikitext103-pinned',
      'dataset_revision': 'a' * 40,
      'mask_rate': 0.5,
      'candidate_k': 64,
    }

  def test_streams_per_window_rows_and_binds_pairing_digest(self):
    with tempfile.TemporaryDirectory() as directory:
      writer = ConditionalDenoisingRecordWriter(
        output_dir=directory, rank=0, metadata=self._metadata())
      writer.append(
        batch={
          'input_ids': torch.zeros(2, 8, dtype=torch.long),
          'source_document_index': torch.tensor([12, 12]),
          'source_document_sha256': ['b' * 64, 'b' * 64],
          'source_chunk_index': torch.tensor([0, 1]),
        },
        metrics={
          'nll_sum': torch.tensor([4.5, 6.25]),
          'active_tokens': torch.tensor([3, 5]),
          'candidate_hits': torch.tensor([2, 4]),
          'retained_mass_sum': torch.tensor([1.5, 3.5]),
        },
        batch_index=7)
      digest = 'c' * 64
      summary = writer.finalize(pairing_digest_sha256=digest)
      final_path = Path(directory) / summary['path']
      rows = [json.loads(line) for line in final_path.read_text().splitlines()]

      self.assertEqual(len(rows), 2)
      self.assertEqual(rows[0]['document_id'], 'wikitext103-pinned:12')
      self.assertEqual(rows[1]['chunk_index'], 1)
      self.assertEqual(rows[0]['masked_tokens'], 3)
      self.assertAlmostEqual(rows[1]['nll_sum'], 6.25)
      self.assertEqual(rows[0]['pairing_digest_sha256'], digest)
      self.assertEqual(summary['num_records'], 2)
      self.assertEqual(summary['total_masked_tokens'], 8)
      self.assertEqual(
        summary['sha256'], hashlib.sha256(final_path.read_bytes()).hexdigest())

      manifest_path = write_record_manifest(
        output_dir=directory,
        metadata=self._metadata(),
        rank_summaries=[summary],
        pairing_digest={'sha256': digest, 'world_size': 1})
      manifest = json.loads(Path(manifest_path).read_text())
      self.assertEqual(manifest['num_records'], 2)
      self.assertEqual(manifest['total_masked_tokens'], 8)
      self.assertEqual(manifest['rank_files'][0]['sha256'], summary['sha256'])

  def test_rejects_dataset_without_document_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      writer = ConditionalDenoisingRecordWriter(
        output_dir=directory, rank=0, metadata=self._metadata())
      try:
        with self.assertRaisesRegex(ValueError, 'document-local'):
          writer.append(
            batch={'input_ids': torch.zeros(1, 8, dtype=torch.long)},
            metrics={
              'nll_sum': torch.tensor([1.0]),
              'active_tokens': torch.tensor([1]),
              'candidate_hits': torch.tensor([1]),
              'retained_mass_sum': torch.tensor([0.9]),
            },
            batch_index=0)
      finally:
        writer.close()

  def test_schema_v2_streams_causal_scores_and_support_grid(self):
    with tempfile.TemporaryDirectory() as directory:
      writer = ConditionalDenoisingRecordWriter(
        output_dir=directory,
        rank=0,
        metadata=self._metadata(),
        schema_version=2,
        support_candidate_ks=(32, 64, 128, 256))
      writer.append(
        batch={
          'input_ids': torch.zeros(1, 8, dtype=torch.long),
          'source_document_index': torch.tensor([12]),
          'source_document_sha256': ['b' * 64],
          'source_chunk_index': torch.tensor([0]),
        },
        metrics={
          'nll_sum': torch.tensor([4.0]),
          'structured_marginal_nll_sum': torch.tensor([4.5]),
          'factorized_backbone_nll_sum': torch.tensor([5.0]),
          'parameter_matched_no_edge_nll_sum': torch.tensor([5.0]),
          'matched_permuted_topology_nll_sum': torch.tensor([4.75]),
          'active_tokens': torch.tensor([4]),
          'candidate_hits': torch.tensor([3]),
          'retained_mass_sum': torch.tensor([3.2]),
          'candidate_support_ks': [32, 64, 128, 256],
          'candidate_support_hits': torch.tensor([[1, 2, 3, 4]]),
          'candidate_support_retained_mass_sum': torch.tensor(
            [[1.0, 2.0, 3.0, 3.8]]),
          'selected_edges': torch.tensor([3]),
          'permuted_changed_edges': torch.tensor([2]),
          'selected_degree_sequence': [[1, 1, 2, 2]],
          'permuted_degree_sequence': [[1, 1, 2, 2]],
          'selected_component_sizes': [[4]],
          'permuted_component_sizes': [[4]],
        },
        batch_index=0)
      summary = writer.finalize(pairing_digest_sha256='c' * 64)
      rows = [json.loads(line) for line in (
        Path(directory) / summary['path']).read_text().splitlines()]
      self.assertEqual(rows[0]['schema_version'], 2)
      self.assertEqual(
        [item['candidate_k'] for item in rows[0]['candidate_support']],
        [32, 64, 128, 256])
      self.assertEqual(rows[0]['factorized_backbone_nll_sum'], 5.0)
      self.assertEqual(rows[0]['selected_degree_sequence'], [1, 1, 2, 2])
      self.assertEqual(rows[0]['permuted_component_sizes'], [4])
      manifest_path = write_record_manifest(
        output_dir=directory,
        metadata=self._metadata(),
        rank_summaries=[summary],
        pairing_digest={'sha256': 'c' * 64, 'world_size': 1},
        schema_version=2)
      self.assertEqual(
        json.loads(Path(manifest_path).read_text())['schema_version'], 2)
      validate_record(rows[0])
      degree_tamper = dict(rows[0])
      degree_tamper['permuted_degree_sequence'] = [0, 2, 2, 2]
      with self.assertRaisesRegex(ValueError, 'preserve degree sequence'):
        validate_record(degree_tamper)
      component_tamper = dict(rows[0])
      component_tamper['permuted_component_sizes'] = [1, 3]
      with self.assertRaisesRegex(ValueError, 'preserve component sizes'):
        validate_record(component_tamper)


if __name__ == '__main__':
  unittest.main()
