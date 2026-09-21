from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from scripts.export_structured_adapter import (
  RELEASE_TENSOR_COUNT,
  export_adapter,
  load_and_validate_adapter_manifest,
  load_adapter_into_head,
  load_adapter_state,
  structured_decoder_identity_from_config,
)


ADAPTER_IDENTITY = {
  'control_identity': 'dynamic_dynamic',
  'topology_mode': 'dynamic',
  'factor_mode': 'dynamic',
  'candidate_k': 64,
  'independent_mode': False,
  'topology_weight': 0.1,
}


def _structured_config(**overrides):
  structured = {
    'topology_mode': 'dynamic',
    'factor_mode': 'dynamic',
    'top_k': 64,
    'rank': 16,
    'time_embed_dim': 64,
    'topology_dim': 128,
    'local_window': 2,
    'num_anchor_slots': 16,
    'contextual_neighbors': 4,
    'component_size_cap': 32,
    'independent_mode': False,
    'min_edge_score': None,
    'fixed_edges': None,
    'fixed_edge_path': None,
    'training': {
      'objective_name': 'conditional_denoising_nll_not_diffusion_elbo',
      'topology_weight': 0.1,
      'factorized_aux_weight': 0.0,
      'topology_strategy': 'gold_reveal_influence',
      'topology_temperature': 0.25,
      'topology_minimum_choices': 2,
      'topology_edge_weight': 1.0,
      'topology_anchor_weight': 0.25,
      'topology_slot_weight': 0.25,
      'topology_on_validation': False,
    },
  }
  training = overrides.pop('training', None)
  structured.update(overrides)
  if training is not None:
    structured['training'].update(training)
  return structured


class ExportStructuredAdapterTest(unittest.TestCase):

  def _sha256(self, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

  def _checkpoint(self, path: Path) -> dict[str, torch.Tensor]:
    adapter = {
      'weight': torch.arange(6.0).reshape(2, 3),
      'bias': torch.tensor([1.0, 2.0]),
    }
    torch.save({
      'global_step': 7,
      'hyper_parameters': {
        'config': {
          'model': {
            'structured_decoder': {
              **_structured_config(),
            },
          },
        },
      },
      'state_dict': {
        **{
          f'backbone.layer_{index}.weight': torch.ones(1)
          for index in range(RELEASE_TENSOR_COUNT)
        },
        **{
          f'structured_head.{key}': value
          for key, value in adapter.items()
        },
      },
    }, path)
    return adapter

  def test_export_is_portable_and_strictly_rehydrates_head(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      expected = self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest_path = root / 'adapter-manifest.json'

      manifest = export_adapter(
        checkpoint,
        output,
        manifest_path,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=RELEASE_TENSOR_COUNT,
        **ADAPTER_IDENTITY)

      self.assertEqual(set(load_file(output)), set(expected))
      with safe_open(output, framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
      self.assertEqual(metadata['control_identity'], 'dynamic_dynamic')
      self.assertEqual(metadata['topology_mode'], 'dynamic')
      self.assertEqual(metadata['factor_mode'], 'dynamic')
      self.assertEqual(metadata['candidate_k'], '64')
      self.assertEqual(metadata['independent_mode'], 'false')
      self.assertEqual(metadata['topology_weight'], '0.1')
      self.assertEqual(
        metadata['source_checkpoint_sha256'], self._sha256(checkpoint))
      self.assertEqual(
        metadata['source_checkpoint_size_bytes'],
        str(checkpoint.stat().st_size))
      self.assertEqual(metadata['source_checkpoint_global_step'], '7')
      self.assertEqual(
        metadata['omitted_frozen_backbone_tensor_count'],
        str(RELEASE_TENSOR_COUNT))
      self.assertEqual(
        metadata['structured_decoder_identity_sha256'],
        manifest['structured_decoder_identity_sha256'])
      self.assertEqual(manifest['schema_version'], 4)
      expected_identity, expected_digest = (
        structured_decoder_identity_from_config(_structured_config()))
      self.assertEqual(
        manifest['structured_decoder_identity'], expected_identity)
      self.assertEqual(
        manifest['structured_decoder_identity_sha256'], expected_digest)
      self.assertEqual(manifest['source_checkpoint_global_step'], 7)
      self.assertEqual(manifest['adapter_tensor_count'], 2)
      self.assertEqual(
        manifest['omitted_frozen_backbone_tensor_count'],
        RELEASE_TENSOR_COUNT)
      self.assertNotIn(str(root), manifest_path.read_text())
      head = torch.nn.Linear(3, 2)
      load_adapter_into_head(
        head, output,
        manifest_path=manifest_path,
        expected_identity=expected_identity,
        expected_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=self._sha256(manifest_path))
      load_and_validate_adapter_manifest(
        manifest_path, output,
        expected_identity=expected_identity,
        expected_adapter_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=self._sha256(manifest_path))
      for key, value in expected.items():
        torch.testing.assert_close(head.state_dict()[key], value)

  def test_export_fails_closed_on_state_or_provenance_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest = root / 'manifest.json'
      with self.assertRaisesRegex(ValueError, 'global-step mismatch'):
        export_adapter(
          checkpoint, output, manifest,
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=8,
          expected_backbone_tensors=RELEASE_TENSOR_COUNT,
          **ADAPTER_IDENTITY)

      payload = torch.load(
        checkpoint, map_location='cpu', weights_only=False)
      payload['state_dict']['noise.weight'] = torch.ones(1)
      torch.save(payload, checkpoint)
      with self.assertRaisesRegex(ValueError, 'outside backbone'):
        export_adapter(
          checkpoint, output, manifest,
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=7,
          expected_backbone_tensors=RELEASE_TENSOR_COUNT,
          **ADAPTER_IDENTITY)

  def test_export_rejects_control_mode_or_candidate_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      common = {
        'expected_checkpoint_sha256': self._sha256(checkpoint),
        'expected_global_step': 7,
        'expected_backbone_tensors': RELEASE_TENSOR_COUNT,
      }
      with self.assertRaisesRegex(ValueError, 'requires topology/factor'):
        export_adapter(
          checkpoint, root / 'adapter.safetensors', root / 'manifest.json',
          control_identity='dynamic_dynamic', topology_mode='fixed',
          factor_mode='dynamic', candidate_k=64, independent_mode=False,
          topology_weight=0.1, **common)
      with self.assertRaisesRegex(ValueError, 'candidate_k must be a positive'):
        export_adapter(
          checkpoint, root / 'adapter.safetensors', root / 'manifest.json',
          control_identity='dynamic_dynamic', topology_mode='dynamic',
          factor_mode='dynamic', candidate_k=0, independent_mode=False,
          topology_weight=0.1, **common)

  def test_export_rejects_identity_not_proven_by_checkpoint(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      payload = torch.load(
        checkpoint, map_location='cpu', weights_only=False)
      payload['hyper_parameters']['config']['model'][
        'structured_decoder']['top_k'] = 32
      torch.save(payload, checkpoint)
      with self.assertRaisesRegex(
          ValueError, 'checkpoint structured-decoder identity mismatch'):
        export_adapter(
          checkpoint,
          root / 'adapter.safetensors',
          root / 'manifest.json',
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=7,
          expected_backbone_tensors=RELEASE_TENSOR_COUNT,
          **ADAPTER_IDENTITY)

  def test_export_rejects_independent_or_topology_weight_mislabel(self):
    for field, value in (
        ('independent_mode', True), ('topology_weight', 0.0)):
      with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'last.ckpt'
        self._checkpoint(checkpoint)
        payload = torch.load(
          checkpoint, map_location='cpu', weights_only=False)
        structured = payload['hyper_parameters']['config']['model'][
          'structured_decoder']
        if field == 'topology_weight':
          structured['training'][field] = value
        else:
          structured[field] = value
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(
            ValueError, 'checkpoint structured-decoder identity mismatch'):
          export_adapter(
            checkpoint, root / 'adapter.safetensors', root / 'manifest.json',
            expected_checkpoint_sha256=self._sha256(checkpoint),
            expected_global_step=7,
            expected_backbone_tensors=RELEASE_TENSOR_COUNT,
            **ADAPTER_IDENTITY)

  def test_manifest_rejects_wrong_static_k_or_independent_runtime(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest_path = root / 'manifest.json'
      manifest = export_adapter(
        checkpoint, output, manifest_path,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=RELEASE_TENSOR_COUNT,
        **ADAPTER_IDENTITY)
      runtime_configs = [
        _structured_config(
          topology_mode='fixed', factor_mode='fixed',
          training={'topology_weight': 0.0}),
        _structured_config(top_k=128),
        _structured_config(independent_mode=True),
      ]
      for structured in runtime_configs:
        _, inferred_digest = structured_decoder_identity_from_config(
          structured)
        self.assertNotEqual(
          inferred_digest, manifest['structured_decoder_identity_sha256'])
        identity, _ = structured_decoder_identity_from_config(structured)
        with self.assertRaisesRegex(
            ValueError, 'identity differs from runtime config'):
          load_and_validate_adapter_manifest(
            manifest_path, output, expected_identity=identity,
            expected_adapter_sha256=manifest['adapter_sha256'],
            expected_manifest_sha256=self._sha256(manifest_path))

  def test_manifest_cannot_relabel_same_shape_static_adapter_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest_path = root / 'manifest.json'
      manifest = export_adapter(
        checkpoint, output, manifest_path,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=RELEASE_TENSOR_COUNT,
        **ADAPTER_IDENTITY)
      with safe_open(output, framework='pt', device='cpu') as handle:
        metadata = dict(handle.metadata())
      metadata.update({
        'control_identity': 'static_static',
        'topology_mode': 'fixed',
        'factor_mode': 'fixed',
        'topology_weight': '0.0',
      })
      forged_adapter = root / 'relabeled.safetensors'
      save_file(load_file(output), forged_adapter, metadata=metadata)
      forged_manifest = dict(manifest)
      forged_manifest.update({
        'adapter_file': forged_adapter.name,
        'adapter_sha256': self._sha256(forged_adapter),
        'adapter_size_bytes': forged_adapter.stat().st_size,
      })
      forged_manifest_path = root / 'relabeled-manifest.json'
      forged_manifest_path.write_text(
        json.dumps(forged_manifest, indent=2, sort_keys=True) + '\n')
      expected_identity, _ = structured_decoder_identity_from_config(
        _structured_config())
      with self.assertRaisesRegex(
          ValueError, 'safetensors metadata differs'):
        load_and_validate_adapter_manifest(
          forged_manifest_path, forged_adapter,
          expected_identity=expected_identity,
          expected_adapter_sha256=self._sha256(forged_adapter),
          expected_manifest_sha256=self._sha256(forged_manifest_path))

  def test_manifest_hash_and_all_derivable_claims_are_verified(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest_path = root / 'manifest.json'
      manifest = export_adapter(
        checkpoint, output, manifest_path,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=RELEASE_TENSOR_COUNT,
        **ADAPTER_IDENTITY)
      expected_identity, _ = structured_decoder_identity_from_config(
        _structured_config())
      trusted_manifest_sha = self._sha256(manifest_path)

      manifest_path.write_text(manifest_path.read_text() + ' ')
      with self.assertRaisesRegex(ValueError, 'manifest SHA256 mismatch'):
        load_and_validate_adapter_manifest(
          manifest_path, output,
          expected_identity=expected_identity,
          expected_adapter_sha256=manifest['adapter_sha256'],
          expected_manifest_sha256=trusted_manifest_sha)

      float_tensor_schema = copy.deepcopy(manifest['tensor_schema'])
      first_tensor = next(iter(float_tensor_schema.values()))
      first_tensor['shape'] = [float(value) for value in first_tensor['shape']]
      cases = (
        (('schema_version',), 4.0),
        (('adapter_size_bytes',), float(manifest['adapter_size_bytes'])),
        (('source_checkpoint_sha256',), '0' * 64),
        (('source_checkpoint_size_bytes',), 1),
        (('source_checkpoint_global_step',), 999),
        (('source_state_dict_tensor_count',), 999),
        (('omitted_frozen_backbone_tensor_count',), 1),
        (('adapter_tensor_count',), float(manifest['adapter_tensor_count'])),
        (('adapter_parameter_count',),
         float(manifest['adapter_parameter_count'])),
        (('adapter_tensor_bytes',), float(manifest['adapter_tensor_bytes'])),
        (('tensor_schema',), {}),
        (('tensor_schema',), float_tensor_schema),
        (('adapter_namespace_in_source',), 'forged.*'),
        (('adapter_namespace_in_file',), 'forged'),
        (('released_backbone',), {}),
        (('released_backbone', 'source_size_bytes'),
         float(manifest['released_backbone']['source_size_bytes'])),
        (('required_loader',), 'forged_loader'),
      )
      for field_path, bad_value in cases:
        forged = copy.deepcopy(manifest)
        target = forged
        for component in field_path[:-1]:
          target = target[component]
        target[field_path[-1]] = bad_value
        forged_path = root / f'forged-{field_path[-1]}.json'
        forged_path.write_text(json.dumps(forged, sort_keys=True))
        with self.subTest(field=field_path), self.assertRaises(ValueError):
          load_and_validate_adapter_manifest(
            forged_path, output,
            expected_identity=expected_identity,
            expected_adapter_sha256=manifest['adapter_sha256'],
            expected_manifest_sha256=self._sha256(forged_path))

  def test_export_rejects_noncanonical_step_or_release_count(self):
    for global_step in (True, 7.0, '7'):
      with self.subTest(global_step=global_step), \
          tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'last.ckpt'
        self._checkpoint(checkpoint)
        payload = torch.load(
          checkpoint, map_location='cpu', weights_only=False)
        payload['global_step'] = global_step
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, 'non-negative integer'):
          export_adapter(
            checkpoint, root / 'adapter.safetensors', root / 'manifest.json',
            expected_checkpoint_sha256=self._sha256(checkpoint),
            expected_global_step=7,
            expected_backbone_tensors=RELEASE_TENSOR_COUNT,
            **ADAPTER_IDENTITY)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      with self.assertRaisesRegex(ValueError, 'pinned released-backbone'):
        export_adapter(
          checkpoint, root / 'adapter.safetensors', root / 'manifest.json',
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=7,
          expected_backbone_tensors=RELEASE_TENSOR_COUNT - 1,
          **ADAPTER_IDENTITY)

  def test_same_shape_tensor_byte_replacement_fails_trusted_hash(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest_path = root / 'manifest.json'
      manifest = export_adapter(
        checkpoint, output, manifest_path,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=RELEASE_TENSOR_COUNT,
        **ADAPTER_IDENTITY)
      trusted_adapter_sha = manifest['adapter_sha256']
      payload = bytearray(output.read_bytes())
      payload[-1] ^= 1
      output.write_bytes(payload)
      expected_identity, _ = structured_decoder_identity_from_config(
        _structured_config())
      with self.assertRaisesRegex(ValueError, 'adapter SHA256 mismatch'):
        load_and_validate_adapter_manifest(
          manifest_path, output,
          expected_identity=expected_identity,
          expected_adapter_sha256=trusted_adapter_sha,
          expected_manifest_sha256=self._sha256(manifest_path))

  def test_hash_verification_and_overwrite_protection(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest = root / 'manifest.json'
      export_adapter(
        checkpoint, output, manifest,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=RELEASE_TENSOR_COUNT,
        **ADAPTER_IDENTITY)
      with self.assertRaisesRegex(FileExistsError, '--force'):
        export_adapter(
          checkpoint, output, manifest,
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=7,
          expected_backbone_tensors=RELEASE_TENSOR_COUNT,
          **ADAPTER_IDENTITY)
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        load_adapter_state(output, expected_sha256='0' * 64)

  def test_checkpoint_hash_is_checked_before_deserialization(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'untrusted.ckpt'
      checkpoint.write_bytes(b'not a pickle checkpoint')
      with self.assertRaisesRegex(ValueError, 'checkpoint SHA256 mismatch'):
        export_adapter(
          checkpoint,
          root / 'adapter.safetensors',
          root / 'manifest.json',
          expected_checkpoint_sha256='0' * 64,
          expected_global_step=7,
          expected_backbone_tensors=RELEASE_TENSOR_COUNT,
          **ADAPTER_IDENTITY)


if __name__ == '__main__':
  unittest.main()
