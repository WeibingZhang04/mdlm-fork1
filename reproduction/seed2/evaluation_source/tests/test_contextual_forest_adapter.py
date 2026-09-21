from pathlib import Path
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

from safetensors import safe_open
from safetensors.torch import load_file
import torch

from scripts.export_contextual_forest_adapter import (
  BACKBONE_WRAPPER_METADATA,
  EXPECTATIONS_FIELDS,
  PRODUCTION_ADAPTER_KEYS,
  PRODUCTION_ADAPTER_PARAMETER_COUNT,
  PRODUCTION_ADAPTER_TENSOR_BYTES,
  PRODUCTION_ADAPTER_TENSOR_COUNT,
  build_export_cli_result,
  export_contextual_forest_adapter,
  load_authenticated_backbone_wrapper,
  load_production_expectations,
)
from scripts.export_structured_adapter import (
  RELEASED_BACKBONE_IDENTITY,
  RELEASE_TENSOR_COUNT,
  canonical_sha256,
  load_adapter_into_head,
  structured_decoder_identity_from_config,
  tensor_state_content_sha256,
  tensor_state_schema,
)
from scripts.replay_contextual_forest_adapter import (
  EXPECTED_RUNTIME_CONFIG,
  RUNTIME_CONFIG_IDENTITY_SHA256,
  _write_report_exclusive,
  authenticate_and_load_replay_model,
  attest_replay_sources,
  attest_repository,
  attest_runtime_environment,
  validate_seed_201_runtime_config,
)
from scripts.verify_contextual_forest_adapter import (
  verify_contextual_forest_adapter,
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


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
  structured.update(overrides)
  return structured


def _diffusion_class_for_loader_test():
  """Import the real Diffusion class without optional training dependencies."""

  class MetricStub(torch.nn.Module):

    def __init__(self, *args, **kwargs):
      del args, kwargs
      super().__init__()

  hydra = types.ModuleType('hydra')
  hydra.__path__ = []
  hydra_utils = types.ModuleType('hydra.utils')
  hydra.utils = hydra_utils
  lightning = types.ModuleType('lightning')
  lightning.LightningModule = torch.nn.Module
  torchmetrics = types.ModuleType('torchmetrics')
  torchmetrics.Metric = MetricStub
  torchmetrics.MetricCollection = MetricStub
  torchmetrics.aggregation = types.SimpleNamespace(MeanMetric=MetricStub)
  evaluation = types.ModuleType('evaluation')
  evaluation.conditional_denoising_records = types.ModuleType(
    'conditional_denoising_records')
  evaluation.causal_denoising = types.ModuleType('causal_denoising')
  stubs = {
    'hydra': hydra,
    'hydra.utils': hydra_utils,
    'lightning': lightning,
    'torchmetrics': torchmetrics,
    'evaluation': evaluation,
  }
  for name in (
      'crf_utils', 'dataloader', 'models', 'noise_schedule',
      'structured_objective', 'structured_pairing', 'structured_training',
      'utils'):
    stubs[name] = types.ModuleType(name)
  module_path = Path(__file__).resolve().parents[1] / 'diffusion.py'
  spec = importlib.util.spec_from_file_location(
    '_diffusion_structured_adapter_loader_test', module_path)
  if spec is None or spec.loader is None:
    raise AssertionError('could not construct diffusion module spec')
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, stubs):
    spec.loader.exec_module(module)
  return module.Diffusion


class TinyBackbone(torch.nn.Module):

  def __init__(self):
    super().__init__()
    for index in range(RELEASE_TENSOR_COUNT):
      self.register_buffer(
        f'tensor_{index:03d}', torch.tensor([float(index)]))


class TinyContextualForest(torch.nn.Module):

  def __init__(
      self,
      *,
      head: torch.nn.Module | None = None,
      structured_config=None,
  ):
    super().__init__()
    self.backbone = TinyBackbone()
    self.structured_head = head or torch.nn.Linear(3, 2)
    self.structured_config = structured_config or _structured_config()


class ContextualForestAdapterTest(unittest.TestCase):

  def _seed_201_config(self, *, cache: Path, output: Path):
    config = {}
    for dotted_path, value in EXPECTED_RUNTIME_CONFIG.items():
      current = config
      fields = dotted_path.split('.')
      for field in fields[:-1]:
        current = current.setdefault(field, {})
      current[fields[-1]] = copy.deepcopy(value)
    config['data']['cache_dir'] = str(cache.resolve())
    config['checkpointing']['save_dir'] = str(output.resolve())
    return config

  def test_production_inventory_matches_contextual_forest_small(self):
    from models.structured_decoder import ContextualCouplingForestHead

    head = ContextualCouplingForestHead(
      hidden_size=768,
      vocab_size=50_258,
      top_k=64,
      rank=16,
      time_embed_dim=64,
      topology_dim=128,
      local_window=2,
      num_anchor_slots=16,
      contextual_neighbors=4,
      component_size_cap=32,
      topology_mode='dynamic',
      factor_mode='dynamic',
      independent_mode=False,
      min_edge_score=None)
    state = head.state_dict()
    self.assertEqual(set(state), PRODUCTION_ADAPTER_KEYS)
    self.assertEqual(len(state), PRODUCTION_ADAPTER_TENSOR_COUNT)
    self.assertEqual(
      sum(value.numel() for value in state.values()),
      PRODUCTION_ADAPTER_PARAMETER_COUNT)
    self.assertEqual(
      sum(value.numel() * value.element_size() for value in state.values()),
      PRODUCTION_ADAPTER_TENSOR_BYTES)
    self.assertEqual({value.dtype for value in state.values()}, {torch.float32})

  def _checkpoint(
      self,
      path: Path,
      model: TinyContextualForest,
      *,
      adapter_state=None,
      extra_state=None,
  ):
    adapter_state = adapter_state or {
      'bias': torch.tensor([1.0, 2.0]),
      'weight': torch.arange(6.0).reshape(2, 3),
    }
    state = {
      **{
        f'backbone.{key}': value.detach().clone()
        for key, value in model.backbone.state_dict().items()
      },
      **{
        f'structured_head.{key}': value.detach().clone()
        for key, value in adapter_state.items()
      },
    }
    if extra_state:
      state.update(extra_state)
    torch.save({
      'global_step': 17,
      'hyper_parameters': {
        'config': {
          'model': {
            'structured_decoder': copy.deepcopy(model.structured_config),
          },
        },
      },
      'state_dict': state,
    }, path)
    return adapter_state

  def _export(self, root: Path, *, model=None, checkpoint=None):
    model = model or TinyContextualForest()
    checkpoint = checkpoint or root / 'last.ckpt'
    if not checkpoint.exists():
      self._checkpoint(checkpoint, model)
    adapter = root / 'adapter.safetensors'
    manifest = root / 'adapter-manifest.json'
    payload = export_contextual_forest_adapter(
      checkpoint,
      adapter,
      manifest,
      expected_checkpoint_sha256=_sha256(checkpoint),
      expected_global_step=17,
      model=model,
      expected_adapter_tensor_count=2,
      expected_adapter_parameter_count=8,
      expected_adapter_tensor_bytes=32,
      development_mode=True)
    return model, adapter, manifest, payload

  def _authenticated_expectations(
      self,
      root: Path,
      *,
      model: TinyContextualForest,
      checkpoint_sha256: str,
  ):
    wrapper_path = root / 'released-backbone-wrapper.pt'
    wrapper_state = {
      f'backbone.{key}': value.detach().clone()
      for key, value in model.backbone.state_dict().items()
    }
    torch.save({
      'metadata': dict(BACKBONE_WRAPPER_METADATA),
      'state_dict': wrapper_state,
    }, wrapper_path)
    normalized_state = {
      key.removeprefix('backbone.'): value
      for key, value in wrapper_state.items()
    }
    _, identity_sha = structured_decoder_identity_from_config(
      model.structured_config)
    wrapper_metadata = dict(BACKBONE_WRAPPER_METADATA)
    expectations = {
      'artifact_role': 'contextual_forest_adapter_expectations',
      'schema_version': 2,
      'source_checkpoint_sha256': checkpoint_sha256,
      'source_checkpoint_global_step': 17,
      'adapter_tensor_count': 2,
      'adapter_parameter_count': 8,
      'adapter_tensor_bytes': 32,
      'structured_decoder_identity_sha256': identity_sha,
      'released_backbone': dict(RELEASED_BACKBONE_IDENTITY),
      'backbone_wrapper': {
        'sha256': _sha256(wrapper_path),
        'size_bytes': wrapper_path.stat().st_size,
        'envelope_keys': ['metadata', 'state_dict'],
        'state_namespace': 'backbone.*',
        'tensor_count': len(normalized_state),
        'parameter_count': sum(
          value.numel() for value in normalized_state.values()),
        'tensor_bytes': sum(
          value.numel() * value.element_size()
          for value in normalized_state.values()),
        'tensor_schema_sha256': canonical_sha256(
          tensor_state_schema(normalized_state)),
        'tensor_content_sha256': tensor_state_content_sha256(
          normalized_state),
        'metadata': wrapper_metadata,
        'metadata_sha256': canonical_sha256(wrapper_metadata),
      },
    }
    expectations_path = root / 'expectations.json'
    expectations_path.write_text(
      json.dumps(expectations, indent=2, sort_keys=True) + '\n')
    authenticated = load_production_expectations(
      expectations_path, expected_sha256=_sha256(expectations_path))
    return wrapper_path, expectations_path, expectations, authenticated

  def _production_export(self, root: Path):
    model = TinyContextualForest()
    checkpoint = root / 'last.ckpt'
    self._checkpoint(checkpoint, model)
    _, _, expectations, authenticated = self._authenticated_expectations(
      root, model=model, checkpoint_sha256=_sha256(checkpoint))
    adapter = root / 'adapter.safetensors'
    manifest_path = root / 'adapter-manifest.json'
    manifest = export_contextual_forest_adapter(
      checkpoint,
      adapter,
      manifest_path,
      expected_checkpoint_sha256=_sha256(checkpoint),
      expected_global_step=17,
      model=model,
      expected_adapter_tensor_count=2,
      expected_adapter_parameter_count=8,
      expected_adapter_tensor_bytes=32,
      expectations=authenticated)
    return (
      model, adapter, manifest_path, manifest, expectations, authenticated)

  def test_export_is_deterministic_sorted_non_pickle_and_anonymous(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest()
      checkpoint = root / 'last.ckpt'
      expected = self._checkpoint(checkpoint, model)
      first = root / 'first'
      second = root / 'second'
      first.mkdir()
      second.mkdir()
      _, first_adapter, first_manifest, manifest = self._export(
        first, model=model, checkpoint=checkpoint)
      _, second_adapter, second_manifest, _ = self._export(
        second, model=model, checkpoint=checkpoint)

      self.assertEqual(first_adapter.read_bytes(), second_adapter.read_bytes())
      self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
      self.assertFalse(first_adapter.read_bytes().startswith(b'PK'))
      self.assertEqual(set(load_file(first_adapter)), set(expected))
      with safe_open(
          first_adapter, framework='pt', device='cpu') as handle:
        self.assertEqual(sorted(handle.keys()), list(handle.keys()))
        metadata = handle.metadata()
      self.assertEqual(metadata['source_namespace'], 'structured_head.')
      self.assertEqual(metadata['file_namespace'], 'prefix-stripped')
      self.assertEqual(
        metadata['source_checkpoint_sha256'], _sha256(checkpoint))
      self.assertEqual(manifest['adapter_tensor_count'], 2)
      self.assertEqual(manifest['adapter_parameter_count'], 8)
      self.assertEqual(manifest['adapter_tensor_bytes'], 32)
      self.assertEqual(
        manifest['released_backbone'], RELEASED_BACKBONE_IDENTITY)
      self.assertNotIn(str(root), first_manifest.read_text())
      self.assertNotIn('pickle', first_manifest.read_text().lower())

  def test_export_checks_checkpoint_hash_before_torch_load(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'untrusted.ckpt'
      checkpoint.write_bytes(b'not a trusted pickle')
      with mock.patch(
          'scripts.export_structured_adapter.torch.load') as load_mock:
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA256 mismatch'):
          export_contextual_forest_adapter(
            checkpoint,
            root / 'adapter.safetensors',
            root / 'manifest.json',
            expected_checkpoint_sha256='0' * 64,
            expected_global_step=17,
            model=TinyContextualForest(),
            expected_adapter_tensor_count=2,
            expected_adapter_parameter_count=8,
            expected_adapter_tensor_bytes=32,
            development_mode=True)
      load_mock.assert_not_called()

  def test_export_rejects_namespace_head_schema_and_inventory_mismatches(self):
    cases = (
      (
        'namespace',
        {'extra_state': {'noise.weight': torch.ones(1)}},
        'outside backbone'),
      (
        'key',
        {'adapter_state': {
          'weight': torch.zeros(2, 3), 'offset': torch.zeros(2)}},
        'key mismatch'),
      (
        'shape',
        {'adapter_state': {
          'weight': torch.zeros(3, 2), 'bias': torch.zeros(2)}},
        'shape mismatch'),
      (
        'dtype',
        {'adapter_state': {
          'weight': torch.zeros(2, 3, dtype=torch.float64),
          'bias': torch.zeros(2, dtype=torch.float64)}},
        'dtype mismatch'),
      (
        'nested_prefix',
        {'adapter_state': {
          'weight': torch.zeros(2, 3),
          'bias': torch.zeros(2),
          'backbone.hidden': torch.zeros(1)}},
        'forbidden namespace'),
    )
    for name, checkpoint_kwargs, message in cases:
      with self.subTest(name=name), \
          tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = TinyContextualForest()
        checkpoint = root / 'last.ckpt'
        self._checkpoint(checkpoint, model, **checkpoint_kwargs)
        with self.assertRaisesRegex(ValueError, message):
          self._export(root, model=model, checkpoint=checkpoint)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest()
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint, model)
      with self.assertRaisesRegex(ValueError, 'parameter_count mismatch'):
        export_contextual_forest_adapter(
          checkpoint,
          root / 'adapter.safetensors',
          root / 'manifest.json',
          expected_checkpoint_sha256=_sha256(checkpoint),
          expected_global_step=17,
          model=model,
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=9,
          expected_adapter_tensor_bytes=32,
          development_mode=True)

  def test_export_proves_frozen_backbone_values(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest()
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint, model)
      payload = torch.load(
        checkpoint, map_location='cpu', weights_only=False)
      payload['state_dict']['backbone.tensor_000'] += 1
      torch.save(payload, checkpoint)
      with self.assertRaisesRegex(ValueError, 'backbone value mismatch'):
        self._export(root, model=model, checkpoint=checkpoint)

  def test_production_export_and_verify_require_authenticated_expectations(self):
    root = Path('/not-opened')
    common = {
      'expected_adapter_tensor_count': 2,
      'expected_adapter_parameter_count': 8,
      'expected_adapter_tensor_bytes': 32,
    }
    with self.assertRaisesRegex(ValueError, 'expectations are required'):
      export_contextual_forest_adapter(
        root / 'checkpoint.ckpt',
        root / 'adapter.safetensors',
        root / 'manifest.json',
        expected_checkpoint_sha256='0' * 64,
        expected_global_step=17,
        model=TinyContextualForest(),
        **common)
    with self.assertRaisesRegex(ValueError, 'expectations are required'):
      verify_contextual_forest_adapter(
        root / 'adapter.safetensors',
        root / 'manifest.json',
        expected_adapter_sha256='0' * 64,
        expected_manifest_sha256='0' * 64,
        model=TinyContextualForest(),
        **common)

  def test_authenticated_wrapper_uses_one_weights_only_byte_snapshot(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest()
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint, model)
      wrapper_path, _, _, authenticated = self._authenticated_expectations(
        root, model=model, checkpoint_sha256=_sha256(checkpoint))
      original_read_bytes = Path.read_bytes
      original_wrapper_bytes = wrapper_path.read_bytes()
      resolved_wrapper_path = wrapper_path.resolve()

      def read_then_replace(path):
        payload = original_read_bytes(path)
        if path == resolved_wrapper_path:
          path.write_bytes(b'replaced-after-authenticated-read')
        return payload

      with mock.patch.object(Path, 'read_bytes', new=read_then_replace), \
          mock.patch(
            'scripts.export_contextual_forest_adapter.torch.load',
            wraps=torch.load) as load_mock:
        state, attestation = load_authenticated_backbone_wrapper(
          wrapper_path, expectations=authenticated)
      self.assertEqual(wrapper_path.read_bytes(), b'replaced-after-authenticated-read')
      self.assertEqual(
        attestation['backbone_wrapper_sha256'],
        hashlib.sha256(original_wrapper_bytes).hexdigest())
      self.assertEqual(set(state), set(model.backbone.state_dict()))
      self.assertEqual(load_mock.call_count, 1)
      self.assertIsInstance(load_mock.call_args.args[0], io.BytesIO)
      self.assertIs(load_mock.call_args.kwargs['weights_only'], True)

  def test_authenticated_wrapper_rejects_closed_metadata_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest()
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint, model)
      wrapper_path, _, expectations, _ = self._authenticated_expectations(
        root, model=model, checkpoint_sha256=_sha256(checkpoint))
      wrapper = torch.load(
        wrapper_path, map_location='cpu', weights_only=True)
      wrapper['metadata']['ema_used'] = True
      torch.save(wrapper, wrapper_path)
      expectations['backbone_wrapper']['sha256'] = _sha256(wrapper_path)
      expectations['backbone_wrapper']['size_bytes'] = (
        wrapper_path.stat().st_size)
      expectations_path = root / 'mutated-expectations.json'
      expectations_path.write_text(
        json.dumps(expectations, indent=2, sort_keys=True) + '\n')
      authenticated = load_production_expectations(
        expectations_path, expected_sha256=_sha256(expectations_path))
      with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
        load_authenticated_backbone_wrapper(
          wrapper_path, expectations=authenticated)

  def test_export_cli_result_contains_no_absolute_paths(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _, adapter, manifest_path, manifest = self._export(root)
      result = build_export_cli_result(adapter, manifest_path, manifest)
      rendered = json.dumps(result, sort_keys=True)
      self.assertNotIn(str(root), rendered)
      self.assertEqual(result['adapter']['filename'], adapter.name)
      self.assertEqual(result['manifest']['filename'], manifest_path.name)
      self.assertNotIn('path', result['adapter'])
      self.assertNotIn('path', result['manifest'])

  def test_export_rejects_local_fixed_edge_path_from_anonymous_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest(
        structured_config=_structured_config(
          topology_mode='fixed',
          factor_mode='fixed',
          fixed_edge_path='/private/training/topology.pt'))
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint, model)
      with self.assertRaisesRegex(ValueError, 'would disclose a local path'):
        self._export(root, model=model, checkpoint=checkpoint)

  def test_verifier_strict_loads_and_rejects_model_schema_mismatches(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _, adapter, manifest_path, manifest = self._export(root)
      manifest_sha = _sha256(manifest_path)
      expected_state = load_file(adapter)
      target = TinyContextualForest()
      result = verify_contextual_forest_adapter(
        adapter,
        manifest_path,
        expected_adapter_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=manifest_sha,
        model=target,
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32,
        development_mode=True)
      self.assertTrue(result['strict_load'])
      for key, value in expected_state.items():
        torch.testing.assert_close(target.structured_head.state_dict()[key], value)

      mismatched_models = (
        ('key', TinyContextualForest(
          head=torch.nn.Sequential(torch.nn.Linear(3, 2))), 'key mismatch'),
        ('shape', TinyContextualForest(
          head=torch.nn.Linear(4, 2)), 'shape mismatch'),
        ('dtype', TinyContextualForest(
          head=torch.nn.Linear(3, 2).double()), 'dtype mismatch'),
      )
      for name, mismatched, message in mismatched_models:
        with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
          verify_contextual_forest_adapter(
            adapter,
            manifest_path,
            expected_adapter_sha256=manifest['adapter_sha256'],
            expected_manifest_sha256=manifest_sha,
            model=mismatched,
            expected_adapter_tensor_count=2,
            expected_adapter_parameter_count=8,
            expected_adapter_tensor_bytes=32,
            development_mode=True)

  def test_verifier_rejects_identity_hash_and_backbone_tampering(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _, adapter, manifest_path, manifest = self._export(root)
      common = {
        'adapter_path': adapter,
        'manifest_path': manifest_path,
        'expected_adapter_sha256': manifest['adapter_sha256'],
        'expected_manifest_sha256': _sha256(manifest_path),
        'expected_adapter_tensor_count': 2,
        'expected_adapter_parameter_count': 8,
        'expected_adapter_tensor_bytes': 32,
        'development_mode': True,
      }
      with self.assertRaisesRegex(ValueError, 'identity differs'):
        verify_contextual_forest_adapter(
          model=TinyContextualForest(
            structured_config=_structured_config(top_k=128)),
          **common)
      with self.assertRaisesRegex(ValueError, 'adapter SHA256 mismatch'):
        verify_contextual_forest_adapter(
          model=TinyContextualForest(),
          **{**common, 'expected_adapter_sha256': '0' * 64})

      forged = json.loads(manifest_path.read_text())
      forged['released_backbone']['revision'] = '0' * 40
      forged_path = root / 'forged-manifest.json'
      forged_path.write_text(json.dumps(forged, sort_keys=True) + '\n')
      with self.assertRaisesRegex(ValueError, 'released-backbone identity'):
        verify_contextual_forest_adapter(
          adapter,
          forged_path,
          expected_adapter_sha256=manifest['adapter_sha256'],
          expected_manifest_sha256=_sha256(forged_path),
          model=TinyContextualForest(),
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=8,
          expected_adapter_tensor_bytes=32,
          development_mode=True)

  def test_legacy_main_head_and_diffusion_loaders_reject_schema_v5(self):
    diffusion_class = _diffusion_class_for_loader_test()
    main_source = (
      Path(__file__).resolve().parents[1] / 'main.py').read_text()
    self.assertIn('return diffusion.Diffusion(', main_source)
    self.assertNotIn('verify_contextual_forest_adapter', main_source)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model, adapter, manifest_path, manifest, _, _ = (
        self._production_export(root))
      identity, _ = structured_decoder_identity_from_config(
        model.structured_config)
      target = torch.nn.Linear(3, 2)
      with mock.patch.object(
          target, 'load_state_dict', wraps=target.load_state_dict) as load_mock:
        with self.assertRaisesRegex(
            ValueError, 'schema-v5.*verify_contextual_forest_adapter'):
          load_adapter_into_head(
            target,
            adapter,
            manifest_path=manifest_path,
            expected_identity=identity,
            expected_sha256=manifest['adapter_sha256'],
            expected_manifest_sha256=_sha256(manifest_path))
      load_mock.assert_not_called()

      for backbone_case in ('unattested', 'mutated'):
        with self.subTest(backbone_case=backbone_case):
          runtime = type('Runtime', (), {})()
          runtime.backbone = TinyBackbone()
          runtime.structured_backbone_provenance = None
          if backbone_case == 'mutated':
            runtime.backbone.tensor_000 += 1
          runtime.structured_head = torch.nn.Linear(3, 2)
          with mock.patch(
              'safetensors.torch.load') as tensor_load, mock.patch.object(
                runtime.structured_head,
                'load_state_dict',
                wraps=runtime.structured_head.load_state_dict) as head_load:
            with self.assertRaisesRegex(
                ValueError, 'schema-v4 adapters only'):
              diffusion_class._load_structured_adapter_checkpoint(
                runtime,
                str(adapter),
                expected_sha256=manifest['adapter_sha256'],
                manifest_path=str(manifest_path),
                expected_manifest_sha256=_sha256(manifest_path),
                structured_config=model.structured_config)
          tensor_load.assert_not_called()
          head_load.assert_not_called()

  def test_schema_v4_head_and_diffusion_loaders_remain_compatible(self):
    diffusion_class = _diffusion_class_for_loader_test()

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model, adapter, manifest_path, manifest = self._export(root)
      expected_state = load_file(adapter)
      identity, _ = structured_decoder_identity_from_config(
        model.structured_config)

      target = torch.nn.Linear(3, 2)
      load_adapter_into_head(
        target,
        adapter,
        manifest_path=manifest_path,
        expected_identity=identity,
        expected_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=_sha256(manifest_path))
      for key, value in expected_state.items():
        torch.testing.assert_close(target.state_dict()[key], value)

      runtime = type('Runtime', (), {})()
      runtime.structured_head = torch.nn.Linear(3, 2)
      diffusion_class._load_structured_adapter_checkpoint(
        runtime,
        str(adapter),
        expected_sha256=manifest['adapter_sha256'],
        manifest_path=str(manifest_path),
        expected_manifest_sha256=_sha256(manifest_path),
        structured_config=model.structured_config)
      self.assertEqual(runtime.structured_adapter_manifest['schema_version'], 4)
      for key, value in expected_state.items():
        torch.testing.assert_close(
          runtime.structured_head.state_dict()[key], value)

  def test_production_verifier_rejects_v4_and_accepts_attested_v5(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model, adapter, manifest_path, manifest, _, authenticated = (
        self._production_export(root))
      result = verify_contextual_forest_adapter(
        adapter,
        manifest_path,
        expected_adapter_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=_sha256(manifest_path),
        model=TinyContextualForest(),
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32,
        expectations=authenticated)
      self.assertEqual(manifest['schema_version'], 5)
      self.assertEqual(result['schema_version'], 1)
      self.assertTrue(result['strict_load'])

      checkpoint = root / 'last.ckpt'
      legacy_adapter = root / 'legacy-adapter.safetensors'
      legacy_manifest_path = root / 'legacy-adapter-manifest.json'
      legacy_manifest = export_contextual_forest_adapter(
        checkpoint,
        legacy_adapter,
        legacy_manifest_path,
        expected_checkpoint_sha256=_sha256(checkpoint),
        expected_global_step=17,
        model=model,
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32,
        development_mode=True)
      with self.assertRaisesRegex(ValueError, 'schema-v4 adapter'):
        verify_contextual_forest_adapter(
          legacy_adapter,
          legacy_manifest_path,
          expected_adapter_sha256=legacy_manifest['adapter_sha256'],
          expected_manifest_sha256=_sha256(legacy_manifest_path),
          model=TinyContextualForest(),
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=8,
          expected_adapter_tensor_bytes=32,
          expectations=authenticated)

  def test_authenticated_replay_accepts_only_attested_schema_v5(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _, adapter, manifest_path, manifest, _, authenticated = (
        self._production_export(root))
      runtime = TinyContextualForest()
      result = authenticate_and_load_replay_model(
        model=runtime,
        adapter_path=adapter,
        manifest_path=manifest_path,
        expected_adapter_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=_sha256(manifest_path),
        expectations=authenticated,
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32)
      self.assertTrue(result['strict_load'])
      self.assertEqual(manifest['schema_version'], 5)
      torch.testing.assert_close(
        runtime.structured_head.weight,
        torch.arange(6.0).reshape(2, 3))

      mutated = TinyContextualForest()
      mutated.backbone.tensor_000 += 1
      with mock.patch.object(
          mutated.structured_head,
          'load_state_dict',
          wraps=mutated.structured_head.load_state_dict) as load_mock:
        with self.assertRaisesRegex(ValueError, 'tensor content differs'):
          authenticate_and_load_replay_model(
            model=mutated,
            adapter_path=adapter,
            manifest_path=manifest_path,
            expected_adapter_sha256=manifest['adapter_sha256'],
            expected_manifest_sha256=_sha256(manifest_path),
            expectations=authenticated,
            expected_adapter_tensor_count=2,
            expected_adapter_parameter_count=8,
            expected_adapter_tensor_bytes=32)
      load_mock.assert_not_called()

      checkpoint = root / 'last.ckpt'
      legacy_adapter = root / 'legacy.safetensors'
      legacy_manifest_path = root / 'legacy-manifest.json'
      legacy_manifest = export_contextual_forest_adapter(
        checkpoint,
        legacy_adapter,
        legacy_manifest_path,
        expected_checkpoint_sha256=_sha256(checkpoint),
        expected_global_step=17,
        model=TinyContextualForest(),
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32,
        development_mode=True)
      with self.assertRaisesRegex(ValueError, 'schema-v4 adapter'):
        authenticate_and_load_replay_model(
          model=TinyContextualForest(),
          adapter_path=legacy_adapter,
          manifest_path=legacy_manifest_path,
          expected_adapter_sha256=legacy_manifest['adapter_sha256'],
          expected_manifest_sha256=_sha256(legacy_manifest_path),
          expectations=authenticated,
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=8,
          expected_adapter_tensor_bytes=32)

  def test_seed_201_runtime_config_and_report_are_fail_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      cache = root / 'cache'
      output = root / 'fresh-attempt'
      cache.mkdir()
      config = self._seed_201_config(cache=cache, output=output)
      identity = validate_seed_201_runtime_config(
        config, data_cache_dir=cache, output_dir=output)
      self.assertEqual(
        identity['identity_sha256'], RUNTIME_CONFIG_IDENTITY_SHA256)
      self.assertNotIn(str(root), json.dumps(identity, sort_keys=True))

      drifted = copy.deepcopy(config)
      drifted['seed'] = 202
      with self.assertRaisesRegex(ValueError, 'configuration drifted'):
        validate_seed_201_runtime_config(
          drifted, data_cache_dir=cache, output_dir=output)
      with self.assertRaisesRegex(ValueError, 'data cache path drifted'):
        validate_seed_201_runtime_config(
          config, data_cache_dir=root / 'other-cache', output_dir=output)

      output.mkdir()
      report = _write_report_exclusive(output, {'status': 'complete'})
      self.assertEqual(report.name, 'authenticated-seed-201-replay.json')
      with self.assertRaises(FileExistsError):
        _write_report_exclusive(output, {'status': 'replacement'})

  def test_seed_201_replay_rejects_repository_and_config_source_drift(self):
    revision = 'a' * 40
    with mock.patch(
        'scripts.replay_contextual_forest_adapter._git_output',
        side_effect=[revision, '']):
      self.assertEqual(
        attest_repository(revision),
        {'revision': revision, 'clean': True})
    with mock.patch(
        'scripts.replay_contextual_forest_adapter._git_output',
        return_value='b' * 40):
      with self.assertRaisesRegex(ValueError, 'revision mismatch'):
        attest_repository(revision)
    with mock.patch(
        'scripts.replay_contextual_forest_adapter._git_output',
        side_effect=[revision, ' M configs/config.yaml']):
      with self.assertRaisesRegex(ValueError, 'not clean'):
        attest_repository(revision)

    sources = attest_replay_sources()
    self.assertEqual(
      sources['model_config']['filename'], 'contextual-forest-small.yaml')
    self.assertEqual(
      sources['data_config']['filename'], 'openwebtext-streaming.yaml')
    with mock.patch(
        'scripts.replay_contextual_forest_adapter._sha256_file',
        return_value='0' * 64):
      with self.assertRaisesRegex(ValueError, 'model_config SHA256 mismatch'):
        attest_replay_sources()

  def test_seed_201_replay_authenticates_runtime_environment(self):
    versions = {
      'lightning': '2.2.1',
      'torchmetrics': '1.3.2',
      'transformers': '4.38.2',
      'datasets': '2.18.0',
      'hydra-core': '1.3.2',
      'omegaconf': '2.3.0',
      'safetensors': '0.4.2',
      'tokenizers': '0.15.2',
      'fsspec': '2024.2.0',
    }
    patches = (
      mock.patch(
        'scripts.replay_contextual_forest_adapter.'
        'importlib.metadata.version', side_effect=versions.__getitem__),
      mock.patch(
        'scripts.replay_contextual_forest_adapter.'
        'platform.python_version', return_value='3.10.12'),
      mock.patch(
        'scripts.replay_contextual_forest_adapter.'
        'platform.platform',
        return_value='Linux-6.8.0-1066-gcp-x86_64-with-glibc2.35'),
      mock.patch.object(torch, '__version__', '2.5.1+cu121'),
      mock.patch.object(torch.version, 'cuda', '12.1'),
      mock.patch(
        'scripts.replay_contextual_forest_adapter.'
        'subprocess.check_output',
        return_value='NVIDIA L4, 580.173.02, 23034\n'),
    )
    with (patches[0], patches[1], patches[2], patches[3], patches[4],
          patches[5]):
      environment = attest_runtime_environment()
    self.assertEqual(environment['gpu']['name'], 'NVIDIA L4')
    self.assertEqual(environment['packages'], versions)

    drifted = dict(versions)
    drifted['transformers'] = '99.0.0'
    with mock.patch(
        'scripts.replay_contextual_forest_adapter.'
        'importlib.metadata.version', side_effect=drifted.__getitem__):
      with self.assertRaisesRegex(RuntimeError, 'transformers drifted'):
        attest_runtime_environment()

  def test_authenticated_expectations_bind_source_config_and_inventory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      model = TinyContextualForest()
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint, model)
      _, _, expectations, authenticated = (
        self._authenticated_expectations(
          root, model=model, checkpoint_sha256=_sha256(checkpoint))
      )
      self.assertEqual(set(expectations), EXPECTATIONS_FIELDS)
      adapter = root / 'adapter.safetensors'
      manifest_path = root / 'adapter-manifest.json'
      manifest = export_contextual_forest_adapter(
        checkpoint,
        adapter,
        manifest_path,
        expected_checkpoint_sha256=_sha256(checkpoint),
        expected_global_step=17,
        model=model,
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32,
        expectations=authenticated)
      self.assertEqual(manifest['schema_version'], 5)
      provenance = manifest['production_provenance']
      self.assertEqual(
        provenance['production_expectations_file_sha256'],
        authenticated.file_sha256)
      self.assertEqual(
        provenance['production_expectations_identity_sha256'],
        authenticated.identity_sha256)
      result = verify_contextual_forest_adapter(
        adapter,
        manifest_path,
        expected_adapter_sha256=manifest['adapter_sha256'],
        expected_manifest_sha256=_sha256(manifest_path),
        model=TinyContextualForest(),
        expected_adapter_tensor_count=2,
        expected_adapter_parameter_count=8,
        expected_adapter_tensor_bytes=32,
        expectations=authenticated)
      self.assertEqual(
        result['source_checkpoint_sha256'],
        expectations['source_checkpoint_sha256'])
      self.assertEqual(
        result['production_expectations_file_sha256'],
        authenticated.file_sha256)
      self.assertEqual(
        result['production_expectations_identity_sha256'],
        authenticated.identity_sha256)
      self.assertNotIn(str(root), json.dumps(result, sort_keys=True))

      mutated_model = TinyContextualForest()
      mutated_model.backbone.tensor_000 += 1
      with self.assertRaisesRegex(ValueError, 'tensor content differs'):
        verify_contextual_forest_adapter(
          adapter,
          manifest_path,
          expected_adapter_sha256=manifest['adapter_sha256'],
          expected_manifest_sha256=_sha256(manifest_path),
          model=mutated_model,
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=8,
          expected_adapter_tensor_bytes=32,
          expectations=authenticated)

      with self.assertRaisesRegex(ValueError, 'tensor content differs'):
        export_contextual_forest_adapter(
          checkpoint,
          root / 'mutated.safetensors',
          root / 'mutated.json',
          expected_checkpoint_sha256=_sha256(checkpoint),
          expected_global_step=17,
          model=mutated_model,
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=8,
          expected_adapter_tensor_bytes=32,
          expectations=authenticated)

      equivalent_path = root / 'equivalent-expectations.json'
      equivalent_path.write_text(json.dumps(expectations, sort_keys=True))
      equivalent = load_production_expectations(
        equivalent_path, expected_sha256=_sha256(equivalent_path))
      self.assertEqual(equivalent.identity_sha256, authenticated.identity_sha256)
      self.assertNotEqual(equivalent.file_sha256, authenticated.file_sha256)
      with self.assertRaisesRegex(ValueError, 'production provenance differs'):
        verify_contextual_forest_adapter(
          adapter,
          manifest_path,
          expected_adapter_sha256=manifest['adapter_sha256'],
          expected_manifest_sha256=_sha256(manifest_path),
          model=TinyContextualForest(),
          expected_adapter_tensor_count=2,
          expected_adapter_parameter_count=8,
          expected_adapter_tensor_bytes=32,
          expectations=equivalent)


if __name__ == '__main__':
  unittest.main()
