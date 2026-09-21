#!/usr/bin/env python3
"""Evaluate sealed fresh-training choices once on document-disjoint test data.

Inputs are the development-only selection from fresh_pair_statistics, its
explicit checkpoint files, and the manifest from prepare_staged_fresh_data.
All input byte hashes are mandatory. A commitment is written before test
tokens are read. This evaluates conditional likelihood, not generation.

Head forwards retain FP32 and the training batch shape; forest inference and
residual normalization use FP64. The frozen released backbone always encodes
one example per call. No optimizer, checkpoint selection, or tuning runs here.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch

from evaluation.dependence_diagnostics import decompose_structured_log_probability
from evaluation.fresh_pair_statistics import ARMS, canonical_sha256, summarize_test
from models.contextual_unary import ContextualUnaryAdapter
from models.structured_decoder import StructuredDecoderOutput
from scripts.run_staged_fresh_training import (
  SOURCE_FILES, backbone_runtime_identity, online_backbone_batch, read_documents,
)
from scripts.run_staged_real_overfit import file_sha256, fixed_masks, make_head, token_digest
from structured_objective import (
  factorized_token_log_probability, infer_structured_distribution,
  structured_token_log_probability,
)

TEST_SEED_OFFSET = 1_900_000


def _is_sha(value):
  return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_authenticated(path: Path, expected: str) -> bytes:
  if not _is_sha(expected):
    raise ValueError('expected digest must be a lowercase SHA256')
  payload = path.read_bytes()
  if hashlib.sha256(payload).hexdigest() != expected:
    raise ValueError(f'{path.name}: SHA256 mismatch')
  return payload


def load_selection(path: Path, expected: str) -> dict:
  selection = json.loads(read_authenticated(path, expected))
  if (selection.get('artifact') != 'fresh_pair_checkpoint_selection'
      or selection.get('schema_version') != 1 or selection.get('selection_split') != 'dev'
      or selection.get('arms') != list(ARMS)):
    raise ValueError('expected schema-1 development-only checkpoint selection')
  payload = {key: value for key, value in selection.items() if key != 'selection_sha256'}
  if canonical_sha256(payload) != selection.get('selection_sha256'):
    raise ValueError('checkpoint selection digest mismatch')
  seeds, rates = selection['training_seeds'], selection['mask_rates']
  if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
      or seeds != sorted(set(seeds)) or not rates
      or any(type(rate) not in (int, float) or not 0 < rate < 1 for rate in rates)
      or rates != sorted(set(rates))):
    raise ValueError('invalid selection seed or mask-rate grid')
  selected = selection['selected']
  expected_pairs = {(seed, arm) for seed in seeds for arm in ARMS}
  if (len(selected) != len(expected_pairs)
      or {(row['training_seed'], row['arm']) for row in selected} != expected_pairs):
    raise ValueError('selection must cover each arm and seed exactly once')
  for row in selected:
    if (type(row['checkpoint_step']) is not int or row['checkpoint_step'] < 0
        or not _is_sha(row['checkpoint_sha256'])):
      raise ValueError('invalid selected checkpoint step/hash')
  if not set(selection['dev_document_ids']) <= set(selection['excluded_test_document_ids']):
    raise ValueError('selection does not exclude its development documents')
  return selection


def load_split_manifest(path: Path, expected: str, selection: dict) -> dict:
  manifest = json.loads(read_authenticated(path, expected))
  if manifest.get('artifact') != 'fresh_staged_openwebtext_splits' or manifest.get('schema_version') != 1:
    raise ValueError('expected fresh staged source-document split manifest')
  payload = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
  if canonical_sha256(payload) != manifest.get('manifest_sha256'):
    raise ValueError('data manifest canonical digest mismatch')
  groups = {split: manifest['splits'][split]['source_document_sha256'] for split in ('train', 'dev', 'test')}
  windows = {split: manifest['splits'][split]['prefix_token_ids_sha256'] for split in groups}
  for split, documents in groups.items():
    if (not documents or len(documents) != len(set(documents))
        or len(documents) != manifest['counts'][split]
        or any(not _is_sha(value) for value in documents)
        or len(windows[split]) != len(documents)
        or len(set(windows[split])) != len(documents)
        or any(not _is_sha(value) for value in windows[split])):
      raise ValueError(f'invalid {split} document identities in manifest')
  forbidden = set(groups['train']) | set(groups['dev']) | set(manifest['excluded_old_debug_document_sha256'])
  if set(groups['train']) & set(groups['dev']) or set(groups['test']) & forbidden:
    raise ValueError('data manifest contains overlapping source documents')
  if (set(windows['train']) & set(windows['dev'])
      or set(windows['test']) & (set(windows['train']) | set(windows['dev']))):
    raise ValueError('data manifest contains overlapping token windows')
  # Exclude the complete prepared train/dev sets, even when a pilot trained
  # on only their prefixes. The selected test set is always evaluated in full.
  if not forbidden <= set(selection['excluded_test_document_ids']):
    raise ValueError('selection omits prepared training, development or old-debug exclusions')
  return manifest


def load_selected_heads(paths, selection, manifest):
  """Authenticate before deserialization, then enforce exact state schemas."""
  from scripts.export_structured_adapter import validate_tensor_state_against_module
  wanted = {row['checkpoint_sha256'] for row in selection['selected']}
  supplied = {}
  for path in paths:
    digest = file_sha256(path)
    if digest not in wanted or digest in supplied:
      raise ValueError('supply each distinct selected checkpoint exactly once, with no extras')
    supplied[digest] = path
  if set(supplied) != wanted:
    raise ValueError('a selected checkpoint file is missing')
  heads, metadata = {}, {}
  for digest, path in supplied.items():
    # Read the exact authenticated bytes, avoiding a hash/load path race.
    payload = read_authenticated(path, digest)
    # Older pilot metadata serialized the harmless TorchVersion str subclass.
    # Do not weaken weights_only or allow arbitrary checkpoint-defined types.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
      state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    del payload
    if state.get('schema_version') != 1 or set(state.get('heads', {})) != set(ARMS):
      raise ValueError('invalid fresh-training checkpoint schema or arms')
    identity, config = state['identity'], state['config']
    seed = config['seed']
    if (config['backbone_batch_size'] != 1 or sorted(config['mask_rates']) != selection['mask_rates']
        or len(config['mask_rates']) != len(selection['mask_rates'])):
      raise ValueError('checkpoint must use the selected rate grid and serial backbone')
    if identity['head_config']['seed'] != seed or identity['head_config']['component_size_cap'] != 0:
      raise ValueError('checkpoint head config has an inconsistent seed or graph cap')
    for split in ('train', 'dev'):
      source = identity[f'{split}_source']
      count = source['examples']
      documents = manifest['splits'][split]['source_document_sha256'][:count]
      if (source['file_sha256'] != manifest['file_sha256'][f'{split}.jsonl']
          or source['length'] != manifest['length'] or count != len(documents)
          or source['document_ids_sha256'] != canonical_sha256(documents)):
        raise ValueError(f'checkpoint {split} data identity differs from split manifest')
      if split == 'dev' and set(documents) != set(selection['dev_document_ids']):
        raise ValueError('selected development documents differ from checkpoint data')
    pinned = identity['source_sha256']
    if set(pinned) != set(SOURCE_FILES):
      raise ValueError('checkpoint implementation source manifest is incomplete')
    for source_path in SOURCE_FILES:
      if file_sha256(REPO_ROOT / source_path) != pinned[source_path]:
        raise ValueError(f'checkpoint implementation source changed: {source_path}')
    current = {'identity': identity, 'config': config}
    if seed in metadata and canonical_sha256(current) != canonical_sha256(metadata[seed]):
      raise ValueError('selected states for one seed have different training identities/configs')
    metadata[seed] = current
    for chosen in selection['selected']:
      if chosen['checkpoint_sha256'] != digest:
        continue
      if chosen['training_seed'] != seed or chosen['checkpoint_step'] != state['completed_steps']:
        raise ValueError('selected seed/step differs from checkpoint contents')
      arm = chosen['arm']
      head = make_head(arm, **identity['head_config'])
      tensors = state['heads'][arm]
      validate_tensor_state_against_module(tensors, head, context=f'{seed}/{arm}')
      if any(not bool(torch.isfinite(value).all()) for value in tensors.values() if value.is_floating_point()):
        raise ValueError('selected head contains nonfinite parameters')
      head.load_state_dict(tensors, strict=True)
      head.requires_grad_(False).eval()
      heads[(seed, arm)] = head
    del state
  if set(heads) != {(seed, arm) for seed in selection['training_seeds'] for arm in ARMS}:
    raise ValueError('loaded heads do not cover selection')
  return heads, metadata


def verify_backbone(model, metadata, device):
  """Bind the selected state to the authenticated released encoder wrapper."""
  device_type = torch.device(device).type
  for info in metadata.values():
    identity = info['identity']
    if canonical_sha256(identity['backbone_provenance']) != canonical_sha256(model.structured_backbone_provenance):
      raise ValueError('backbone provenance differs from the training checkpoint')
    if canonical_sha256(identity['backbone_runtime_config']) != canonical_sha256(backbone_runtime_identity(model)):
      raise ValueError('resolved backbone/noise configuration differs from training')
    head_config = identity['head_config']
    if (head_config['hidden_size'] != model.structured_head.hidden_size
        or head_config['vocab_size'] != model.structured_head.vocab_size):
      raise ValueError('backbone/head dimensions differ')
    runtime = identity['runtime']
    if (str(runtime['torch_version']) != str(torch.__version__)
        or runtime['device_type'] != device_type
        or runtime['cached_output_dtype'] != 'float32'
        or runtime['backbone_cuda_autocast'] != 'bfloat16'):
      raise ValueError('test runtime differs from the training precision/runtime contract')
    if device_type == 'cuda' and runtime['gpu_name'] != torch.cuda.get_device_name(device):
      raise ValueError('test GPU type differs from the training checkpoint')
  model.requires_grad_(False).eval()
  if any(parameter.requires_grad for parameter in model.parameters()):
    raise AssertionError('backbone evaluation model is not strictly frozen')


class _Float64ResidualOutput(StructuredDecoderOutput):
  """Same forward distribution, using FP64 residual decoder arithmetic."""

  def residual_log_probs(self, unary_logits):
    tail = unary_logits.double().clone()
    tail.scatter_(-1, self.candidate_ids, -torch.inf)
    return tail - tail.logsumexp(-1, keepdim=True)


@torch.no_grad()
def score_head_float64(head, batch):
  """Preserve FP32 forward/support, refine only probability arithmetic."""
  args = batch['hidden'], batch['logits'], batch['sigma'], batch['active']
  targets, active = batch['targets'], batch['active']
  expected_ids = batch['logits'].topk(head.top_k, dim=-1).indices
  if isinstance(head, ContextualUnaryAdapter):
    lattice = head.candidate_lattice(*args)
    ids = lattice.candidate_ids
    matches = ids == targets[..., None]
    explicit = matches.any(-1)
    states = torch.where(explicit, matches.long().argmax(-1), torch.full_like(targets, ids.shape[-1]))
    node = lattice.unary_log_potentials.double().log_softmax(-1).gather(-1, states[..., None]).squeeze(-1)
    tail = batch['logits'].double().clone()
    tail.scatter_(-1, ids, -torch.inf)
    correction = batch['logits'].gather(-1, targets[..., None]).squeeze(-1).double() - tail.logsumexp(-1)
    node = node + torch.where(active & ~explicit, correction, torch.zeros_like(correction))
    marginal = torch.where(active, node, torch.zeros_like(node)).sum(-1)
    joint, dependence = marginal, torch.zeros_like(marginal)
    error = torch.zeros_like(marginal)
    edges = torch.zeros_like(active.sum(-1))
  else:
    original = head(*args, factor_mode='dynamic')
    fields = {field.name: getattr(original, field.name) for field in dataclasses.fields(original)}
    for name in ('unary_log_potentials', 'pair_left_factors', 'pair_right_factors'):
      fields[name] = fields[name].double()
    output = _Float64ResidualOutput(**fields)
    ids = output.candidate_ids
    inference = infer_structured_distribution(output, active, backend='low_rank')
    decomposition = decompose_structured_log_probability(output, batch['logits'], targets, active, inference)
    joint = structured_token_log_probability(output, batch['logits'], targets, active, inference)
    marginal, dependence = decomposition.marginal_log_probability, decomposition.dependence_log_probability
    error = (joint - marginal - dependence).abs()
    torch.testing.assert_close(joint, marginal + dependence, atol=1e-8, rtol=1e-10)
    edges = output.edge_mask.sum(-1)
  if not torch.equal(ids, expected_ids):
    raise AssertionError('head changed the target-independent base candidate support')
  for score in (joint, marginal, dependence):
    if not bool(torch.isfinite(score).all()):
      raise FloatingPointError('nonfinite held-out log probability')
  return {'joint_log_probability': joint, 'marginal_log_probability': marginal,
          'dependence_log_probability': dependence, 'decomposition_error_nats': error,
          'edge_count': edges, 'candidate_ids': ids,
          'explicit_target_count': ((ids == targets[..., None]).any(-1) & active).sum(-1)}


def validate_test_tokens(tokens, documents, selection, manifest):
  expected = manifest['splits']['test']['source_document_sha256']
  if documents != expected:
    raise ValueError('test documents differ from the full prespecified source-order split')
  if [canonical_sha256(row.tolist()) for row in tokens] != manifest['splits']['test']['prefix_token_ids_sha256']:
    raise ValueError('test token prefixes differ from the prespecified split')
  forbidden = set(selection['excluded_test_document_ids'])
  windows = set(selection['dev_clean_token_sha256s'])
  if set(documents) & forbidden or any(token_digest(row) in windows for row in tokens):
    raise ValueError('test reuses an excluded document or development token window')


@torch.no_grad()
def evaluate_selected(model, heads, metadata, tokens, documents, selection, *, device, record_sink=None):
  """Evaluate all selected arms on each shared corruption; never fit anything."""
  records = []
  choices = {(row['training_seed'], row['arm']): row for row in selection['selected']}
  for seed in selection['training_seeds']:
    info = metadata[seed]
    batch_size = info['config']['batch_size']
    seed_heads = {arm: heads[(seed, arm)].to(device).eval() for arm in ARMS}
    # Use the training rate order, which also fixes the held-out seed mapping.
    for rate_index, rate in enumerate(info['config']['mask_rates']):
      corruption_seed = seed + TEST_SEED_OFFSET + rate_index
      masks = fixed_masks(tokens, rate, corruption_seed)
      for offset in range(0, len(tokens), batch_size):
        clean, active = tokens[offset:offset + batch_size], masks[offset:offset + batch_size]
        batch = online_backbone_batch(model, clean, active, rate, device, backbone_batch_size=1)
        # Calculate this once, in FP64, then repeat identical numbers in all arms.
        base = factorized_token_log_probability(batch['logits'].double(), batch['targets'], batch['active'])
        support = None
        for arm in ARMS:
          scores = score_head_float64(seed_heads[arm], batch)
          if support is not None and not torch.equal(support, scores['candidate_ids']):
            raise AssertionError('paired arms do not share candidate support')
          support = scores['candidate_ids']
          for local in range(len(clean)):
            index = offset + local
            row = {
              'split': 'test', 'arm': arm, 'training_seed': seed,
              'checkpoint_step': choices[(seed, arm)]['checkpoint_step'],
              'checkpoint_sha256': choices[(seed, arm)]['checkpoint_sha256'],
              'selection_sha256': selection['selection_sha256'],
              'dataset': info['identity']['dataset'], 'document_id': documents[index],
              'example_index': index, 'mask_rate': rate, 'corruption_seed': corruption_seed,
              'clean_token_sha256': token_digest(clean[local]),
              'mask_sha256': token_digest(active[local]),
              'corrupted_token_sha256': token_digest(batch['corrupted'][local]),
              'candidate_ids_sha256': token_digest(scores['candidate_ids'][local]),
              'active_token_count': int(active[local].sum()),
              'actual_mask_fraction': float(active[local].float().mean()),
              'diffusion_time': rate / (1.0 - model.noise.eps),
              'sigma': float(batch['sigma'][local]),
              'backbone_log_probability': float(base[local]),
              **{key: float(scores[key][local]) for key in (
                'joint_log_probability', 'marginal_log_probability',
                'dependence_log_probability', 'decomposition_error_nats')},
              'edge_count': int(scores['edge_count'][local]),
              'explicit_target_count': int(scores['explicit_target_count'][local]),
            }
            canonical_sha256(row)  # Reject nonfinite JSON before recording it.
            records.append(row)
            if record_sink is not None:
              record_sink(row)
        del batch, scores, support
    for head in seed_heads.values():
      head.cpu()
  return records


def _write_json(path, value):
  with path.open('x') as handle:
    json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write('\n')


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('selection', 'data-manifest', 'test-jsonl', 'expectations'):
    parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument(f'--expected-{name}-sha256', required=True)
  parser.add_argument('--selected-checkpoint', type=Path, action='append', required=True)
  parser.add_argument('--backbone-checkpoint', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--data-cache-dir', type=Path)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--bootstrap-replicates', type=int, default=5000)
  return parser.parse_args(argv)


def main(argv=None):
  args = _args(argv)
  if args.threads < 1 or args.bootstrap_replicates < 1:
    raise ValueError('thread/bootstrap counts must be positive')
  selection = load_selection(args.selection, args.expected_selection_sha256)
  manifest = load_split_manifest(args.data_manifest, args.expected_data_manifest_sha256, selection)
  if args.expected_test_jsonl_sha256 != manifest['file_sha256']['test.jsonl']:
    raise ValueError('expected test hash differs from prespecified data manifest')
  heads, metadata = load_selected_heads(args.selected_checkpoint, selection, manifest)
  torch.set_num_threads(args.threads)
  if torch.device(args.device).type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
  from scripts.export_contextual_forest_adapter import build_production_model, load_production_expectations
  expectations = load_production_expectations(args.expectations, expected_sha256=args.expected_expectations_sha256)
  # Reserve this output namespace before expensive model loading and refuse
  # reruns into an existing result. No test file has been opened at this point.
  args.output_dir.mkdir(parents=True, exist_ok=False)
  model = build_production_model(
    model_config='contextual-forest-small', data_config='train_openwebtext_pinned',
    backbone_checkpoint=args.backbone_checkpoint, expectations=expectations,
    overrides=[f'model.length={manifest["length"]}'], runtime_mode='ppl_eval',
    data_cache_dir=args.data_cache_dir, checkpoint_save_dir=args.output_dir).to(args.device).eval()
  verify_backbone(model, metadata, args.device)
  commitment = {
    'artifact': 'fresh_pair_test_commitment', 'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'test_tokens_read': False,
    'selection_sha256': selection['selection_sha256'],
    'selection_file_sha256': args.expected_selection_sha256,
    'data_manifest_file_sha256': args.expected_data_manifest_sha256,
    'test_file_sha256': args.expected_test_jsonl_sha256,
    'selected': selection['selected'], 'test_examples': manifest['counts']['test'],
    'test_seed_offset': TEST_SEED_OFFSET, 'seed_mapping': 'training seed + offset + training rate-list index',
    'backbone_batch_size': 1, 'head_forward_dtype': 'float32',
    'forest_inference_dtype': 'float64', 'residual_normalization_dtype': 'float64',
    'backbone_provenance': model.structured_backbone_provenance,
    'source_sha256': {path: file_sha256(REPO_ROOT / path) for path in (
      'scripts/evaluate_staged_fresh_selected.py', 'evaluation/fresh_pair_statistics.py')},
    'training_identity_sha256_by_seed': {str(seed): canonical_sha256(info) for seed, info in metadata.items()},
    'bootstrap_replicates': args.bootstrap_replicates, 'bootstrap_seed': 1701,
    'interpretation': 'conditional likelihood on a source-order pilot; not generation quality',
  }
  _write_json(args.output_dir / 'commitment.json', commitment)
  tokens, documents, source = read_documents(
    args.test_jsonl, args.expected_test_jsonl_sha256, manifest['counts']['test'], manifest['length'],
    model.structured_head.vocab_size, model.mask_index, model.tokenizer)
  validate_test_tokens(tokens, documents, selection, manifest)
  with (args.output_dir / 'test-records.jsonl').open('x') as handle:
    def record_sink(row):
      handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
      handle.flush()
    records = evaluate_selected(model, heads, metadata, tokens, documents, selection,
                                device=args.device, record_sink=record_sink)
  summary = summarize_test(records, selection, bootstrap_replicates=args.bootstrap_replicates)
  summary['test_source'] = source
  summary['commitment_file_sha256'] = file_sha256(args.output_dir / 'commitment.json')
  summary['test_records_file_sha256'] = file_sha256(args.output_dir / 'test-records.jsonl')
  summary['max_decomposition_error_nats'] = max(row['decomposition_error_nats'] for row in records)
  _write_json(args.output_dir / 'results.json', summary)
  print(json.dumps({'event': 'test_evaluation_complete', 'output_dir': str(args.output_dir),
                    'records': len(records), 'selection_sha256': selection['selection_sha256']}), flush=True)


if __name__ == '__main__':
  main()
