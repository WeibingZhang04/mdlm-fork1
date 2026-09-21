"""Paired, provenance-oriented sampling helpers for diffusion generation.

This module deliberately does not instantiate a model or parse Hydra config.
It turns explicit prompt specifications into initial diffusion states, invokes
the existing sampling kernels, and returns JSON-serializable records.  The CLI
entry point lives in ``scripts/run_generation_pilot.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import random
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from evaluation.generation_metrics import (
  paired_token_metrics,
  repetition_rate,
  summarize_token_metrics,
)


DEFAULT_SAMPLING_MODES = (
  'factorized',
  'structured_marginal',
  'structured_joint',
)
SAMPLING_MODES = (
  DEFAULT_SAMPLING_MODES[0],
  'factorized_confidence_gated',
  *DEFAULT_SAMPLING_MODES[1:],
)
STRUCTURED_SAMPLING_MODES = frozenset({
  'structured_marginal',
  'structured_joint',
})


@dataclass(frozen=True)
class PromptSpec:
  """One fixed initial diffusion state shared by every sampling arm."""

  prompt_id: str
  initial_token_ids: tuple[int, ...]
  active_mask: tuple[bool, ...]
  reference_token_ids: tuple[int, ...] | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.prompt_id:
      raise ValueError('prompt_id must be non-empty')
    if not self.initial_token_ids:
      raise ValueError('initial_token_ids must be non-empty')
    if len(self.initial_token_ids) != len(self.active_mask):
      raise ValueError('initial_token_ids and active_mask lengths differ')
    if not any(self.active_mask):
      raise ValueError('active_mask must select at least one token')
    if (self.reference_token_ids is not None
        and len(self.reference_token_ids) != len(self.initial_token_ids)):
      raise ValueError('reference_token_ids length differs from prompt length')


@dataclass(frozen=True)
class PairedSampleSpec:
  """A prompt replicate with a seed that is invariant across sampling arms."""

  sample_index: int
  pair_key: str
  pair_seed: int
  prompt: PromptSpec


def stable_sha256(payload: Any) -> str:
  """Hash a JSON-compatible object under a canonical serialization."""
  encoded = json.dumps(
    payload, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


def _encode_text(tokenizer, text: str) -> list[int]:
  encoded = tokenizer(
    text, add_special_tokens=False, return_attention_mask=False)
  values = encoded['input_ids'] if isinstance(encoded, dict) else encoded.input_ids
  if values and isinstance(values[0], list):
    values = values[0]
  return [int(token) for token in values]


def _padding_token_id(tokenizer) -> int:
  for name in ('eos_token_id', 'pad_token_id'):
    value = getattr(tokenizer, name, None)
    if value is not None:
      return int(value)
  raise ValueError('tokenizer has neither eos_token_id nor pad_token_id')


def prompt_from_record(
    record: dict[str, Any],
    *,
    tokenizer,
    mask_token_id: int,
    sequence_length: int,
    line_number: int,
) -> PromptSpec:
  """Parse one JSONL prompt using one of three unambiguous schemas.

  Supported schemas are:

  * explicit ``input_ids`` + ``active_mask`` (+ optional full reference ids),
  * ``text`` + ``mask_token_indices`` for deterministic infilling, and
  * ``prompt`` for prefix-conditioned generation to ``sequence_length``.
  """
  if sequence_length <= 0:
    raise ValueError('sequence_length must be positive')
  prompt_id = str(record.get('id', f'line-{line_number}'))
  metadata = dict(record.get('metadata', {}))

  if 'input_ids' in record:
    input_ids = tuple(int(value) for value in record['input_ids'])
    active_mask = tuple(bool(value) for value in record.get('active_mask', ()))
    if len(input_ids) != sequence_length:
      raise ValueError(
        f'{prompt_id}: input_ids must have sequence_length '
        f'{sequence_length}, found {len(input_ids)}')
    if len(active_mask) != sequence_length:
      raise ValueError(
        f'{prompt_id}: active_mask must have sequence_length '
        f'{sequence_length}, found {len(active_mask)}')
    if any(
        token == mask_token_id and not active
        for token, active in zip(input_ids, active_mask)):
      raise ValueError(
        f'{prompt_id}: mask token appears at a position not selected by '
        'active_mask')
    reference = record.get('reference_token_ids')
    reference_ids = (
      tuple(int(value) for value in reference)
      if reference is not None else None)
    initial = tuple(
      mask_token_id if active else token
      for token, active in zip(input_ids, active_mask))
    return PromptSpec(
      prompt_id=prompt_id,
      initial_token_ids=initial,
      active_mask=active_mask,
      reference_token_ids=reference_ids,
      metadata=metadata)

  if 'text' in record:
    if 'mask_token_indices' not in record:
      raise ValueError(f'{prompt_id}: text schema requires mask_token_indices')
    encoded = _encode_text(tokenizer, str(record['text']))[:sequence_length]
    valid_length = len(encoded)
    reference = encoded + [
      _padding_token_id(tokenizer)
    ] * (sequence_length - valid_length)
    mask_indices = [int(index) for index in record['mask_token_indices']]
    if len(mask_indices) != len(set(mask_indices)):
      raise ValueError(f'{prompt_id}: duplicate mask_token_indices')
    if any(index < 0 or index >= valid_length for index in mask_indices):
      raise ValueError(
        f'{prompt_id}: mask_token_indices must address encoded text '
        f'positions [0, {valid_length})')
    active_mask = tuple(index in set(mask_indices)
                        for index in range(sequence_length))
    initial = tuple(
      mask_token_id if active else token
      for token, active in zip(reference, active_mask))
    metadata.update({
      'schema': 'text_infilling',
      'encoded_text_length': valid_length,
    })
    return PromptSpec(
      prompt_id=prompt_id,
      initial_token_ids=initial,
      active_mask=active_mask,
      reference_token_ids=tuple(reference),
      metadata=metadata)

  if 'prompt' in record:
    prefix = _encode_text(tokenizer, str(record['prompt']))
    if len(prefix) >= sequence_length:
      raise ValueError(
        f'{prompt_id}: prompt encodes to {len(prefix)} tokens, leaving '
        'no position to generate')
    initial = tuple(prefix + [mask_token_id] * (sequence_length - len(prefix)))
    active_mask = tuple(
      index >= len(prefix) for index in range(sequence_length))
    metadata.update({
      'schema': 'prefix_generation',
      'prefix_token_count': len(prefix),
    })
    return PromptSpec(
      prompt_id=prompt_id,
      initial_token_ids=initial,
      active_mask=active_mask,
      reference_token_ids=None,
      metadata=metadata)

  raise ValueError(
    f'{prompt_id}: expected input_ids, text, or prompt schema')


def unconditional_prompt(
    *,
    mask_token_id: int,
    sequence_length: int,
) -> PromptSpec:
  return PromptSpec(
    prompt_id='unconditional',
    initial_token_ids=(int(mask_token_id),) * sequence_length,
    active_mask=(True,) * sequence_length,
    metadata={'schema': 'unconditional'})


def expand_paired_samples(
    prompts: Sequence[PromptSpec],
    *,
    num_samples: int,
    base_seed: int,
) -> list[PairedSampleSpec]:
  """Cycle prompt records deterministically to create a fixed paired pilot."""
  if not prompts:
    raise ValueError('at least one prompt is required')
  if num_samples <= 0:
    raise ValueError('num_samples must be positive')
  result = []
  replicate_counts = {prompt.prompt_id: 0 for prompt in prompts}
  for sample_index in range(num_samples):
    prompt = prompts[sample_index % len(prompts)]
    replicate = replicate_counts[prompt.prompt_id]
    replicate_counts[prompt.prompt_id] += 1
    result.append(PairedSampleSpec(
      sample_index=sample_index,
      pair_key=f'{prompt.prompt_id}/replicate-{replicate:04d}',
      pair_seed=int(base_seed + sample_index),
      prompt=prompt))
  return result


def pairing_digest(samples: Sequence[PairedSampleSpec]) -> str:
  """Cryptographic commitment to ordered prompts, masks, and pair seeds."""
  return stable_sha256([
    {
      'sample_index': item.sample_index,
      'pair_key': item.pair_key,
      'pair_seed': item.pair_seed,
      'prompt_id': item.prompt.prompt_id,
      'initial_token_ids': item.prompt.initial_token_ids,
      'active_mask': item.prompt.active_mask,
      'reference_token_ids': item.prompt.reference_token_ids,
      'prompt_metadata': item.prompt.metadata,
    }
    for item in samples
  ])


def batch_seed(samples: Sequence[PairedSampleSpec]) -> int:
  """Stable batch RNG seed; batch size is therefore provenance-relevant."""
  digest = stable_sha256([
    {'pair_key': item.pair_key, 'pair_seed': item.pair_seed}
    for item in samples
  ])
  return int(digest[:16], 16) % (2 ** 63 - 1)


def seed_everything(seed: int, device: torch.device) -> None:
  random.seed(seed)
  np.random.seed(seed % (2 ** 32))
  torch.manual_seed(seed)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)


def _finish_sampling(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Apply the configured final denoising evaluation exactly as Diffusion._sample."""
  if model.sampler == 'analytic':
    return model._denoiser_update(x, t)
  if model.config.backbone == 'crf_dit':
    conditioning = model.noise(t)[0]
    sigma_1d = model._process_sigma(conditioning)
    return model._compute_crf_marginals(x, sigma_1d).argmax(dim=-1)
  if (model.structured_enabled
      and model.structured_sampling_mode in STRUCTURED_SAMPLING_MODES):
    conditioning = model.noise(t)[0]
    return model._structured_clean_sample(x, conditioning)
  conditioning = model.noise(t)[0]
  return model.forward(x, conditioning).argmax(dim=-1)


def _normalize_snapshot_call_indices(
    values: Iterable[int],
) -> tuple[int, ...]:
  """Validate and canonicalize requested trajectory model-call indices."""
  result = []
  for value in values:
    if isinstance(value, bool) or not isinstance(value, int):
      raise TypeError('snapshot call indices must be integers')
    if value < 0:
      raise ValueError('snapshot call indices must be non-negative')
    result.append(value)
  return tuple(sorted(set(result)))


class _TrajectoryRecorder:
  """Capture JSON-serializable raw batch states without consuming RNG."""

  def __init__(
      self,
      *,
      initial_tokens: torch.Tensor,
      mask_token_id: int,
      requested_call_indices: Iterable[int],
  ) -> None:
    self.requested_call_indices = _normalize_snapshot_call_indices(
      requested_call_indices)
    self._requested = set(self.requested_call_indices)
    self._mask_token_id = int(mask_token_id)
    self._active_mask = initial_tokens.eq(self._mask_token_id)
    self._selected_recorded: set[int] = set()
    self.snapshots: list[dict[str, Any]] = []

  def record(
      self,
      tokens: torch.Tensor,
      *,
      model_call_index: int,
      timestep: torch.Tensor | None,
      stage: str,
      force: bool = False,
  ) -> None:
    if (not force
        and (model_call_index not in self._requested
             or model_call_index in self._selected_recorded)):
      return
    if model_call_index in self._requested:
      self._selected_recorded.add(model_call_index)
    active_mask = self._active_mask.to(tokens.device)
    unresolved = active_mask & tokens.eq(self._mask_token_id)
    timestep_values = None
    if timestep is not None:
      timestep_cpu = timestep.detach().to('cpu')
      if timestep_cpu.ndim == 1:
        timestep_cpu = timestep_cpu[:, None]
      timestep_values = [
        [float(value) for value in row]
        for row in timestep_cpu.tolist()
      ]
    self.snapshots.append({
      'model_call_index': int(model_call_index),
      'stage': stage,
      'timestep': timestep_values,
      'token_ids': [
        [int(token) for token in row]
        for row in tokens.detach().to('cpu').tolist()
      ],
      'active_mask': active_mask.detach().to('cpu').tolist(),
      'unresolved_active_mask': unresolved.detach().to('cpu').tolist(),
      'unresolved_active_count': int(unresolved.sum().item()),
    })


@torch.no_grad()
def _sample_from_initial_state(
    model,
    initial_tokens: torch.Tensor,
    *,
    nfe_budget: int,
    eps: float = 1e-5,
    trajectory_recorder: _TrajectoryRecorder | None = None,
) -> tuple[torch.Tensor, int]:
  """Sample from explicit masks while leaving observed tokens untouched.

  ``nfe_budget`` includes the optional final denoising call.  Structured
  sampling can finish all masks early, so the returned measured NFE may be
  lower than the requested upper bound.
  """
  if initial_tokens.ndim != 2:
    raise ValueError('initial_tokens must be a [batch, length] matrix')
  if nfe_budget <= 0:
    raise ValueError('nfe_budget must be positive')
  if model.sampler not in {'ddpm', 'analytic'}:
    raise ValueError(
      'paired harness currently supports predictor=ddpm or analytic; '
      f'found {model.sampler!r}')
  final_call = int(bool(model.config.sampling.noise_removal))
  num_steps = nfe_budget - final_call
  if num_steps <= 0:
    raise ValueError(
      'nfe_budget must exceed the final noise-removal call; use at least 2')

  x = initial_tokens.clone().to(model.device)
  observed = initial_tokens.ne(model.mask_index).to(model.device)
  observed_values = initial_tokens.to(model.device)
  timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
  dt = (1 - eps) / num_steps
  measured_nfe = 0
  if trajectory_recorder is not None:
    trajectory_recorder.record(
      x,
      model_call_index=0,
      timestep=None,
      stage='initial',
      force=True)

  structured_path = (
    bool(model.structured_enabled)
    and model.structured_sampling_mode in STRUCTURED_SAMPLING_MODES)
  handle = None
  original_structured_backbone_output = None
  if structured_path:
    # The structured path intentionally calls backbone.encode/decode directly
    # to reuse hidden states, so a hook on backbone.__call__ does not see an
    # NFE.  Wrap the single shared encode/decode entry point instead.
    original_structured_backbone_output = model._structured_backbone_output

    def counted_structured_backbone_output(*args, **kwargs):
      nonlocal measured_nfe
      measured_nfe += 1
      return original_structured_backbone_output(*args, **kwargs)

    model._structured_backbone_output = counted_structured_backbone_output
  else:
    def count_backbone_calls(unused_module, unused_inputs, unused_output):
      nonlocal measured_nfe
      measured_nfe += 1

    handle = model.backbone.register_forward_hook(count_backbone_calls)
  try:
    for index in range(num_steps):
      t = timesteps[index] * torch.ones(
        x.shape[0], 1, device=model.device)
      calls_before_update = measured_nfe
      if model.sampler == 'ddpm':
        x = model._ddpm_update(x, t, dt)
      else:
        x = model._analytic_update(x, t, dt)
      if (trajectory_recorder is not None
          and measured_nfe - calls_before_update not in {0, 1}):
        raise RuntimeError(
          'trajectory capture requires at most one model call per sampler '
          'state transition')
      # Fail closed if a future core sampler starts modifying evidence.
      if not torch.equal(x[observed], observed_values[observed]):
        raise AssertionError('sampling kernel modified observed prompt tokens')
      if trajectory_recorder is not None:
        is_final_state = not final_call and index == num_steps - 1
        trajectory_recorder.record(
          x,
          model_call_index=measured_nfe,
          timestep=t,
          stage='final' if is_final_state else 'denoising',
          force=is_final_state)

    if final_call:
      t = timesteps[-1] * torch.ones(
        x.shape[0], 1, device=model.device)
      calls_before_update = measured_nfe
      x = _finish_sampling(model, x, t)
      if (trajectory_recorder is not None
          and measured_nfe - calls_before_update not in {0, 1}):
        raise RuntimeError(
          'trajectory capture requires at most one model call per final '
          'state transition')
      # The ordinary factorized final argmax is written for unconditional
      # generation and proposes values at every position.  Explicit evidence
      # is part of this harness's contract, so clamp it after that proposal.
      x = torch.where(observed, observed_values, x)
      if trajectory_recorder is not None:
        trajectory_recorder.record(
          x,
          model_call_index=measured_nfe,
          timestep=t,
          stage='final',
          force=True)
    if not torch.equal(x[observed], observed_values[observed]):
      raise AssertionError('final denoiser modified observed prompt tokens')
  finally:
    if handle is not None:
      handle.remove()
    if original_structured_backbone_output is not None:
      model._structured_backbone_output = original_structured_backbone_output
  return x, measured_nfe


@torch.no_grad()
def sample_from_initial_state(
    model,
    initial_tokens: torch.Tensor,
    *,
    nfe_budget: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, int]:
  """Sample without trajectory capture; this is the ordinary harness path."""
  return _sample_from_initial_state(
    model,
    initial_tokens,
    nfe_budget=nfe_budget,
    eps=eps)


@torch.no_grad()
def sample_trajectory_from_initial_state(
    model,
    initial_tokens: torch.Tensor,
    *,
    nfe_budget: int,
    snapshot_call_indices: Iterable[int],
    eps: float = 1e-5,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
  """Sample once and capture selected states from that single NFE run.

  Model-call index zero and the actual final state are always captured.  Other
  states are selected by the number of model calls completed, not by loop
  iteration or by a state from a separately sampled NFE budget.  If structured
  sampling resolves every mask early, a requested index beyond the measured
  NFE is reported as missing rather than synthesized.
  """
  recorder = _TrajectoryRecorder(
    initial_tokens=initial_tokens,
    mask_token_id=model.mask_index,
    requested_call_indices=snapshot_call_indices)
  generated, measured_nfe = _sample_from_initial_state(
    model,
    initial_tokens,
    nfe_budget=nfe_budget,
    eps=eps,
    trajectory_recorder=recorder)
  captured = {
    int(snapshot['model_call_index']) for snapshot in recorder.snapshots
  }
  return generated, measured_nfe, {
    'schema_version': 1,
    'trajectory_scope': 'single_nfe_run',
    'mask_token_id': int(model.mask_index),
    'batch_size': int(initial_tokens.shape[0]),
    'sequence_length': int(initial_tokens.shape[1]),
    'requested_nfe_budget': int(nfe_budget),
    'measured_nfe': int(measured_nfe),
    'snapshot_call_indices_requested': list(
      recorder.requested_call_indices),
    'snapshot_call_indices_captured': sorted(captured),
    'snapshot_call_indices_missing': [
      value for value in recorder.requested_call_indices
      if value not in captured
    ],
    'snapshots': recorder.snapshots,
  }


def _validate_exact_replay_records(
    samples: Sequence[PairedSampleSpec],
    expected_records: Sequence[dict[str, Any]],
    *,
    sampling_mode: str,
    nfe_budget: int,
) -> tuple[int, list[list[int]], int]:
  """Fail closed unless records identify the complete ordered source batch."""
  if not samples:
    raise ValueError('trajectory replay requires a non-empty source batch')
  if len(expected_records) != len(samples):
    raise ValueError(
      'trajectory replay must receive every record from the original batch')
  expected_batch_seed = batch_seed(samples)
  expected_final_token_ids = []
  expected_measured_nfe = None
  for row_index, (sample, record) in enumerate(zip(samples, expected_records)):
    context = f'expected_records[{row_index}]'
    if not isinstance(record, dict):
      raise TypeError(f'{context} must be an object')
    identity = {
      'sample_index': sample.sample_index,
      'pair_key': sample.pair_key,
      'pair_seed': sample.pair_seed,
      'prompt_id': sample.prompt.prompt_id,
      'prompt_metadata': sample.prompt.metadata,
      'initial_token_ids': list(sample.prompt.initial_token_ids),
      'active_mask': list(sample.prompt.active_mask),
      'reference_token_ids': (
        list(sample.prompt.reference_token_ids)
        if sample.prompt.reference_token_ids is not None else None),
    }
    for field, value in identity.items():
      if record.get(field) != value:
        raise ValueError(
          f'{context}.{field} does not match the ordered source batch')
    if record.get('sampling_mode') != sampling_mode:
      raise ValueError(f'{context}.sampling_mode does not match replay mode')
    if record.get('requested_nfe_budget') != int(nfe_budget):
      raise ValueError(
        f'{context}.requested_nfe_budget does not match replay budget')
    if record.get('batch_seed') != expected_batch_seed:
      raise ValueError(
        f'{context}.batch_seed does not match the complete source batch')
    timing = record.get('timing')
    if not isinstance(timing, dict):
      raise ValueError(f'{context}.timing must identify the source batch')
    timing_identity = {
      'batch_seed': expected_batch_seed,
      'batch_size': len(samples),
      'requested_nfe_budget': int(nfe_budget),
    }
    for field, value in timing_identity.items():
      if timing.get(field) != value:
        raise ValueError(
          f'{context}.timing.{field} does not match the source batch')
    measured_nfe = record.get('measured_nfe')
    if (isinstance(measured_nfe, bool)
        or not isinstance(measured_nfe, int)
        or measured_nfe < 0):
      raise ValueError(f'{context}.measured_nfe must be non-negative integer')
    if timing.get('measured_nfe') != measured_nfe:
      raise ValueError(
        f'{context}.timing.measured_nfe does not match its record')
    if expected_measured_nfe is None:
      expected_measured_nfe = measured_nfe
    elif measured_nfe != expected_measured_nfe:
      raise ValueError('measured_nfe differs within the source batch')
    final_tokens = record.get('sample_token_ids')
    if (not isinstance(final_tokens, list)
        or len(final_tokens) != len(sample.prompt.initial_token_ids)
        or any(isinstance(token, bool) or not isinstance(token, int)
               for token in final_tokens)):
      raise ValueError(
        f'{context}.sample_token_ids must be a full integer token row')
    expected_final_token_ids.append([int(token) for token in final_tokens])
  assert expected_measured_nfe is not None
  return expected_batch_seed, expected_final_token_ids, expected_measured_nfe


@torch.no_grad()
def replay_sampling_group_trajectory(
    model,
    samples: Sequence[PairedSampleSpec],
    *,
    sampling_mode: str,
    nfe_budget: int,
    expected_records: Sequence[dict[str, Any]],
    snapshot_call_indices: Iterable[int],
    device: torch.device,
) -> dict[str, Any]:
  """Exactly replay one complete original batch and capture its trajectory.

  The original ordered batch is provenance-critical because ``batch_seed`` is
  a commitment to every pair in that order.  The replay refuses partial,
  reordered, or differently configured records and raises unless its final raw
  token ids and measured NFE exactly match the persisted source records.
  """
  if sampling_mode not in SAMPLING_MODES:
    raise ValueError(f'unsupported sampling mode {sampling_mode!r}')
  if not samples:
    raise ValueError('trajectory replay requires a non-empty source batch')
  if torch.device(model.device) != torch.device(device):
    raise ValueError('replay device must match model.device')
  lengths = {len(item.prompt.initial_token_ids) for item in samples}
  if len(lengths) != 1:
    raise ValueError('all prompts in a replay batch must have equal length')
  rng_seed, expected_final, expected_measured_nfe = (
    _validate_exact_replay_records(
      samples,
      expected_records,
      sampling_mode=sampling_mode,
      nfe_budget=nfe_budget))
  initial = torch.tensor(
    [item.prompt.initial_token_ids for item in samples],
    dtype=torch.long,
    device=device)
  declared_active = torch.tensor(
    [item.prompt.active_mask for item in samples],
    dtype=torch.bool,
    device=device)
  if not torch.equal(initial.eq(model.mask_index), declared_active):
    raise ValueError(
      'initial mask tokens do not exactly match the declared active masks')

  previous_sampling_mode = model.structured_sampling_mode
  model.structured_sampling_mode = sampling_mode
  seed_everything(rng_seed, device)
  try:
    generated, measured_nfe, trajectory = (
      sample_trajectory_from_initial_state(
        model,
        initial,
        nfe_budget=nfe_budget,
        snapshot_call_indices=snapshot_call_indices))
  finally:
    model.structured_sampling_mode = previous_sampling_mode

  actual_final = [
    [int(token) for token in row]
    for row in generated.detach().to('cpu').tolist()
  ]
  if measured_nfe != expected_measured_nfe:
    raise AssertionError(
      'exact trajectory replay measured NFE mismatch: '
      f'expected {expected_measured_nfe}, found {measured_nfe}')
  if actual_final != expected_final:
    mismatch_rows = [
      index for index, (actual, expected) in enumerate(
        zip(actual_final, expected_final))
      if actual != expected
    ]
    raise AssertionError(
      'exact trajectory replay final token-id mismatch in batch rows '
      f'{mismatch_rows}')

  trajectory.update({
    'artifact_type': 'exact_batch_generation_trajectory',
    'sampling_mode': sampling_mode,
    'batch_seed': int(rng_seed),
    'batch_size': len(samples),
    'batch_pairing_digest': pairing_digest(samples),
    'batch_order': [
      {
        'batch_row_index': row_index,
        'sample_index': sample.sample_index,
        'pair_key': sample.pair_key,
        'pair_seed': sample.pair_seed,
        'prompt_id': sample.prompt.prompt_id,
      }
      for row_index, sample in enumerate(samples)
    ],
    'expected_final_token_ids': expected_final,
    'final_token_ids': actual_final,
    'final_token_ids_match_expected': True,
  })
  return trajectory


def _active_values(values: Sequence[int], mask: Sequence[bool]) -> list[int]:
  return [int(value) for value, active in zip(values, mask) if active]


def run_sampling_group(
    model,
    samples: Sequence[PairedSampleSpec],
    *,
    sampling_mode: str,
    nfe_budget: int,
    tokenizer,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Run one paired batch and return per-sample plus batch metadata."""
  if sampling_mode not in SAMPLING_MODES:
    raise ValueError(f'unsupported sampling mode {sampling_mode!r}')
  if not samples:
    raise ValueError('sampling group must be non-empty')
  lengths = {len(item.prompt.initial_token_ids) for item in samples}
  if len(lengths) != 1:
    raise ValueError('all prompts in a sampling batch must have equal length')
  previous_sampling_mode = model.structured_sampling_mode
  model.structured_sampling_mode = sampling_mode
  initial = torch.tensor(
    [item.prompt.initial_token_ids for item in samples],
    dtype=torch.long,
    device=device)
  declared_active = torch.tensor(
    [item.prompt.active_mask for item in samples],
    dtype=torch.bool,
    device=device)
  if not torch.equal(initial.eq(model.mask_index), declared_active):
    raise ValueError(
      'initial mask tokens do not exactly match the declared active masks')
  rng_seed = batch_seed(samples)
  seed_everything(rng_seed, device)

  if device.type == 'cuda':
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
  start = time.perf_counter()
  try:
    generated, measured_nfe = sample_from_initial_state(
      model, initial, nfe_budget=nfe_budget)
  finally:
    model.structured_sampling_mode = previous_sampling_mode
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
  elapsed = time.perf_counter() - start
  peak_memory = (
    int(torch.cuda.max_memory_allocated(device))
    if device.type == 'cuda' else None)

  generated_cpu = generated.detach().cpu().tolist()
  active_token_count = sum(sum(item.prompt.active_mask) for item in samples)
  unresolved = sum(
    token == model.mask_index
    for row in generated_cpu for token in row)
  if unresolved:
    raise RuntimeError(
      f'sampling left {unresolved} unresolved mask tokens; refusing to '
      'decode or persist an incomplete batch')
  batch_metadata = {
    'batch_seed': rng_seed,
    'batch_size': len(samples),
    'requested_nfe_budget': int(nfe_budget),
    'measured_nfe': int(measured_nfe),
    'wall_clock_seconds': float(elapsed),
    'active_tokens': int(active_token_count),
    'active_tokens_per_second': (
      float(active_token_count / elapsed) if elapsed > 0 else None),
    'sequence_tokens_per_second': (
      float(generated.numel() / elapsed) if elapsed > 0 else None),
    'peak_memory_bytes': peak_memory,
    'unresolved_mask_tokens': int(unresolved),
  }

  records = []
  decoded = tokenizer.batch_decode(generated_cpu)
  for item, token_ids, text in zip(samples, generated_cpu, decoded):
    active_ids = _active_values(token_ids, item.prompt.active_mask)
    reference_active = (
      _active_values(
        item.prompt.reference_token_ids, item.prompt.active_mask)
      if item.prompt.reference_token_ids is not None else None)
    metrics: dict[str, Any] = {
      'repetition_rate': {
        str(n): repetition_rate(active_ids, n=n) for n in (1, 2, 4)
      },
    }
    if reference_active is not None:
      metrics.update(paired_token_metrics(active_ids, reference_active))
    records.append({
      'schema_version': 1,
      'sample_index': item.sample_index,
      'pair_key': item.pair_key,
      'pair_seed': item.pair_seed,
      'prompt_id': item.prompt.prompt_id,
      'prompt_metadata': item.prompt.metadata,
      'sampling_mode': sampling_mode,
      'requested_nfe_budget': int(nfe_budget),
      'measured_nfe': int(measured_nfe),
      'batch_seed': rng_seed,
      'initial_token_ids': list(item.prompt.initial_token_ids),
      'active_mask': list(item.prompt.active_mask),
      'reference_token_ids': (
        list(item.prompt.reference_token_ids)
        if item.prompt.reference_token_ids is not None else None),
      'sample_token_ids': [int(token) for token in token_ids],
      'sample_active_token_ids': active_ids,
      'text': text,
      'metrics': metrics,
      'timing': batch_metadata,
    })
  return records, batch_metadata


def summarize_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Aggregate one mode/budget group from its JSON-serializable records."""
  if not records:
    raise ValueError('cannot summarize an empty group')
  generated = [record['sample_active_token_ids'] for record in records]
  reference = []
  for record in records:
    target = record['reference_token_ids']
    if target is None:
      reference = None
      break
    reference.append(_active_values(target, record['active_mask']))
  result = summarize_token_metrics(generated, reference=reference)
  unique_batches = {
    (record['batch_seed'], record['timing']['wall_clock_seconds']):
    record['timing']
    for record in records
  }
  batch_values = list(unique_batches.values())
  elapsed = sum(float(batch['wall_clock_seconds']) for batch in batch_values)
  active_tokens = sum(int(batch['active_tokens']) for batch in batch_values)
  result.update({
    'sampling_mode': records[0]['sampling_mode'],
    'requested_nfe_budget': records[0]['requested_nfe_budget'],
    'pairing_digest': stable_sha256([
      {
        'sample_index': record['sample_index'],
        'pair_key': record['pair_key'],
        'pair_seed': record['pair_seed'],
        'prompt_id': record['prompt_id'],
        'initial_token_ids': record['initial_token_ids'],
        'active_mask': record['active_mask'],
        'reference_token_ids': record['reference_token_ids'],
        'prompt_metadata': record['prompt_metadata'],
      }
      for record in records
    ]),
    'num_batches': len(batch_values),
    'wall_clock_seconds': elapsed,
    'active_tokens_per_second': (
      float(active_tokens / elapsed) if elapsed > 0 else None),
    'peak_memory_bytes': max(
      (batch['peak_memory_bytes'] for batch in batch_values
       if batch['peak_memory_bytes'] is not None),
      default=None),
    'measured_nfe_values': sorted({
      int(batch['measured_nfe']) for batch in batch_values
    }),
    'unresolved_mask_tokens': sum(
      int(batch['unresolved_mask_tokens']) for batch in batch_values),
  })
  return result


def iter_batches(
    values: Sequence[PairedSampleSpec],
    batch_size: int,
) -> Iterable[Sequence[PairedSampleSpec]]:
  if batch_size <= 0:
    raise ValueError('batch_size must be positive')
  for offset in range(0, len(values), batch_size):
    yield values[offset:offset + batch_size]
