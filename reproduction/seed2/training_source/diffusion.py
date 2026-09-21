import hashlib
import json
import math
import os
import typing
import warnings
from dataclasses import dataclass

import fsspec
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import crf_utils
import dataloader
from evaluation import conditional_denoising_records
from evaluation import causal_denoising
import models
import noise_schedule
import structured_objective
import structured_pairing
import structured_training
import utils

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


def _loglinear_time_from_mask_rate(mask_rate: float, eps: float) -> float:
  """Invert ``q(x_t != x_0) = (1 - eps) t`` for a fixed eval rate."""
  if not 0.0 < mask_rate < 1.0:
    raise ValueError('mask_rate must be strictly between 0 and 1')
  if not 0.0 < eps < 1.0:
    raise ValueError('loglinear eps must be strictly between 0 and 1')
  timestep = mask_rate / (1.0 - eps)
  if timestep > 1.0:
    raise ValueError('mask_rate exceeds the loglinear schedule maximum')
  return timestep


def _structured_training_rng_seeds(
    base_seed: int, epoch: int, rank: int) -> tuple[int, int]:
  """Domain-separate paired corruption and topology-teacher RNG streams."""
  if min(base_seed, epoch, rank) < 0:
    raise ValueError('structured RNG seed inputs must be non-negative')
  modulus = 2 ** 63 - 1
  corruption_seed = (
    int(base_seed) * 1_000_003 + int(epoch) * 10_007 + int(rank)) % modulus
  topology_seed = (corruption_seed + 4_294_967_291) % modulus
  return corruption_seed, topology_seed


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class RatioMetric(torchmetrics.Metric):
  """Distributed ratio of explicitly supplied numerators/denominators."""

  full_state_update = False

  def __init__(self):
    super().__init__()
    self.add_state(
      'numerator', default=torch.tensor(0.0, dtype=torch.float64),
      dist_reduce_fx='sum')
    self.add_state(
      'denominator', default=torch.tensor(0.0, dtype=torch.float64),
      dist_reduce_fx='sum')

  def update(self, numerator, denominator):
    self.numerator += torch.as_tensor(
      numerator, device=self.numerator.device,
      dtype=self.numerator.dtype).detach()
    self.denominator += torch.as_tensor(
      denominator, device=self.denominator.device,
      dtype=self.denominator.dtype).detach()

  def compute(self):
    return torch.where(
      self.denominator > 0,
      self.numerator / self.denominator.clamp_min(1),
      torch.zeros_like(self.numerator))


class DistributedSumMetric(torchmetrics.Metric):
  """Distributed sum used to expose coverage denominators."""

  full_state_update = False

  def __init__(self):
    super().__init__()
    self.add_state(
      'total', default=torch.tensor(0.0, dtype=torch.float64),
      dist_reduce_fx='sum')

  def update(self, value):
    self.total += torch.as_tensor(
      value, device=self.total.device, dtype=self.total.dtype).detach()

  def compute(self):
    return self.total


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.monitor_metric = str(
      self.config.model.get('monitor_metric', 'val/nll'))
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'crf_dit':
      self.backbone = models.crf_decoder.CRFDiT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.structured_head = None
    self.structured_enabled = False
    self.structured_config = None
    self.structured_training_config = None
    self.structured_backbone_mode = 'joint'
    self.structured_deterministic_backbone = True
    self.structured_sampling_mode = 'factorized'
    self.register_buffer(
      'structured_fixed_edge_index', None, persistent=True)
    self._last_structured_metrics = {}
    self._last_structured_metric_updates = {}
    self._structured_validation_pairing_digest = None
    self._last_structured_example_metrics = {}
    configured_mask_rate = self.config.eval.get(
      'structured_mask_rate', None)
    self.structured_eval_mask_rate = (
      None if configured_mask_rate is None
      else float(configured_mask_rate))
    configured_corruption_seed = self.config.eval.get(
      'corruption_seed', None)
    self.structured_eval_corruption_seed = (
      None if configured_corruption_seed is None
      else int(configured_corruption_seed))
    self._structured_validation_generator = None
    self._structured_training_corruption_generator = None
    self._structured_training_topology_generator = None
    record_cfg = self.config.eval.get('conditional_records', {})
    self.conditional_records_enabled = bool(
      record_cfg.get('enabled', False))
    self.conditional_record_config = record_cfg
    self.conditional_record_schema_version = int(
      record_cfg.get('schema_version', 1))
    self.conditional_record_support_candidate_ks = tuple(
      int(value) for value in record_cfg.get(
        'support_candidate_ks', []))
    self._conditional_record_writer = None
    self._initialize_structured_decoder()

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    structured_metrics = torchmetrics.MetricCollection({
      'conditional_nll_per_masked_token': RatioMetric(),
      'candidate_recall': RatioMetric(),
      'retained_unary_mass': RatioMetric(),
      'topology_edge_coverage': RatioMetric(),
      'topology_edge_coverage_denominator': DistributedSumMetric(),
      'topology_anchor_coverage': RatioMetric(),
      'topology_anchor_coverage_denominator': DistributedSumMetric(),
      'topology_slot_coverage': RatioMetric(),
      'topology_slot_coverage_denominator': DistributedSumMetric(),
    })
    structured_metrics.set_dtype(torch.float64)
    self.structured_train_metrics = structured_metrics.clone()
    self.structured_valid_metrics = structured_metrics.clone()
    self.structured_test_metrics = structured_metrics.clone()

    # Generative perplexity is optional.  Keep its external tokenizer lazy so
    # conditional-denoising and paired-generation runs do not depend on an
    # unused, mutable Hub download during model construction.
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = None
    if self.config.eval.compute_generative_perplexity:
      self.eval_model_tokenizer = transformers.AutoTokenizer.\
        from_pretrained(self.gen_ppl_eval_model_name_or_path)
      if self.eval_model_tokenizer.pad_token is None:
        self.eval_model_tokenizer.pad_token =\
            self.eval_model_tokenizer.eos_token
        self.eval_model_tokenizer.pad_token_id =\
            self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._trainable_model_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.crf_reveal = self.config.sampling.get(
      'crf_reveal', 'independent')
    self.crf_unigram_aux_enabled = False
    self.crf_unigram_aux_start_weight = 0.0
    self.crf_unigram_aux_end_weight = 0.0
    self.crf_unigram_aux_warmup_steps = 0
    if self.config.backbone == 'crf_dit':
      aux_cfg = self.config.model.crf.get('unigram_aux', {})
      self.crf_unigram_aux_enabled = bool(
        aux_cfg.get('enabled', False))
      self.crf_unigram_aux_start_weight = float(
        aux_cfg.get('start_weight', 0.0))
      self.crf_unigram_aux_end_weight = float(
        aux_cfg.get('end_weight', 0.1))
      self.crf_unigram_aux_warmup_steps = int(
        aux_cfg.get('warmup_steps', 10000))
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _initialize_structured_decoder(self):
    """Install the opt-in coupling head without changing ordinary DIT runs."""
    structured_cfg = self.config.model.get('structured_decoder', None)
    if structured_cfg is None or not bool(
        structured_cfg.get('enabled', False)):
      return
    if self.config.backbone != 'dit':
      raise ValueError(
        'contextual structured decoding currently requires backbone=dit')

    self.structured_enabled = True
    self.structured_config = structured_cfg
    training_cfg = structured_cfg.get('training', {})
    sampling_cfg = structured_cfg.get('sampling', {})
    self.structured_training_config = training_cfg
    self.structured_backbone_mode = str(
      training_cfg.get('backbone_mode', 'joint'))
    if self.structured_backbone_mode not in {'frozen', 'joint'}:
      raise ValueError(
        'structured backbone_mode must be frozen or joint')
    self.structured_deterministic_backbone = bool(
      training_cfg.get('deterministic_backbone', True))
    sampling_mode = sampling_cfg.get('mode', None)
    legacy_use_joint = sampling_cfg.get('use_joint', None)
    if sampling_mode is not None and legacy_use_joint is not None:
      raise ValueError(
        'set structured sampling.mode, not both mode and use_joint')
    if sampling_mode is None:
      sampling_mode = (
        'structured_joint' if bool(legacy_use_joint) else 'factorized')
      if legacy_use_joint is not None:
        warnings.warn(
          'structured sampling.use_joint is deprecated; use '
          'model.structured_decoder.sampling.mode=structured_joint',
          stacklevel=2)
    self.structured_sampling_mode = (
      structured_training.validate_structured_sampling_mode(
        str(sampling_mode)))

    self.structured_head = (
      models.structured_decoder.ContextualCouplingForestHead(
        hidden_size=self.config.model.hidden_size,
        vocab_size=self.vocab_size,
        top_k=int(structured_cfg.get('top_k', 64)),
        rank=int(structured_cfg.get('rank', 16)),
        time_embed_dim=int(structured_cfg.get('time_embed_dim', 64)),
        topology_dim=int(structured_cfg.get('topology_dim', 128)),
        local_window=int(structured_cfg.get('local_window', 2)),
        num_anchor_slots=int(
          structured_cfg.get('num_anchor_slots', 16)),
        contextual_neighbors=int(
          structured_cfg.get('contextual_neighbors', 4)),
        component_size_cap=int(
          structured_cfg.get('component_size_cap', 32)),
        topology_mode=str(
          structured_cfg.get('topology_mode', 'dynamic')),
        factor_mode=str(structured_cfg.get('factor_mode', 'dynamic')),
        independent_mode=bool(
          structured_cfg.get('independent_mode', False)),
        min_edge_score=structured_cfg.get('min_edge_score', None)))

    checkpoint_path = training_cfg.get('backbone_checkpoint', None)
    require_checkpoint = bool(
      training_cfg.get('require_pretrained_backbone', False))
    eval_checkpoint = self.config.eval.get('checkpoint_path', None)
    resume_checkpoint = self.config.checkpointing.get(
      'resume_ckpt_path', None)
    full_checkpoint_will_load = bool(
      (self.config.mode != 'train' and eval_checkpoint)
      or (self.config.checkpointing.get('resume_from_ckpt', False)
          and resume_checkpoint
          and utils.fsspec_exists(str(resume_checkpoint))))
    if checkpoint_path:
      self._load_structured_backbone_checkpoint(
        str(checkpoint_path),
        strict=bool(training_cfg.get('strict_backbone_checkpoint', True)),
        use_ema=bool(training_cfg.get('use_ema_backbone', False)))
    elif require_checkpoint and not full_checkpoint_will_load:
      raise ValueError(
        'a pretrained backbone is required; set '
        'model.structured_decoder.training.backbone_checkpoint')
    elif (self.structured_backbone_mode == 'frozen'
          and not full_checkpoint_will_load):
      warnings.warn(
        'structured head is freezing a backbone without loading a '
        'checkpoint; this is suitable only for smoke tests', stacklevel=2)

    if self.structured_backbone_mode == 'frozen':
      self.backbone.requires_grad_(False)

    adapter_checkpoint = self.config.eval.get('adapter_checkpoint', None)
    adapter_sha256 = self.config.eval.get('adapter_sha256', None)
    adapter_manifest = self.config.eval.get('adapter_manifest', None)
    adapter_manifest_sha256 = self.config.eval.get(
      'adapter_manifest_sha256', None)
    if (adapter_sha256 or adapter_manifest or adapter_manifest_sha256) \
        and not adapter_checkpoint:
      raise ValueError(
        'eval adapter hashes and manifest require '
        'adapter_checkpoint')
    if adapter_checkpoint:
      if (not adapter_sha256 or not adapter_manifest
          or not adapter_manifest_sha256):
        raise ValueError(
          'eval.adapter_checkpoint requires adapter_sha256, '
          'adapter_manifest, and adapter_manifest_sha256')
      if eval_checkpoint:
        raise ValueError(
          'set only one of eval.checkpoint_path and '
          'eval.adapter_checkpoint')
      if self.config.mode == 'train':
        raise ValueError(
          'eval.adapter_checkpoint is evaluation-only')
      self._load_structured_adapter_checkpoint(
        str(adapter_checkpoint),
        expected_sha256=str(adapter_sha256),
        manifest_path=str(adapter_manifest),
        expected_manifest_sha256=str(adapter_manifest_sha256),
        structured_config=structured_cfg)

    fixed_edges = structured_cfg.get('fixed_edges', None)
    fixed_edge_path = structured_cfg.get('fixed_edge_path', None)
    if fixed_edges is not None and fixed_edge_path:
      raise ValueError('set only one of fixed_edges and fixed_edge_path')
    if fixed_edge_path:
      with fsspec.open(str(fixed_edge_path), 'rb') as handle:
        try:
          fixed_edges = torch.load(
            handle, map_location='cpu', weights_only=True)
        except TypeError:
          handle.seek(0)
          fixed_edges = torch.load(handle, map_location='cpu')
      if isinstance(fixed_edges, dict):
        if 'edge_index' not in fixed_edges:
          raise ValueError(
            'fixed-edge checkpoint must contain edge_index')
        fixed_edges = fixed_edges['edge_index']
    if fixed_edges is not None:
      if not torch.is_tensor(fixed_edges):
        fixed_edges = [list(edge) for edge in fixed_edges]
      is_empty = (
        fixed_edges.numel() == 0 if torch.is_tensor(fixed_edges)
        else len(fixed_edges) == 0)
      if is_empty:
        fixed_edges = torch.empty(0, 2, dtype=torch.long)
      else:
        fixed_edges = torch.as_tensor(fixed_edges, dtype=torch.long)
      if fixed_edges.ndim != 2 or fixed_edges.shape[-1] != 2:
        raise ValueError('configured fixed edges must have shape [E,2]')
      if (fixed_edges.numel()
          and (int(fixed_edges.min()) < 0
               or int(fixed_edges.max()) >= self.config.model.length)):
        raise ValueError('configured fixed edge is outside model length')
      self.structured_fixed_edge_index = fixed_edges

  def _load_structured_backbone_checkpoint(
      self, path, strict=True, use_ema=False):
    """Load only ``backbone.*`` weights from an MDLM Lightning checkpoint."""
    with fsspec.open(path, 'rb') as handle:
      try:
        checkpoint = torch.load(
          handle, map_location='cpu', weights_only=False)
      except TypeError:
        handle.seek(0)
        checkpoint = torch.load(handle, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    backbone_state = {
      key[len('backbone.'):]: value
      for key, value in state_dict.items()
      if key.startswith('backbone.')
    }
    if not backbone_state:
      backbone_keys = set(self.backbone.state_dict())
      backbone_state = {
        key: value for key, value in state_dict.items()
        if key in backbone_keys}
    if not backbone_state:
      raise ValueError(f'no backbone weights found in {path}')
    incompatible = self.backbone.load_state_dict(
      backbone_state, strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
      raise ValueError(
        'backbone checkpoint mismatch: '
        f'missing={incompatible.missing_keys}, '
        f'unexpected={incompatible.unexpected_keys}')
    if use_ema:
      ema_state = checkpoint.get('ema', None)
      named_parameters = list(self.backbone.named_parameters())
      shadow_parameters = (
        structured_training.validated_ema_shadow_parameters(
          ema_state=ema_state,
          expected_parameters=[
            parameter for _, parameter in named_parameters],
          context=f'backbone checkpoint {path}',
          allow_extra=True))
      ema_backbone_state = {
        name: shadow.detach().to(dtype=parameter.dtype)
        for (name, parameter), shadow in zip(
          named_parameters, shadow_parameters)}
      self.backbone.load_state_dict(ema_backbone_state, strict=False)

  def _load_structured_adapter_checkpoint(
      self, path, *, expected_sha256, manifest_path,
      expected_manifest_sha256, structured_config):
    """Load an adapter only after manifest, byte, and config verification."""
    with fsspec.open(path, 'rb') as handle:
      payload = handle.read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
      raise ValueError(
        f'structured adapter SHA256 mismatch: expected '
        f'{expected_sha256}, found {actual_sha256}')
    with fsspec.open(manifest_path, 'rb') as handle:
      manifest_bytes = handle.read()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
      raise ValueError(
        f'structured adapter manifest SHA256 mismatch: expected '
        f'{expected_manifest_sha256}, found {actual_manifest_sha256}')
    try:
      manifest_payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
      raise ValueError(
        'structured adapter manifest is not valid JSON') from error
    if (not isinstance(manifest_payload, dict)
        or manifest_payload.get('schema_version') != 4):
      raise ValueError(
        'Diffusion adapter loading supports schema-v4 adapters only; use '
        'verify_contextual_forest_adapter for schema-v5 artifacts with '
        'authenticated expectations and actual-backbone attestation')
    from scripts.export_structured_adapter import (
      safetensors_metadata_from_bytes,
      structured_decoder_identity_from_config,
      validate_tensor_state_against_module,
      validate_adapter_manifest_payload,
    )
    runtime_identity, runtime_identity_sha256 = (
      structured_decoder_identity_from_config(structured_config))
    from safetensors.torch import load
    state = load(payload)
    validated_manifest = validate_adapter_manifest_payload(
      manifest_payload,
      adapter_filename=str(path).rstrip('/').rsplit('/', 1)[-1],
      adapter_sha256=actual_sha256,
      adapter_size_bytes=len(payload),
      adapter_metadata=safetensors_metadata_from_bytes(payload),
      adapter_state=state,
      expected_identity=runtime_identity)
    if (validated_manifest['structured_decoder_identity_sha256']
        != runtime_identity_sha256):
      raise AssertionError('validated structured adapter digest drifted')
    if not state:
      raise ValueError('structured adapter contains no tensors')
    prefixed = [
      key for key in state
      if key.startswith(('backbone.', 'structured_head.'))]
    if prefixed:
      raise ValueError(
        'structured adapter must use prefix-stripped head keys; '
        f'found {prefixed[:5]}')
    validate_tensor_state_against_module(
      state, self.structured_head, context='structured adapter')
    try:
      self.structured_head.load_state_dict(state, strict=True)
    except RuntimeError as error:
      raise ValueError(
        f'structured adapter state mismatch: {error}') from error
    self.structured_adapter_manifest = validated_manifest

  def _trainable_model_parameters(self):
    """Stable parameter order shared by the optimizer and EMA."""
    modules = [self.backbone, self.noise]
    if self.structured_head is not None:
      modules.append(self.structured_head)
    for module in modules:
      for parameter in module.parameters():
        if parameter.requires_grad:
          yield parameter

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'
    if self.config.backbone == 'crf_dit':
      assert self.parameterization == 'subs', \
        'CRF backbone only supports subs parameterization'
      assert self.T == 0, \
        'CRF backbone only supports continuous time (T=0)'
      assert self.crf_reveal in {
        'independent', 'sequential', 'gated',
        'gated_stochastic'}, \
        f'Unknown CRF reveal mode: {self.crf_reveal}'
      assert self.crf_unigram_aux_warmup_steps >= 0, \
        'CRF unigram auxiliary warmup must be non-negative'
    if self.structured_enabled:
      if self.parameterization != 'subs' or self.T != 0:
        raise ValueError(
          'contextual forests currently support continuous-time SUBS only')
      training_cfg = self.structured_training_config
      structured_training.validate_structured_objective_name(
        str(training_cfg.get('objective_name', '')))
      expected_monitor = (
        'val/structured/conditional_nll_per_masked_token')
      if self.monitor_metric != expected_monitor:
        raise ValueError(
          'structured model.monitor_metric must be '
          f'{expected_monitor!r}, got {self.monitor_metric!r}')
      topology_strategy = str(
        training_cfg.get('topology_strategy', 'gold_reveal_influence'))
      if topology_strategy not in {'none', 'gold_reveal_influence'}:
        raise ValueError(
          'topology_strategy must be none or gold_reveal_influence')
      for name in (
          'structured_nll_weight', 'factorized_aux_weight',
          'topology_weight', 'backbone_lr_multiplier'):
        if float(training_cfg.get(name, 0.0)) < 0:
          raise ValueError(f'{name} must be nonnegative')
      if (self.structured_sampling_mode != 'factorized'
          and self.sampler != 'ddpm'):
        raise ValueError(
          'structured and confidence-gated sampling requires '
          'sampling.predictor=ddpm')
      if (self.structured_sampling_mode != 'factorized'
          and bool(self.config.sampling.semi_ar)):
        raise ValueError(
          'structured marginal/joint sampling does not support '
          'semi-AR strides')
      if ((self.structured_eval_mask_rate is None)
          != (self.structured_eval_corruption_seed is None)):
        raise ValueError(
          'eval.structured_mask_rate and eval.corruption_seed must be set '
          'together for a pinned conditional-denoising evaluation')
      if self.structured_eval_mask_rate is not None:
        if not 0.0 < self.structured_eval_mask_rate < 1.0:
          raise ValueError(
            'eval.structured_mask_rate must be strictly between 0 and 1')
        if self.structured_eval_corruption_seed < 0:
          raise ValueError('eval.corruption_seed must be non-negative')
        if self.config.noise.type != 'loglinear':
          raise ValueError(
            'fixed mask-rate evaluation currently requires '
            'noise.type=loglinear')
        if self.change_of_variables or self.importance_sampling:
          raise ValueError(
            'fixed mask-rate evaluation requires direct time sampling')
        maximum_rate = 1.0 - float(self.noise.eps)
        if self.structured_eval_mask_rate > maximum_rate:
          raise ValueError(
            f'eval.structured_mask_rate exceeds the loglinear maximum '
            f'{maximum_rate}')
      if self.conditional_records_enabled:
        if self.config.mode != 'ppl_eval':
          raise ValueError(
            'eval.conditional_records is only supported in ppl_eval mode')
        if self.structured_eval_mask_rate is None:
          raise ValueError(
            'conditional records require a fixed mask rate and '
            'corruption seed')
        if bool(self.config.eval.generate_samples):
          raise ValueError(
            'conditional record evaluation requires '
            'eval.generate_samples=false')
        if not bool(self.config.data.get(
            'require_pinned_provenance', False)):
          raise ValueError(
            'conditional records require pinned dataset provenance')
        boundary_mode = str(self.config.data.get(
          'valid_document_boundary_mode', 'concatenate'))
        if boundary_mode == 'concatenate':
          raise ValueError(
            'conditional records require document-local validation '
            'windows')
        required_record_fields = (
          'protocol_id', 'job_id', 'arm', 'train_seed')
        missing_record_fields = [
          field for field in required_record_fields
          if self.conditional_record_config.get(field, None) in (None, '')]
        if missing_record_fields:
          raise ValueError(
            'eval.conditional_records is missing '
            f'{missing_record_fields}')
        train_seed = self.conditional_record_config.get('train_seed')
        if (not isinstance(train_seed, int) or isinstance(train_seed, bool)
            or train_seed < 0):
          raise ValueError(
            'eval.conditional_records.train_seed must be a '
            'non-negative integer')
        if self.conditional_record_schema_version not in {1, 2}:
          raise ValueError(
            'eval.conditional_records.schema_version must be 1 or 2')
        support_ks = self.conditional_record_support_candidate_ks
        if self.conditional_record_schema_version == 2:
          if (not support_ks or tuple(sorted(set(support_ks))) != support_ks
              or support_ks[-1] > self.vocab_size
              or support_ks[0] < 1):
            raise ValueError(
              'schema-v2 support_candidate_ks must be increasing, unique, '
              'non-empty, and within the vocabulary')
        elif support_ks:
          raise ValueError(
            'support_candidate_ks require conditional record schema v2')

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      ema_state = checkpoint.get('ema', None)
      disable_eval_ema = bool(
        self.config.mode != 'train'
        and self.config.eval.get('disable_ema', False))
      if disable_eval_ema:
        if ema_state is None:
          warnings.warn(
            'checkpoint has no EMA state and eval.disable_ema=true; '
            'continuing with raw parameters', stacklevel=2)
        self.ema = None
      else:
        shadows = structured_training.validated_ema_shadow_parameters(
          ema_state=ema_state,
          expected_parameters=list(self._trainable_model_parameters()),
          context='full Lightning checkpoint',
          allow_extra=False)
        normalized_ema_state = dict(ema_state)
        normalized_ema_state['shadow_params'] = shadows
        self.ema.load_state_dict(normalized_ema_state)
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if isinstance(dl.dataset, torch.utils.data.IterableDataset):
        # Streaming datasets own iteration, worker sharding, and epoch state;
        # they have no length or random-access index for our resumable sampler.
        updated_dls.append(dl)
        continue
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=(self.config.loader.num_workers > 0)))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._trainable_model_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(
      batch['input_ids'], attention_mask, phase=prefix)
    loss = losses.loss

    if prefix not in {'train', 'val', 'test'}:
      raise ValueError(f'Invalid prefix: {prefix}')
    if not self.structured_enabled:
      if prefix == 'train':
        metrics = self.train_metrics
      elif prefix == 'val':
        metrics = self.valid_metrics
      else:
        metrics = self.test_metrics
      metrics.update(losses.nlls, losses.token_mask)
      self.log_dict(metrics,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)
    if self.structured_enabled and self._last_structured_metrics:
      if prefix == 'train':
        weighted_metrics = self.structured_train_metrics
      elif prefix == 'val':
        weighted_metrics = self.structured_valid_metrics
      else:
        weighted_metrics = self.structured_test_metrics
      for name, update_arguments in (
          self._last_structured_metric_updates.items()):
        weighted_metrics[name].update(*update_arguments)
      self.log_dict({
        f'{prefix}/structured/{name}': metric
        for name, metric in weighted_metrics.items()
      }, on_step=False, on_epoch=True, sync_dist=True)
      self.log_dict({
        f'{prefix}/structured/{name}': value
        for name, value in self._last_structured_metrics.items()
      }, on_step=(prefix == 'train'), on_epoch=True, sync_dist=True,
        batch_size=batch['input_ids'].shape[0])
    return loss

  def on_train_epoch_start(self):
    if (self.structured_enabled
        and self.structured_deterministic_backbone):
      # Influence distillation compares two corruption views; disabling
      # dropout prevents stochastic view noise from becoming a topology label.
      self.backbone.eval()
    else:
      self.backbone.train()
    if self.structured_head is not None:
      self.structured_head.train()
    self.noise.train()
    if self.structured_enabled:
      generator_device = (
        self.device if self.device.type == 'cuda'
        else torch.device('cpu'))
      # Keep forward corruptions independent of topology-teacher sampling.
      # Otherwise dynamic-topology arms consume an extra global-RNG draw and
      # cease to be paired with fixed-topology controls after the first batch.
      corruption_seed, topology_seed = _structured_training_rng_seeds(
        int(self.config.seed), int(self.current_epoch), int(self.global_rank))
      self._structured_training_corruption_generator = torch.Generator(
        device=generator_device)
      self._structured_training_corruption_generator.manual_seed(
        corruption_seed)
      self._structured_training_topology_generator = torch.Generator(
        device=generator_device)
      self._structured_training_topology_generator.manual_seed(
        topology_seed)

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    if (self.config.backbone == 'crf_dit'
        and self.crf_unigram_aux_enabled):
      self.log(name='trainer/crf_unigram_aux_nll',
               value=self._last_crf_unigram_aux_nll,
               on_step=True,
               on_epoch=False,
               sync_dist=True)
      self.log(name='trainer/crf_unigram_aux_weight',
               value=self._last_crf_unigram_aux_weight,
               on_step=True,
               on_epoch=False,
               sync_dist=True)
    return loss

  def _conditional_record_metadata(self):
    return {
      'protocol_id': str(
        self.conditional_record_config.get('protocol_id')),
      'job_id': str(self.conditional_record_config.get('job_id')),
      'arm': str(self.conditional_record_config.get('arm')),
      'train_seed': int(
        self.conditional_record_config.get('train_seed')),
      'corruption_seed': int(self.structured_eval_corruption_seed),
      'dataset': str(self.config.data.valid),
      'dataset_revision': str(self.config.data.valid_revision),
      'mask_rate': float(self.structured_eval_mask_rate),
      'candidate_k': int(self.structured_config.get('top_k')),
    }

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(self._trainable_model_parameters())
      self.ema.copy_to(self._trainable_model_parameters())
    self.backbone.eval()
    if self.structured_head is not None:
      self.structured_head.eval()
    self.noise.eval()
    if self.structured_enabled:
      if self.structured_eval_corruption_seed is not None:
        generator_device = (
          self.device if self.device.type == 'cuda'
          else torch.device('cpu'))
        self._structured_validation_generator = torch.Generator(
          device=generator_device)
        self._structured_validation_generator.manual_seed(
          self.structured_eval_corruption_seed + int(self.global_rank))
      else:
        self._structured_validation_generator = None
      self._structured_validation_pairing_digest = (
        structured_pairing.StructuredValidationPairingDigest())
      if (self.conditional_records_enabled
          and not bool(self.trainer.sanity_checking)):
        self._conditional_record_writer = (
          conditional_denoising_records.
          ConditionalDenoisingRecordWriter(
            output_dir=str(self.config.checkpointing.save_dir),
            rank=int(self.global_rank),
            metadata=self._conditional_record_metadata(),
            schema_version=self.conditional_record_schema_version,
            support_candidate_ks=(
              self.conditional_record_support_candidate_ks)))
      else:
        self._conditional_record_writer = None
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='val')
    if self._conditional_record_writer is not None:
      self._conditional_record_writer.append(
        batch=batch,
        metrics=self._last_structured_example_metrics,
        batch_index=batch_idx)
    return loss

  def on_validation_epoch_end(self):
    if (self.structured_enabled
        and self._structured_validation_pairing_digest is not None):
      local_record = (
        self._structured_validation_pairing_digest.rank_record(
          int(self.global_rank)))
      if (torch.distributed.is_available()
          and torch.distributed.is_initialized()):
        rank_records = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_records, local_record)
      else:
        rank_records = [local_record]
      payload = structured_pairing.combine_rank_records(
        rank_records,
        epoch=int(self.current_epoch),
        step=int(self.global_step),
        sanity_checking=bool(self.trainer.sanity_checking))
      if self._conditional_record_writer is not None:
        local_summary = self._conditional_record_writer.finalize(
          pairing_digest_sha256=payload['sha256'])
        if (torch.distributed.is_available()
            and torch.distributed.is_initialized()):
          rank_summaries = [None] * torch.distributed.get_world_size()
          torch.distributed.all_gather_object(
            rank_summaries, local_summary)
        else:
          rank_summaries = [local_summary]
        if self.trainer.is_global_zero:
          conditional_denoising_records.write_record_manifest(
            output_dir=str(self.config.checkpointing.save_dir),
            metadata=self._conditional_record_metadata(),
            rank_summaries=rank_summaries,
            pairing_digest=payload,
            schema_version=self.conditional_record_schema_version)
        self._conditional_record_writer = None
      if self.trainer.is_global_zero:
        structured_pairing.write_pairing_digest(
          str(self.config.checkpointing.save_dir), payload)
      self._structured_validation_pairing_digest = None
      self._structured_validation_generator = None
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(self._trainable_model_parameters())

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    parameter_groups = None
    if self.structured_enabled:
      training_cfg = self.structured_training_config
      configured_head_lr = training_cfg.get('head_lr', None)
      head_lr = (self.config.optim.lr if configured_head_lr is None
                 else float(configured_head_lr))
      backbone_lr = (
        self.config.optim.lr
        * float(training_cfg.get('backbone_lr_multiplier', 0.05)))
      head_and_noise = [
        parameter for module in (self.structured_head, self.noise)
        for parameter in module.parameters() if parameter.requires_grad]
      backbone_parameters = [
        parameter for parameter in self.backbone.parameters()
        if parameter.requires_grad]
      parameter_groups = []
      if head_and_noise:
        parameter_groups.append({
          'params': head_and_noise, 'lr': head_lr,
          'name': 'structured_head'})
      if backbone_parameters:
        parameter_groups.append({
          'params': backbone_parameters, 'lr': backbone_lr,
          'name': 'backbone'})
    else:
      parameter_groups = list(self._trainable_model_parameters())
    optimizer = torch.optim.AdamW(
      parameter_groups,
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': self.monitor_metric,
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if self.eval_model_tokenizer is None:
      raise RuntimeError(
        'generative-perplexity tokenizer was not initialized; set '
        'eval.compute_generative_perplexity=true before model construction')
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance, *, generator=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device, generator=generator) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _crf_unigram_aux_weight(self):
    """Current scheduled weight for the CRF pruning-head objective."""
    trainer = getattr(self, '_trainer', None)
    step = int(getattr(trainer, 'global_step', 0))
    return crf_utils.linear_warmup_weight(
      step=step,
      start_weight=self.crf_unigram_aux_start_weight,
      end_weight=self.crf_unigram_aux_end_weight,
      warmup_steps=self.crf_unigram_aux_warmup_steps)

  def _crf_gated_update(self, x, p_x0, move_chance_t,
                        move_chance_s):
    """Reveal the most-confident masked sites at the DDPM step count."""
    if self.crf_reveal == 'sequential':
      reveal = crf_utils.sequential_reveal_mask(
        x=x,
        probabilities=p_x0,
        mask_index=self.mask_index)
    else:
      reveal = crf_utils.confident_reveal_mask(
        x=x,
        probabilities=p_x0,
        mask_index=self.mask_index,
        move_chance_t=move_chance_t,
        move_chance_s=move_chance_s)

    token_probs = p_x0.clone()
    token_probs[..., self.mask_index] = 0
    if self.crf_reveal in {'sequential', 'gated'}:
      proposed_tokens = token_probs.argmax(dim=-1)
    elif self.crf_reveal == 'gated_stochastic':
      proposed_tokens = _sample_categorical(token_probs)
    else:
      raise ValueError(
        f'_crf_gated_update called for mode {self.crf_reveal}')
    return torch.where(reveal, proposed_tokens, x)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      if self.config.backbone == 'crf_dit':
        sigma_1d = self._process_sigma(sigma_t)
        p_x0 = self._compute_crf_marginals(x, sigma_1d)
      else:
        p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    if (self.config.backbone == 'crf_dit'
        and self.crf_reveal != 'independent'):
      return p_x0, self._crf_gated_update(
        x, p_x0, move_chance_t, move_chance_s)

    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  @torch.no_grad()
  def _structured_clean_sample(self, x, conditioning):
    """Draw from exact node marginals or the exact joint forest law."""
    active_mask = x.eq(self.mask_index)
    if not bool(active_mask.any().item()):
      return x
    if self.structured_sampling_mode == 'factorized':
      raise RuntimeError(
        '_structured_clean_sample requires a structured sampling mode')
    output, unary_logits = self._structured_head_output(
      tokens=x,
      conditioning=conditioning,
      active_mask=active_mask,
      force_no_grad_backbone=True)
    if self.structured_sampling_mode == 'structured_joint':
      clean = structured_objective.sample_structured_tokens(
        output=output,
        unary_logits=unary_logits,
        active_mask=active_mask,
        num_samples=1)[:, 0]
    elif self.structured_sampling_mode == 'structured_marginal':
      clean = structured_objective.sample_structured_marginal_tokens(
        output=output,
        unary_logits=unary_logits,
        active_mask=active_mask,
        num_samples=1)[:, 0]
    else:
      raise RuntimeError(
        f'unsupported structured sampling mode '
        f'{self.structured_sampling_mode!r}')
    return torch.where(active_mask, clean, x)

  @torch.no_grad()
  def _factorized_confidence_gated_ddpm_update(self, x, p_x0,
                                                move_chance_t,
                                                move_chance_s):
    """Stochastically sample factorized tokens at confidence-ranked sites.

    This is a selection-only control.  It uses the unchanged backbone
    marginals for both token confidence and independent categorical token
    draws.  The gate deterministically reveals the most-confident masked
    positions while matching the absorbing schedule's per-row remaining-mask
    count up to integer rounding.  No coupling-head value enters this update.
    """
    return crf_utils.factorized_confidence_gated_update(
      x=x,
      probabilities=p_x0,
      mask_index=self.mask_index,
      move_chance_t=move_chance_t,
      move_chance_s=move_chance_s)

  @torch.no_grad()
  def _structured_ddpm_update(self, x, t, dt):
    """Sample structured token identities, then apply the reveal kernel."""
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    proposed_clean = self._structured_clean_sample(x, sigma_t)
    reveal_probability = (
      (move_chance_t - move_chance_s)
      / move_chance_t.clamp_min(1e-12)).clamp(0.0, 1.0)
    reveal = (
      torch.rand(x.shape, device=x.device)
      < reveal_probability[:, None])
    reveal = reveal & x.eq(self.mask_index)
    return torch.where(reveal, proposed_clean, x)

  def _ddpm_update(self, x, t, dt):
    if (self.structured_enabled
        and structured_training.uses_structured_token_distribution(
          self.structured_sampling_mode)):
      return self._structured_ddpm_update(x, t, dt)
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t

    if self.config.backbone == 'crf_dit':
      p_x0 = self._compute_crf_marginals(
        x, unet_conditioning)
    else:
      log_p_x0 = self.forward(x, unet_conditioning)
      p_x0 = log_p_x0.exp()

    assert move_chance_t.ndim == p_x0.ndim
    if (self.structured_enabled
        and self.structured_sampling_mode
        == 'factorized_confidence_gated'):
      return self._factorized_confidence_gated_ddpm_update(
        x, p_x0, move_chance_t, move_chance_s)
    if (self.config.backbone == 'crf_dit'
        and self.crf_reveal != 'independent'):
      return self._crf_gated_update(
        x, p_x0, move_chance_t, move_chance_s)

    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  @torch.no_grad()
  def _compute_crf_marginals(self, x, sigma):
    """Compute per-position marginals under the first-order CRF.

    Uses the encoder for top-K candidate selection, the CRF
    decoder for transition matrices, and forward-backward DP
    for exact marginals over the pruned candidate set.

    Args:
      x: (batch, seq_len) noisy token indices
      sigma: (batch,) noise level (1-D, already processed)
    Returns:
      marginals: (batch, seq_len, vocab_size) probabilities
    """
    sigma = self._process_sigma(sigma)
    H, c = self.backbone.encode(x, sigma)
    batch, seq_len, _ = H.shape
    K = self.backbone.top_k
    device = x.device

    # --- Unigram logits for top-K selection ---
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      unigram_logits = self.backbone.output_layer(H, c)
    unigram_logits = unigram_logits.float()
    unigram_logits[:, :, self.mask_index] = self.neg_infinity

    # For unmasked positions, force x_t as the only candidate
    unmasked = (x != self.mask_index)
    if unmasked.any():
      det_logits = torch.full_like(
        unigram_logits, self.neg_infinity)
      det_logits.scatter_(-1, x.unsqueeze(-1), 0.0)
      unigram_logits = torch.where(
        unmasked.unsqueeze(-1), det_logits, unigram_logits)

    _, top_k_indices = unigram_logits.topk(K, dim=-1)
    # At observed positions top-k still contains K-1 arbitrary entries with
    # very negative pruning scores. Mark them invalid so they cannot carry
    # forward/backward probability into neighboring positions.
    candidate_valid = (
      (~unmasked).unsqueeze(-1)
      | top_k_indices.eq(x.unsqueeze(-1)))

    # --- Position 0: emission from start embedding ---
    pos0_dummy = torch.zeros(
      batch, 1, dtype=torch.long, device=device)
    pos0_pos = torch.zeros(
      1, dtype=torch.long, device=device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      pos0_logits = self.backbone.crf_decoder(
        pos0_dummy, pos0_pos, H, use_start_for_first=True)
    pos0_logits = pos0_logits.float().squeeze(1)
    pos0_logits[:, self.mask_index] = self.neg_infinity
    pos0_log_probs = F.log_softmax(pos0_logits, dim=-1)
    emission_0 = torch.gather(
      pos0_log_probs, 1, top_k_indices[:, 0, :])

    # --- Transitions for positions 1..N-1 ---
    if seq_len > 1:
      prev_candidates = top_k_indices[:, :-1, :]
      prev_flat = prev_candidates.reshape(
        batch, (seq_len - 1) * K)

      pos_ids = torch.arange(
        1, seq_len, device=device).repeat_interleave(K)

      curr_topk = top_k_indices[:, 1:, :]
      curr_flat = curr_topk.unsqueeze(2).expand(
        -1, -1, K, -1).reshape(
          batch, (seq_len - 1) * K, K)

      def decode_query_chunk(start, end):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          return self.backbone.crf_decoder.forward_batched(
            prev_flat[:, start:end], pos_ids[start:end], H)

      # A dense [B, (N-1)K, V] tensor is many GB at the intended
      # N=1024, K=64, V~50k scale. Query chunking retains the exact
      # full-vocabulary log normalizer while bounding peak decoder output.
      selected_log_probs = crf_utils.chunked_normalized_gather(
        logits_fn=decode_query_chunk,
        gather_indices=curr_flat,
        query_chunk_size=(
          self.backbone.inference_query_chunk_size),
        excluded_index=self.mask_index,
        neg_infinity=self.neg_infinity)
      transitions = selected_log_probs.view(
        batch, seq_len - 1, K, K)
    else:
      transitions = torch.zeros(
        batch, 0, K, K, device=device)

    emission_0, transitions = crf_utils.constrain_chain_potentials(
      emission_0=emission_0,
      transitions=transitions,
      candidate_valid=candidate_valid,
      neg_infinity=self.neg_infinity)

    # --- Forward-backward ---
    marginals_topk = crf_utils.forward_backward(
      emission_0, transitions)

    # --- Scatter marginals to full vocab ---
    full_marginals = torch.zeros(
      batch, seq_len, self.vocab_size,
      device=device, dtype=marginals_topk.dtype)
    full_marginals.scatter_(2, top_k_indices, marginals_topk)

    # SUBS: unmasked positions are deterministic
    if unmasked.any():
      det_dist = torch.zeros_like(full_marginals)
      det_dist.scatter_(-1, x.unsqueeze(-1), 1.0)
      full_marginals = torch.where(
        unmasked.unsqueeze(-1), det_dist, full_marginals)

    return full_marginals

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      elif self.config.backbone == 'crf_dit':
        unet_conditioning = self.noise(t)[0]
        sigma_1d = self._process_sigma(unet_conditioning)
        p_x0 = self._compute_crf_marginals(x, sigma_1d)
        x = p_x0.argmax(dim=-1)
      elif (self.structured_enabled
            and structured_training.uses_structured_token_distribution(
              self.structured_sampling_mode)):
        unet_conditioning = self.noise(t)[0]
        x = self._structured_clean_sample(x, unet_conditioning)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(self._trainable_model_parameters())
      self.ema.copy_to(self._trainable_model_parameters())
    self.backbone.eval()
    if self.structured_head is not None:
      self.structured_head.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(self._trainable_model_parameters())
    self.backbone.train()
    if self.structured_head is not None:
      self.structured_head.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device, *, generator=None):
    _eps_t = torch.rand(n, device=device, generator=generator)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _structured_backbone_output(
      self, tokens, conditioning, force_no_grad=False):
    """Encode once and return hidden states plus mask-excluded raw unaries."""
    backbone_conditioning = self._process_sigma(conditioning)

    def run_backbone():
      hidden_states, time_conditioning = self.backbone.encode(
        tokens, backbone_conditioning)
      unary_logits = self.backbone.decode(
        hidden_states, time_conditioning).float()
      # The absorbing mask is not a clean-token candidate.  This exclusion is
      # based only on the known vocabulary role, never on x0.
      unary_logits = unary_logits.clone()
      unary_logits[:, :, self.mask_index] = -torch.inf
      return hidden_states, unary_logits

    if force_no_grad or self.structured_backbone_mode == 'frozen':
      with torch.no_grad():
        return run_backbone()
    return run_backbone()

  def _structured_head_output(
      self, tokens, conditioning, active_mask,
      force_no_grad_backbone=False, return_hidden_states=False):
    hidden_states, unary_logits = self._structured_backbone_output(
      tokens=tokens,
      conditioning=conditioning,
      force_no_grad=force_no_grad_backbone)
    if conditioning.ndim == 2 and conditioning.shape[-1] == 1:
      head_timestep = conditioning[:, 0]
    else:
      head_timestep = conditioning
    fixed_edge_mask = None
    if self.structured_fixed_edge_index is not None:
      fixed_edges = self.structured_fixed_edge_index
      fixed_edge_mask = (
        active_mask[:, fixed_edges[:, 0]]
        & active_mask[:, fixed_edges[:, 1]])
    output = self.structured_head(
      hidden_states=hidden_states,
      unary_logits=unary_logits,
      timestep=head_timestep,
      active_mask=active_mask,
      fixed_edge_index=self.structured_fixed_edge_index,
      fixed_edge_mask=fixed_edge_mask)
    if return_hidden_states:
      return output, unary_logits, hidden_states
    return output, unary_logits

  def _forward_pass_structured(
      self, x0, attention_mask, *, record_pairing=False):
    """Train the forest adapter on a real forward-corrupted text batch.

    The returned objective is conditional denoising NLL per masked token plus
    optional topology distillation and factorized-unary auxiliary terms.  It
    is intentionally not reported as a diffusion ELBO.
    """
    if record_pairing:
      corruption_generator = self._structured_validation_generator
    elif self.training:
      if self._structured_training_corruption_generator is None:
        raise RuntimeError(
          'structured training corruption generator was not initialized')
      corruption_generator = self._structured_training_corruption_generator
    else:
      corruption_generator = None
    fixed_mask_rate = (
      self.structured_eval_mask_rate if record_pairing else None)
    if fixed_mask_rate is not None:
      # Under LogLinearNoise, q(x_t != x_0) = (1 - eps) * t.  Deriving t
      # from the requested rate keeps both the corruption and the head's
      # timestep conditioning on the same pinned forward-process point.
      t = torch.full(
        (x0.shape[0],),
        _loglinear_time_from_mask_rate(
          fixed_mask_rate, float(self.noise.eps)),
        device=x0.device,
        dtype=torch.float32)
      sigma, _ = self.noise(t)
      conditioning = sigma[:, None]
      move_chance = torch.full(
        (x0.shape[0], 1), fixed_mask_rate,
        device=x0.device, dtype=torch.float32)
    else:
      t = self._sample_t(
        x0.shape[0], x0.device, generator=corruption_generator)
    if fixed_mask_rate is not None:
      pass
    elif self.change_of_variables:
      conditioning = t[:, None]
      f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
      f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]
    else:
      sigma, _ = self.noise(t)
      conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])
    xt = self.q_xt(
      x0, move_chance, generator=corruption_generator)
    if attention_mask is None:
      attention_mask = torch.ones_like(x0, dtype=torch.bool)
    else:
      attention_mask = attention_mask.bool()
    active_mask = xt.eq(self.mask_index) & attention_mask
    if record_pairing:
      if self._structured_validation_pairing_digest is None:
        raise RuntimeError(
          'structured validation pairing digest was not initialized')
      self._structured_validation_pairing_digest.update(
        clean_input_ids=x0,
        attention_mask=attention_mask,
        sampled_times=t,
        corrupted_input_ids=xt,
        active_mask=active_mask)

    output, unary_logits, hidden_states = self._structured_head_output(
      tokens=xt,
      conditioning=conditioning,
      active_mask=active_mask,
      return_hidden_states=True)
    denoising = structured_training.structured_denoising_loss(
      output=output,
      unary_logits=unary_logits,
      clean_tokens=x0,
      active_mask=active_mask)
    target_is_explicit = output.candidate_ids.eq(
      x0[:, :, None]).any(dim=-1)
    self._last_structured_example_metrics = {
      'nll_sum': denoising.per_example_nll.detach(),
      'active_tokens': active_mask.sum(dim=-1).detach(),
      'candidate_hits': (
        target_is_explicit & active_mask).sum(dim=-1).detach(),
      'retained_mass_sum': output.retained_mass.masked_fill(
        ~active_mask, 0.0).sum(dim=-1).detach(),
    }
    if (record_pairing and self.conditional_records_enabled
        and self.conditional_record_schema_version == 2):
      head_timestep = (
        conditioning[:, 0]
        if conditioning.ndim == 2 and conditioning.shape[-1] == 1
        else conditioning)
      causal_metrics = causal_denoising.causal_denoising_metrics(
        head=self.structured_head,
        primary_output=output,
        hidden_states=hidden_states,
        unary_logits=unary_logits,
        timestep=head_timestep,
        clean_tokens=x0,
        active_mask=active_mask,
        candidate_ks=self.conditional_record_support_candidate_ks,
        topology_permutation_seed=(
          int(self.structured_eval_corruption_seed) + 1_000_000_007),
        structured_joint_nll_sum=denoising.per_example_nll.detach())
      self._last_structured_example_metrics.update(
        causal_metrics.as_record_metrics())

    training_cfg = self.structured_training_config
    factorized_auxiliary = structured_training.factorized_denoising_nll(
      unary_logits=unary_logits,
      clean_tokens=x0,
      active_mask=active_mask)
    topology_zero = torch.where(
      torch.isfinite(output.proposal_scores),
      output.proposal_scores,
      torch.zeros_like(output.proposal_scores)).sum() * 0.0
    topology_loss = topology_zero
    topology_edge_loss = topology_zero
    topology_anchor_loss = topology_zero
    topology_slot_loss = topology_zero
    topology_valid_examples = topology_zero.detach()
    topology_influence = topology_zero.detach()
    topology_edge_coverage_numerator = topology_zero.detach()
    topology_edge_coverage_denominator = topology_zero.detach()
    topology_anchor_coverage_numerator = topology_zero.detach()
    topology_anchor_coverage_denominator = topology_zero.detach()
    topology_slot_coverage_numerator = topology_zero.detach()
    topology_slot_coverage_denominator = topology_zero.detach()
    topology_weight = float(training_cfg.get('topology_weight', 0.0))
    topology_strategy = str(
      training_cfg.get('topology_strategy', 'gold_reveal_influence'))
    train_topology = (
      topology_weight > 0.0
      and topology_strategy == 'gold_reveal_influence'
      and output.topology_mode == 'dynamic'
      and not output.independent_mode
      and (self.training
           or bool(training_cfg.get('topology_on_validation', False))))
    if train_topology:
      topology_generator = (
        self._structured_training_topology_generator
        if self.training else None)
      sources = structured_training.sample_active_sources(
        active_mask, generator=topology_generator)
      revealed_xt = xt.clone()
      valid_sources = sources >= 0
      batch_index = torch.arange(x0.shape[0], device=x0.device)
      revealed_xt[
        batch_index[valid_sources], sources[valid_sources]
      ] = x0[batch_index[valid_sources], sources[valid_sources]]
      _, revealed_logits = self._structured_backbone_output(
        tokens=revealed_xt,
        conditioning=conditioning,
        force_no_grad=True)
      topology = (
        structured_training.gold_reveal_influence_topology_loss(
          output=output,
          base_unary_logits=unary_logits.detach(),
          revealed_unary_logits=revealed_logits,
          clean_tokens=x0,
          active_mask=active_mask,
          source_positions=sources,
          temperature=float(
            training_cfg.get('topology_temperature', 0.25)),
          minimum_choices=int(
            training_cfg.get('topology_minimum_choices', 2)),
          edge_weight=float(
            training_cfg.get('topology_edge_weight', 1.0)),
          anchor_weight=float(
            training_cfg.get('topology_anchor_weight', 0.25)),
          slot_weight=float(
            training_cfg.get('topology_slot_weight', 0.25))))
      topology_loss = topology.loss
      topology_edge_loss = topology.edge_loss
      topology_anchor_loss = topology.anchor_loss
      topology_slot_loss = topology.slot_loss
      topology_valid_examples = topology.valid_examples
      topology_influence = topology.mean_influence
      topology_edge_coverage_numerator = (
        topology.edge_coverage_numerator)
      topology_edge_coverage_denominator = (
        topology.edge_coverage_denominator)
      topology_anchor_coverage_numerator = (
        topology.anchor_coverage_numerator)
      topology_anchor_coverage_denominator = (
        topology.anchor_coverage_denominator)
      topology_slot_coverage_numerator = (
        topology.slot_coverage_numerator)
      topology_slot_coverage_denominator = (
        topology.slot_coverage_denominator)

    structured_weight = float(
      training_cfg.get('structured_nll_weight', 1.0))
    factorized_weight = float(
      training_cfg.get('factorized_aux_weight', 0.0))
    total_loss = (
      structured_weight * denoising.loss
      + factorized_weight * factorized_auxiliary
      + topology_weight * topology_loss)
    attention_tokens = attention_mask.sum().clamp_min(1)
    self._last_structured_metrics = {
      'loss': total_loss.detach(),
      'active_fraction': (
        denoising.active_tokens.detach() / attention_tokens),
      'selected_edges': output.edge_mask.sum(dim=-1).float().mean().detach(),
      'factorized_aux_nll': factorized_auxiliary.detach(),
      'topology_loss': topology_loss.detach(),
      'topology_edge_loss': topology_edge_loss.detach(),
      'topology_anchor_loss': topology_anchor_loss.detach(),
      'topology_slot_loss': topology_slot_loss.detach(),
      'topology_valid_examples': topology_valid_examples.detach(),
      'topology_teacher_influence': topology_influence.detach(),
    }
    self._last_structured_metric_updates = {
      'conditional_nll_per_masked_token': (
        denoising.nll_sum.detach(), denoising.active_tokens.detach()),
      'candidate_recall': (
        denoising.candidate_hits.detach(),
        denoising.active_tokens.detach()),
      'retained_unary_mass': (
        denoising.retained_mass_sum.detach(),
        denoising.active_tokens.detach()),
      'topology_edge_coverage': (
        topology_edge_coverage_numerator.detach(),
        topology_edge_coverage_denominator.detach()),
      'topology_edge_coverage_denominator': (
        topology_edge_coverage_denominator.detach(),),
      'topology_anchor_coverage': (
        topology_anchor_coverage_numerator.detach(),
        topology_anchor_coverage_denominator.detach()),
      'topology_anchor_coverage_denominator': (
        topology_anchor_coverage_denominator.detach(),),
      'topology_slot_coverage': (
        topology_slot_coverage_numerator.detach(),
        topology_slot_coverage_denominator.detach()),
      'topology_slot_coverage_denominator': (
        topology_slot_coverage_denominator.detach(),),
    }
    return total_loss, denoising.distributed_nll, active_mask

  def _forward_pass_diffusion(self, x0, return_components=False):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)

    unigram_logits = None
    if self.config.backbone == 'crf_dit':
      sigma_1d = self._process_sigma(unet_conditioning)
      crf_outputs = self.backbone.forward_crf_train(
        xt, sigma_1d, x0,
        return_unigram_logits=self.crf_unigram_aux_enabled)
      if self.crf_unigram_aux_enabled:
        crf_logits, unigram_logits = crf_outputs
      else:
        crf_logits = crf_outputs
      model_output = self._subs_parameterization(
        logits=crf_logits, xt=xt)
    else:
      model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      constant = torch.log1p(
        - torch.exp(- self.noise.sigma_min))
      primary_loss = log_p_theta * constant
      aux_token_weight = -constant
    else:
      aux_token_weight = dsigma / torch.expm1(sigma)
      primary_loss = -log_p_theta * aux_token_weight[:, None]

    unigram_aux_loss = torch.zeros_like(primary_loss)
    unigram_aux_weight = 0.0
    if unigram_logits is not None:
      unigram_aux_loss = crf_utils.unigram_denoising_loss(
        logits=unigram_logits,
        xt=xt,
        x0=x0,
        mask_index=self.mask_index,
        token_weight=aux_token_weight)
      unigram_aux_weight = self._crf_unigram_aux_weight()

    objective_loss = (
      primary_loss + unigram_aux_weight * unigram_aux_loss)
    if return_components:
      return (objective_loss, primary_loss,
              unigram_aux_loss, unigram_aux_weight)
    return objective_loss

  def _loss(self, x0, attention_mask, phase=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.structured_enabled:
      (structured_loss, metric_losses,
       structured_token_mask) = self._forward_pass_structured(
         input_tokens, attention_mask,
         record_pairing=(phase == 'val'))
      return Loss(
        loss=structured_loss,
        nlls=metric_losses,
        token_mask=structured_token_mask)
    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      objective_losses = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
      metric_losses = objective_losses
    else:
      if self.config.backbone == 'crf_dit':
        (objective_losses, metric_losses,
         unigram_aux_losses,
         unigram_aux_weight) = self._forward_pass_diffusion(
           input_tokens, return_components=True)
      else:
        objective_losses = self._forward_pass_diffusion(input_tokens)
        metric_losses = objective_losses

    nlls = metric_losses * attention_mask
    objective_nlls = objective_losses * attention_mask
    count = attention_mask.sum()

    token_objective = objective_nlls.sum() / count
    if (self.config.backbone == 'crf_dit'
        and self.crf_unigram_aux_enabled):
      self._last_crf_unigram_aux_nll = (
        unigram_aux_losses.detach() * attention_mask).sum() / count
      self._last_crf_unigram_aux_weight = unigram_aux_weight

    return Loss(loss=token_objective,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(self._trainable_model_parameters())
      self.ema.copy_to(self._trainable_model_parameters())
    self.backbone.eval()
    if self.structured_head is not None:
      self.structured_head.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(self._trainable_model_parameters())
    self.backbone.train()
    if self.structured_head is not None:
      self.structured_head.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
