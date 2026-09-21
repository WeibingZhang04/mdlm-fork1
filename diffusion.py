import itertools
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

import dataloader
import models
import noise_schedule
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


def _structured_training_rng_seeds(
    base_seed: int, epoch: int, rank: int) -> tuple[int, int]:
  """Domain-separate paired corruption and topologyq-teacher RNG streams."""
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
    tokenizer: transformers.PreTrainedTokenizer,
    initialize_pretrained_backbone: bool = True):
    super().__init__()
    self.save_hyperparameters(ignore=['initialize_pretrained_backbone'])
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
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
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.structured_head = None
    self.structured_enabled = False
    self.structured_sampling_mode = 'factorized'
    self.structured_config = None
    self.structured_training_config = None
    self._structured_training_corruption_generator = None
    self._structured_training_topology_generator = None
    self._last_structured_topology_metrics = {}
    self._last_structured_metric_updates = {}
    self._initialize_structured_decoder(initialize_pretrained_backbone)

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

    if self.structured_enabled:
      conditional_metrics = torchmetrics.MetricCollection({
        'conditional_nll_per_masked_token': RatioMetric(),
      })
      conditional_metrics.set_dtype(torch.float64)
      self.structured_train_metrics = conditional_metrics.clone(prefix='train/')
      self.structured_valid_metrics = conditional_metrics.clone(prefix='val/')
      self.structured_test_metrics = conditional_metrics.clone(prefix='test/')
      diagnostic_metrics = {
        name: RatioMetric() for name in (
          'factorized_nll_per_masked_token', 'candidate_recall',
          'retained_unary_mass', 'active_fraction',
          'teacher_eligible_fraction', 'topology_loss_per_batch')}
      diagnostic_metrics.update({
        'active_tokens': DistributedSumMetric(),
        'teacher_examples': DistributedSumMetric()})
      for view in ('edge', 'anchor', 'slot'):
        diagnostic_metrics[f'topology_{view}_loss'] = RatioMetric()
        diagnostic_metrics[f'topology_{view}_coverage'] = RatioMetric()
        for count in ('valid_examples', 'coverage_numerator', 'coverage_denominator'):
          diagnostic_metrics[f'topology_{view}_{count}'] = DistributedSumMetric()
      # Different diagnostics have different update arguments/denominators.
      diagnostics = torchmetrics.MetricCollection(
        diagnostic_metrics, compute_groups=False)
      diagnostics.set_dtype(torch.float64)
      self.structured_train_diagnostics = diagnostics.clone(prefix='train/structured/')
      self.structured_valid_diagnostics = diagnostics.clone(prefix='val/structured/')
      self.structured_test_diagnostics = diagnostics.clone(prefix='test/structured/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = None
    if self.config.eval.get('compute_generative_perplexity', True):
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
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _initialize_structured_decoder(self, initialize_pretrained_backbone=True):
    """Initialize CCF; a full model checkpoint supplies its own backbone."""
    structured_cfg = self.config.model.get('structured_decoder', None)
    if structured_cfg is None or not bool(
        structured_cfg.get('enabled', False)):
      return
    if self.config.backbone != 'dit' or self.parameterization != 'subs':
      raise ValueError('CCF training requires backbone=dit and parameterization=subs')
    if (self.config.T != 0 or self.importance_sampling
        or self.change_of_variables):
      raise ValueError('the restored CCF path requires continuous-time uniform sampling')
    training_cfg = structured_cfg.get('training', {})
    if training_cfg.get('backbone_mode', 'frozen') != 'frozen':
      raise ValueError('the restored four-arm training path requires a frozen backbone')
    if self.config.training.ema > 0:
      raise ValueError('set training.ema=0 for the restored four-arm training path')
    if bool(training_cfg.get('use_ema_backbone', False)):
      raise ValueError('the restored four-arm path loads raw checkpoint weights, not EMA')
    if not bool(training_cfg.get('strict_backbone_checkpoint', True)):
      raise ValueError('CCF backbone checkpoint loading must be strict')
    if not bool(training_cfg.get('deterministic_backbone', True)):
      raise ValueError('the frozen four-arm backbone must use evaluation mode')
    topology_weight = float(training_cfg.get('topology_weight', 0.0))
    if not math.isfinite(topology_weight) or topology_weight < 0:
      raise ValueError('topology_weight must be finite and non-negative')
    if training_cfg.get('topology_strategy', 'gold_reveal_influence') != 'gold_reveal_influence':
      raise ValueError('only gold_reveal_influence topology supervision is restored')
    checkpoint_path = (training_cfg.get('backbone_checkpoint', None)
                       if initialize_pretrained_backbone else None)
    if initialize_pretrained_backbone and not checkpoint_path and bool(
        training_cfg.get('require_pretrained_backbone', True)):
      raise ValueError(
        'set model.structured_decoder.training.backbone_checkpoint; '
        'require_pretrained_backbone=false is only for random-backbone smoke tests')

    from models.structured_decoder import ContextualCouplingForestHead
    from structured_training import validate_structured_sampling_mode

    self.structured_sampling_mode = validate_structured_sampling_mode(
      str(structured_cfg.get('sampling', {}).get('mode', 'factorized')))
    if self.structured_sampling_mode == 'structured_joint':
      if self.sampler != 'ddpm':
        raise ValueError('structured_joint sampling requires sampling.predictor=ddpm')
      if self.config.sampling.get('semi_ar', False):
        raise ValueError('structured_joint sampling does not support semi-AR strides')

    self.structured_config = structured_cfg
    self.structured_training_config = training_cfg
    self.structured_head = (
      ContextualCouplingForestHead(
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
        factor_embedding_mode=str(
          structured_cfg.get('factor_embedding_mode', 'shared')),
        factor_conditioner_hidden_dim=structured_cfg.get(
          'factor_conditioner_hidden_dim', 0),
        independent_mode=bool(
          structured_cfg.get('independent_mode', False)),
        min_edge_score=structured_cfg.get('min_edge_score', None)))

    if checkpoint_path:
      self._load_structured_backbone_checkpoint(str(checkpoint_path))
    elif initialize_pretrained_backbone:
      warnings.warn('CCF is freezing a random backbone (smoke tests only)',
                    stacklevel=2)
    self.backbone.requires_grad_(False)
    self.backbone.eval()
    self.structured_enabled = True

  def _load_structured_backbone_checkpoint(self, path):
    """Load raw backbone weights from a trusted MDLM checkpoint, strictly."""
    with fsspec.open(path, 'rb') as handle:
      checkpoint = torch.load(handle, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    backbone_state = {
      key[len('backbone.'):]: value
      for key, value in state_dict.items()
      if key.startswith('backbone.')
    }
    if not backbone_state:
      # Bare DiT state dicts are also accepted. Do not drop unexpected keys.
      backbone_state = state_dict
    self.backbone.load_state_dict(backbone_state, strict=True)

  @torch.no_grad()
  def _structured_backbone_output(self, tokens, conditioning):
    """Return frozen features and raw logits for the clean-token vocabulary."""
    hidden_states, time_conditioning = self.backbone.encode(
      tokens, self._process_sigma(conditioning))
    unary_logits = self.backbone.decode(
      hidden_states, time_conditioning).float().clone()
    # The absorbing mask is never a clean-token candidate.
    unary_logits[:, :, self.mask_index] = -torch.inf
    return hidden_states, unary_logits

  def train(self, mode=True):
    super().train(mode)
    if self.structured_enabled:
      # Lightning recursively enables training after validation. Frozen CCF
      # features must remain deterministic, including backbone dropout.
      self.backbone.eval()
    return self

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

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
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
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

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
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if self.structured_enabled:
      metrics_by_prefix = {
        'train': self.structured_train_metrics,
        'val': self.structured_valid_metrics,
        'test': self.structured_test_metrics,
      }
      if prefix not in metrics_by_prefix:
        raise ValueError(f'Invalid prefix: {prefix}')
      metrics = metrics_by_prefix[prefix]
      updates = self._last_structured_metric_updates
      # Sum joint NLLs and active-token counts before dividing across batches.
      metrics.update(*updates['conditional_nll_per_masked_token'])
      self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
      diagnostics = {
        'train': self.structured_train_diagnostics,
        'val': self.structured_valid_diagnostics,
        'test': self.structured_test_diagnostics,
      }[prefix]
      for name, arguments in updates.items():
        if name != 'conditional_nll_per_masked_token':
          diagnostics[name].update(*arguments)
      self.log_dict(diagnostics, on_step=False, on_epoch=True, sync_dist=True)
      return loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train(not self.structured_enabled)
    if self.structured_enabled:
      self.structured_head.train()
    self.noise.train()
    if self.structured_enabled:
      corruption_seed, topology_seed = _structured_training_rng_seeds(
        int(self.config.get('seed', 1)), int(self.current_epoch),
        int(self.global_rank))
      self._structured_training_corruption_generator = torch.Generator(
        device=self.device).manual_seed(corruption_seed)
      self._structured_training_topology_generator = torch.Generator(
        device=self.device).manual_seed(topology_seed)

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
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
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    if self.structured_enabled:
      configured_head_lr = self.structured_training_config.get('head_lr', None)
      head_lr = (self.config.optim.lr if configured_head_lr is None
                 else float(configured_head_lr))
      if not math.isfinite(head_lr) or head_lr < 0:
        raise ValueError('head_lr must be finite and non-negative')
      # The restored CCF path freezes the backbone at initialization.
      head_and_noise = [
        parameter for module in (self.structured_head, self.noise)
        for parameter in module.parameters() if parameter.requires_grad]
      parameter_groups = [{
        'params': head_and_noise, 'lr': head_lr, 'name': 'structured_head'}]
    else:
      parameter_groups = itertools.chain(
        self.backbone.parameters(), self.noise.parameters())
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
      'monitor': (self.config.model.get(
        'monitor_metric', 'val/conditional_nll_per_masked_token')
        if self.structured_enabled else 'val/loss'),
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
      raise ValueError('set eval.compute_generative_perplexity=true before model construction')
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

  def q_xt(self, x, move_chance, generator=None):
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
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  @torch.no_grad()
  def _structured_clean_sample(self, x, conditioning):
    """Draw clean identities jointly from the forest, preserving revealed tokens."""
    from structured_objective import sample_structured_tokens

    active_mask = x.eq(self.mask_index)
    if not bool(active_mask.any().item()):
      return x
    output, unary_logits = self._structured_head_output(
      tokens=x, conditioning=conditioning, active_mask=active_mask)
    clean = sample_structured_tokens(
      output=output, unary_logits=unary_logits,
      active_mask=active_mask, num_samples=1)[:, 0]
    return torch.where(active_mask, clean, x)

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
    if self.structured_sampling_mode == 'structured_joint':
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
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

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
      elif self.structured_sampling_mode == 'structured_joint':
        unet_conditioning = self.noise(t)[0]
        x = self._structured_clean_sample(x, unet_conditioning)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Preserve the caller's modes, including the permanently frozen backbone.
    modules = [self.backbone, self.noise]
    if self.structured_head is not None:
      modules.append(self.structured_head)
    training_modes = [module.training for module in modules]
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    for module in modules:
      module.eval()
    try:
      samples = self._sample(num_steps=num_steps, eps=eps)
    finally:
      if self.ema:
        self.ema.restore(itertools.chain(
          self.backbone.parameters(),
          self.noise.parameters()))
      for module, training in zip(modules, training_modes):
        module.train(training)
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

  def _sample_t(self, n, device, generator=None):
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

  def _forward_pass_diffusion(self, x0):
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
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _structured_head_output(self, tokens, conditioning, active_mask):
    hidden_states, unary_logits = self._structured_backbone_output(
      tokens, conditioning)
    # CCF receives the actual noise level even when MDLM's backbone is
    # configured without timestep conditioning.
    output = self.structured_head(
      hidden_states=hidden_states,
      unary_logits=unary_logits,
      timestep=conditioning.squeeze(-1),
      active_mask=active_mask)
    return output, unary_logits

  def _structured_topology_loss(
      self, output, unary_logits, xt, x0, conditioning, active_mask):
    """Supervise original-context proposals with a frozen gold-reveal pass."""
    from structured_training import (
      gold_reveal_influence_topology_loss, sample_active_sources)

    cfg = self.structured_training_config
    enabled = (
      float(cfg.get('topology_weight', 0.0)) > 0.0
      and output.topology_mode == 'dynamic'
      and not output.independent_mode
      and (self.training or bool(cfg.get('topology_on_validation', False))))
    if not enabled:
      return None
    generator = self._structured_training_topology_generator if self.training else None
    if self.training and generator is None:
      raise RuntimeError('initialize CCF topology RNG in on_train_epoch_start')
    sources = sample_active_sources(active_mask, generator=generator)
    revealed_xt = xt.clone()
    valid_sources = sources >= 0
    batch_index = torch.arange(x0.shape[0], device=x0.device)
    revealed_xt[batch_index[valid_sources], sources[valid_sources]] = (
      x0[batch_index[valid_sources], sources[valid_sources]])
    # Same frozen backbone, a second no-grad pass. Never send revealed_xt
    # into the primary structured head or use it to choose that head's edges.
    _, revealed_logits = self._structured_backbone_output(revealed_xt, conditioning)
    return gold_reveal_influence_topology_loss(
      output=output,
      base_unary_logits=unary_logits.detach(),
      revealed_unary_logits=revealed_logits,
      clean_tokens=x0,
      active_mask=active_mask,
      source_positions=sources,
      temperature=float(cfg.get('topology_temperature', 0.25)),
      minimum_choices=int(cfg.get('topology_minimum_choices', 2)),
      edge_weight=float(cfg.get('topology_edge_weight', 1.0)),
      anchor_weight=float(cfg.get('topology_anchor_weight', 0.25)),
      slot_weight=float(cfg.get('topology_slot_weight', 0.25)))

  @torch.no_grad()
  def _record_structured_diagnostics(
      self, denoising, unary_logits, x0, active_mask, attention_mask, topology):
    from structured_training import factorized_denoising_nll

    active_tokens = denoising.active_tokens.detach()
    factorized_nll = factorized_denoising_nll(unary_logits, x0, active_mask)
    updates = {
      'conditional_nll_per_masked_token': (denoising.nll_sum.detach(), active_tokens),
      'factorized_nll_per_masked_token': (factorized_nll * active_tokens, active_tokens),
      'candidate_recall': (denoising.candidate_hits.detach(), active_tokens),
      'retained_unary_mass': (denoising.retained_mass_sum.detach(), active_tokens),
      'active_fraction': (active_tokens, attention_mask.bool().sum()),
      'active_tokens': (active_tokens,),
    }
    zero = denoising.loss.detach().new_zeros(())
    teacher_examples = zero if topology is None else zero.new_tensor(x0.shape[0])
    eligible = zero if topology is None else topology.edge_coverage_denominator.detach()
    updates['teacher_examples'] = (teacher_examples,)
    updates['teacher_eligible_fraction'] = (eligible, teacher_examples)
    # Total auxiliary loss combines means over different populations; report its
    # batch mean explicitly rather than inventing a shared example denominator.
    updates['topology_loss_per_batch'] = (
      zero if topology is None else topology.loss.detach(),
      zero if topology is None else zero.new_ones(()))
    for view in ('edge', 'anchor', 'slot'):
      loss = zero if topology is None else getattr(topology, f'{view}_loss').detach()
      count = zero if topology is None else getattr(
        topology, f'{view}_valid_examples').detach()
      numerator = zero if topology is None else getattr(
        topology, f'{view}_coverage_numerator').detach()
      denominator = zero if topology is None else getattr(
        topology, f'{view}_coverage_denominator').detach()
      updates[f'topology_{view}_loss'] = (loss * count, count)
      updates[f'topology_{view}_valid_examples'] = (count,)
      updates[f'topology_{view}_coverage'] = (numerator, denominator)
      updates[f'topology_{view}_coverage_numerator'] = (numerator,)
      updates[f'topology_{view}_coverage_denominator'] = (denominator,)
    self._last_structured_metric_updates = updates

  def _forward_pass_structured(self, x0, attention_mask):
    """Conditional joint denoising NLL per masked token, not a diffusion ELBO."""
    from structured_training import structured_denoising_loss

    generator = None
    if self.training:
      generator = self._structured_training_corruption_generator
      if generator is None:
        raise RuntimeError('initialize CCF corruption RNG in on_train_epoch_start')
    t = self._sample_t(x0.shape[0], x0.device, generator=generator)
    sigma, _ = self.noise(t)
    conditioning = sigma[:, None]
    move_chance = 1 - torch.exp(-conditioning)
    xt = self.q_xt(x0, move_chance, generator=generator)
    active_mask = xt.eq(self.mask_index) & attention_mask.bool()
    output, unary_logits = self._structured_head_output(
      xt, conditioning, active_mask)
    denoising = structured_denoising_loss(
      output=output, unary_logits=unary_logits,
      clean_tokens=x0, active_mask=active_mask)
    # Preserve a zero-gradient connection when topology supervision is off.
    topology_zero = torch.where(
      torch.isfinite(output.proposal_scores), output.proposal_scores,
      torch.zeros_like(output.proposal_scores)).sum() * 0.0
    topology = self._structured_topology_loss(
      output, unary_logits, xt, x0, conditioning, active_mask)
    topology_loss = topology_zero if topology is None else topology.loss
    # Keep detached components/coverage available for the logging connection.
    self._last_structured_topology_metrics = {} if topology is None else {
      name: getattr(topology, name).detach() for name in (
        'loss', 'edge_loss', 'anchor_loss', 'slot_loss', 'valid_examples',
        'mean_influence', 'edge_coverage_numerator', 'edge_coverage_denominator',
        'anchor_coverage_numerator', 'anchor_coverage_denominator',
        'slot_coverage_numerator', 'slot_coverage_denominator')
    }
    weight = float(self.structured_training_config.get('topology_weight', 0.0))
    total_loss = denoising.loss + topology_zero + weight * topology_loss
    self._record_structured_diagnostics(
      denoising, unary_logits, x0, active_mask, attention_mask, topology)
    # The reported conditional NLL excludes the auxiliary topology objective.
    return Loss(loss=total_loss, nlls=denoising.distributed_nll,
                token_mask=active_mask)

  def _loss(self, x0, attention_mask):
    if self.structured_enabled and attention_mask is None:
      attention_mask = torch.ones_like(x0, dtype=torch.bool)
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.structured_enabled:
      return self._forward_pass_structured(input_tokens, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
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
    if self.structured_sampling_mode == 'structured_joint':
      raise ValueError('structured_joint sampling does not support semi-AR strides')
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
