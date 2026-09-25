"""Matched continued MDLM training and full-support joint denoising.

This module does not change the frozen-backbone experiment. Its contextual
objective is a time-weighted joint denoising loss, not a claimed diffusion ELBO.
Top-K identities are discrete; their values and the complete residual mass
remain differentiable. Checkpoints always couple the tuned backbone and head.
"""
from __future__ import annotations

import copy
import random

import torch
from torch import nn

from chain_crf.backbone import FrozenMDLM, ROOT, load_tokenizer
from chain_crf.core import build_candidates, gold_log_prob
from chain_crf.data import canonical_hash
from chain_crf.heads import ContextualPairHead

SCHEMA = "chain_crf_joint_finetune_v1"


class TrainableMDLM(nn.Module):
    """Native released DIT with gradients and ordinary training-mode dropout."""

    def __init__(self, encoder, tokenizer, model_spec, initialization):
        super().__init__()
        self.encoder = encoder.requires_grad_(True)
        self.tokenizer = tokenizer
        self.model_spec = copy.deepcopy(model_spec)
        self.initialization = copy.deepcopy(initialization)
        self.vocab_size = model_spec["vocab_size"]
        self.hidden_size = model_spec["hidden_size"]
        self.mask_id = self.mask_index = model_spec["mask_id"]
        self.noise_eps = .001
        self.provenance = {"weights": "joint_finetuned_raw_no_ema",
                           "initialization": self.initialization,
                           "model_spec": self.model_spec}

    @classmethod
    def from_release(cls, checkpoint=None, *, device="cuda", cache_dir=None):
        from omegaconf import OmegaConf
        released = FrozenMDLM(checkpoint, device=device, cache_dir=cache_dir)
        spec = {"kind": "native_mdlm_dit", "vocab_size": released.vocab_size,
                "hidden_size": released.hidden_size, "mask_id": released.mask_id,
                "time_conditioning": False,
                "model_config": OmegaConf.to_container(
                    OmegaConf.load(ROOT / "configs/model/small.yaml"), resolve=True)}
        model = cls(released.encoder, released.tokenizer, spec, released.provenance)
        # FrozenMDLM.eval() propagates to its encoder, so explicitly restore
        # normal module training behavior after transferring that encoder.
        return model.train()

    def forward(self, tokens, time):
        _check_inputs(tokens, time)
        hidden, conditioning = self.encoder.encode(
            tokens, torch.zeros(len(tokens), device=tokens.device))
        logits = self.encoder.decode(hidden, conditioning).float()
        valid = torch.arange(self.vocab_size, device=logits.device) != self.mask_id
        return {"log_probs": logits.masked_fill(~valid, -torch.inf).log_softmax(-1),
                "hidden": hidden.float()}


def _check_inputs(tokens, time):
    if tokens.ndim != 2 or time.shape != (len(tokens),):
        raise ValueError("Expected tokens[B,L] and time[B]")
    if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
        raise ValueError("Diffusion time must be finite and in [0,1]")


class SyntheticTrainableMDLM(nn.Module):
    """Trainable offline fixture; its outputs are never scientific results."""

    def __init__(self, vocab_size=17, hidden_size=12, dropout=.1, seed=17, device="cpu"):
        super().__init__()
        if vocab_size < 3 or hidden_size < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid synthetic model dimensions")
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.mask_id = self.mask_index = vocab_size - 1
        self.noise_eps, self.tokenizer = .001, None
        generator = torch.Generator().manual_seed(seed)
        self.embeddings = nn.Parameter(torch.randn(vocab_size, hidden_size, generator=generator))
        self.projection = nn.Parameter(torch.randn(hidden_size, vocab_size, generator=generator) * .2)
        self.dropout = nn.Dropout(dropout)
        self.model_spec = {"kind": "synthetic_trainable", "vocab_size": vocab_size,
                           "hidden_size": hidden_size, "mask_id": self.mask_id,
                           "dropout": dropout, "seed": seed, "time_conditioning": False}
        self.initialization = {"synthetic_only": True, "seed": seed,
                               "vocab_size": vocab_size, "hidden_size": hidden_size}
        self.provenance = {"weights": "joint_finetuned_raw_no_ema", "synthetic_only": True,
                           "initialization": self.initialization, "model_spec": self.model_spec}
        self.to(device)

    def forward(self, tokens, time):
        _check_inputs(tokens, time)
        hidden = self.embeddings[tokens]
        hidden = self.dropout(hidden + .2 * hidden.mean(1, keepdim=True))
        logits = hidden @ self.projection
        valid = torch.arange(self.vocab_size, device=tokens.device) != self.mask_id
        return {"log_probs": logits.masked_fill(~valid, -torch.inf).log_softmax(-1), "hidden": hidden}


def make_joint_head(arm, backbone, rank=32, mlp_size=128, gate_init_std=.01):
    if arm == "independent":
        return None
    if arm != "contextual" or gate_init_std <= 0:
        raise ValueError("Expected independent or contextual arm and positive gate initialization")
    head = ContextualPairHead(backbone.vocab_size, backbone.hidden_size, rank, mlp_size)
    # Exact zero pairs, but a nonconstant gate avoids the balanced-context
    # dead-gradient trap of zero-right AND zero-final-gate initialization.
    nn.init.zeros_(head.right.weight)
    nn.init.normal_(head.gate[-1].weight, std=gate_init_std)
    nn.init.zeros_(head.gate[-1].bias)
    return head.to(next(backbone.parameters()).device)


def corrupt_joint(clean, mask_id, generator, noise_eps=.001, time_eps=.001):
    """Antithetic uniform times across the *effective* batch, then Bernoulli.

    CPU RNG is separate from dropout. All-visible rows/batches are retained;
    redrawing them would change the ordinary MDLM training distribution.
    """
    if clean.device.type != "cpu" or clean.ndim != 2 or len(clean) < 1:
        raise ValueError("Corruption expects a nonempty CPU clean-token batch")
    if not 0 < time_eps < 1 or not 0 <= noise_eps < 1:
        raise ValueError("Invalid time/noise epsilon")
    times = time_eps + (1-time_eps) * (
        (torch.rand(len(clean), generator=generator) + torch.arange(len(clean))) / len(clean))
    active = torch.rand(clean.shape, generator=generator) < ((1-noise_eps)*times[:, None])
    return clean.masked_fill(active, mask_id), active, times


def sequence_nll(prediction, corrupted, clean, mask_id, *, head=None, time=None, k=64):
    """Unweighted clean-sequence NLL [B], summing only masked predictions."""
    active = corrupted.eq(mask_id)
    lp = prediction["log_probs"]
    base = -lp.gather(-1, clean[..., None]).squeeze(-1).masked_fill(~active, 0).sum(-1)
    if head is None:
        return base, {"base_nll_sum": base.detach().sum(), "masked_tokens": active.sum()}
    packet = build_candidates(lp, corrupted, mask_id, k, gold=clean)
    edge = head(packet.candidate_ids, prediction["hidden"], time)
    active_edges = active[:, :-1] | active[:, 1:]
    edge = edge.masked_fill(~active_edges[..., None, None], 0)
    nll = -gold_log_prob(packet, edge)
    covered = (packet.candidate_ids[..., :-1] == clean[..., None]).any(-1)
    return nll, {"base_nll_sum": base.detach().sum(), "masked_tokens": active.sum(),
                 "gold_covered": (covered & active).sum(),
                 "tail_mass_sum": packet.unary[..., -1].exp().masked_fill(~active, 0).detach().sum()}


def weighted_denoising_loss(nll, times, length, *, effective_batch_size=None):
    if length < 1 or bool(((times <= 0) | ~torch.isfinite(times)).any()) or nll.shape != times.shape:
        raise ValueError("Positive sequence length/times and matching NLL shape required")
    denominator = (len(nll) if effective_batch_size is None else effective_batch_size) * length
    if denominator < 1:
        raise ValueError("Effective batch size must be positive")
    return (nll / times).sum() / denominator


def make_optimizer(backbone, head, *, backbone_lr=1e-5, head_lr=1e-4, warmup_steps=100):
    if min(backbone_lr, head_lr) <= 0 or warmup_steps < 0:
        raise ValueError("Invalid optimizer configuration")
    groups = [{"params": list(backbone.parameters()), "lr": backbone_lr, "name": "backbone"}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": head_lr, "name": "head"})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1., (step+1)/max(1, warmup_steps)))
    return optimizer, scheduler


def train_update(backbone, head, optimizer, scheduler, clean, mask_rng, *, k=64,
                 microbatch_size=1, gradient_clip=1., corruption=None):
    if microbatch_size < 1 or gradient_clip <= 0:
        raise ValueError("Positive microbatch size and gradient clip required")
    backbone.train()
    if head is not None:
        head.train()
    optimizer.zero_grad(set_to_none=True)
    corrupted, active, times = (corrupt_joint(clean.cpu(), backbone.mask_id, mask_rng, backbone.noise_eps)
                                if corruption is None else corruption)
    device = next(backbone.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    totals = {"weighted_loss": 0., "nll_sum": 0., "tokens": clean.numel(), "masked_tokens": 0}
    for offset in range(0, len(clean), microbatch_size):
        part = slice(offset, offset+microbatch_size)
        x, y, t = (value[part].to(device) for value in (corrupted, clean, times))
        nll, metrics = sequence_nll(backbone(x, t), x, y, backbone.mask_id, head=head, time=t, k=k)
        loss = weighted_denoising_loss(nll, t, clean.shape[1], effective_batch_size=len(clean))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite joint-training loss")
        loss.backward()
        totals["weighted_loss"] += float(loss.detach())
        totals["nll_sum"] += float(nll.detach().sum())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + float(value)
    # Per-group clipping leaves the backbone's clipping rule identical in
    # both arms; an extra head must not shrink the backbone update globally.
    for group in optimizer.param_groups:
        norm = nn.utils.clip_grad_norm_(group["params"], gradient_clip, error_if_nonfinite=True)
        totals[f"{group['name']}_grad_norm"] = float(norm)
    optimizer.step()
    scheduler.step()
    if device.type == "cuda":
        totals["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        totals["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return totals


@torch.no_grad()
def evaluate_joint(backbone, head, tokens, *, k=64, batch_size=4, microbatch_size=1, seed=100003):
    if min(batch_size, microbatch_size, len(tokens)) < 1:
        raise ValueError("Nonempty data and positive batch sizes required")
    was_training = backbone.training
    head_was_training = head.training if head is not None else None
    backbone.eval()
    if head is not None:
        head.eval()
    rng = torch.Generator().manual_seed(seed)
    device = next(backbone.parameters()).device
    totals = {"weighted_nll_sum": 0., "nll_sum": 0., "tokens": 0, "masked_tokens": 0}
    try:
        for offset in range(0, len(tokens), batch_size):
            clean = tokens[offset:offset+batch_size]
            corrupted, _, times = corrupt_joint(clean.cpu(), backbone.mask_id, rng, backbone.noise_eps)
            for sub in range(0, len(clean), microbatch_size):
                part = slice(sub, sub+microbatch_size)
                x, y, t = (v[part].to(device) for v in (corrupted, clean, times))
                nll, metrics = sequence_nll(backbone(x, t), x, y, backbone.mask_id, head=head, time=t, k=k)
                totals["weighted_nll_sum"] += float((nll/t).sum())
                totals["nll_sum"] += float(nll.sum())
                totals["tokens"] += y.numel()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + float(value)
    finally:
        backbone.train(was_training)
        if head is not None:
            head.train(head_was_training)
    totals["weighted_loss"] = totals["weighted_nll_sum"] / totals["tokens"]
    totals["nll_per_masked_token"] = totals["nll_sum"] / max(1, totals["masked_tokens"])
    return totals


def checkpoint_payload(backbone, head, optimizer, scheduler, stream, mask_rng,
                       *, step, identity, best_dev=None):
    config = identity["config"]
    if (identity["model_spec"] != backbone.model_spec
            or identity["initialization_backbone"] != backbone.initialization
            or (config["arm"] == "independent") != (head is None)):
        raise ValueError("Checkpoint architecture/initialization/arm mismatch")
    return {"schema": SCHEMA, "identity": copy.deepcopy(identity),
            "identity_sha256": canonical_hash(identity), "config": copy.deepcopy(config),
            "backbone_state": backbone.state_dict(), "pair_state": head.state_dict() if head is not None else None,
            "step": step, "trained_tokens": step*config["batch_size"]*config["length"],
            "best_dev": best_dev, "training": {
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "stream": stream.state_dict(), "mask_rng": mask_rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                "python_rng": random.getstate()}}


def validate_checkpoint(payload):
    if payload.get("schema") != SCHEMA:
        raise ValueError("Not a coupled joint-training checkpoint")
    identity = payload["identity"]
    config = payload["config"]
    if payload.get("identity_sha256") != canonical_hash(identity) or identity.get("config") != config:
        raise ValueError("Joint checkpoint identity/configuration checksum mismatch")
    if config["arm"] not in ("independent", "contextual"):
        raise ValueError("Unknown joint training arm")
    if (config["arm"] == "independent") != (payload.get("pair_state") is None):
        raise ValueError("Joint checkpoint arm/pair-state mismatch")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("Invalid completed optimizer step")
    if payload["trained_tokens"] != payload["step"]*config["batch_size"]*config["length"]:
        raise ValueError("Joint checkpoint token exposure mismatch")
    return identity


def _strict_state(module, state):
    expected = module.state_dict()
    if expected.keys() != state.keys():
        raise ValueError("Checkpoint state keys differ from coupled architecture")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype differs: {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Nonfinite checkpoint parameter: {name}")
    module.load_state_dict(state, strict=True)


def models_from_checkpoint(payload, *, device="cpu", cache_dir=None):
    """Construct both modules from one file; no separate release/head mixing."""
    identity = validate_checkpoint(payload)
    spec, init, config = identity["model_spec"], identity["initialization_backbone"], payload["config"]
    if spec["kind"] == "synthetic_trainable":
        backbone = SyntheticTrainableMDLM(spec["vocab_size"], spec["hidden_size"], spec["dropout"], spec["seed"])
        if backbone.model_spec != spec or backbone.initialization != init:
            raise ValueError("Synthetic initialization/specification mismatch")
    elif spec["kind"] == "native_mdlm_dit":
        from omegaconf import OmegaConf
        from models.dit import DIT
        from scripts.prepare_released_mdlm_owt import RELEASE_SHA256
        from chain_crf.backbone import TOKENIZER_REVISION
        if (init.get("source_safetensors_sha256") != RELEASE_SHA256
                or init.get("tokenizer_revision") != TOKENIZER_REVISION
                or spec["time_conditioning"] is not False
                or spec["vocab_size"] != 50258 or spec["mask_id"] != 50257):
            raise ValueError("Native initialization/tokenizer specification mismatch")
        tokenizer = load_tokenizer(cache_dir)
        if tokenizer.vocab_size != spec["mask_id"]:
            raise ValueError("Tokenizer vocabulary differs from joint checkpoint")
        encoder = DIT(OmegaConf.create({"model": spec["model_config"]}), spec["vocab_size"])
        backbone = TrainableMDLM(encoder, tokenizer, spec, init)
    else:
        raise ValueError("Unsupported coupled backbone architecture")
    head = make_joint_head(config["arm"], backbone, config["rank"], config["mlp_size"], config["gate_init_std"])
    _strict_state(backbone, payload["backbone_state"])
    if head is not None:
        _strict_state(head, payload["pair_state"])
        head.to(device).eval()
    backbone.to(device).eval()
    return backbone, head


def restore_training(payload, backbone, head, optimizer, scheduler, stream, mask_rng, identity):
    saved = validate_checkpoint(payload)
    if saved != identity:
        raise ValueError("Resume identity differs: data, source, architecture, runtime or optimizer configuration")
    if backbone.model_spec != saved["model_spec"] or backbone.initialization != saved["initialization_backbone"]:
        raise ValueError("Resume backbone does not match coupled model identity")
    _strict_state(backbone, payload["backbone_state"])
    if (head is None) != (payload["pair_state"] is None):
        raise ValueError("Resume head/arm mismatch")
    if head is not None:
        _strict_state(head, payload["pair_state"])
    state = payload["training"]
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    stream.load_state_dict(state["stream"])
    mask_rng.set_state(state["mask_rng"].cpu())
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs from the training checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    return payload["step"], payload["best_dev"]


def evaluation_components(backbone, head, control="joint"):
    """Pair-disabled keeps the *tuned* backbone; own-marginal keeps its CRF."""
    if control not in ("joint", "own-marginal", "pair-disabled"):
        raise ValueError("Unknown coupled-checkpoint evaluation control")
    if head is None:
        if control == "own-marginal":
            raise ValueError("Independent continuation has no CRF own-marginal control")
        return None, "backbone", "joint"
    if control == "pair-disabled":
        return None, "backbone", "joint"
    return head, "contextual", "marginal" if control == "own-marginal" else "joint"
