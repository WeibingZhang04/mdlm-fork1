"""Frozen native MDLM-OWT encoder, without legacy structured-decoder setup.

Weights and tokenizer are pinned to the releases used by this repository.
No remote Python is executed. The returned hidden states are detached, but
not inference tensors, so trainable heads may safely save them for backward.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import torch
from torch import nn

TOKENIZER_REPOSITORY = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_tokenizer(cache_dir=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        TOKENIZER_REPOSITORY, revision=TOKENIZER_REVISION,
        cache_dir=None if cache_dir is None else str(cache_dir),
        trust_remote_code=False,
    )


class FrozenMDLM(nn.Module):
    """``model(tokens, time) -> {log_probs, hidden}``; time is normalized t.

    The released model has time_conditioning=false: its encoder receives
    zeros, exactly as Diffusion._process_sigma does. Heads can still use t.
    ``checkpoint`` may be the pinned safetensors file or its repository-made
    Lightning envelope. If omitted, only the pinned release is downloaded.
    """

    def __init__(self, checkpoint=None, *, device="cuda", cache_dir=None):
        super().__init__()
        from omegaconf import OmegaConf
        from models.dit import DIT, FLASH_ATTN_AVAILABLE
        from scripts.prepare_released_mdlm_owt import (
            download_release, verify_release_file, validate_backbone_state,
            RELEASE_REPOSITORY, RELEASE_REVISION, RELEASE_SHA256,
        )
        self.tokenizer = load_tokenizer(cache_dir)
        self.mask_id = self.tokenizer.vocab_size
        self.mask_index = self.mask_id
        self.vocab_size = self.mask_id + 1
        self.hidden_size = 768
        self.noise_eps = 1e-3
        if self.vocab_size != 50258:
            raise ValueError("Pinned MDLM requires the 50257-token GPT-2 vocabulary")
        path = Path(checkpoint) if checkpoint else download_release(
            None if cache_dir is None else Path(cache_dir))
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            verify_release_file(path)
            state = validate_backbone_state(load_file(str(path), device="cpu"))
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            metadata = payload.get("metadata", {})
            if metadata.get("source_sha256") != RELEASE_SHA256:
                raise ValueError("Checkpoint must be the verified prepare_released_mdlm_owt wrapper")
            state = validate_backbone_state(payload["state_dict"])
        config = OmegaConf.create({"model": OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/model/small.yaml"), resolve=True)})
        self.encoder = DIT(config, vocab_size=self.vocab_size)
        self.encoder.load_state_dict({k.removeprefix("backbone."): v
                                     for k, v in state.items()}, strict=True)
        self.encoder.requires_grad_(False)
        self.to(device)
        self.eval()
        self.provenance = {
            "repository": RELEASE_REPOSITORY, "revision": RELEASE_REVISION,
            "source_safetensors_sha256": RELEASE_SHA256,
            "loaded_file_sha256": file_sha256(path),
            "tokenizer": TOKENIZER_REPOSITORY,
            "tokenizer_revision": TOKENIZER_REVISION,
            "mask_id": self.mask_id, "vocab_size": self.vocab_size,
            "time_conditioning": False, "weights": "released_raw_no_ema",
            "native_encoder": "models.dit.DIT", "flash_attention_available": FLASH_ATTN_AVAILABLE,
            "native_cuda_precision": "float32 parameters; bfloat16 autocast transformer blocks and output layer",
            "head_input_precision": "float32 detached hidden states and log probabilities",
            "encoder_source_sha256": file_sha256(ROOT / "models/dit.py"),
            "model_config_sha256": file_sha256(ROOT / "configs/model/small.yaml"),
        }

    def train(self, mode=True):
        # A training driver cannot accidentally enable encoder dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, time: torch.Tensor):
        if tokens.ndim != 2 or time.numel() != len(tokens):
            raise ValueError("Expected tokens[B,L] and normalized time[B]")
        if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("Diffusion time must be finite and in [0,1]")
        hidden, conditioning = self.encoder.encode(
            tokens, torch.zeros(len(tokens), device=tokens.device))
        logits = self.encoder.decode(hidden, conditioning).float()
        logits[..., self.mask_id] = -torch.inf
        return {"log_probs": logits.log_softmax(-1).detach(),
                "hidden": hidden.float().detach()}


class SyntheticBackbone(nn.Module):
    """Small deterministic offline plumbing fixture; never a real-data result."""

    def __init__(self, vocab_size=17, hidden_size=12, device="cpu", seed=17):
        super().__init__()
        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.mask_id = self.mask_index = vocab_size - 1
        self.noise_eps = 1e-3
        self.tokenizer = None
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("embeddings", torch.randn(vocab_size, hidden_size, generator=generator))
        self.register_buffer("projection", torch.randn(hidden_size, vocab_size, generator=generator) * .2)
        self.to(device)
        self.provenance = {"synthetic_only": True, "seed": seed, "vocab_size": vocab_size,
                           "hidden_size": hidden_size}

    @torch.no_grad()
    def forward(self, tokens, time):
        hidden = self.embeddings[tokens]
        hidden = hidden + .2 * hidden.mean(1, keepdim=True)
        logits = hidden @ self.projection
        logits[..., self.mask_id] = -torch.inf
        return {"log_probs": logits.log_softmax(-1), "hidden": hidden}
