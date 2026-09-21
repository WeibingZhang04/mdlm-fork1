import math

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import (
  DDiTBlock, DDitFinalLayer, EmbeddingLayer,
  TimestepEmbedder, Rotary, LayerNorm
)


class CRFDecoderBlock(nn.Module):
  """Cross-attention decoder block for CRF transitions.

  Each block: LayerNorm → cross-attention(Q=decoder, KV=encoder) → residual
              → LayerNorm → FFN → residual
  """

  def __init__(self, decoder_dim, n_heads, encoder_dim,
               mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.head_dim = decoder_dim // n_heads

    self.norm1 = LayerNorm(decoder_dim)
    self.q_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)
    self.kv_proj = nn.Linear(
      encoder_dim, 2 * decoder_dim, bias=False)
    self.out_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)

    self.norm2 = LayerNorm(decoder_dim)
    self.mlp = nn.Sequential(
      nn.Linear(decoder_dim, mlp_ratio * decoder_dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * decoder_dim, decoder_dim, bias=True))
    self.dropout_p = dropout

  def forward(self, x, encoder_hidden):
    """Cross-attention from decoder queries to encoder KV.

    Args:
      x: (batch, L_dec, decoder_dim)
      encoder_hidden: (batch, L_enc, encoder_dim)
    Returns:
      (batch, L_dec, decoder_dim)
    """
    residual = x
    x = self.norm1(x)

    batch, L_dec, _ = x.shape
    L_enc = encoder_hidden.shape[1]

    q = self.q_proj(x).view(
      batch, L_dec, self.n_heads, self.head_dim).transpose(1, 2)
    kv = self.kv_proj(encoder_hidden)
    k, v = kv.chunk(2, dim=-1)
    k = k.view(
      batch, L_enc, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(
      batch, L_enc, self.n_heads, self.head_dim).transpose(1, 2)

    attn_out = F.scaled_dot_product_attention(
      q, k, v,
      dropout_p=self.dropout_p if self.training else 0.0)
    attn_out = attn_out.transpose(1, 2).contiguous().view(
      batch, L_dec, -1)
    x = residual + self.out_proj(attn_out)

    x = x + self.mlp(self.norm2(x))
    return x


class CRFDecoder(nn.Module):
  """First-order CRF decoder with cross-attention to encoder
  hidden states.

  Computes log P(x_{0,i} | x_{0,i-1}, x_t) for each position i:
    1. Embed previous token x_{0,i-1} + position embedding
    2. Cross-attend to encoder hidden states H
    3. Project to vocabulary logits
  """

  def __init__(self, decoder_dim, n_heads, encoder_dim,
               n_layers, vocab_size, max_seq_len, dropout=0.1):
    super().__init__()
    self.decoder_dim = decoder_dim
    self.vocab_size = vocab_size

    self.token_embed = nn.Embedding(vocab_size, decoder_dim)
    self.pos_embed = nn.Embedding(max_seq_len, decoder_dim)
    self.start_embed = nn.Parameter(
      torch.randn(decoder_dim) * 0.02)

    if encoder_dim != decoder_dim:
      self.encoder_proj = nn.Linear(
        encoder_dim, decoder_dim, bias=False)
    else:
      self.encoder_proj = nn.Identity()

    self.blocks = nn.ModuleList([
      CRFDecoderBlock(decoder_dim, n_heads, decoder_dim,
                      dropout=dropout)
      for _ in range(n_layers)
    ])

    self.norm = LayerNorm(decoder_dim)
    self.output_proj = nn.Linear(decoder_dim, vocab_size)
    nn.init.zeros_(self.output_proj.weight)
    nn.init.zeros_(self.output_proj.bias)

  def forward(self, prev_tokens, positions, encoder_hidden,
              use_start_for_first=True):
    """Teacher-forced forward: prev_tokens contains ground truth.

    Args:
      prev_tokens: (batch, L) previous token indices.
          Position 0 is ignored when use_start_for_first=True.
      positions: (L,) or (batch, L) position indices
          (the output positions, i.e. which position we predict)
      encoder_hidden: (batch, L_enc, encoder_dim)
      use_start_for_first: replace position-0 embedding
          with the learnable start embedding
    Returns:
      logits: (batch, L, vocab_size)
    """
    encoder_hidden = self.encoder_proj(encoder_hidden)

    x = self.token_embed(prev_tokens)
    if use_start_for_first:
      x[:, 0, :] = self.start_embed

    if positions.ndim == 1:
      x = x + self.pos_embed(positions)[None, :, :]
    else:
      x = x + self.pos_embed(positions)

    for block in self.blocks:
      x = block(x, encoder_hidden)

    x = self.norm(x)
    return self.output_proj(x)

  def forward_batched(self, token_indices, positions,
                      encoder_hidden):
    """Batched forward for inference where query length
    may differ from encoder length.

    Args:
      token_indices: (batch, L_q) token indices
      positions: (L_q,) or (batch, L_q) position indices
      encoder_hidden: (batch, L_enc, encoder_dim)
    Returns:
      logits: (batch, L_q, vocab_size)
    """
    encoder_hidden = self.encoder_proj(encoder_hidden)

    x = self.token_embed(token_indices)
    if positions.ndim == 1:
      x = x + self.pos_embed(positions)[None, :, :]
    else:
      x = x + self.pos_embed(positions)

    for block in self.blocks:
      x = block(x, encoder_hidden)

    x = self.norm(x)
    return self.output_proj(x)


class CRFDiT(nn.Module):
  """DiT encoder + CRF decoder backbone.

  The encoder processes noisy input x_t into hidden states H.
  The CRF decoder cross-attends to H to produce first-order
  transition probabilities P(x_{0,i} | x_{0,i-1}, x_t).
  """

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    encoder_dim = config.model.hidden_size

    # --- Encoder (same architecture as DIT) ---
    self.vocab_embed = EmbeddingLayer(encoder_dim, vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      encoder_dim // config.model.n_heads)

    self.blocks = nn.ModuleList([
      DDiTBlock(encoder_dim,
                config.model.n_heads,
                config.model.cond_dim,
                dropout=config.model.dropout)
      for _ in range(config.model.n_blocks)
    ])

    self.output_layer = DDitFinalLayer(
      encoder_dim, vocab_size, config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

    # --- CRF decoder ---
    crf_cfg = config.model.crf
    self.crf_decoder = CRFDecoder(
      decoder_dim=crf_cfg.decoder_dim,
      n_heads=crf_cfg.decoder_heads,
      encoder_dim=encoder_dim,
      n_layers=crf_cfg.decoder_layers,
      vocab_size=vocab_size,
      max_seq_len=config.model.length,
      dropout=config.model.dropout)
    self.top_k = crf_cfg.top_k
    self.inference_query_chunk_size = int(
      crf_cfg.get('inference_query_chunk_size', 0))

  def encode(self, indices, sigma):
    """Encode noisy input x_t into hidden states.

    Args:
      indices: (batch, seq_len) token indices
      sigma: (batch,) noise level (1-D)
    Returns:
      H: (batch, seq_len, hidden_size) encoder hidden states
      c: (batch, cond_dim) timestep conditioning
    """
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))
    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, rotary_cos_sin, c, seqlens=None)

    return x, c

  def forward(self, indices, sigma):
    """Unigram logits (same interface as DIT)."""
    H, c = self.encode(indices, sigma)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      return self.output_layer(H, c)

  def forward_crf_train(self, xt, sigma, x0,
                        return_unigram_logits=False):
    """Teacher-forced CRF training forward pass.

    Args:
      xt: (batch, seq_len) noisy token indices
      sigma: (batch,) noise level (1-D)
      x0: (batch, seq_len) clean token indices
      return_unigram_logits: also return the encoder's unigram head. This is
          used by the optional auxiliary denoising objective that trains the
          same head later used to choose the CRF's top-K candidates.
    Returns:
      logits: (batch, seq_len, vocab_size) CRF transition logits.
          logits[b,i,v] corresponds to unnormalised
          log P(x_{0,i}=v | x_{0,i-1}=x0[b,i-1], x_t).
      If return_unigram_logits=True, returns
          (logits, unigram_logits) with matching shapes.
    """
    H, c = self.encode(xt, sigma)

    batch, seq_len = x0.shape
    # Shifted x0: [dummy, x0[0], ..., x0[N-2]]
    # Position 0 will use the learnable start embedding.
    prev_tokens = torch.cat([
      x0[:, :1],   # placeholder; replaced by start_embed
      x0[:, :-1]
    ], dim=1)
    positions = torch.arange(seq_len, device=x0.device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      logits = self.crf_decoder(
        prev_tokens, positions, H, use_start_for_first=True)

      if return_unigram_logits:
        unigram_logits = self.output_layer(H, c)

    if return_unigram_logits:
      return logits, unigram_logits
    return logits
