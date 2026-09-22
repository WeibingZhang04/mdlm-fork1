#!/usr/bin/env python3
"""Score saved generations with both MDLM and first-EOS GPT-2 PPL rules.

The ``mdlm_original`` mask follows yuntian-group/mdlm diffusion.py at
https://github.com/yuntian-group/mdlm/blob/2833ce236649c57877d76e4f08950886b1eef0fa/diffusion.py#L573-L582.
The ``stop_at_eos`` mask follows the four-arm generation scorer at
https://github.com/WeibingZhang04/mdlm-fork1/blob/69b8649cead3d20398c407332b9147a9c9e6d74d/evaluation/generation_metrics.py#L340-L405.

Both rules use the same GPT-2 Large weights, tokenizer, logits, and samples
within one invocation. This scores saved text independently of sampling or
training; it does not alter the model's generation path.

The original MDLM CLI did not pin GPT-2 Large and could drop an incomplete
last evaluation batch. This standalone script defaults to the historical
four-arm GPT-2 Large revision and scores every supplied sample. Batch size 1
avoids padding and incomplete-batch differences when checking MDLM's rule.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional


DEFAULT_REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'
METHODS = ('mdlm_original', 'stop_at_eos')


def mdlm_original_mask(token_ids: list[int], eos_id: int) -> list[bool]:
  """Original MDLM ``first_eos + token_mask`` for next-token losses.

  Both operands in that expression are Boolean tensors, so their sum is a
  Boolean OR, not a numeric weight of two. Padding is intentionally included
  here because the original method applies its mask to the padded batch.
  """
  eos_seen = 0
  mask = []
  for token_id in token_ids:
    eos_seen += token_id == eos_id
    mask.append(eos_seen == 1 or token_id != eos_id)
  return mask[1:]


def stop_at_eos_mask(
    token_ids: list[int], attention_mask: list[int], eos_id: int,
    bos_id: Optional[int]) -> list[bool]:
  """Score real tokens through first nonleading EOS, including that EOS."""
  if len(token_ids) != len(attention_mask):
    raise ValueError('token IDs and attention mask have different lengths')
  first_valid = next(
    (index for index, valid in enumerate(attention_mask) if valid), None)
  if first_valid is None:
    return [False] * max(0, len(token_ids) - 1)
  first_eos = len(token_ids)
  for index, (token_id, valid) in enumerate(zip(token_ids, attention_mask)):
    if not valid or token_id != eos_id:
      continue
    if index == first_valid and bos_id is not None and token_id == bos_id:
      continue
    first_eos = index
    break
  return [bool(attention_mask[index]) and index <= first_eos
          for index in range(1, len(token_ids))]


def _read_texts(path: Path, text_field: str) -> list[str]:
  texts = []
  with path.open(encoding='utf-8') as handle:
    for line_number, line in enumerate(handle, 1):
      if not line.strip():
        continue
      record = json.loads(line)
      if not isinstance(record, dict) or not isinstance(
          record.get(text_field), str):
        raise ValueError(
          f'{path}:{line_number} needs a string {text_field!r} field')
      texts.append(record[text_field])
  if not texts:
    raise ValueError(f'no samples found in {path}')
  return texts


def score_texts(
    texts: list[str], *, methods: tuple[str, ...], revision: str,
    batch_size: int, max_length: int, device: str) -> dict:
  """Evaluate either rule alone or both from one set of GPT-2 logits."""
  if not texts:
    raise ValueError('at least one text sample is required')
  if not methods or any(method not in METHODS for method in methods):
    raise ValueError(f'methods must be chosen from {METHODS}')
  if batch_size < 1 or max_length < 2:
    raise ValueError('batch size must be positive and max length at least two')

  import torch
  import torch.nn.functional as F
  from transformers import AutoModelForCausalLM, AutoTokenizer

  model_name = 'gpt2-large'
  tokenizer = AutoTokenizer.from_pretrained(
    model_name, revision=revision, use_fast=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'right'
  tokenizer.truncation_side = 'right'
  if tokenizer.eos_token_id is None:
    raise ValueError('GPT-2 tokenizer has no EOS token ID')
  model = AutoModelForCausalLM.from_pretrained(
    model_name, revision=revision, torch_dtype=torch.float32).to(device).eval()

  totals = {method: {'nll_contributions': [], 'scored_tokens': 0}
            for method in methods}
  with torch.no_grad():
    for offset in range(0, len(texts), batch_size):
      batch = texts[offset:offset + batch_size]
      encoded = tokenizer(
        batch, return_tensors='pt', return_token_type_ids=False,
        return_attention_mask=True, truncation=True, padding=True,
        max_length=max_length, add_special_tokens=True)
      input_ids = encoded['input_ids'].to(device)
      attention = encoded['attention_mask'].to(device)
      if input_ids.shape[1] < 2:
        continue  # A causal LM cannot score the first token by itself.
      logits = model(input_ids=input_ids, attention_mask=attention).logits
      losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), input_ids[:, 1:],
        reduction='none')
      for row in range(len(batch)):
        ids = input_ids[row].tolist()
        attn = attention[row].tolist()
        masks = {}
        if 'mdlm_original' in methods:
          masks['mdlm_original'] = mdlm_original_mask(
            ids, tokenizer.eos_token_id)
        if 'stop_at_eos' in methods:
          masks['stop_at_eos'] = stop_at_eos_mask(
            ids, attn, tokenizer.eos_token_id, tokenizer.bos_token_id)
        for method, mask in masks.items():
          chosen = losses[row][torch.tensor(mask, dtype=torch.bool,
                                             device=losses.device)]
          count = len(chosen)
          if count:
            # Match the fork scorer's per-sequence FP32 mean and weighted
            # aggregation. MDLM's metric also aggregates weighted token NLL.
            contribution = (float(chosen.mean().item()) * count
                            if method == 'stop_at_eos'
                            else float(chosen.sum().item()))
            totals[method]['nll_contributions'].append(contribution)
          totals[method]['scored_tokens'] += count

  scores = {}
  for method, total in totals.items():
    count = total['scored_tokens']
    mean_nll = (math.fsum(total['nll_contributions']) / count
                if count else None)
    scores[method] = {
      'label': ('MDLM original token mask' if method == 'mdlm_original'
                else 'Stop at first nonleading EOS'),
      'scored_tokens': count,
      'mean_nll_nats': mean_nll,
      'perplexity': math.exp(mean_nll) if mean_nll is not None else None,
    }
  return {
    'sample_count': len(texts),
    'reference_model': model_name,
    'reference_revision': revision,
    'batch_size': batch_size,
    'max_length': max_length,
    'device': device,
    'scores': scores,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input-jsonl', type=Path, required=True,
                      help='One JSON object with a text field per sample.')
  parser.add_argument('--text-field', default='text')
  parser.add_argument('--method', choices=('both',) + METHODS, default='both')
  parser.add_argument('--revision', default=DEFAULT_REVISION,
                      help='GPT-2 Large model and tokenizer revision.')
  parser.add_argument('--batch-size', type=int, default=1)
  parser.add_argument('--max-length', type=int, default=1024)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--output-json', type=Path,
                      help='Write the labeled result as JSON; otherwise print it.')
  args = parser.parse_args()
  methods = METHODS if args.method == 'both' else (args.method,)
  result = score_texts(
    _read_texts(args.input_jsonl, args.text_field), methods=methods,
    revision=args.revision, batch_size=args.batch_size,
    max_length=args.max_length, device=args.device)
  payload = json.dumps(result, indent=2) + '\n'
  if args.output_json is None:
    print(payload, end='')
  else:
    args.output_json.write_text(payload, encoding='utf-8')
    print(args.output_json)


if __name__ == '__main__':
  main()
