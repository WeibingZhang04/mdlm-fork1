#!/usr/bin/env python3
"""Score generated-text files with a pinned GPT-2-large reference model.

This reports *generative perplexity*: the perplexity of generated text under
an external autoregressive language model.  It is not MDLM's diffusion-ELBO
test perplexity from Table 2 of the MDLM paper.

Accepted inputs (detected automatically):
  * raw .txt files (one sample per file),
  * MDLM stdout containing ``Text samples: [...]``,
  * preview files with ``=== condition [sample N] ===`` headings,
  * .jsonl files whose objects contain a ``text`` field, and
  * .json files containing text strings, a list, or a ``samples`` list.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = 'gpt2-large'
DEFAULT_REVISION = '32b71b12589c2f8d625668d2335a01cac3249519'
PREVIEW_HEADING = re.compile(r'^===\s*(.*?)\s*===\s*$', re.MULTILINE)
SAMPLE_SUFFIX = re.compile(r'\s+sample\s+\d+\s*$', re.IGNORECASE)


def _nonempty(texts: Iterable[str], *, source: Path) -> list[str]:
  result = [str(text) for text in texts if str(text).strip()]
  if not result:
    raise ValueError(f'no non-empty samples found in {source}')
  return result


def _parse_mdlm_stdout(content: str, path: Path) -> dict[str, list[str]]:
  samples = []
  for line in content.splitlines():
    marker = 'Text samples:'
    if marker not in line:
      continue
    payload = line.split(marker, 1)[1].strip()
    try:
      parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError) as error:
      raise ValueError(
        f'could not parse the Text samples list in {path}') from error
    if isinstance(parsed, str):
      samples.append(parsed)
    elif isinstance(parsed, (list, tuple)):
      if not all(isinstance(value, str) for value in parsed):
        raise ValueError(f'Text samples in {path} contains a non-string')
      samples.extend(parsed)
    else:
      raise ValueError(f'Text samples in {path} is not a string or list')
  return {'all': _nonempty(samples, source=path)}


def _parse_preview(content: str, path: Path) -> dict[str, list[str]]:
  matches = list(PREVIEW_HEADING.finditer(content))
  groups: dict[str, list[str]] = defaultdict(list)
  for index, match in enumerate(matches):
    start = match.end()
    stop = matches[index + 1].start() if index + 1 < len(matches) else len(content)
    label = SAMPLE_SUFFIX.sub('', match.group(1)).strip() or 'all'
    sample = content[start:stop].strip()
    if sample:
      groups[label].append(sample)
  if not groups:
    raise ValueError(f'no preview samples found in {path}')
  return dict(groups)


def _json_label(record: dict) -> str:
  for field in ('condition', 'arm', 'sampling_mode', 'mode'):
    value = record.get(field)
    if value is not None and str(value).strip():
      return str(value).strip()
  return 'all'


def _parse_jsonl(content: str, path: Path) -> dict[str, list[str]]:
  groups: dict[str, list[str]] = defaultdict(list)
  for line_number, line in enumerate(content.splitlines(), start=1):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as error:
      raise ValueError(f'invalid JSON in {path}:{line_number}') from error
    if not isinstance(record, dict) or not isinstance(record.get('text'), str):
      raise ValueError(f'{path}:{line_number} must contain a string text field')
    groups[_json_label(record)].append(record['text'])
  if not groups:
    raise ValueError(f'no JSONL samples found in {path}')
  return {label: _nonempty(texts, source=path)
          for label, texts in groups.items()}


def _collect_json(value, groups: dict[str, list[str]], label: str = 'all') -> None:
  if isinstance(value, str):
    groups[label].append(value)
  elif isinstance(value, list):
    for item in value:
      _collect_json(item, groups, label)
  elif isinstance(value, dict):
    if isinstance(value.get('text'), str):
      groups[_json_label(value)].append(value['text'])
    elif 'samples' in value:
      _collect_json(value['samples'], groups, label)
    else:
      raise ValueError('JSON object has neither a text field nor samples')
  else:
    raise ValueError('JSON samples must be strings, lists, or objects')


def load_text_groups(path: Path, input_format: str = 'auto') -> dict[str, list[str]]:
  """Load one input file and return condition -> decoded sample texts."""
  path = path.expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  content = path.read_text(errors='strict')
  selected = input_format
  if selected == 'auto':
    if path.suffix.lower() == '.jsonl':
      selected = 'jsonl'
    elif path.suffix.lower() == '.json':
      selected = 'json'
    elif 'Text samples:' in content:
      selected = 'mdlm-log'
    elif PREVIEW_HEADING.search(content):
      selected = 'preview'
    else:
      selected = 'raw'

  if selected == 'mdlm-log':
    return _parse_mdlm_stdout(content, path)
  if selected == 'preview':
    return _parse_preview(content, path)
  if selected == 'jsonl':
    return _parse_jsonl(content, path)
  if selected == 'json':
    try:
      value = json.loads(content)
    except json.JSONDecodeError as error:
      raise ValueError(f'invalid JSON in {path}') from error
    groups: dict[str, list[str]] = defaultdict(list)
    _collect_json(value, groups)
    return {label: _nonempty(texts, source=path)
            for label, texts in groups.items()}
  if selected == 'raw':
    return {'all': _nonempty([content], source=path)}
  raise ValueError(f'unknown input format: {selected}')


def _markdown_table(rows: list[dict[str, object]]) -> str:
  headers = ('path', 'condition', 'samples', 'GPT2 tokens',
             'GPT2 NLL (nats/token)', 'GPT2 Gen. PPL')
  lines = [
    '| ' + ' | '.join(headers) + ' |',
    '| ' + ' | '.join('---' for _ in headers) + ' |',
  ]
  for row in rows:
    lines.append('| ' + ' | '.join(str(row[header]) for header in headers) + ' |')
  return '\n'.join(lines)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
  print(f'Wrote TSV: {path}')


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Print GPT-2-large generative perplexity for text files.',
    epilog=(
      'This is the external-reference Gen. PPL metric, not the MDLM '
      'diffusion-ELBO test PPL reported in Table 2.'))
  parser.add_argument('paths', nargs='+', type=Path)
  parser.add_argument(
    '--input-format', choices=('auto', 'raw', 'mdlm-log', 'preview',
                              'jsonl', 'json'), default='auto')
  parser.add_argument('--output', type=Path, help='also save the table as TSV')
  parser.add_argument('--model', default=DEFAULT_MODEL)
  parser.add_argument('--revision', default=DEFAULT_REVISION)
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--max-length', type=int, default=1024)
  parser.add_argument(
    '--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = parse_args(argv)
  loaded = []
  for requested_path in args.paths:
    resolved = requested_path.expanduser().resolve()
    for condition, texts in load_text_groups(
        resolved, input_format=args.input_format).items():
      loaded.append((resolved, condition, texts))

  # Keep heavyweight ML imports out of format parsing and --help.
  from evaluation.generation_metrics import TransformersReferenceLMScorer

  scorer = TransformersReferenceLMScorer(
    args.model,
    revision=args.revision,
    device=args.device,
    batch_size=args.batch_size,
    max_length=args.max_length,
    dtype=args.dtype)

  rows = []
  for path, condition, texts in loaded:
    scores = scorer.score(texts)
    scored = [score for score in scores
              if score.mean_nll_nats is not None and score.token_count > 0]
    token_count = sum(score.token_count for score in scored)
    if token_count == 0:
      mean_nll = None
      perplexity = None
    else:
      mean_nll = math.fsum(
        score.mean_nll_nats * score.token_count for score in scored
      ) / token_count
      perplexity = math.exp(min(mean_nll, 80.0))
    rows.append({
      'path': str(path),
      'condition': condition,
      'samples': len(texts),
      'GPT2 tokens': token_count,
      'GPT2 NLL (nats/token)': (
        'NA' if mean_nll is None else f'{mean_nll:.6f}'),
      'GPT2 Gen. PPL': (
        'NA' if perplexity is None else f'{perplexity:.3f}'),
    })

  print(_markdown_table(rows))
  if args.output is not None:
    _write_tsv(args.output, rows)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
