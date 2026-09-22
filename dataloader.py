import functools
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import data_provenance
import utils

LOGGER = utils.get_logger(__name__)


# Bump this whenever the on-disk representation or its validation semantics
# change.  In particular, v2 records the fingerprint assigned by
# ``load_from_disk`` rather than the transient pre-save Dataset fingerprint.
PINNED_PROCESSED_CACHE_SCHEMA_VERSION = 2


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


_WIKITEXT_ARTICLE_HEADING = re.compile(
  r'^\s*=\s+[^=].*?\s+=\s*$')


def _coalesce_wikitext_articles(dataset):
  """Recover WikiText articles from its line-oriented raw representation."""
  documents = []
  current_lines = []
  current_start = None

  def flush(stop):
    if not current_lines:
      return
    text = ''.join(current_lines)
    documents.append({
      'text': text,
      '_source_start_index': current_start,
      '_source_stop_index': stop,
      '_source_document_sha256': hashlib.sha256(
        text.encode('utf-8')).hexdigest(),
    })

  for index, example in enumerate(dataset):
    text = example['text']
    if _WIKITEXT_ARTICLE_HEADING.match(text) and current_lines:
      flush(index)
      current_lines = []
      current_start = None
    if text.strip():
      if current_start is None:
        current_start = index
      current_lines.append(text)
  flush(len(dataset))
  if not documents:
    raise ValueError('WikiText split contains no non-empty articles')
  return datasets.Dataset.from_list(documents)


def _source_row_count(dataset, *, streaming):
  if streaming:
    return None
  try:
    return len(dataset)
  except TypeError:
    return None


def _select_source_window(dataset, window, *, streaming):
  if window is None:
    return dataset
  start, stop = window
  if streaming:
    return dataset.skip(start).take(stop - start)
  return dataset.select(range(start, stop))


def _dataset_fingerprint(dataset):
  fingerprint = getattr(dataset, '_fingerprint', None)
  return str(fingerprint) if fingerprint is not None else None


def _runtime_provenance_path(provenance_dir, role, key):
  if not provenance_dir:
    raise ValueError(
      'pinned provenance requires a writable provenance_dir')
  if not role:
    raise ValueError('pinned provenance requires a non-empty role')
  return (
    str(provenance_dir).rstrip('/')
    + f'/{role}-{key}.json')


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False,
    revision=None, dataset_name_or_path=None, dataset_config_name=None,
    source_split=None, source_window=None, expected_source_num_rows=None,
    text_field=None, document_boundary_mode='concatenate',
    trust_remote_code=False, require_pinned_provenance=False,
    tokenizer_name_or_path=None, tokenizer_revision=None,
    provenance_dir=None, provenance_role=None,
    disjoint_window_proof=None):
  """Load and tokenize one split under either the legacy or pinned protocol.

  The pinned protocol is enabled by ``require_pinned_provenance``.  It
  requires immutable Hub commits, an expected source row count, a runtime
  provenance artifact, and a content-addressed processed cache.  Document
  mode emits only full model-length windows drawn from one source document;
  incomplete tails are intentionally dropped rather than padded into a
  bidirectional backbone that has no padding-attention mask.
  """
  if document_boundary_mode not in {
      'concatenate', 'source_document', 'wikitext_articles'}:
    raise ValueError(
      'document_boundary_mode must be concatenate, source_document, or '
      'wikitext_articles')
  if document_boundary_mode != 'concatenate' and not wrap:
    raise ValueError('document-boundary preservation requires wrap=true')

  source_num_rows = None
  if expected_source_num_rows is not None:
    source_num_rows = int(expected_source_num_rows)
    if source_num_rows <= 0:
      raise ValueError('expected_source_num_rows must be positive')
  normalized_window = data_provenance.normalize_window(
    source_window, field=f'{provenance_role or mode}_source_window',
    source_num_rows=source_num_rows)

  specification = None
  provenance_key = None
  if require_pinned_provenance:
    if not dataset_name_or_path or not source_split:
      raise ValueError(
        'pinned provenance requires dataset_name_or_path and source_split')
    revision = data_provenance.require_commit_revision(
      revision, field=f'{provenance_role or mode}_revision')
    tokenizer_revision = data_provenance.require_commit_revision(
      tokenizer_revision, field='tokenizer_revision')
    if not tokenizer_name_or_path:
      raise ValueError(
        'pinned provenance requires tokenizer_name_or_path')
    if source_num_rows is None:
      raise ValueError(
        'pinned provenance requires expected_source_num_rows')
    specification = {
      'logical_dataset_name': dataset_name,
      'dataset_name_or_path': str(dataset_name_or_path),
      'dataset_config_name': (
        None if dataset_config_name is None
        else str(dataset_config_name)),
      'source_split': str(source_split),
      'source_revision': revision,
      'source_num_rows': source_num_rows,
      'source_window': (
        None if normalized_window is None
        else list(normalized_window)),
      'text_field': str(text_field or 'text'),
      'trust_remote_code': bool(trust_remote_code),
      'tokenizer_name_or_path': str(tokenizer_name_or_path),
      'tokenizer_revision': tokenizer_revision,
      'tokenizer_class': type(tokenizer).__name__,
      'tokenizer_vocab_size': int(tokenizer.vocab_size),
      'block_size': int(block_size),
      'wrap': bool(wrap),
      'document_boundary_mode': document_boundary_mode,
      'processed_cache_schema_version': (
        PINNED_PROCESSED_CACHE_SCHEMA_VERSION),
      'document_remainder_policy': (
        'drop_incomplete_tail_and_documents_shorter_than_payload'
        if document_boundary_mode != 'concatenate' else None),
      'detokenizer': (
        'wikitext_v1' if dataset_name.startswith('wikitext')
        else ('scientific_papers_v1'
              if (dataset_name.startswith('scientific_papers')
                  or dataset_name_or_path == 'armanc/scientific_papers')
              else None)),
      'datasets_version': datasets.__version__,
      'transformers_version': transformers.__version__,
      'disjoint_window_proof': disjoint_window_proof,
    }
    provenance_key = data_provenance.cache_key(specification)

  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped'
  if provenance_key is not None:
    filename += f'_{provenance_key}'
  filename += '.dat'
  _path = os.path.join(cache_dir, filename)
  cache_provenance_path = _path + '.provenance.json'
  
  # A streaming run must not silently fall back to a previously materialized
  # map-style cache at the same path.
  if not streaming and utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    cached = datasets.load_from_disk(_path)
    if require_pinned_provenance:
      if not utils.fsspec_exists(cache_provenance_path):
        raise RuntimeError(
          f'pinned cache exists without provenance: {_path}')
      manifest = data_provenance.validate_manifest(
        data_provenance.read_manifest(cache_provenance_path),
        expected_specification=specification)
      expected_fingerprint = manifest['observed'].get(
        'processed_fingerprint')
      if (expected_fingerprint is not None
          and _dataset_fingerprint(cached) != expected_fingerprint):
        raise RuntimeError(
          f'processed dataset fingerprint mismatch for {_path}')
      runtime_path = _runtime_provenance_path(
        provenance_dir, provenance_role, provenance_key)
      data_provenance.write_manifest(runtime_path, manifest)
    return cached.with_format('torch')
  if streaming:
    LOGGER.info(f'Streaming {dataset_name} split {mode}.')
  else:
    LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  revision_kwargs = {} if revision is None else {'revision': revision}

  if dataset_name_or_path is not None:
    load_kwargs = {
      'split': source_split,
      'cache_dir': cache_dir,
      'streaming': streaming,
      **revision_kwargs,
    }
    if trust_remote_code:
      load_kwargs['trust_remote_code'] = True
    dataset = datasets.load_dataset(
      dataset_name_or_path,
      name=dataset_config_name,
      **load_kwargs)
    data = dataset
  elif dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir,
      **revision_kwargs)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir,
      **revision_kwargs)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir, **revision_kwargs)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming,
      **revision_kwargs)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming,
      **revision_kwargs)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      **revision_kwargs)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      **revision_kwargs)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming,
      **revision_kwargs)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming,
      **revision_kwargs)

  if dataset_name_or_path is None:
    if dataset_name in ['lambada', 'openwebtext-train',
                        'openwebtext-valid']:
      data = dataset
    else:
      data = dataset[mode]

  observed_source_num_rows = _source_row_count(
    data, streaming=streaming)
  if (source_num_rows is not None
      and observed_source_num_rows is not None
      and observed_source_num_rows != source_num_rows):
    raise RuntimeError(
      f'{dataset_name} source row count mismatch at revision {revision}: '
      f'expected {source_num_rows}, observed {observed_source_num_rows}')
  raw_fingerprint = _dataset_fingerprint(data)
  data = _select_source_window(
    data, normalized_window, streaming=streaming)
  observed_window_num_rows = _source_row_count(
    data, streaming=streaming)
  if (normalized_window is not None
      and observed_window_num_rows is not None
      and observed_window_num_rows != (
        normalized_window[1] - normalized_window[0])):
    raise RuntimeError(
      f'{dataset_name} row window did not materialize its pinned size')

  if document_boundary_mode == 'wikitext_articles':
    if streaming:
      raise ValueError('wikitext_articles mode requires map-style loading')
    data = _coalesce_wikitext_articles(data)

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif (dataset_name.startswith('scientific_papers')
        or dataset_name_or_path == 'armanc/scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  selected_text_field = text_field
  if selected_text_field is None:
    if dataset_name == 'ptb':
      selected_text_field = 'sentence'
    elif 'scientific_papers' in dataset_name:
      selected_text_field = 'article'
    else:
      selected_text_field = 'text'

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  def preprocess_document_windows(example, indices):
    raw_texts = list(example[selected_text_field])
    texts = list(raw_texts)
    if detokenizer is not None:
      texts = _apply_detokenizer(detokenizer)(texts)
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    encoded = tokenizer(
      texts,
      add_special_tokens=False,
      return_attention_mask=False,
      return_token_type_ids=False)['input_ids']
    payload_size = block_size - 2
    result = {
      'input_ids': [],
      'attention_mask': [],
      'source_document_index': [],
      'source_document_sha256': [],
      'source_chunk_index': [],
      'source_document_token_count': [],
    }
    window_start = normalized_window[0] if normalized_window else 0
    source_starts = example.get('_source_start_index')
    source_hashes = example.get('_source_document_sha256')
    for batch_index, token_ids in enumerate(encoded):
      if source_starts is None:
        document_index = window_start + int(indices[batch_index])
      else:
        document_index = int(source_starts[batch_index])
      if source_hashes is None:
        document_hash = hashlib.sha256(
          raw_texts[batch_index].encode('utf-8')).hexdigest()
      else:
        document_hash = str(source_hashes[batch_index])
      full_chunks = len(token_ids) // payload_size
      for chunk_index in range(full_chunks):
        start = chunk_index * payload_size
        chunk = token_ids[start: start + payload_size]
        result['input_ids'].append([BOS] + chunk + [EOS])
        result['attention_mask'].append([1] * block_size)
        result['source_document_index'].append(document_index)
        result['source_document_sha256'].append(document_hash)
        result['source_chunk_index'].append(chunk_index)
        result['source_document_token_count'].append(len(token_ids))
    return result

  if document_boundary_mode != 'concatenate':
    remove_columns = list(data.column_names or [])
    map_kwargs = {
      'batched': True,
      'with_indices': True,
      'remove_columns': remove_columns,
    }
    if not streaming:
      map_kwargs.update({
        'num_proc': num_proc,
        'load_from_cache_file': True,
        'desc': 'Tokenizing document-local windows',
      })
    chunked_dataset = data.map(
      preprocess_document_windows, **map_kwargs)
    if not streaming and len(chunked_dataset) == 0:
      raise RuntimeError(
        f'{dataset_name} produced no full document-local windows of '
        f'length {block_size}')
  else:
    if streaming:
      tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True)
    else:
      tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc='Tokenizing')
    if dataset_name == 'ptb':
      tokenized_dataset = tokenized_dataset.remove_columns(
        'sentence')
    elif 'scientific_papers' in dataset_name:
      tokenized_dataset = tokenized_dataset.remove_columns([
        'article', 'abstract', 'section_names'])
    elif dataset_name == 'ag_news':
      tokenized_dataset = tokenized_dataset.remove_columns(
        ['text', 'label'])
    else:
      tokenized_dataset = tokenized_dataset.remove_columns(
        'text')

  if not wrap:
    if not streaming:
      tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  if document_boundary_mode == 'concatenate':
    group_texts = functools.partial(
      _group_texts, block_size=block_size, bos=BOS, eos=EOS)
    if streaming:
      chunked_dataset = tokenized_dataset.map(
        group_texts,
        batched=True)
    else:
      chunked_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc='Grouping')

  if not streaming:
    chunked_dataset.save_to_disk(_path)
    # Hugging Face gives a Dataset a new fingerprint when it is persisted and
    # loaded back from Arrow.  The provenance record must describe the object
    # future runs will validate, not the transient in-memory object that wrote
    # the cache.  Reloading here also makes the first and subsequent callers
    # observe identical dataset metadata.
    chunked_dataset = datasets.load_from_disk(_path)

  if require_pinned_provenance:
    observed = {
      'source_num_rows': observed_source_num_rows,
      'window_num_rows': observed_window_num_rows,
      'document_num_rows_after_boundary_recovery': (
        _source_row_count(data, streaming=streaming)),
      'processed_num_sequences': (
        _source_row_count(chunked_dataset, streaming=streaming)),
      'raw_fingerprint': raw_fingerprint,
      'window_fingerprint': _dataset_fingerprint(data),
      'processed_fingerprint': _dataset_fingerprint(chunked_dataset),
    }
    manifest = data_provenance.build_manifest(
      specification=specification, observed=observed)
    runtime_path = _runtime_provenance_path(
      provenance_dir, provenance_role, provenance_key)
    data_provenance.write_manifest(runtime_path, manifest)
    if not streaming:
      data_provenance.write_manifest(cache_provenance_path, manifest)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  tokenizer_revision = getattr(config.data, 'tokenizer_revision', None)
  if bool(getattr(config.data, 'require_pinned_provenance', False)):
    tokenizer_revision = data_provenance.require_commit_revision(
      tokenizer_revision, field='data.tokenizer_revision')
  revision_kwargs = (
    {} if tokenizer_revision is None
    else {'revision': tokenizer_revision})
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased', **revision_kwargs)
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path, **revision_kwargs)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer


class _TorchIterableDatasetAdapter(torch.utils.data.IterableDataset):
  """Expose a Hugging Face streaming dataset to PyTorch as iterable-style."""

  def __init__(self, dataset):
    super().__init__()
    self.dataset = dataset

  def __iter__(self):
    return iter(self.dataset)

  def set_epoch(self, epoch):
    if hasattr(self.dataset, 'set_epoch'):
      self.dataset.set_epoch(epoch)


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')

  data_cfg = config.data

  def data_option(name, default=None):
    if hasattr(data_cfg, 'get'):
      return data_cfg.get(name, default)
    return getattr(data_cfg, name, default)

  def source_option(role, name, default=None):
    return data_option(f'{role}_{name}', default)

  require_pinned = bool(
    data_option('require_pinned_provenance', False))
  provenance_dir = data_option('provenance_dir', None)
  if require_pinned and not provenance_dir:
    provenance_dir = os.path.join(
      str(config.checkpointing.save_dir), 'data_provenance')

  disjoint_proof = None
  if bool(data_option('require_disjoint_train_valid_windows', False)):
    equality_fields = (
      'dataset_name_or_path', 'dataset_config_name', 'source_split',
      'revision', 'expected_source_num_rows')
    mismatches = [
      field for field in equality_fields
      if source_option('train', field) != source_option('valid', field)
    ]
    if mismatches:
      raise ValueError(
        'disjoint row-window proof requires identical train/valid source '
        f'fields; mismatched: {mismatches}')
    disjoint_proof = data_provenance.disjoint_window_proof(
      dataset_name_or_path=source_option('train', 'dataset_name_or_path'),
      dataset_config_name=source_option('train', 'dataset_config_name'),
      split=source_option('train', 'source_split'),
      revision=source_option('train', 'revision'),
      source_num_rows=int(source_option(
        'train', 'expected_source_num_rows')),
      train_window=list(source_option('train', 'source_window')),
      heldout_window=list(source_option('valid', 'source_window')))

  shared_dataset_kwargs = {
    'require_pinned_provenance': require_pinned,
    'tokenizer_name_or_path': data_option(
      'tokenizer_name_or_path', None),
    'tokenizer_revision': data_option('tokenizer_revision', None),
    'provenance_dir': provenance_dir,
    'disjoint_window_proof': disjoint_proof,
  }

  def role_dataset_kwargs(role):
    window = source_option(role, 'source_window', None)
    return {
      'dataset_name_or_path': source_option(
        role, 'dataset_name_or_path', None),
      'dataset_config_name': source_option(
        role, 'dataset_config_name', None),
      'source_split': source_option(role, 'source_split', None),
      'source_window': None if window is None else list(window),
      'expected_source_num_rows': source_option(
        role, 'expected_source_num_rows', None),
      'text_field': source_option(role, 'text_field', None),
      'document_boundary_mode': source_option(
        role, 'document_boundary_mode', 'concatenate'),
      'trust_remote_code': bool(source_option(
        role, 'trust_remote_code', False)),
      'provenance_role': role,
      **shared_dataset_kwargs,
    }

  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=bool(config.data.streaming),
      revision=getattr(config.data, 'train_revision', None),
      **role_dataset_kwargs('train'))
    if (config.data.streaming
        and not isinstance(
          train_set, torch.utils.data.IterableDataset)):
      train_set = _TorchIterableDatasetAdapter(train_set)
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    # Evaluation is deliberately finite and map-style, even when the training
    # corpus is streamed.  This keeps validation/test length and ordering
    # stable for checkpoint selection and comparable metrics.
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=False,
      revision=getattr(config.data, 'valid_revision', None),
      **role_dataset_kwargs('valid'))

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=(config.loader.num_workers > 0))
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if (valid_seed is None
        or not bool(data_option('shuffle_validation', True))):
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=(config.loader.num_workers > 0))
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
