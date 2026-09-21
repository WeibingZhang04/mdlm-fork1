"""Hierarchical paired inference for schema-v2 causal-denoising records."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NLL_METRICS = frozenset({
  'nll_sum',
  'structured_marginal_nll_sum',
  'factorized_backbone_nll_sum',
  'parameter_matched_no_edge_nll_sum',
  'matched_permuted_topology_nll_sum',
})


@dataclass(frozen=True)
class ContrastTerm:
  """One coefficient × arm/metric term in a paired NLL contrast."""

  arm: str
  metric: str
  coefficient: float

  def __post_init__(self):
    if not self.arm:
      raise ValueError('contrast arm must be non-empty')
    if self.metric not in NLL_METRICS:
      raise ValueError(f'unsupported causal NLL metric {self.metric!r}')
    if not math.isfinite(self.coefficient) or self.coefficient == 0:
      raise ValueError('contrast coefficient must be finite and nonzero')


def _record_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
  return (
    row['arm'], row['train_seed'], row['corruption_seed'], row['dataset'],
    row['dataset_revision'], float(row['mask_rate']), row['candidate_k'],
    row['document_id'], row['document_index'], row['document_sha256'],
    row['chunk_index'])


def _deduplicated_v2_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
  result: dict[tuple[Any, ...], dict[str, Any]] = {}
  for raw in records:
    row = dict(raw)
    if row.get('schema_version') != 2:
      raise ValueError('causal hierarchical analysis requires schema-v2 rows')
    key = _record_key(row)
    previous = result.get(key)
    if previous is not None and previous != row:
      raise ValueError(f'conflicting duplicate causal record: {key}')
    result[key] = row
  if not result:
    raise ValueError('causal hierarchical analysis has no records')
  return list(result.values())


def _percentile_interval(
    values: np.ndarray, confidence_level: float) -> tuple[float, float]:
  tail = (1.0 - confidence_level) / 2.0
  return tuple(float(value) for value in np.quantile(
    values, [tail, 1.0 - tail], method='linear'))


def _bootstrap_matrices(
    matrices: Mapping[tuple[str, str, float, int], Mapping[int, np.ndarray]],
    *,
    num_resamples: int,
    rng_seed: int,
    confidence_level: float,
    estimand: str,
) -> dict[str, Any]:
  if not matrices:
    raise ValueError('hierarchical bootstrap has no strata')
  if num_resamples < 1:
    raise ValueError('num_resamples must be positive')
  if not 0.0 < confidence_level < 1.0:
    raise ValueError('confidence_level must lie in (0,1)')
  first = next(iter(matrices.values()))
  train_seeds = sorted(first)
  if not train_seeds:
    raise ValueError('hierarchical bootstrap has no adapter seeds')
  if any(sorted(per_train) != train_seeds for per_train in matrices.values()):
    raise ValueError('bootstrap strata have different adapter seeds')

  rng = np.random.default_rng(rng_seed)
  stratum_distributions = {
    stratum: np.empty(num_resamples, dtype=np.float64)
    for stratum in matrices}
  pooled = np.empty(num_resamples, dtype=np.float64)
  train_count = len(train_seeds)
  for start in range(0, num_resamples, 256):
    stop = min(start + 256, num_resamples)
    count = stop - start
    # The same top-level adapter-seed resample is used across every stratum.
    train_indices = rng.integers(
      0, train_count, size=(count, train_count))
    pooled_chunk = np.zeros(count, dtype=np.float64)
    for stratum, per_train in matrices.items():
      matrix = np.stack([per_train[seed] for seed in train_seeds])
      document_count = matrix.shape[1]
      document_indices = rng.integers(
        0, document_count,
        size=(count, train_count, document_count))
      sampled = matrix[train_indices[:, :, None], document_indices]
      means = sampled.mean(axis=(1, 2))
      stratum_distributions[stratum][start:stop] = means
      pooled_chunk += means
    pooled[start:stop] = pooled_chunk / len(matrices)

  conditions = {}
  for stratum, distribution in stratum_distributions.items():
    per_train = matrices[stratum]
    point = math.fsum(
      float(values.mean()) for values in per_train.values()) / len(per_train)
    lower, upper = _percentile_interval(distribution, confidence_level)
    dataset, revision, mask_rate, adapter_k = stratum
    key = f'{dataset}|mask={mask_rate:.6f}|adapter_k={adapter_k}'
    conditions[key] = {
      'dataset': dataset,
      'dataset_revision': revision,
      'mask_rate': mask_rate,
      'adapter_candidate_k': adapter_k,
      'estimate': point,
      'ci_lower': lower,
      'ci_upper': upper,
    }
  pooled_point = math.fsum(
    math.fsum(float(values.mean()) for values in per_train.values())
    / len(per_train)
    for per_train in matrices.values()) / len(matrices)
  pooled_lower, pooled_upper = _percentile_interval(
    pooled, confidence_level)
  return {
    'method': 'hierarchical_paired_percentile_bootstrap',
    'estimand': estimand,
    'nesting': [
      'average corruption replications within source document',
      'resample adapter training seeds with replacement',
      'resample source documents within sampled adapter seed',
      'equal-weight frozen dataset x mask-rate strata',
    ],
    'top_level_resampling_unit': 'adapter_training_seed',
    'num_adapter_seeds': len(train_seeds),
    'num_strata': len(matrices),
    'num_resamples': num_resamples,
    'rng': 'NumPy Generator(PCG64)',
    'rng_seed': rng_seed,
    'confidence_level': confidence_level,
    'pooled': {
      'estimate': pooled_point,
      'ci_lower': pooled_lower,
      'ci_upper': pooled_upper,
    },
    'conditions': conditions,
  }


def _verify_pairing_digests(
    records: Sequence[Mapping[str, Any]], selected_arms: set[str]) -> None:
  digests: dict[tuple[Any, ...], set[str]] = {}
  for row in records:
    if row['arm'] not in selected_arms:
      continue
    key = (
      row['corruption_seed'], row['dataset'], row['dataset_revision'],
      float(row['mask_rate']), row['candidate_k'])
    digests.setdefault(key, set()).add(row['pairing_digest_sha256'])
  mismatched = [key for key, values in digests.items() if len(values) != 1]
  if mismatched:
    raise ValueError(
      f'pairing digests differ across arms/adapter seeds for {mismatched[0]}')


def contrast_document_matrices(
    records: Iterable[Mapping[str, Any]],
    terms: Sequence[ContrastTerm],
) -> dict[tuple[str, str, float, int], dict[int, np.ndarray]]:
  """Build document-paired matrices for any predeclared linear contrast."""
  records = _deduplicated_v2_records(records)
  terms = tuple(terms)
  if len(terms) < 2:
    raise ValueError('a causal contrast requires at least two terms')
  term_keys = [(term.arm, term.metric) for term in terms]
  if len(set(term_keys)) != len(term_keys):
    raise ValueError('causal contrast contains duplicate arm/metric terms')
  selected_arms = {term.arm for term in terms}
  missing_arms = selected_arms - {row['arm'] for row in records}
  if missing_arms:
    raise ValueError(f'causal records are missing arms {sorted(missing_arms)}')
  _verify_pairing_digests(records, selected_arms)

  selected = [row for row in records if row['arm'] in selected_arms]
  train_seeds = sorted({row['train_seed'] for row in selected})
  eval_seeds = sorted({row['corruption_seed'] for row in selected})
  strata = sorted({
    (row['dataset'], row['dataset_revision'], float(row['mask_rate']),
     row['candidate_k']) for row in selected})

  # Aggregate document-local chunks before dividing by masked-token count.
  totals: dict[tuple[Any, ...], dict[str, float]] = {}
  window_sets: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
  for row in selected:
    document_key = (
      row['arm'], row['train_seed'], row['corruption_seed'], row['dataset'],
      row['dataset_revision'], float(row['mask_rate']), row['candidate_k'],
      row['document_id'], row['document_index'])
    value = totals.setdefault(document_key, {'masked_tokens': 0.0})
    value['masked_tokens'] += float(row['masked_tokens'])
    for metric in {term.metric for term in terms if term.arm == row['arm']}:
      value[metric] = value.get(metric, 0.0) + float(row[metric])
    cell_key = document_key[:7]
    window_sets.setdefault(cell_key, set()).add((
      row['document_id'], row['document_index'], row['document_sha256'],
      row['chunk_index']))

  matrices = {}
  for stratum in strata:
    dataset, revision, mask_rate, adapter_k = stratum
    expected_windows = []
    expected_documents = []
    for term in terms:
      for train_seed in train_seeds:
        for eval_seed in eval_seeds:
          cell = (
            term.arm, train_seed, eval_seed, dataset, revision, mask_rate,
            adapter_k)
          windows = window_sets.get(cell)
          if not windows:
            raise ValueError(f'incomplete causal factorial cell {cell}')
          expected_windows.append(windows)
          expected_documents.append({(item[0], item[1]) for item in windows})
    reference_windows = expected_windows[0]
    if any(value != reference_windows for value in expected_windows):
      raise ValueError(f'window identities differ in stratum {stratum}')
    reference_documents = expected_documents[0]
    if any(value != reference_documents for value in expected_documents):
      raise ValueError(f'document identities differ in stratum {stratum}')

    ordered_documents = sorted(reference_documents)
    per_train = {}
    for train_seed in train_seeds:
      document_values = []
      for document_id, document_index in ordered_documents:
        replication_values = []
        for eval_seed in eval_seeds:
          contrast = 0.0
          for term in terms:
            key = (
              term.arm, train_seed, eval_seed, dataset, revision, mask_rate,
              adapter_k, document_id, document_index)
            value = totals[key]
            if value['masked_tokens'] <= 0:
              raise ValueError(f'document has no masked tokens: {key}')
            contrast += (
              term.coefficient * value[term.metric]
              / value['masked_tokens'])
          replication_values.append(contrast)
        document_values.append(
          math.fsum(replication_values) / len(replication_values))
      per_train[train_seed] = np.asarray(
        document_values, dtype=np.float64)
    matrices[stratum] = per_train
  return matrices


def aggregate_causal_contrast(
    records: Iterable[Mapping[str, Any]],
    *,
    name: str,
    terms: Sequence[ContrastTerm],
    num_resamples: int = 20_000,
    rng_seed: int = 2701,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
  """Aggregate a paired NLL contrast with adapter seed at the top level."""
  terms = tuple(terms)
  matrices = contrast_document_matrices(records, terms)
  expression = ' + '.join(
    f'{term.coefficient:+g}*{term.arm}:{term.metric}' for term in terms)
  result = _bootstrap_matrices(
    matrices,
    num_resamples=num_resamples,
    rng_seed=rng_seed,
    confidence_level=confidence_level,
    estimand=expression)
  return {
    'name': name,
    'terms': [
      {'arm': term.arm, 'metric': term.metric,
       'coefficient': term.coefficient}
      for term in terms],
    'analysis': result,
  }


def _support_document_matrices(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    support_k: int,
    value_name: str,
) -> dict[tuple[str, str, float, int], dict[int, np.ndarray]]:
  selected = [row for row in records if row['arm'] == arm]
  if not selected:
    raise ValueError(f'causal records have no arm {arm!r}')
  train_seeds = sorted({row['train_seed'] for row in selected})
  eval_seeds = sorted({row['corruption_seed'] for row in selected})
  strata = sorted({
    (row['dataset'], row['dataset_revision'], float(row['mask_rate']),
     row['candidate_k']) for row in selected})
  totals: dict[tuple[Any, ...], dict[str, float]] = {}
  documents_by_cell: dict[tuple[Any, ...], set[tuple[str, int]]] = {}
  for row in selected:
    support = {
      entry['candidate_k']: entry for entry in row['candidate_support']}
    if support_k not in support:
      raise ValueError(f'candidate support record omits K={support_k}')
    key = (
      row['train_seed'], row['corruption_seed'], row['dataset'],
      row['dataset_revision'], float(row['mask_rate']), row['candidate_k'],
      row['document_id'], row['document_index'])
    value = totals.setdefault(key, {
      'masked_tokens': 0.0, 'candidate_hits': 0.0,
      'retained_mass_sum': 0.0})
    value['masked_tokens'] += float(row['masked_tokens'])
    value['candidate_hits'] += float(support[support_k]['candidate_hits'])
    value['retained_mass_sum'] += float(
      support[support_k]['retained_mass_sum'])
    documents_by_cell.setdefault(key[:6], set()).add((
      row['document_id'], row['document_index']))

  matrices = {}
  for stratum in strata:
    dataset, revision, mask_rate, adapter_k = stratum
    document_sets = []
    for train_seed in train_seeds:
      for eval_seed in eval_seeds:
        cell = (
          train_seed, eval_seed, dataset, revision, mask_rate, adapter_k)
        documents = documents_by_cell.get(cell)
        if not documents:
          raise ValueError(f'incomplete support factorial cell {cell}')
        document_sets.append(documents)
    reference = document_sets[0]
    if any(documents != reference for documents in document_sets):
      raise ValueError(f'support document identities differ in {stratum}')
    per_train = {}
    for train_seed in train_seeds:
      values = []
      for document_id, document_index in sorted(reference):
        replicated = []
        for eval_seed in eval_seeds:
          key = (
            train_seed, eval_seed, dataset, revision, mask_rate, adapter_k,
            document_id, document_index)
          value = totals[key]
          if value['masked_tokens'] <= 0:
            raise ValueError(f'document has no masked tokens: {key}')
          replicated.append(value[value_name] / value['masked_tokens'])
        values.append(math.fsum(replicated) / len(replicated))
      per_train[train_seed] = np.asarray(values, dtype=np.float64)
    matrices[stratum] = per_train
  return matrices


def aggregate_candidate_support(
    records: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    num_resamples: int = 20_000,
    rng_seed: int = 3701,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
  """Aggregate K=32/64/128/256 recall and retained-mass diagnostics."""
  records = _deduplicated_v2_records(records)
  selected = [row for row in records if row['arm'] == arm]
  if not selected:
    raise ValueError(f'causal records have no arm {arm!r}')
  support_ks = tuple(
    entry['candidate_k'] for entry in selected[0]['candidate_support'])
  for row in selected[1:]:
    observed = tuple(
      entry['candidate_k'] for entry in row['candidate_support'])
    if observed != support_ks:
      raise ValueError('candidate support K grid differs across records')
  by_k = {}
  for support_k in support_ks:
    by_metric = {}
    for offset, (metric, value_name) in enumerate((
        ('candidate_recall', 'candidate_hits'),
        ('retained_unary_mass', 'retained_mass_sum'))):
      matrices = _support_document_matrices(
        selected, arm=arm, support_k=support_k, value_name=value_name)
      by_metric[metric] = _bootstrap_matrices(
        matrices,
        num_resamples=num_resamples,
        rng_seed=rng_seed + support_k * 10 + offset,
        confidence_level=confidence_level,
        estimand=f'{metric} at support K={support_k}')
    by_k[str(support_k)] = by_metric
  return {
    'arm': arm,
    'support_candidate_ks': list(support_ks),
    'by_candidate_k': by_k,
  }


def aggregate_topology_permutation_diagnostic(
    records: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    minimum_pooled_changed_edge_fraction: float,
    minimum_condition_changed_edge_fraction: float,
) -> dict[str, Any]:
  """Gate that the matched topology permutation actually changes edges.

  Fractions are edge-weighted because the scientific question is how many
  selected pair factors were reassigned.  Results are reported both pooled
  and for every frozen dataset/mask-rate stratum; an empty forest fails closed.
  """
  for name, value in (
      ('minimum_pooled_changed_edge_fraction',
       minimum_pooled_changed_edge_fraction),
      ('minimum_condition_changed_edge_fraction',
       minimum_condition_changed_edge_fraction)):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
      raise ValueError(f'{name} must be finite and lie in [0,1]')
  records = _deduplicated_v2_records(records)
  selected = [row for row in records if row['arm'] == arm]
  if not selected:
    raise ValueError(f'causal records have no arm {arm!r}')

  totals: dict[tuple[str, str, float, int], list[int]] = {}
  for row in selected:
    selected_edges = row.get('selected_edges')
    changed_edges = row.get('permuted_changed_edges')
    if (not isinstance(selected_edges, int) or isinstance(selected_edges, bool)
        or not isinstance(changed_edges, int) or isinstance(changed_edges, bool)
        or selected_edges < 0 or changed_edges < 0
        or changed_edges > selected_edges):
      raise ValueError('invalid selected/permuted changed-edge counts')
    selected_degrees = row.get('selected_degree_sequence')
    permuted_degrees = row.get('permuted_degree_sequence')
    selected_components = row.get('selected_component_sizes')
    permuted_components = row.get('permuted_component_sizes')
    for name, signature, positive in (
        ('selected_degree_sequence', selected_degrees, False),
        ('permuted_degree_sequence', permuted_degrees, False),
        ('selected_component_sizes', selected_components, True),
        ('permuted_component_sizes', permuted_components, True)):
      if (not isinstance(signature, (list, tuple)) or not signature
          or any(not isinstance(value, int) or isinstance(value, bool)
                 or value < int(positive) for value in signature)
          or list(signature) != sorted(signature)):
        raise ValueError(f'invalid {name} in causal record')
    masked_tokens = row.get('masked_tokens')
    if (not isinstance(masked_tokens, int) or isinstance(masked_tokens, bool)
        or masked_tokens <= 0):
      raise ValueError('invalid masked_tokens in causal record')
    if (len(selected_degrees) != masked_tokens
        or len(permuted_degrees) != masked_tokens
        or sum(selected_degrees) != 2 * selected_edges
        or sum(permuted_degrees) != 2 * selected_edges
        or sum(selected_components) != masked_tokens
        or sum(permuted_components) != masked_tokens):
      raise ValueError('causal record topology signatures are inconsistent')
    degree_preserved = list(selected_degrees) == list(permuted_degrees)
    components_preserved = (
      list(selected_components) == list(permuted_components))
    key = (
      row['dataset'], row['dataset_revision'], float(row['mask_rate']),
      row['candidate_k'])
    cell = totals.setdefault(key, [0, 0, 0, 0, 0])
    cell[0] += selected_edges
    cell[1] += changed_edges
    cell[2] += 1
    cell[3] += int(degree_preserved)
    cell[4] += int(components_preserved)

  conditions = {}
  pooled_selected = 0
  pooled_changed = 0
  condition_passes = []
  condition_degree_passes = []
  condition_component_passes = []
  pooled_records = 0
  pooled_degree_preserved = 0
  pooled_components_preserved = 0
  for (dataset, revision, mask_rate, adapter_k), (
      selected_edges, changed_edges, num_records, degree_preserved_records,
      component_preserved_records) in sorted(totals.items()):
    if selected_edges <= 0:
      raise ValueError(
        'topology permutation diagnostic encountered a condition with no '
        'selected edges')
    fraction = changed_edges / selected_edges
    passed = fraction >= minimum_condition_changed_edge_fraction
    condition_passes.append(passed)
    degree_passed = degree_preserved_records == num_records
    component_passed = component_preserved_records == num_records
    condition_degree_passes.append(degree_passed)
    condition_component_passes.append(component_passed)
    key = f'{dataset}|mask={mask_rate:.6f}|adapter_k={adapter_k}'
    conditions[key] = {
      'dataset': dataset,
      'dataset_revision': revision,
      'mask_rate': mask_rate,
      'adapter_candidate_k': adapter_k,
      'num_records': num_records,
      'degree_sequence_preserved_records': degree_preserved_records,
      'component_sizes_preserved_records': component_preserved_records,
      'degree_sequence_preserved_every_record': degree_passed,
      'component_sizes_preserved_every_record': component_passed,
      'selected_edges': selected_edges,
      'changed_edges': changed_edges,
      'changed_edge_fraction': fraction,
      'minimum_changed_edge_fraction': (
        minimum_condition_changed_edge_fraction),
      'passed': passed,
    }
    pooled_selected += selected_edges
    pooled_changed += changed_edges
    pooled_records += num_records
    pooled_degree_preserved += degree_preserved_records
    pooled_components_preserved += component_preserved_records
  if pooled_selected <= 0:
    raise ValueError('topology permutation diagnostic has no selected edges')
  pooled_fraction = pooled_changed / pooled_selected
  pooled_passed = pooled_fraction >= minimum_pooled_changed_edge_fraction
  return {
    'arm': arm,
    'estimand': 'edge_weighted_fraction_of_selected_edges_reassigned',
    'pooled': {
      'num_records': pooled_records,
      'degree_sequence_preserved_records': pooled_degree_preserved,
      'component_sizes_preserved_records': pooled_components_preserved,
      'degree_sequence_preserved_every_record': (
        pooled_degree_preserved == pooled_records),
      'component_sizes_preserved_every_record': (
        pooled_components_preserved == pooled_records),
      'selected_edges': pooled_selected,
      'changed_edges': pooled_changed,
      'changed_edge_fraction': pooled_fraction,
      'minimum_changed_edge_fraction': (
        minimum_pooled_changed_edge_fraction),
      'passed': pooled_passed,
    },
    'conditions': conditions,
    'gate': {
      'degree_sequence_preserved_every_record': (
        bool(condition_degree_passes) and all(condition_degree_passes)),
      'component_sizes_preserved_every_record': (
        bool(condition_component_passes) and all(condition_component_passes)),
      'pooled_fraction_passed': pooled_passed,
      'every_condition_fraction_passed': all(condition_passes),
      'passed': (
        pooled_passed
        and all(condition_passes)
        and bool(condition_degree_passes)
        and all(condition_degree_passes)
        and all(condition_component_passes)),
    },
  }


def aggregate_technical_diagnostics(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_arms: Sequence[str],
    expected_train_seeds: Sequence[int],
    expected_support_ks: Sequence[int],
    no_edge_absolute_tolerance: float = 1e-10,
    support_monotonicity_tolerance: float = 1e-6,
) -> dict[str, Any]:
  """Summarize result-direction-neutral validity checks for causal routing.

  These checks establish that the four-arm paired experiment produced a
  numerically usable, structurally non-degenerate artifact. They never test
  whether any NLL contrast is positive, negative, large, or significant.
  """
  rows = _deduplicated_v2_records(records)
  arms = list(expected_arms)
  if not arms or len(arms) != len(set(arms)):
    raise ValueError('expected_arms must be non-empty and unique')
  observed_arms = sorted({row['arm'] for row in rows})
  if set(observed_arms) != set(arms):
    raise ValueError(
      f'causal technical diagnostics expected arms {sorted(arms)}, '
      f'found {observed_arms}')
  train_seeds = list(expected_train_seeds)
  if (not train_seeds or train_seeds != sorted(set(train_seeds))
      or any(not isinstance(value, int) or isinstance(value, bool)
             or value <= 0 for value in train_seeds)):
    raise ValueError(
      'expected_train_seeds must be increasing positive integers')
  observed_train_seeds = sorted({row['train_seed'] for row in rows})
  if observed_train_seeds != train_seeds:
    raise ValueError(
      f'causal technical diagnostics expected train seeds {train_seeds}, '
      f'found {observed_train_seeds}')
  support_ks = list(expected_support_ks)
  if (not support_ks or support_ks != sorted(set(support_ks))
      or any(not isinstance(value, int) or isinstance(value, bool)
             or value <= 0 for value in support_ks)):
    raise ValueError('expected_support_ks must be increasing positive integers')
  for name, value in (
      ('no_edge_absolute_tolerance', no_edge_absolute_tolerance),
      ('support_monotonicity_tolerance', support_monotonicity_tolerance)):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or float(value) < 0.0):
      raise ValueError(f'{name} must be finite and non-negative')

  pairing: dict[tuple[Any, ...], set[str]] = {}
  paired_units: dict[tuple[Any, ...], dict[tuple[str, int], int]] = {}
  topology: dict[tuple[str, str, str, float, int], list[int]] = {}
  maximum_no_edge_error = 0.0
  support_grid_complete = True
  support_monotone = True
  finite_statistics = True
  for row in rows:
    pairing_key = (
      row['corruption_seed'], row['dataset'], row['dataset_revision'],
      float(row['mask_rate']), row['candidate_k'])
    pairing.setdefault(pairing_key, set()).add(
      row['pairing_digest_sha256'])
    paired_unit_key = (
      row['corruption_seed'], row['dataset'], row['dataset_revision'],
      float(row['mask_rate']), row['candidate_k'], row['document_id'],
      row['document_index'], row['document_sha256'], row['chunk_index'])
    arm_seed = (row['arm'], row['train_seed'])
    unit = paired_units.setdefault(paired_unit_key, {})
    previous_masked = unit.get(arm_seed)
    if previous_masked is not None and previous_masked != row['masked_tokens']:
      raise ValueError(
        'conflicting masked-token counts within a paired arm/seed unit')
    unit[arm_seed] = int(row['masked_tokens'])
    topology_key = (
      row['arm'], row['dataset'], row['dataset_revision'],
      float(row['mask_rate']), row['candidate_k'])
    topology_totals = topology.setdefault(topology_key, [0, 0, 0])
    topology_totals[0] += int(row['selected_edges'])
    topology_totals[1] += int(row['masked_tokens'])
    topology_totals[2] += 1

    no_edge_error = abs(
      float(row['factorized_backbone_nll_sum'])
      - float(row['parameter_matched_no_edge_nll_sum']))
    maximum_no_edge_error = max(maximum_no_edge_error, no_edge_error)
    finite_statistics = finite_statistics and all(math.isfinite(float(
      row[field])) for field in NLL_METRICS | {
        'masked_tokens', 'selected_edges', 'permuted_changed_edges'})

    support = row['candidate_support']
    observed_support_ks = [entry['candidate_k'] for entry in support]
    support_grid_complete = (
      support_grid_complete and observed_support_ks == support_ks)
    for previous, current in zip(support, support[1:]):
      if current['candidate_hits'] < previous['candidate_hits']:
        support_monotone = False
      if (float(current['retained_mass_sum'])
          + float(support_monotonicity_tolerance)
          < float(previous['retained_mass_sum'])):
        support_monotone = False

  mismatched_pairing = [
    key for key, digests in pairing.items() if len(digests) != 1]
  expected_arm_seed_cells = {
    (arm, train_seed) for arm in arms for train_seed in train_seeds}
  incomplete_paired_units = [
    key for key, values in paired_units.items()
    if set(values) != expected_arm_seed_cells]
  masked_token_mismatched_units = [
    key for key, values in paired_units.items()
    if len(set(values.values())) != 1]
  topology_conditions = {}
  nonempty_conditions = []
  for (arm, dataset, revision, mask_rate, adapter_k), (
      selected_edges, masked_tokens, num_records) in sorted(topology.items()):
    if masked_tokens <= 0:
      raise ValueError('topology diagnostic condition has no masked tokens')
    nonempty = selected_edges > 0
    nonempty_conditions.append(nonempty)
    key = (
      f'{arm}|{dataset}|mask={mask_rate:.6f}|adapter_k={adapter_k}')
    topology_conditions[key] = {
      'arm': arm,
      'dataset': dataset,
      'dataset_revision': revision,
      'mask_rate': mask_rate,
      'adapter_candidate_k': adapter_k,
      'num_records': num_records,
      'masked_tokens': masked_tokens,
      'selected_edges': selected_edges,
      'selected_edges_per_masked_token': selected_edges / masked_tokens,
      'nonempty': nonempty,
    }

  return {
    'expected_arms': arms,
    'observed_arms': observed_arms,
    'num_records': len(rows),
    'finite_statistics': {'passed': finite_statistics},
    'pairing': {
      'expected_train_seeds': train_seeds,
      'observed_train_seeds': observed_train_seeds,
      'num_conditions': len(pairing),
      'mismatched_conditions': len(mismatched_pairing),
      'num_paired_units': len(paired_units),
      'expected_arm_train_cells_per_unit': len(expected_arm_seed_cells),
      'incomplete_paired_units': len(incomplete_paired_units),
      'masked_token_mismatched_units': len(masked_token_mismatched_units),
      'passed': (
        not mismatched_pairing
        and not incomplete_paired_units
        and not masked_token_mismatched_units),
    },
    'no_edge_identity': {
      'maximum_absolute_error': maximum_no_edge_error,
      'absolute_tolerance': float(no_edge_absolute_tolerance),
      'passed': maximum_no_edge_error <= float(no_edge_absolute_tolerance),
    },
    'candidate_support': {
      'expected_candidate_ks': support_ks,
      'grid_complete': support_grid_complete,
      'monotonicity_absolute_tolerance': float(
        support_monotonicity_tolerance),
      'monotone_within_tolerance': support_monotone,
    },
    'topology_structure': {
      'conditions': topology_conditions,
      'every_condition_nonempty': all(nonempty_conditions),
      'passed': bool(nonempty_conditions) and all(nonempty_conditions),
    },
  }
