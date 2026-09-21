import copy
import hashlib
import unittest

from evaluation.fresh_pair_statistics import ARMS, select_checkpoints, summarize_test


def sha(value):
  return hashlib.sha256(str(value).encode()).hexdigest()


def observation(arm, seed, step, document, rate, *, split='dev', gain=0.1, dependence=0.0):
  count = 2 if rate == 0.25 else 4
  joint = (-2.0 + gain) * count
  return {
    'split': split, 'arm': arm, 'training_seed': seed, 'checkpoint_step': step,
    'checkpoint_sha256': sha((seed, step)), 'dataset': 'corpus', 'document_id': document,
    'clean_token_sha256': sha(('clean', document)), 'mask_rate': rate,
    'corruption_seed': seed + 900000 + int(rate * 100),
    'mask_sha256': sha(('mask', document, rate, seed)),
    'corrupted_token_sha256': sha(('corrupted', document, rate, seed)),
    'active_token_count': count, 'backbone_log_probability': -2.0 * count,
    'joint_log_probability': joint, 'marginal_log_probability': joint - dependence * count,
    'dependence_log_probability': dependence * count,
  }


def dev_records(seeds=(1, 2)):
  rows = []
  gains = {'unary': {10: 0.1, 20: 0.2}, 'shared': {10: 0.2, 20: 0.3},
           'directional': {10: 0.5, 20: 0.4}}
  for seed in seeds:
    for step in (10, 20):
      for document in ('dev-0', 'dev-1'):
        for rate in (0.25, 0.5):
          for arm in ARMS:
            rows.append(observation(arm, seed, step, document, rate,
                                    gain=gains[arm][step], dependence=0.0 if arm == 'unary' else 0.05))
  return rows


def make_test_records(selection):
  rows = []
  for selected in selection['selected']:
    arm, seed, step = selected['arm'], selected['training_seed'], selected['checkpoint_step']
    for index, document in enumerate(('test-0', 'test-1')):
      gain = {'unary': 0.1, 'shared': 0.2, 'directional': 0.3 + index * 0.1}[arm]
      for rate in selection['mask_rates']:
        row = observation(arm, seed, step, document, rate, split='test',
                          gain=gain, dependence=0.0 if arm == 'unary' else 0.05)
        row['selection_sha256'] = selection['selection_sha256']
        rows.append(row)
  return rows


class FreshPairStatisticsTest(unittest.TestCase):

  def setUp(self):
    self.dev = dev_records()
    self.selection = select_checkpoints(self.dev, excluded_test_document_ids=['train-0', 'old-debug-0'])
    self.test = make_test_records(self.selection)

  def test_each_arm_selects_its_dev_checkpoint_before_test(self):
    for selected in self.selection['selected']:
      self.assertEqual(selected['checkpoint_step'], 10 if selected['arm'] == 'directional' else 20)
    before = copy.deepcopy(self.selection)
    result = summarize_test(self.test, self.selection, bootstrap_replicates=200)
    self.assertEqual(self.selection, before)
    metrics = result['overall']['metrics_nats_per_masked_token']
    self.assertAlmostEqual(metrics['backbone_nll']['estimate'], 2.0)
    self.assertAlmostEqual(metrics['directional_joint_gain']['estimate'], 0.35)
    self.assertAlmostEqual(metrics['directional_singleton_gain']['estimate'], 0.30)
    self.assertAlmostEqual(metrics['directional_dependence_gain']['estimate'], 0.05)
    self.assertAlmostEqual(metrics['directional_vs_unary']['estimate'], 0.25)
    self.assertAlmostEqual(metrics['directional_vs_shared']['estimate'], 0.15)
    self.assertEqual(result['overall']['documents'], 2)
    self.assertEqual(result['overall']['training_seeds'], 2)
    self.assertEqual(result['overall']['paired_observations'], 8)
    self.assertTrue(result['bootstrap']['training_seed_uncertainty_estimated'])
    self.assertEqual(set(result['by_mask_rate']), {'0.25', '0.5'})

  def test_selection_is_token_weighted_and_ties_choose_earlier(self):
    rows = []
    for arm in ARMS:
      for step in (10, 20):
        for document, count in [('small', 1), ('large', 9)]:
          row = observation(arm, 1, step, document, 0.5)
          row['active_token_count'] = count
          row['backbone_log_probability'] = -2 * count
          # Step10 has higher document-mean NLL, but lower total token NLL.
          nll = (10 if document == 'small' else 0) if step == 10 else (0 if document == 'small' else 18)
          row['joint_log_probability'] = row['marginal_log_probability'] = -nll
          row['dependence_log_probability'] = 0
          rows.append(row)
    selected = select_checkpoints(rows)
    self.assertTrue(all(row['checkpoint_step'] == 10 for row in selected['selected']))
    for row in rows:
      row['joint_log_probability'] = row['marginal_log_probability'] = -1.0
    tied = select_checkpoints(rows)
    self.assertTrue(all(row['checkpoint_step'] == 10 for row in tied['selected']))

  def test_missing_or_drifting_development_rows_are_rejected(self):
    malformed = [self.dev[:-1], self.dev + [self.dev[0]]]
    changed = copy.deepcopy(self.dev)
    changed[-1]['mask_sha256'] = sha('changed')
    malformed.append(changed)
    for rows in malformed:
      with self.assertRaises(ValueError):
        select_checkpoints(rows)

  def test_test_pairing_checkpoint_and_selection_bindings_are_checked(self):
    for field, value in [('mask_sha256', sha('wrong')), ('active_token_count', 999),
                         ('checkpoint_sha256', sha('wrong')), ('selection_sha256', sha('wrong'))]:
      rows = copy.deepcopy(self.test)
      rows[0][field] = value
      with self.assertRaises(ValueError):
        summarize_test(rows, self.selection, bootstrap_replicates=10)
    with self.assertRaisesRegex(ValueError, 'missing a paired arm'):
      summarize_test(self.test[:-1], self.selection, bootstrap_replicates=10)
    with self.assertRaisesRegex(ValueError, 'duplicate'):
      summarize_test(self.test + [self.test[0]], self.selection, bootstrap_replicates=10)

  def test_reused_documents_or_dev_windows_are_rejected(self):
    for document in ('dev-0', 'train-0', 'old-debug-0'):
      rows = copy.deepcopy(self.test)
      rows[0]['document_id'] = document
      with self.assertRaisesRegex(ValueError, 'reuses'):
        summarize_test(rows, self.selection, bootstrap_replicates=10)
    rows = copy.deepcopy(self.test)
    rows[0]['clean_token_sha256'] = self.dev[0]['clean_token_sha256']
    with self.assertRaisesRegex(ValueError, 'reuses'):
      summarize_test(rows, self.selection, bootstrap_replicates=10)

  def test_selection_cannot_be_modified_after_sealing(self):
    modified = copy.deepcopy(self.selection)
    modified['selected'][0]['checkpoint_step'] += 10
    with self.assertRaisesRegex(ValueError, 'digest mismatch'):
      summarize_test(self.test, modified, bootstrap_replicates=10)

  def test_rate_repeats_stay_in_one_document_bootstrap_cluster(self):
    first = summarize_test(self.test, self.selection, bootstrap_replicates=500, bootstrap_seed=3)
    duplicated = copy.deepcopy(self.test)
    extra = copy.deepcopy(self.test)
    for row in extra:
      row['corruption_seed'] += 100
    duplicated.extend(extra)
    second = summarize_test(duplicated, self.selection, bootstrap_replicates=500, bootstrap_seed=3)
    # Repeating every observation within its document must not narrow the CI.
    self.assertEqual(first['overall']['metrics_nats_per_masked_token'],
                     second['overall']['metrics_nats_per_masked_token'])

  def test_one_seed_is_reported_as_document_uncertainty_only(self):
    selected = select_checkpoints(dev_records(seeds=(1,)))
    result = summarize_test(make_test_records(selected), selected, bootstrap_replicates=100)
    self.assertFalse(result['bootstrap']['training_seed_uncertainty_estimated'])
    self.assertEqual(result['overall']['training_seeds'], 1)

  def test_invalid_probability_scores_are_rejected(self):
    for field, value in [('joint_log_probability', 1.0),
                         ('dependence_log_probability', 123.0),
                         ('backbone_log_probability', float('nan'))]:
      rows = copy.deepcopy(self.dev)
      rows[0][field] = value
      with self.assertRaises(ValueError):
        select_checkpoints(rows)


if __name__ == '__main__':
  unittest.main()
