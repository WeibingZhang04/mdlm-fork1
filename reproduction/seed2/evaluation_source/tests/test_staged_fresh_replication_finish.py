"""The replication finish workflow scores only newly selected seeds."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from deploy.gcloud import finish_staged_fresh_replication_v1 as finish


class ReplicationFinishTest(unittest.TestCase):
  @staticmethod
  def single_seed_sidecar(root, seed, run_name):
    return {
      'selected_checkpoint_files': [
        {'arm': arm, 'training_seed': seed,
         'path': str(root / run_name / 'checkpoints' / f'step-{step:06d}.pt')}
        for arm, step in [('shared', 900), ('directional', 900), ('unary', 1000)]
      ],
      'selection_file_sha256': 'a' * 64,
    }

  def test_seal_all_runs_but_score_only_new_unique_checkpoint_files(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      files = []
      for seed in (2, 3):
        for arm, step in [('shared', 900), ('directional', 900), ('unary', 1000)]:
          files.append({'arm': arm, 'training_seed': seed,
                         'path': str(root / f'fresh-replication-seed{seed}-retry1-v1'
                                     / 'checkpoints' / f'step-{step:06d}.pt')})
      sidecar = {'selected_checkpoint_files': files, 'selection_file_sha256': 'a' * 64}
      result = ({'overall': {}}, {'training_seeds': [1, 2, 3], 'records': 4608})
      with mock.patch.object(finish, 'seal', return_value=({}, sidecar)) as seal, \
           mock.patch.object(finish, 'evaluate') as evaluate, \
           mock.patch.object(finish, 'aggregate', return_value=result) as aggregate, \
           mock.patch.object(finish, 'file_sha256', return_value='b' * 64), \
           mock.patch.dict(finish.os.environ), redirect_stdout(io.StringIO()):
        finish.main(['--experiment-root', str(root)])
      self.assertEqual(seal.call_count, 2)
      self.assertEqual([path.name for path in seal.call_args_list[0].args[0]],
                       ['fresh-replication-seed2-retry1-v1', 'fresh-replication-seed3-retry1-v1'])
      self.assertEqual([path.name for path in seal.call_args_list[1].args[0]],
                       ['fresh-pilot-resume-v1', 'fresh-replication-seed2-retry1-v1',
                        'fresh-replication-seed3-retry1-v1'])
      command = evaluate.call_args.args[0]
      selected = [command[index + 1] for index, item in enumerate(command) if item == '--selected-checkpoint']
      self.assertEqual(len(selected), 4)
      self.assertEqual(selected, sorted({item['path'] for item in files}))
      self.assertFalse(any('fresh-pilot-resume-v1' in path for path in selected))
      shards = aggregate.call_args.args[4]
      self.assertEqual(len(shards), 2)
      self.assertEqual(shards[0][1], finish.PILOT_MANIFEST_SHA)
      self.assertEqual(shards[0][3], finish.PILOT_TEST_SHA)
      self.assertEqual(shards[0][2], root / 'fresh-test-v1')
      self.assertEqual(aggregate.call_args.args[5], root / 'fresh-test-seed123-v1')

  def test_incomplete_training_stops_before_test_access(self):
    with tempfile.TemporaryDirectory() as directory, \
         mock.patch.object(finish, 'seal', side_effect=ValueError('incomplete')) as seal, \
         mock.patch.object(finish, 'evaluate') as evaluate, \
         mock.patch.object(finish, 'aggregate') as aggregate, mock.patch.dict(finish.os.environ):
      with self.assertRaisesRegex(ValueError, 'incomplete'):
        finish.main(['--experiment-root', directory])
      self.assertEqual(seal.call_count, 1)
      evaluate.assert_not_called()
      aggregate.assert_not_called()

  def test_resumed_seed2_keeps_seed3_in_its_fresh_namespace(self):
    with tempfile.TemporaryDirectory() as directory, \
         mock.patch.object(finish, 'seal', side_effect=ValueError('incomplete')) as seal, \
         mock.patch.object(finish, 'evaluate') as evaluate, mock.patch.dict(finish.os.environ):
      with self.assertRaisesRegex(ValueError, 'incomplete'):
        finish.main(['--experiment-root', directory, '--seed2-resumed'])
      self.assertEqual([path.name for path in seal.call_args.args[0]],
                       ['fresh-replication-seed2-resume-v1',
                        'fresh-replication-seed3-retry1-v1'])
      evaluate.assert_not_called()

  def test_existing_outputs_are_never_overwritten(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(finish, 'seal') as seal:
      (Path(directory) / 'fresh-selection-seed23-v1').mkdir()
      with self.assertRaises(FileExistsError):
        finish.main(['--experiment-root', directory])
      seal.assert_not_called()

  def test_single_seed_routes_only_requested_run_and_never_combines(self):
    for seed in (2, 3):
      for run_tag in ('v1', 'retry1-v1'):
        for resumed in (False, True):
          with self.subTest(seed=seed, run_tag=run_tag, resumed=resumed), \
               tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_name = (f'fresh-replication-seed{seed}-resume-v1'
                        if seed == 2 and resumed else f'fresh-replication-seed{seed}-{run_tag}')
            sidecar = self.single_seed_sidecar(root, seed, run_name)
            events = io.StringIO()
            with mock.patch.object(finish, 'seal', return_value=({}, sidecar)) as seal, \
                 mock.patch.object(finish, 'evaluate') as evaluate, \
                 mock.patch.object(finish, 'aggregate') as aggregate, \
                 mock.patch.object(finish, 'file_sha256') as file_sha256, \
                 mock.patch.dict(finish.os.environ), redirect_stdout(events):
              arguments = ['--experiment-root', str(root), '--single-seed', str(seed),
                           '--run-tag', run_tag]
              if resumed:
                arguments.append('--seed2-resumed')
              finish.main(arguments)
            seal.assert_called_once_with(
              [root / run_name], root / 'fresh-data-v1/manifest.json', finish.DATA_SHA,
              root / f'fresh-selection-seed{seed}-v1')
            evaluate.assert_called_once()
            command = evaluate.call_args.args[0]
            option = lambda name: command[command.index(name) + 1]
            self.assertEqual(option('--selection'), str(root / f'fresh-selection-seed{seed}-v1/selection.json'))
            self.assertEqual(option('--output-dir'), str(root / f'fresh-test-seed{seed}-v1'))
            self.assertEqual(option('--expected-selection-sha256'), sidecar['selection_file_sha256'])
            self.assertEqual(option('--expected-data-manifest-sha256'), finish.DATA_SHA)
            self.assertEqual(option('--expected-test-jsonl-sha256'), finish.TEST_SHA)
            selected = [command[index + 1] for index, item in enumerate(command) if item == '--selected-checkpoint']
            self.assertEqual(selected, sorted({item['path'] for item in sidecar['selected_checkpoint_files']}))
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(run_name in path for path in selected))
            aggregate.assert_not_called()
            file_sha256.assert_not_called()  # No combined seal or original-test shard is opened.
            final = json.loads(events.getvalue().splitlines()[-1])
            self.assertEqual(final['event'], 'replication_seed_test_complete')
            self.assertEqual(final['training_seeds'], [seed])
            self.assertEqual(final['output_dir'], str(root / f'fresh-test-seed{seed}-v1'))

  def test_single_seed_does_not_touch_original_other_seed_or_combined_namespaces(self):
    for seed in (2, 3):
      with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        untouched = [root / 'fresh-test-v1/results.json',
                     root / 'fresh-selection-v1/selection-manifest.json']
        for other in (5 - seed, 23, 123):
          untouched.extend([root / f'fresh-selection-seed{other}-v1/selection.json',
                            root / f'fresh-test-seed{other}-v1/results.json'])
        for index, path in enumerate(untouched):
          path.parent.mkdir()
          path.write_text(f'immutable original artifact {index}\n')
        original_bytes = {path: path.read_bytes() for path in untouched}
        sidecar = self.single_seed_sidecar(root, seed, f'fresh-replication-seed{seed}-retry1-v1')
        with mock.patch.object(finish, 'seal', return_value=({}, sidecar)) as seal, \
             mock.patch.object(finish, 'evaluate'), mock.patch.object(finish, 'aggregate') as aggregate, \
             mock.patch.dict(finish.os.environ), redirect_stdout(io.StringIO()):
          finish.main(['--experiment-root', str(root), '--single-seed', str(seed)])
        seal.assert_called_once()
        aggregate.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in untouched}, original_bytes)
    self.assertEqual(finish.PILOT_MANIFEST_SHA, '7d425ba57cf1b5f419c8e90a3d7ddfe1e8afec6e2995ec46773b9a30e1bb5644')
    self.assertEqual(finish.PILOT_TEST_SHA, '538bf373dddcba7c0ed03a80f6f3bf594ccf1293f31895d1be98483a2f284cad')

  def test_single_seed_authentication_failure_prevents_test_access(self):
    for seed in (2, 3):
      for error in ('incomplete development grid', 'final-ledger hash mismatch'):
        with self.subTest(seed=seed, error=error), tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(finish, 'seal', side_effect=ValueError(error)) as seal, \
             mock.patch.object(finish, 'evaluate') as evaluate, \
             mock.patch.object(finish, 'aggregate') as aggregate, mock.patch.dict(finish.os.environ):
          with self.assertRaisesRegex(ValueError, error):
            finish.main(['--experiment-root', directory, '--single-seed', str(seed)])
          seal.assert_called_once()
          evaluate.assert_not_called()
          aggregate.assert_not_called()

  def test_wrong_seed_identity_prevents_test_access(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      # A correctly named directory cannot substitute a different sealed seed.
      sidecar = self.single_seed_sidecar(root, 3, 'fresh-replication-seed2-retry1-v1')
      with mock.patch.object(finish, 'seal', return_value=({}, sidecar)), \
           mock.patch.object(finish, 'evaluate') as evaluate, \
           mock.patch.object(finish, 'aggregate') as aggregate, mock.patch.dict(finish.os.environ):
        with self.assertRaisesRegex(ValueError, 'identities do not match'):
          finish.main(['--experiment-root', directory, '--single-seed', '2'])
        evaluate.assert_not_called()
        aggregate.assert_not_called()

  def test_single_seed_refuses_existing_or_dangling_symlink_outputs(self):
    for seed in (2, 3):
      for prefix in ('fresh-selection', 'fresh-test'):
        for symlink in (False, True):
          with self.subTest(seed=seed, prefix=prefix, symlink=symlink), \
               tempfile.TemporaryDirectory() as directory, \
               mock.patch.object(finish, 'seal') as seal, \
               mock.patch.object(finish, 'evaluate') as evaluate, \
               mock.patch.object(finish, 'aggregate') as aggregate:
            path = Path(directory) / f'{prefix}-seed{seed}-v1'
            if symlink:
              path.symlink_to(Path(directory) / 'absent-output-target', target_is_directory=True)
            else:
              path.mkdir()
            with self.assertRaises(FileExistsError):
              finish.main(['--experiment-root', directory, '--single-seed', str(seed)])
            seal.assert_not_called()
            evaluate.assert_not_called()
            aggregate.assert_not_called()


if __name__ == '__main__':
  unittest.main()
