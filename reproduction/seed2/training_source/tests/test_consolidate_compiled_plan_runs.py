import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import consolidate_compiled_plan_runs as consolidate


def _write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + '\n')


class ConsolidateCompiledPlanRunsTest(unittest.TestCase):

  def test_copies_complete_partition_union_and_reuses_verified_jobs(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      canonical = root / 'canonical'
      source0 = root / 'source0'
      source1 = root / 'source1'
      destination = root / 'destination'
      relative_plan = Path('experiments/protocol/plans/frozen')
      plan_dir = source0 / relative_plan
      job_ids = ['eval--a', 'eval--b']
      _write_json(plan_dir / 'compiled-plan.json', {'job_ids': job_ids})
      for index, job_id in enumerate(job_ids):
        artifact = canonical / 'experiments/protocol/runs' / job_id
        _write_json(plan_dir / 'jobs' / f'{job_id}.json', {
          'job_id': job_id,
          'artifact_dir': str(artifact),
        })
        source = [source0, source1][index]
        marker = {
          'job_id': job_id,
          'job_execution_sha256': f'execution-{index}',
          'outputs': [{
            'name': 'records',
            'relative_path': 'records.jsonl',
            'sha256': f'output-{index}',
            'size_bytes': index + 1,
          }],
        }
        run = source / artifact.relative_to(canonical)
        _write_json(run / consolidate.SUCCESS_MARKER, marker)
        (run / 'records.jsonl').write_text(f'{index}\n')

      output0 = root / 'consolidation-0.json'
      argv = [
        'consolidate', '--plan-dir', str(plan_dir),
        '--source-root', str(source0), '--source-root', str(source1),
        '--destination-root', str(destination),
        '--canonical-root', str(canonical), '--output', str(output0),
      ]
      with mock.patch.object(sys, 'argv', argv):
        self.assertEqual(consolidate.main(), 0)
      result = json.loads(output0.read_text())
      self.assertEqual(result['num_jobs'], 2)
      self.assertEqual(result['num_copied'], 2)
      self.assertEqual(result['num_reused'], 0)
      self.assertTrue((destination / relative_plan / 'compiled-plan.json').is_file())

      output1 = root / 'consolidation-1.json'
      argv[-1] = str(output1)
      with mock.patch.object(sys, 'argv', argv):
        self.assertEqual(consolidate.main(), 0)
      replay = json.loads(output1.read_text())
      self.assertEqual(replay['num_copied'], 0)
      self.assertEqual(replay['num_reused'], 2)

  def test_rejects_conflicting_successful_duplicates(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      canonical = root / 'canonical'
      source0 = root / 'source0'
      source1 = root / 'source1'
      plan_dir = source0 / 'plans/frozen'
      job_id = 'eval--a'
      artifact = canonical / 'runs' / job_id
      _write_json(plan_dir / 'compiled-plan.json', {'job_ids': [job_id]})
      _write_json(plan_dir / 'jobs' / f'{job_id}.json', {
        'job_id': job_id,
        'artifact_dir': str(artifact),
      })
      for index, source in enumerate((source0, source1)):
        _write_json(
          source / artifact.relative_to(canonical) / consolidate.SUCCESS_MARKER,
          {
            'job_id': job_id,
            'job_execution_sha256': 'execution',
            'outputs': [{
              'name': 'records',
              'relative_path': 'records.jsonl',
              'sha256': f'conflict-{index}',
              'size_bytes': 1,
            }],
          })
      argv = [
        'consolidate', '--plan-dir', str(plan_dir),
        '--source-root', str(source0), '--source-root', str(source1),
        '--destination-root', str(root / 'destination'),
        '--canonical-root', str(canonical), '--output', str(root / 'out.json'),
      ]
      with mock.patch.object(sys, 'argv', argv):
        with self.assertRaisesRegex(ValueError, 'conflicting successful copies'):
          consolidate.main()


if __name__ == '__main__':
  unittest.main()
