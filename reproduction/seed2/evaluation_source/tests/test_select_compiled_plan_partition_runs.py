import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import select_compiled_plan_partition_runs as select


def _write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + '\n')


def _write_run(root: Path, relative: Path, job_id: str, value: str) -> None:
  run = root / relative
  _write_json(run / select.SUCCESS_MARKER, {
    'job_id': job_id,
    'job_execution_sha256': 'execution',
    'outputs': [{
      'name': 'records',
      'relative_path': 'records.jsonl',
      'sha256': value,
      'size_bytes': 2,
    }],
  })
  (run / 'records.jsonl').write_text(f'{value}\n')


class SelectCompiledPlanPartitionRunsTest(unittest.TestCase):

  def test_base_priority_and_disjoint_fallback_with_conflict_audit(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      canonical = root / 'canonical'
      base = root / 'base'
      source0 = root / 'source0'
      source1 = root / 'source1'
      destination = root / 'selected'
      plan_dir = base / 'plans/frozen'
      job_ids = ['eval--base', 'eval--fallback']
      _write_json(plan_dir / 'compiled-plan.json', {'job_ids': job_ids})
      for job_id in job_ids:
        artifact = canonical / 'runs' / job_id
        _write_json(plan_dir / 'jobs' / f'{job_id}.json', {
          'job_id': job_id, 'artifact_dir': str(artifact)})
      _write_run(base, Path('runs/eval--base'), 'eval--base', 'base')
      _write_run(source0, Path('runs/eval--base'), 'eval--base', 'other')
      _write_run(
        source0, Path('runs/eval--fallback'), 'eval--fallback', 'first')
      _write_run(
        source1, Path('runs/eval--fallback'), 'eval--fallback', 'second')

      output = root / 'selection.json'
      argv = [
        'select', '--plan-dir', str(plan_dir), '--base-root', str(base),
        '--source-root', str(source0), '--source-root', str(source1),
        '--destination-root', str(destination),
        '--canonical-root', str(canonical), '--output', str(output),
      ]
      with mock.patch.object(sys, 'argv', argv):
        self.assertEqual(select.main(), 0)
      result = json.loads(output.read_text())
      self.assertEqual(result['num_selected_from_base'], 1)
      self.assertEqual(result['num_copied'], 1)
      self.assertEqual(result['num_jobs_with_conflicting_candidates'], 2)
      self.assertFalse((destination / 'runs/eval--base').exists())
      self.assertTrue(
        (destination / 'runs/eval--fallback' / select.SUCCESS_MARKER).is_file())
      self.assertEqual(
        result['jobs']['eval--fallback']['selected_source_root'],
        str(source0.resolve()))


if __name__ == '__main__':
  unittest.main()
