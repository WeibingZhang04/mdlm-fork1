import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import structured_utils as utils
from scripts.run_generation_fast import run


class FastEntryTest(unittest.TestCase):
  def test_arguments_restoration_and_provenance(self):
    original = utils.sample_forest_low_rank
    argv = ['unchanged', 'arguments']
    with TemporaryDirectory() as folder:
      def main(received):
        self.assertIs(received, argv)
        self.assertIsNot(utils.sample_forest_low_rank, original)
        return 0
      pilot = SimpleNamespace(_parse_args=lambda _: SimpleNamespace(output_dir=Path(folder)), main=main)
      self.assertEqual(run(pilot, argv), 0)
      record = json.loads((Path(folder) / 'sampler-provenance.json').read_text())
      self.assertEqual(record['sampler_implementation'], 'level_draws')
      self.assertEqual(len(record['source_sha256']), 9)
    self.assertIs(utils.sample_forest_low_rank, original)

  def test_exception_restores_sampler(self):
    original = utils.sample_forest_low_rank
    def fail(_):
      raise RuntimeError('test failure')
    pilot = SimpleNamespace(_parse_args=lambda _: None, main=fail)
    with self.assertRaisesRegex(RuntimeError, 'test failure'):
      run(pilot, [])
    self.assertIs(utils.sample_forest_low_rank, original)
