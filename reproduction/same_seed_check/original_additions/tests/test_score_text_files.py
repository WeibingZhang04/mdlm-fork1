import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'score_text_files.py'
SPEC = importlib.util.spec_from_file_location('score_text_files', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScoreTextFilesParsingTest(unittest.TestCase):

  def test_raw_text(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'sample.txt'
      path.write_text('one generated sample')
      self.assertEqual(MODULE.load_text_groups(path), {
        'all': ['one generated sample'],
      })

  def test_mdlm_stdout(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'output.txt'
      path.write_text("config stuff\nText samples: ['first', 'second\\nline']\n")
      self.assertEqual(MODULE.load_text_groups(path), {
        'all': ['first', 'second\nline'],
      })

  def test_preview_combines_numbered_samples_by_condition(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'preview.txt'
      path.write_text(
        '=== base sample 0 ===\nfirst\n\n'
        '=== base sample 1 ===\nsecond\n\n'
        '=== treatment ===\nthird\n')
      self.assertEqual(MODULE.load_text_groups(path), {
        'base': ['first', 'second'],
        'treatment': ['third'],
      })

  def test_jsonl_groups_records(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'samples.jsonl'
      path.write_text(
        '{"condition":"base","text":"first"}\n'
        '{"condition":"base","text":"second"}\n'
        '{"arm":"dynamic","text":"third"}\n')
      self.assertEqual(MODULE.load_text_groups(path), {
        'base': ['first', 'second'],
        'dynamic': ['third'],
      })


if __name__ == '__main__':
  unittest.main()
