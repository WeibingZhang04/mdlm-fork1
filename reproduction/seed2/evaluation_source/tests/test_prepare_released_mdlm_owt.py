import hashlib
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch
import yaml

from scripts.prepare_released_mdlm_owt import (
  RELEASE_FILENAME,
  RELEASE_REPOSITORY,
  RELEASE_REVISION,
  RELEASE_SHA256,
  convert_release,
  validate_backbone_state,
  verify_release_file,
)


class PrepareReleasedMdlmOwtTest(unittest.TestCase):

  def test_pinned_release_identity_is_immutable(self):
    self.assertEqual(RELEASE_REPOSITORY, 'kuleshov-group/mdlm-owt')
    self.assertEqual(
      RELEASE_REVISION,
      'd0958fa851335ece6c15260ce0025f030673c0fb')
    self.assertEqual(RELEASE_FILENAME, 'model.safetensors')
    self.assertEqual(
      RELEASE_SHA256,
      '47149e73f7552f39ea9776dbe74d925d25237bcf2ed2e2ec03cdff9d51c82aa4')
    config_path = (
      Path(__file__).resolve().parents[1]
      / 'configs/model/contextual-forest-small.yaml')
    training = yaml.safe_load(config_path.read_text())[
      'structured_decoder']['training']
    self.assertIs(training['use_ema_backbone'], False)

  def test_verification_fails_before_parsing_wrong_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory) / RELEASE_FILENAME
      source.write_bytes(b'not a safetensors file')
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        verify_release_file(
          source, expected_sha256='0' * 64, expected_size_bytes=None)

  def test_converter_preserves_backbone_namespace_and_marks_no_ema(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / RELEASE_FILENAME
      tensors = {
        'backbone.layer.bias': torch.tensor([1.0, 2.0]),
        'backbone.layer.weight': torch.arange(6.0).reshape(2, 3),
      }
      save_file(tensors, source)
      expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
      output = root / 'mdlm-owt-backbone.pt'
      result = convert_release(
        source, output,
        expected_sha256=expected_sha256,
        expected_size_bytes=source.stat().st_size,
        expected_tensor_count=2)

      checkpoint = torch.load(output, map_location='cpu', weights_only=False)
      self.assertEqual(set(checkpoint), {'state_dict', 'metadata'})
      self.assertEqual(set(checkpoint['state_dict']), set(tensors))
      for key, value in tensors.items():
        torch.testing.assert_close(checkpoint['state_dict'][key], value)
      self.assertFalse(checkpoint['metadata']['ema_available'])
      self.assertFalse(checkpoint['metadata']['ema_used'])
      self.assertEqual(
        checkpoint['metadata']['required_loader_setting'],
        'use_ema_backbone=false')
      self.assertEqual(result['tensor_count'], 2)
      self.assertNotIn('ema', checkpoint)

  def test_converter_refuses_overwrite_and_non_backbone_keys(self):
    with self.assertRaisesRegex(ValueError, r'only backbone\.\* keys'):
      validate_backbone_state(
        {'layer.weight': torch.ones(1)}, expected_tensor_count=1)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / RELEASE_FILENAME
      save_file({'backbone.weight': torch.ones(1)}, source)
      output = root / 'output.pt'
      output.write_text('keep me')
      expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
      with self.assertRaisesRegex(FileExistsError, '--force'):
        convert_release(
          source, output,
          expected_sha256=expected_sha256,
          expected_size_bytes=source.stat().st_size,
          expected_tensor_count=1)
      self.assertEqual(output.read_text(), 'keep me')


if __name__ == '__main__':
  unittest.main()
