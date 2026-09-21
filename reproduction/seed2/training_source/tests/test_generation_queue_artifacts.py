import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from evaluation.generation_queue_artifacts import (
  QueueLockError,
  SharedQueueLock,
  atomic_rename_directory_new,
  atomic_write_new,
)


class AtomicWriteTest(unittest.TestCase):

  def test_atomic_write_never_overwrites(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'artifact.json'
      atomic_write_new(path, 'first\n')
      with self.assertRaisesRegex(FileExistsError, 'refusing to overwrite'):
        atomic_write_new(path, 'second\n')
      self.assertEqual(path.read_text(), 'first\n')
      self.assertEqual(list(path.parent.glob(f'.{path.name}.tmp-*')), [])

  def test_atomic_directory_publish_never_replaces_even_empty_destination(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / 'staging'
      destination = root / 'final'
      source.mkdir()
      (source / 'bundle.json').write_text('first\n')
      destination.mkdir()
      with self.assertRaises(FileExistsError):
        atomic_rename_directory_new(source, destination)
      self.assertEqual((source / 'bundle.json').read_text(), 'first\n')
      self.assertTrue(destination.is_dir())
      destination.rmdir()
      atomic_rename_directory_new(source, destination)
      self.assertFalse(source.exists())
      self.assertEqual((destination / 'bundle.json').read_text(), 'first\n')


class SharedQueueLockTest(unittest.TestCase):

  def _lock(self, path, *, recover=False, reader=None):
    return SharedQueueLock(
      path,
      queue_id='arxiv-generation-queue',
      dataset_slug='arxiv',
      launch_plan_sha256='a' * 64,
      recover_stale=recover,
      process_identity_reader=(reader or (lambda pid: f'identity-{pid}')))

  def test_active_owner_blocks_second_queue_and_release_allows_next(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'generation-queue.lock'
      first = self._lock(path)
      first.acquire()
      with self.assertRaisesRegex(QueueLockError, 'is active'):
        self._lock(path).acquire()
      self.assertTrue(path.is_file())
      first.release()
      self.assertFalse(path.exists())
      with self._lock(path):
        self.assertTrue(path.is_file())
      self.assertFalse(path.exists())

  def test_stale_lock_is_preserved_without_explicit_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'generation-queue.lock'
      path.write_text(json.dumps({
        'schema_version': 1,
        'artifact': 'frozen_generation_queue_lock',
        'queue_id': 'wikitext-generation-queue',
        'dataset_slug': 'wikitext',
        'pid': 999_999,
        'process_start_identity': 'old-process',
        'launch_plan_sha256': 'b' * 64,
        'created_utc': '2026-08-31T00:00:00+00:00',
        'owner_token': 'old-token',
      }))
      with self.assertRaisesRegex(QueueLockError, 'is stale.*preserved'):
        self._lock(
          path,
          reader=lambda pid: (
            f'identity-{pid}' if pid == os.getpid() else None)).acquire()
      self.assertTrue(path.is_file())

  def test_explicit_stale_recovery_preserves_old_inode(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'generation-queue.lock'
      path.write_text('{invalid json')
      lock = self._lock(
        path, recover=True,
        reader=lambda pid: f'identity-{pid}' if pid == os.getpid() else None)
      with lock:
        self.assertTrue(path.is_file())
      self.assertFalse(path.exists())
      preserved = list(Path(directory).glob(
        'generation-queue.lock.invalid-*.preserved'))
      self.assertEqual(len(preserved), 1)
      self.assertEqual(preserved[0].read_text(), '{invalid json')

  def test_tampered_held_lock_is_not_deleted(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'generation-queue.lock'
      lock = self._lock(path)
      lock.acquire()
      path.write_text(path.read_text() + 'tamper')
      with self.assertRaisesRegex(QueueLockError, 'changed while held'):
        lock.release()
      self.assertTrue(path.is_file())

  def test_recovery_guard_serializes_stale_inspect_rename_create(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'generation-queue.lock'
      path.write_text(json.dumps({
        'schema_version': 1,
        'artifact': 'frozen_generation_queue_lock',
        'queue_id': 'old-queue',
        'dataset_slug': 'wikitext',
        'pid': 999_999,
        'process_start_identity': 'dead-owner',
        'launch_plan_sha256': 'b' * 64,
        'created_utc': '2026-08-31T00:00:00+00:00',
        'owner_token': 'old-token',
      }))
      def reader(pid):
        return f'identity-{pid}' if pid == os.getpid() else None
      first = self._lock(path, recover=True, reader=reader)
      entered_recovery = threading.Event()
      permit_recovery = threading.Event()
      original_recover = first._recover_existing

      def paused_recovery(state):
        entered_recovery.set()
        self.assertTrue(permit_recovery.wait(timeout=5))
        original_recover(state)

      first._recover_existing = paused_recovery
      outcome = []

      def acquire_first():
        try:
          first.acquire()
          outcome.append('acquired')
        except Exception as error:  # pragma: no cover - diagnostic path
          outcome.append(error)

      worker = threading.Thread(target=acquire_first)
      worker.start()
      self.assertTrue(entered_recovery.wait(timeout=5))
      with self.assertRaisesRegex(
          QueueLockError, 'acquisition guard is active'):
        self._lock(path, recover=True, reader=reader).acquire()
      # The stale inode is still present: the second recoverer could not race
      # the first process between inspection and preservation.
      self.assertEqual(json.loads(path.read_text())['queue_id'], 'old-queue')
      permit_recovery.set()
      worker.join(timeout=5)
      self.assertFalse(worker.is_alive())
      self.assertEqual(outcome, ['acquired'])
      first.release()

  def test_stale_or_invalid_recovery_guard_always_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'generation-queue.lock'
      guard = path.with_name(f'{path.name}.acquire-guard')
      def reader(pid):
        return f'identity-{pid}' if pid == os.getpid() else None
      cases = {
        'invalid': '{not json',
        'stale': json.dumps({
          'schema_version': 1,
          'artifact': 'frozen_generation_queue_acquisition_guard',
          'queue_id': 'dead-recoverer',
          'dataset_slug': 'wikitext',
          'pid': 999_999,
          'process_start_identity': 'dead-owner',
          'launch_plan_sha256': 'b' * 64,
          'created_utc': '2026-08-31T00:00:00+00:00',
          'owner_token': 'old-token',
        }),
      }
      for state, content in cases.items():
        with self.subTest(state=state):
          guard.write_text(content)
          with self.assertRaisesRegex(
              QueueLockError, f'acquisition guard is {state}.*preserved'):
            self._lock(path, recover=True, reader=reader).acquire()
          self.assertEqual(guard.read_text(), content)
          self.assertFalse(path.exists())
          guard.unlink()


if __name__ == '__main__':
  unittest.main()
