"""Fail-closed operational artifacts for frozen generation queues.

This module deliberately contains no experiment-specific scientific logic.  It
provides the shared, exclusive queue lock and small cryptographic helpers used
by the WikiText and cross-domain controllers and by post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping
import uuid


LOCK_SCHEMA_VERSION = 1
LOCK_ARTIFACT = 'frozen_generation_queue_lock'
GUARD_ARTIFACT = 'frozen_generation_queue_acquisition_guard'


class QueueLockError(RuntimeError):
  """The shared generation queue lock could not be acquired or released."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b''):
      digest.update(chunk)
  return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
  return json.dumps(
    payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
  ).encode('utf-8')


def canonical_sha256(payload: Any) -> str:
  return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_strict_json(path: Path) -> Any:
  def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f'duplicate JSON key {key!r} in {path}')
      result[key] = value
    return result

  def reject_nonfinite(value):
    raise ValueError(f'non-finite JSON number {value!r} in {path}')

  try:
    return json.loads(
      Path(path).read_text(), object_pairs_hook=reject_duplicates,
      parse_constant=reject_nonfinite)
  except json.JSONDecodeError as error:
    raise ValueError(f'invalid JSON in {path}: {error}') from error


def atomic_write_new(path: Path, content: str) -> None:
  """Atomically create ``path`` without ever replacing an existing inode."""
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}')
  try:
    descriptor = os.open(
      temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
      data = content.encode('utf-8')
      offset = 0
      while offset < len(data):
        offset += os.write(descriptor, data[offset:])
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
    try:
      os.link(temporary, path)
    except FileExistsError as error:
      raise FileExistsError(f'refusing to overwrite {path}') from error
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  finally:
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


def atomic_rename_directory_new(source: Path, destination: Path) -> None:
  """Atomically publish a directory while refusing an existing destination."""
  source = Path(source).expanduser().resolve()
  destination = Path(destination).expanduser().resolve()
  if source.parent != destination.parent:
    raise ValueError('atomic directory publication requires one parent directory')
  libc = ctypes.CDLL(None, use_errno=True)
  source_bytes = os.fsencode(source)
  destination_bytes = os.fsencode(destination)
  if hasattr(libc, 'renameat2'):
    rename = libc.renameat2
    rename.argtypes = (
      ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
      ctypes.c_uint)
    rename.restype = ctypes.c_int
    result = rename(
      -100, source_bytes, -100, destination_bytes, 1)  # RENAME_NOREPLACE
  elif hasattr(libc, 'renamex_np'):
    rename = libc.renamex_np
    rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    result = rename(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
  else:
    raise OSError(
      errno.ENOTSUP,
      'platform lacks an atomic no-replace directory rename primitive')
  if result:
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
      raise FileExistsError(
        error_number, f'refusing to overwrite {destination}', destination)
    raise OSError(error_number, os.strerror(error_number), destination)
  directory_fd = os.open(destination.parent, os.O_RDONLY)
  try:
    os.fsync(directory_fd)
  finally:
    os.close(directory_fd)


def _linux_start_identity(pid: int) -> str | None:
  try:
    raw = (Path('/proc') / str(pid) / 'stat').read_text()
  except FileNotFoundError:
    return None
  except OSError as error:
    raise QueueLockError(f'cannot inspect /proc/{pid}/stat: {error}') from error
  _, separator, tail = raw.rpartition(') ')
  fields = tail.split()
  if not separator or len(fields) <= 19 or not fields[19].isdigit():
    raise QueueLockError(f'/proc/{pid}/stat has no valid process start time')
  return f'linux-start-ticks:{fields[19]}'


def read_process_start_identity(pid: int) -> str | None:
  """Return a PID-reuse-safe identity on Linux and a best effort elsewhere."""
  if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
    return None
  if Path('/proc').is_dir():
    return _linux_start_identity(pid)
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return None
  except PermissionError:
    pass
  try:
    started = subprocess.check_output(
      ['ps', '-o', 'lstart=', '-p', str(pid)], text=True,
      stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError):
    # The production queue is Linux and always takes the /proc branch.  Some
    # sandboxed non-Linux test hosts deny ps(1); retain active-owner semantics
    # there without claiming PID-reuse protection.
    return f'portable-live-pid:{pid}'
  return f'ps-lstart:{started}' if started else None


ProcessIdentityReader = Callable[[int], str | None]


def _strict_mapping(
    payload: object, expected_fields: set[str], *, context: str,
) -> Mapping[str, Any]:
  if not isinstance(payload, Mapping):
    raise QueueLockError(f'{context} must be a JSON object')
  if set(payload) != expected_fields:
    raise QueueLockError(
      f'{context} schema mismatch: '
      f'missing={sorted(expected_fields - set(payload))}, '
      f'unknown={sorted(set(payload) - expected_fields)}')
  return payload


@dataclass
class SharedQueueLock:
  """One experiment-wide O_EXCL lock with explicit active/stale handling."""

  path: Path
  queue_id: str
  dataset_slug: str
  launch_plan_sha256: str
  recover_stale: bool = False
  process_identity_reader: ProcessIdentityReader = read_process_start_identity

  def __post_init__(self) -> None:
    self.path = Path(self.path).expanduser().resolve()
    self.guard_path = self.path.with_name(f'{self.path.name}.acquire-guard')
    self._descriptor: int | None = None
    self._inode: int | None = None
    self._content: bytes | None = None

  def _owner_payload(self, *, artifact: str) -> tuple[dict[str, Any], bytes]:
    owner_identity = self.process_identity_reader(os.getpid())
    if owner_identity is None:
      raise QueueLockError('cannot determine the queue-controller process identity')
    payload = {
      'schema_version': LOCK_SCHEMA_VERSION,
      'artifact': artifact,
      'queue_id': self.queue_id,
      'dataset_slug': self.dataset_slug,
      'pid': os.getpid(),
      'process_start_identity': owner_identity,
      'launch_plan_sha256': self.launch_plan_sha256,
      'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
      'owner_token': uuid.uuid4().hex,
    }
    return payload, json.dumps(
      payload, indent=2, sort_keys=True).encode('utf-8') + b'\n'

  def _existing_state(
      self, path: Path, *, expected_artifact: str,
  ) -> tuple[str, Mapping[str, Any] | None]:
    try:
      payload = load_strict_json(path)
      payload = _strict_mapping(
        payload,
        {
          'schema_version', 'artifact', 'queue_id', 'dataset_slug', 'pid',
          'process_start_identity', 'launch_plan_sha256', 'created_utc',
          'owner_token',
        },
        context='generation queue lock')
      if (payload['schema_version'] != LOCK_SCHEMA_VERSION
          or payload['artifact'] != expected_artifact):
        raise QueueLockError('generation queue owner file has an unsupported schema')
      pid = payload['pid']
      start_identity = payload['process_start_identity']
      if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
          or not isinstance(start_identity, str) or not start_identity):
        raise QueueLockError('generation queue lock has an invalid owner')
      observed = self.process_identity_reader(pid)
      return (
        'active' if observed == start_identity else 'stale', payload)
    except (OSError, TypeError, ValueError, QueueLockError):
      return 'invalid', None

  def _acquire_guard(self) -> tuple[int, int, bytes]:
    """Serialize every inspection/recovery/create transaction.

    The guard itself is never automatically recovered.  A crashed or
    malformed guard therefore fails closed and must be inspected and preserved
    out of band; recursively recovering it would recreate the same TOCTOU the
    guard exists to eliminate.
    """
    _, content = self._owner_payload(artifact=GUARD_ARTIFACT)
    try:
      descriptor = os.open(
        self.guard_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
      state, existing = self._existing_state(
        self.guard_path, expected_artifact=GUARD_ARTIFACT)
      owner = (
        f' queue_id={existing.get("queue_id")!r} '
        f'pid={existing.get("pid")}' if existing is not None else '')
      raise QueueLockError(
        f'generation queue acquisition guard is {state} and was preserved;'
        f'{owner} manual review and out-of-band preservation are required') \
        from error
    try:
      offset = 0
      while offset < len(content):
        offset += os.write(descriptor, content[offset:])
      os.fsync(descriptor)
      return descriptor, os.fstat(descriptor).st_ino, content
    except Exception:
      os.close(descriptor)
      try:
        self.guard_path.unlink()
      except FileNotFoundError:
        pass
      raise

  def _release_guard(
      self, descriptor: int, inode: int, content: bytes,
  ) -> None:
    try:
      try:
        current = self.guard_path.read_bytes()
        stat = self.guard_path.stat()
      except FileNotFoundError as error:
        raise QueueLockError(
          'generation queue acquisition guard disappeared while held') \
          from error
      if stat.st_ino != inode or current != content:
        raise QueueLockError(
          'generation queue acquisition guard changed while held; preserving it')
      self.guard_path.unlink()
      directory_fd = os.open(self.guard_path.parent, os.O_RDONLY)
      try:
        os.fsync(directory_fd)
      finally:
        os.close(directory_fd)
    finally:
      os.close(descriptor)

  def _recover_existing(self, state: str) -> None:
    preserved = self.path.with_name(
      f'{self.path.name}.{state}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}'
      f'-{uuid.uuid4().hex}.preserved')
    if preserved.exists():
      raise QueueLockError(f'stale-lock preservation target exists: {preserved}')
    os.rename(self.path, preserved)

  def acquire(self) -> 'SharedQueueLock':
    if self._descriptor is not None:
      raise QueueLockError('generation queue lock is already held')
    self.path.parent.mkdir(parents=True, exist_ok=True)
    guard = self._acquire_guard()
    acquired = False
    try:
      _, content = self._owner_payload(artifact=LOCK_ARTIFACT)
      for attempt in range(2):
        try:
          descriptor = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
          state, existing = self._existing_state(
            self.path, expected_artifact=LOCK_ARTIFACT)
          owner = (
            f' queue_id={existing.get("queue_id")!r} '
            f'pid={existing.get("pid")}' if existing is not None else '')
          if state == 'active':
            raise QueueLockError(
              f'generation queue lock is active:{owner}') from error
          if not self.recover_stale:
            raise QueueLockError(
              f'generation queue lock is {state} and was preserved; '
              'review it and pass the explicit stale-lock recovery option') \
              from error
          if attempt:
            raise QueueLockError('queue lock reappeared during stale recovery') \
              from error
          # Every conforming acquirer holds guard_path across this inspection,
          # preservation rename, and subsequent O_EXCL create.  No second
          # recoverer can substitute an active inode between those operations.
          self._recover_existing(state)
          continue
        try:
          offset = 0
          while offset < len(content):
            offset += os.write(descriptor, content[offset:])
          os.fsync(descriptor)
          self._descriptor = descriptor
          self._inode = os.fstat(descriptor).st_ino
          self._content = content
          acquired = True
          break
        except Exception:
          os.close(descriptor)
          try:
            self.path.unlink()
          except FileNotFoundError:
            pass
          raise
      if not acquired:
        raise AssertionError('unreachable queue-lock acquisition state')
    finally:
      try:
        self._release_guard(*guard)
      except Exception:
        # Do not unlink a newly created main lock after guard integrity has
        # been lost.  Preserve both artifacts and abandon only our descriptor.
        if self._descriptor is not None:
          os.close(self._descriptor)
          self._descriptor = None
          self._inode = None
          self._content = None
        raise
    return self

  def release(self) -> None:
    if self._descriptor is None or self._content is None or self._inode is None:
      raise QueueLockError('generation queue lock is not held')
    descriptor = self._descriptor
    try:
      try:
        current = self.path.read_bytes()
        stat = self.path.stat()
      except FileNotFoundError as error:
        raise QueueLockError(
          'generation queue lock disappeared while held') from error
      if stat.st_ino != self._inode or current != self._content:
        raise QueueLockError(
          'generation queue lock changed while held; preserving it')
      self.path.unlink()
      directory_fd = os.open(self.path.parent, os.O_RDONLY)
      try:
        os.fsync(directory_fd)
      finally:
        os.close(directory_fd)
    finally:
      os.close(descriptor)
      self._descriptor = None
      self._inode = None
      self._content = None

  def __enter__(self) -> 'SharedQueueLock':
    return self.acquire()

  def __exit__(self, exc_type, exc_value, traceback) -> bool:
    self.release()
    return False
