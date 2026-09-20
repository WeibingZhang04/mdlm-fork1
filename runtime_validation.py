"""Runtime switch for expensive diagnostic tensor scans."""

import os


DEBUG_VALIDATION = os.environ.get(
  'MDLM_DEBUG_VALIDATION', '').lower() in {'1', 'true', 'yes', 'on'}


def enabled() -> bool:
  return DEBUG_VALIDATION
