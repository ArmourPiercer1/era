# Copyright 2026 Google LLC.
"""Load environment variables from a ``.env`` file into ``os.environ``.

This module is imported by other ``implementation/`` modules.  The import
side-effect searches for a ``.env`` file in these locations (in order):

1. The project root (parent of this ``implementation/`` directory).
2. The current working directory.

Variables already present in the environment are never overwritten,
so you can always set them explicitly before importing this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
  """Find and load the first available ``.env`` file."""
  candidates = [
      Path(__file__).resolve().parent.parent / '.env',  # project root
      Path.cwd() / '.env',                               # cwd
  ]

  for path in candidates:
    if not path.is_file():
      continue

    try:
      _read_and_set(path)
      _logger.debug('Loaded environment from %s', path)
      return
    except OSError as exc:
      _logger.warning('Could not read %s: %s', path, exc)

  _logger.debug('No .env file found (looked in %s)', candidates)


def _read_and_set(path: Path) -> None:
  """Parse *path* line by line and set ``os.environ``.

  This is a minimal, zero-dependency ``.env`` parser.  It handles:

  * ``KEY=value`` assignments
  * ``export KEY=value``
  * blank lines and ``#`` comments
  * single- and double-quoted values

  Existing environment variables are never overwritten.
  """
  with open(path, encoding='utf-8') as fh:
    for raw_line in fh:
      line = raw_line.strip()

      # Skip blanks and comments.
      if not line or line.startswith('#'):
        continue

      # Strip optional 'export ' prefix.
      if line.startswith('export '):
        line = line[len('export '):]

      # Split on first '='.
      if '=' not in line:
        _logger.warning(
            'Skipping malformed line in %s (no KEY=VALUE pattern): %r',
            path,
            raw_line.strip(),
        )
        continue
      key, _, value = line.partition('=')

      key = key.strip()
      value = value.strip()

      # Strip surrounding quotes (single or double).
      if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]

      # Never override an already-set variable.
      if key and key not in os.environ:
        os.environ[key] = value


# Load on import.
_load_dotenv()
