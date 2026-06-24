# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sandbox for safely executing untrusted Python code.

Uses bubblewrap (bwrap) to create an isolated Linux namespace with:
  - No network access
  - Read-only access to Python environment and system libraries
  - Writable temp directory for scratch data
  - Resource limits (CPU time, memory)
  - Timeout enforcement
"""

import dataclasses
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Any, List, Optional, Tuple


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured execution result + timing aggregation helpers
# ---------------------------------------------------------------------------

def median(values: List[float]) -> Optional[float]:
  """Return the median of *values*, or ``None`` if empty."""
  if not values:
    return None
  ordered = sorted(values)
  n = len(ordered)
  mid = n // 2
  if n % 2 == 1:
    return ordered[mid]
  return (ordered[mid - 1] + ordered[mid]) / 2.0


def best(values: List[float]) -> Optional[float]:
  """Return the minimum of *values* (best-case timing), or ``None`` if empty."""
  return min(values) if values else None


def trimmed_mean(values: List[float], trim: float = 0.2) -> Optional[float]:
  """Return the mean after dropping a *trim* fraction from each end.

  Falls back to the plain mean when trimming would remove everything.
  """
  if not values:
    return None
  ordered = sorted(values)
  k = int(len(ordered) * trim)
  core = ordered[k:len(ordered) - k] or ordered
  return sum(core) / len(core)


@dataclasses.dataclass
class SandboxResult:
  """Structured result of a sandboxed execution.

  ``timings`` holds the per-repeat wall-clock seconds of the measured calls
  (warmup calls excluded).  Aggregations are exposed as properties so a
  task scoring function can pick median/best as appropriate.
  """

  success: bool
  result: Any = None
  error: Optional[str] = None
  timings: List[float] = dataclasses.field(default_factory=list)
  stderr: str = ''

  @property
  def best(self) -> Optional[float]:
    """Best-case (minimum) measured runtime in seconds."""
    return best(self.timings)

  @property
  def median(self) -> Optional[float]:
    """Median measured runtime in seconds."""
    return median(self.timings)

  @property
  def elapsed(self) -> Optional[float]:
    """Representative runtime (median) in seconds."""
    return median(self.timings)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Sandbox:
  """Abstract sandbox for executing generated code."""

  def __init__(self, timeout_seconds: int = 60):
    """Initialise the sandbox.

    Args:
        timeout_seconds: Default timeout for each ``run()`` call.
    """
    self._default_timeout = timeout_seconds

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: Any,
      timeout_seconds: int = 60,
  ) -> Tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    raise NotImplementedError(
        'Must provide a sandbox for executing untrusted code.')


# ---------------------------------------------------------------------------
# Bubblewrap sandbox implementation
# ---------------------------------------------------------------------------

class BubblewrapSandbox(Sandbox):
  """Sandbox that executes code inside a bubblewrap (bwrap) container.

  Uses Linux user namespaces via bwrap to isolate the execution:
  - All Linux namespaces are unshared (mount, pid, net, ipc, uts, user, cgroup)
  - No network access
  - Filesystem is read-only except for a dedicated temp directory
  - Process tree dies with the parent process
  - Results are communicated back via pickled stdout

  Requires bubblewrap to be installed (``apt install bubblewrap``).
  """

  # Paths on the host system that must be bind-mounted into the sandbox so
  # that the Python interpreter and its dynamically-linked libraries work.
  _READONLY_MOUNTS = [
      '/usr',
      '/lib',
      '/lib64',
      '/bin',
      '/sbin',
      '/etc',
  ]

  def __init__(
      self,
      timeout_seconds: int = 60,
      memory_limit_mb: int = 102400,
  ):
    """Initialise the sandbox.

    Args:
        timeout_seconds: Default timeout applied to each ``run()`` call when no
            explicit timeout is supplied.
        memory_limit_mb: Memory limit for the sandboxed Python process.
    """
    super().__init__(timeout_seconds=timeout_seconds)
    self._memory_limit_bytes = memory_limit_mb * 1024 * 1024
    self._bwrap_path = self._find_bwrap()
    self._python_path = self._find_venv_python()
    self._venv_root = os.path.dirname(os.path.dirname(self._python_path))

  # -----------------------------------------------------------------------
  # Public API
  # -----------------------------------------------------------------------

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: Any = None,
      timeout_seconds: Optional[int] = None,
  ) -> Tuple[Any, bool]:
    """Execute *program* and call *function_to_run* inside the sandbox.

    Thin backward-compatible wrapper over :meth:`run_detailed` that returns
    only ``(result, success)``.

    Returns:
        ``(result, True)`` when the function ran successfully, or
        ``(error_message, False)`` when something went wrong.
    """
    res = self.run_detailed(program, function_to_run, test_input,
                            timeout_seconds)
    if res.success:
      return (res.result, True)
    return (res.error, False)

  def run_detailed(
      self,
      program: str,
      function_to_run: str,
      test_input: Any = None,
      timeout_seconds: Optional[int] = None,
      warmup: int = 0,
      repeats: int = 1,
  ) -> SandboxResult:
    """Execute *program*, call *function_to_run*, and measure its runtime.

    Args:
        program: Full Python source code to execute.  Must define the
            function named by *function_to_run*.
        function_to_run: Name of the function to call after executing
            *program*.
        test_input: Argument passed to ``function_to_run`` (may be ``None``).
        timeout_seconds: Wall-clock timeout.  Falls back to the instance
            default set in ``__init__``.
        warmup: Number of unmeasured warmup calls before timing.
        repeats: Number of measured calls (each timed with ``perf_counter``).

    Returns:
        A :class:`SandboxResult` with ``result``/``timings`` on success, or
        ``success=False`` and ``error`` set on failure.
    """
    if not re.match(r'^[a-zA-Z_]\w*$', function_to_run):
      raise ValueError(
          'function_to_run must be a valid Python identifier, '
          f'got {function_to_run!r}')

    timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
    run_id = uuid.uuid4().hex[:12]
    temp_dir = tempfile.mkdtemp(prefix=f'era_sandbox_{run_id}_')

    try:
      runner_path = self._write_runner(program, function_to_run, test_input,
                                       temp_dir, warmup, repeats)

      cmd = self._build_bwrap_command(runner_path, temp_dir)

      proc = subprocess.run(
          cmd,
          capture_output=True,
          timeout=timeout,
      )

      return self._parse_result(proc.stdout, proc.stderr)

    except subprocess.TimeoutExpired:
      return SandboxResult(
          success=False,
          error=f'Sandbox timed out after {timeout} seconds.')
    except Exception:  # pylint: disable=broad-except
      import traceback
      return SandboxResult(
          success=False,
          error=f'Sandbox internal error:\n{traceback.format_exc()}')
    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)

  # -----------------------------------------------------------------------
  # Internal helpers
  # -----------------------------------------------------------------------

  @staticmethod
  def _find_bwrap() -> str:
    """Return the path to the bwrap binary, or raise if not found."""
    bwrap = shutil.which('bwrap')
    if bwrap is None:
      raise RuntimeError(
          'bwrap (bubblewrap) is required by BubblewrapSandbox. '
          'Install it with:  sudo apt install bubblewrap')
    return bwrap

  @staticmethod
  def _find_venv_python() -> str:
    """Locate the .venv Python interpreter.

    We walk up from this file to discover the project root, then look for
    ``.venv/bin/python``.
    """
    # Prefer sys.executable when it already lives inside a venv.
    if sys.prefix != sys.base_prefix:
      return sys.executable

    # Fallback: navigate relative to this source file.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    venv_python = os.path.join(project_root, '.venv', 'bin', 'python')
    if os.path.isfile(venv_python):
      return venv_python

    raise RuntimeError(
        'BubblewrapSandbox requires the project virtual environment at '
        f'{venv_python}. Create it with: python3 -m venv .venv, then '
        'install dependencies into that environment. Refusing to fall back '
        'to a global Python interpreter because generated code must run in '
        'the controlled project environment.')

  def _write_runner(
      self,
      program: str,
      function_to_run: str,
      test_input: Any,
      temp_dir: str,
      warmup: int = 0,
      repeats: int = 1,
  ) -> str:
    """Write a small runner script into *temp_dir* and return its path.

    The runner:
    1. Prevents bytecode writes (venv is mounted read-only).
    2. Applies the memory limit.
    3. Executes *program* so that its definitions become available.
    4. Runs ``warmup`` unmeasured calls, then ``repeats`` timed calls of
       ``function_to_run(test_input)`` (each wrapped in ``perf_counter``).
    5. Pickles ``{success, result, timings}`` (or ``{success, error}``) to
       stdout.
    """
    runner_code = f'''import os, pickle, resource, sys, time, traceback

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_memory_limit = {self._memory_limit_bytes}
resource.setrlimit(resource.RLIMIT_AS, (_memory_limit, _memory_limit))

_program = {program!r}
_func_name = {function_to_run!r}
_test_input = {test_input!r}
_warmup = {int(warmup)}
_repeats = {int(repeats)}

try:
    _ns = {{"__builtins__": __builtins__}}
    exec(_program, _ns)
    _func = _ns.get(_func_name)
    if _func is None:
        raise NameError(f"Function {{_func_name!r}} not defined by program")
    for _ in range(max(0, _warmup)):
        _func(_test_input)
    _timings = []
    _result = None
    for _ in range(max(1, _repeats)):
        _t0 = time.perf_counter()
        _result = _func(_test_input)
        _t1 = time.perf_counter()
        _timings.append(_t1 - _t0)
    sys.stdout.buffer.write(pickle.dumps(
        {{"success": True, "result": _result, "timings": _timings}}))
except Exception:
    sys.stdout.buffer.write(pickle.dumps({{
        "success": False,
        "error": traceback.format_exc(),
    }}))
'''
    runner_path = os.path.join(temp_dir, 'runner.py')
    with open(runner_path, 'w', encoding='utf-8') as fh:
      fh.write(runner_code)
    return runner_path

  def _build_bwrap_command(
      self,
      runner_path: str,
      temp_dir: str,
  ) -> list:
    """Assemble the bwrap command-line.

    The runner script lives in *temp_dir*, which is bind-mounted as ``/tmp``
    inside the sandbox – so the sandbox sees it at ``/tmp/runner.py``.
    """
    # The runner inside the sandbox is at /tmp/runner.py because we bind
    # temp_dir to /tmp.
    sandbox_runner = '/tmp/' + os.path.basename(runner_path)

    cmd = [self._bwrap_path]

    # System directories (read-only).
    for mount in self._READONLY_MOUNTS:
      if os.path.isdir(mount):
        cmd += ['--ro-bind', mount, mount]

    # Virtual filesystems.
    cmd += ['--proc', '/proc']
    cmd += ['--dev', '/dev']

    # Writable scratch space at /tmp.
    cmd += ['--bind', temp_dir, '/tmp']

    # Venv: mount read-only so the sandbox can import installed packages.
    # We mount a tmpfs over /home first so that the venv is the *only*
    # content visible under /home (no real user data leaks).
    cmd += ['--tmpfs', '/home']
    if os.path.isdir(self._venv_root):
      cmd += ['--ro-bind', self._venv_root, self._venv_root]

    # Isolate every namespace.
    cmd += ['--unshare-all']

    # Kill the sandbox if the parent (bwrap itself) dies.
    cmd += ['--die-with-parent']

    # The Python interpreter and the runner.
    cmd += [self._python_path, sandbox_runner]

    return cmd

  @staticmethod
  def _parse_result(stdout: bytes, stderr: bytes) -> SandboxResult:
    """Deserialise the result dictionary from the sandbox's stdout."""
    stderr_text = stderr.decode('utf-8', errors='replace').strip()

    if not stdout:
      if stderr_text:
        return SandboxResult(
            success=False,
            error=f'Sandbox produced no output on stdout.\nstderr:\n{stderr_text}',
            stderr=stderr_text)
      return SandboxResult(
          success=False,
          error='Sandbox produced no output on stdout and no stderr.',
          stderr=stderr_text)

    try:
      data = pickle.loads(stdout)
    except Exception:  # pylint: disable=broad-except
      raw = stdout.decode('utf-8', errors='replace')[:2000]
      return SandboxResult(
          success=False,
          error=f'Failed to unpickle sandbox result. Raw stdout:\n{raw}',
          stderr=stderr_text)

    success = data.get('success', False)
    if success:
      if stderr_text:
        _LOGGER.warning('Sandbox stderr (run succeeded):\n%s', stderr_text)
      return SandboxResult(
          success=True,
          result=data.get('result'),
          timings=list(data.get('timings') or []),
          stderr=stderr_text)
    return SandboxResult(
        success=False,
        error=data.get('error', 'Unknown error inside sandbox'),
        stderr=stderr_text)


# ---------------------------------------------------------------------------
# Aliases for backward compatibility with notebooks
# ---------------------------------------------------------------------------

# ExecSandbox is the name used in experiment_pipeline.ipynb.
ExecSandbox = BubblewrapSandbox
