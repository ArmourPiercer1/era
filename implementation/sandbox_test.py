# Copyright 2026 Google LLC.
"""Tests for the BubblewrapSandbox."""

import os
import sys
import unittest
from unittest import mock

# Ensure the implementation directory is on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sandbox as sandbox_module
from sandbox import (
    BubblewrapSandbox, ExecSandbox, Sandbox, SandboxResult,
    median, best, trimmed_mean,
)


class BubblewrapSandboxTest(unittest.TestCase):
  """Integration tests for BubblewrapSandbox.

  These tests require ``bwrap`` to be installed and a working ``.venv`` with
  pandas / numpy / scikit-learn available.
  """

  @classmethod
  def setUpClass(cls):
    cls.sandbox = BubblewrapSandbox(timeout_seconds=10)

  # -------------------------------------------------------------------
  # Basic functionality
  # -------------------------------------------------------------------

  def test_simple_return_value(self):
    """A trivial function that returns a constant."""
    program = "def greet(name):\n    return f'Hello, {name}!'\n"
    result, ok = self.sandbox.run(program, 'greet', 'World')
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertEqual(result, 'Hello, World!')

  def test_return_none(self):
    """Function that returns None – matches playground_s3e1 usage."""
    program = "def wrapper(unused_arg):\n    return 42\n"
    result, ok = self.sandbox.run(program, 'wrapper', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertEqual(result, 42)

  def test_arithmetic(self):
    program = "def add(args):\n    return sum(args)\n"
    result, ok = self.sandbox.run(program, 'add', [1, 2, 3, 4])
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertEqual(result, 10)

  # -------------------------------------------------------------------
  # ML library access
  # -------------------------------------------------------------------

  def test_numpy_available(self):
    """The sandbox must be able to import numpy."""
    program = """
import numpy as np
def compute(_):
    return float(np.mean([1.0, 2.0, 3.0]))
"""
    result, ok = self.sandbox.run(program, 'compute', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertAlmostEqual(result, 2.0)

  def test_pandas_available(self):
    """The sandbox must be able to import pandas."""
    program = """
import pandas as pd
def make_df(_):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    return df['a'].tolist()
"""
    result, ok = self.sandbox.run(program, 'make_df', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertEqual(result, [1, 2])

  def test_sklearn_available(self):
    """The sandbox must be able to import scikit-learn."""
    program = """
from sklearn.linear_model import LinearRegression
import numpy as np
def fit_model(_):
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = LinearRegression()
    model.fit(X, y)
    return float(model.predict([[4]])[0])
"""
    result, ok = self.sandbox.run(program, 'fit_model', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertAlmostEqual(result, 8.0, delta=0.1)

  # -------------------------------------------------------------------
  # File I/O through temp directory
  # -------------------------------------------------------------------

  def test_write_and_read_temp_file(self):
    """The sandboxed code can write to /tmp and read it back."""
    program = r"""
import os, tempfile
def write_and_read(_):
    p = os.path.join(tempfile.gettempdir(), 'test.txt')
    with open(p, 'w') as f:
        f.write('hello from sandbox')
    with open(p, 'r') as f:
        return f.read()
"""
    result, ok = self.sandbox.run(program, 'write_and_read', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertEqual(result, 'hello from sandbox')

  def test_csv_roundtrip(self):
    """Mimics the playground_s3e1 pattern of writing/reading CSVs."""
    program = r"""
import os, tempfile, csv, io
def csv_roundtrip(_):
    d = os.path.join(tempfile.gettempdir(), 'mydata')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'data.csv')
    with open(p, 'w') as f:
        f.write('a,b\n1,2\n3,4\n')
    with open(p, 'r') as f:
        return f.read()
"""
    result, ok = self.sandbox.run(program, 'csv_roundtrip', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertIn('a,b', result)
    self.assertIn('1,2', result)

  # -------------------------------------------------------------------
  # Error handling
  # -------------------------------------------------------------------

  def test_timeout(self):
    """Code that sleeps forever should be killed by timeout."""
    program = """
def spin(_):
    import time
    time.sleep(300)
    return 'never'
"""
    result, ok = self.sandbox.run(program, 'spin', None, timeout_seconds=2)
    self.assertFalse(ok)
    self.assertIn('timed out', str(result).lower())

  def test_function_not_defined(self):
    """Calling a function that does not exist should fail."""
    program = "def foo(_):\n    return 1\n"
    result, ok = self.sandbox.run(program, 'nonexistent', None)
    self.assertFalse(ok)
    self.assertIn('not defined', str(result))

  def test_syntax_error(self):
    """Malformed Python should produce a failure."""
    program = "this is not valid python!!!!"
    result, ok = self.sandbox.run(program, 'whatever', None)
    self.assertFalse(ok)

  def test_runtime_exception(self):
    """A runtime error inside the sandbox should be reported."""
    program = """
def explode(_):
    raise ValueError('deliberate explosion')
"""
    result, ok = self.sandbox.run(program, 'explode', None)
    self.assertFalse(ok)
    self.assertIn('ValueError', str(result))
    self.assertIn('deliberate explosion', str(result))

  def test_invalid_function_name_raises_value_error(self):
    """Only valid Python identifiers may be called inside the sandbox."""
    program = "def wrapper(_):\n    return 1\n"
    with self.assertRaises(ValueError):
      self.sandbox.run(program, '123bad', None)
    with self.assertRaises(ValueError):
      self.sandbox.run(program, 'has-dash', None)

  def test_memory_limit_enforced(self):
    """The sandboxed process should fail instead of exhausting host memory."""
    sandbox = BubblewrapSandbox(timeout_seconds=10, memory_limit_mb=256)
    program = """
def allocate(_):
    # Allocate far more than the per-test 256 MB limit.
    data = bytearray(512 * 1024 * 1024)
    return len(data)
"""
    result, ok = sandbox.run(program, 'allocate', None)
    self.assertFalse(ok)
    self.assertIn('MemoryError', str(result))

  def test_successful_run_logs_stderr(self):
    """stderr diagnostics should not be silently discarded."""
    program = """
def warn(_):
    import sys
    print('sandbox warning', file=sys.stderr)
    return 123
"""
    with self.assertLogs('sandbox', level='WARNING') as logs:
      result, ok = self.sandbox.run(program, 'warn', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertEqual(result, 123)
    self.assertIn('sandbox warning', '\n'.join(logs.output))

  # -------------------------------------------------------------------
  # run_detailed: timing
  # -------------------------------------------------------------------

  def test_run_detailed_returns_timings(self):
    """run_detailed records one timing per repeat, all positive."""
    program = "def f(_):\n    return 7\n"
    res = self.sandbox.run_detailed(program, 'f', None, repeats=3)
    self.assertTrue(res.success, msg=f'sandbox failed: {res.error}')
    self.assertEqual(res.result, 7)
    self.assertEqual(len(res.timings), 3)
    self.assertTrue(all(t >= 0 for t in res.timings))
    self.assertIsNotNone(res.median)
    self.assertIsNotNone(res.best)

  def test_run_detailed_warmup_not_counted(self):
    """warmup calls are not included in the measured timings."""
    program = "def f(_):\n    return 1\n"
    res = self.sandbox.run_detailed(program, 'f', None, warmup=2, repeats=3)
    self.assertTrue(res.success, msg=f'sandbox failed: {res.error}')
    self.assertEqual(len(res.timings), 3)

  def test_run_detailed_default_single_repeat(self):
    """Default repeats=1 yields exactly one timing (matches run())."""
    program = "def f(_):\n    return 1\n"
    res = self.sandbox.run_detailed(program, 'f', None)
    self.assertTrue(res.success, msg=f'sandbox failed: {res.error}')
    self.assertEqual(len(res.timings), 1)

  def test_run_detailed_timeout_has_no_timings(self):
    """A timeout returns success=False with empty timings."""
    program = """
def spin(_):
    import time
    time.sleep(300)
    return 'never'
"""
    res = self.sandbox.run_detailed(program, 'spin', None, timeout_seconds=2)
    self.assertFalse(res.success)
    self.assertEqual(res.timings, [])
    self.assertIn('timed out', str(res.error).lower())

  def test_run_detailed_exception_has_no_timings(self):
    """A runtime error returns success=False, traceback captured, no timings."""
    program = """
def explode(_):
    raise ValueError('boom')
"""
    res = self.sandbox.run_detailed(program, 'explode', None)
    self.assertFalse(res.success)
    self.assertEqual(res.timings, [])
    self.assertIn('ValueError', str(res.error))

  def test_run_detailed_invalid_function_name(self):
    res = "def f(_):\n    return 1\n"
    with self.assertRaises(ValueError):
      self.sandbox.run_detailed(res, '1bad', None)

  # -------------------------------------------------------------------
  # Isolation checks
  # -------------------------------------------------------------------

  def test_no_network_access(self):
    """The sandbox must not allow network access."""
    program = """
def try_network(_):
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(('8.8.8.8', 53))
        s.close()
        return 'connected!'
    except OSError as e:
        return f'blocked: {e}'
"""
    result, ok = self.sandbox.run(program, 'try_network', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    # The network should be blocked – either the connection fails or we
    # get a permission / network-unreachable error.
    self.assertIn('blocked', result.lower(),
                  msg=f'Expected network to be blocked, got: {result}')

  def test_no_home_access(self):
    """The sandbox must not expose real user files from the host home dir.

    Sensitive files like ``.bashrc``, ``.ssh``, or ``Documents`` must not
    be visible inside the sandbox.  Mount artefacts created by the venv
    bind-mount (e.g. an empty ``projects/`` parent directory) are harmless.
    """
    program = r"""
import os
def read_home(_):
    home = os.path.expanduser('~')
    try:
        files = os.listdir(home)
    except PermissionError:
        return 'permission denied'
    # Check for real user files that would indicate a leak.
    sensitive = {'.bashrc', '.profile', '.ssh', '.cache', '.config',
                 '.gitconfig', '.bash_history', 'Documents', 'Downloads'}
    leaked = [f for f in files if f in sensitive]
    if leaked:
        return f'LEAK: {leaked}'
    return f'ok, {len(files)} entries (no real user files)'
"""
    result, ok = self.sandbox.run(program, 'read_home', None)
    self.assertTrue(ok, msg=f'sandbox failed: {result}')
    self.assertIn('no real user files', result,
                  msg=f'Expected no real user files in home, got: {result}')

  # -------------------------------------------------------------------
  # Multi-run isolation (no cross-contamination)
  # -------------------------------------------------------------------

  def test_runs_are_isolated(self):
    """State from one run must not leak into the next."""
    program = r"""
import os, tempfile
def write_state(_):
    p = os.path.join(tempfile.gettempdir(), 'state.txt')
    if os.path.exists(p):
        with open(p) as f:
            return f'found: {f.read()}'
    with open(p, 'w') as f:
        f.write('hello')
    return 'wrote'
"""
    # First run writes 'hello'.
    r1, ok1 = self.sandbox.run(program, 'write_state', None)
    self.assertTrue(ok1, msg=f'first run failed: {r1}')
    self.assertEqual(r1, 'wrote')

    # Second run should see a fresh temp directory (no 'hello').
    r2, ok2 = self.sandbox.run(program, 'write_state', None)
    self.assertTrue(ok2, msg=f'second run failed: {r2}')
    self.assertEqual(r2, 'wrote',
                     msg=f'Expected fresh state, got: {r2}')


class ExecSandboxAliasTest(unittest.TestCase):
  """Verify that ExecSandbox is the same class as BubblewrapSandbox."""

  def test_alias_is_same_class(self):
    self.assertIs(ExecSandbox, BubblewrapSandbox)

  def test_alias_instance(self):
    sb = ExecSandbox(timeout_seconds=5)
    self.assertIsInstance(sb, BubblewrapSandbox)


class TimingAggregationTest(unittest.TestCase):
  """Unit tests (Tier A) for the pure timing-aggregation helpers."""

  def test_median_odd(self):
    self.assertEqual(median([3.0, 1.0, 2.0]), 2.0)

  def test_median_even(self):
    self.assertEqual(median([1.0, 2.0, 3.0, 4.0]), 2.5)

  def test_median_empty(self):
    self.assertIsNone(median([]))

  def test_best(self):
    self.assertEqual(best([3.0, 1.0, 2.0]), 1.0)

  def test_best_empty(self):
    self.assertIsNone(best([]))

  def test_trimmed_mean_drops_outliers(self):
    # 0.2 trim of 10 values drops 2 from each end -> mean of middle 6.
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
    self.assertAlmostEqual(trimmed_mean(values, trim=0.2), 4.5)

  def test_trimmed_mean_small_list_falls_back(self):
    self.assertAlmostEqual(trimmed_mean([2.0, 4.0], trim=0.4), 3.0)

  def test_trimmed_mean_empty(self):
    self.assertIsNone(trimmed_mean([]))

  def test_sandbox_result_properties(self):
    res = SandboxResult(success=True, result=1, timings=[3.0, 1.0, 2.0])
    self.assertEqual(res.best, 1.0)
    self.assertEqual(res.median, 2.0)
    self.assertEqual(res.elapsed, 2.0)

  def test_sandbox_result_empty_timings(self):
    res = SandboxResult(success=False, error='x')
    self.assertIsNone(res.best)
    self.assertIsNone(res.median)


class SandboxInterfaceTest(unittest.TestCase):
  """Verify the abstract base class contract."""

  def test_base_class_accepts_timeout(self):
    sb = Sandbox(timeout_seconds=30)
    self.assertEqual(sb._default_timeout, 30)

  def test_base_class_stores_default_timeout(self):
    sb = Sandbox()
    self.assertEqual(sb._default_timeout, 60)

  def test_base_class_cannot_run(self):
    sb = Sandbox()
    with self.assertRaises(NotImplementedError):
      sb.run('x = 1', 'x', None)


class VenvLookupTest(unittest.TestCase):
  """Verify .venv lookup stays strict instead of falling back globally."""

  def test_missing_venv_raises_clear_error(self):
    with mock.patch.object(sandbox_module.sys, 'prefix', '/usr'):
      with mock.patch.object(sandbox_module.sys, 'base_prefix', '/usr'):
        with mock.patch.object(sandbox_module.os.path, 'isfile', return_value=False):
          with self.assertRaises(RuntimeError) as ctx:
            BubblewrapSandbox._find_venv_python()
    message = str(ctx.exception)
    self.assertIn('.venv', message)
    self.assertIn('Refusing to fall back', message)


class PlaygroundIntegrationTest(unittest.TestCase):
  """Verify playground_s3e1 uses the concrete sandbox implementation."""

  def test_playground_imports_exec_sandbox(self):
    import playground_s3e1
    self.assertTrue(hasattr(playground_s3e1, 'ExecSandbox'))

  def test_run_experiment_instantiates_exec_sandbox(self):
    import playground_s3e1

    fake_llm = mock.Mock()
    fake_search = mock.Mock(return_value=(mock.Mock(program='best'), 1.0))

    with mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
      with mock.patch.object(playground_s3e1, 'prepare_data', return_value=[]):
        with mock.patch.object(playground_s3e1, 'GeminiLLM', return_value=fake_llm):
          with mock.patch.object(playground_s3e1, 'ExecSandbox') as exec_cls:
            exec_cls.return_value.run.return_value = (None, False)
            with mock.patch.object(playground_s3e1.futs, 'search', fake_search):
              with mock.patch.object(playground_s3e1.os, 'makedirs'):
                with mock.patch('builtins.open', mock.mock_open(read_data='a,b\n1,2\n')):
                  playground_s3e1.run_experiment(iterations=0)

    exec_cls.assert_called_once_with(timeout_seconds=60)


if __name__ == '__main__':
  unittest.main()
