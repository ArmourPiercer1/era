# Copyright 2026 Google LLC.
"""Tests for the env module (.env file loading)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env


class DotenvParsingTest(unittest.TestCase):
  """Unit tests for the _read_and_set parser."""

  def setUp(self):
    # Save a copy of the environment so we can restore it.
    self._saved = os.environ.copy()

  def tearDown(self):
    os.environ.clear()
    os.environ.update(self._saved)

  def _write_and_load(self, content: str) -> None:
    """Write *content* to a temp .env file and parse it."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.env', delete=False) as fh:
      fh.write(content)
      tmp = fh.name
    try:
      env._read_and_set(Path(tmp))
    finally:
      os.unlink(tmp)

  # -------------------------------------------------------------------
  # Basic parsing
  # -------------------------------------------------------------------

  def test_simple_key_value(self):
    self._write_and_load('FOO=bar')
    self.assertEqual(os.environ['FOO'], 'bar')

  def test_export_syntax(self):
    self._write_and_load('export FOO=bar')
    self.assertEqual(os.environ['FOO'], 'bar')

  def test_double_quoted_value(self):
    self._write_and_load('FOO="hello world"')
    self.assertEqual(os.environ['FOO'], 'hello world')

  def test_single_quoted_value(self):
    self._write_and_load("FOO='hello world'")
    self.assertEqual(os.environ['FOO'], 'hello world')

  def test_value_with_equals(self):
    self._write_and_load('FOO=a=b=c')
    self.assertEqual(os.environ['FOO'], 'a=b=c')

  def test_multiple_keys(self):
    self._write_and_load('A=1\nB=2\nC=3')
    self.assertEqual(os.environ['A'], '1')
    self.assertEqual(os.environ['B'], '2')
    self.assertEqual(os.environ['C'], '3')

  # -------------------------------------------------------------------
  # Comments and blank lines
  # -------------------------------------------------------------------

  def test_comment_line_is_ignored(self):
    self._write_and_load('# this is a comment\nFOO=bar')
    self.assertEqual(os.environ['FOO'], 'bar')
    self.assertNotIn('# this is a comment', os.environ)

  def test_blank_lines_are_ignored(self):
    self._write_and_load('\n\nFOO=bar\n\n')
    self.assertEqual(os.environ['FOO'], 'bar')

  def test_inline_comment_not_supported(self):
    """The parser does NOT strip inline # comments (by design – keys may
    contain #)."""
    self._write_and_load('FOO=bar # not a comment')
    self.assertEqual(os.environ['FOO'], 'bar # not a comment')

  def test_empty_value(self):
    self._write_and_load('FOO=')
    self.assertEqual(os.environ['FOO'], '')

  # -------------------------------------------------------------------
  # Never overwrite
  # -------------------------------------------------------------------

  def test_existing_var_not_overwritten(self):
    os.environ['FOO'] = 'original'
    self._write_and_load('FOO=from_file')
    self.assertEqual(os.environ['FOO'], 'original')

  def test_malformed_line_warns(self):
    with self.assertLogs('env', level='WARNING') as logs:
      self._write_and_load('not-a-key-value-line')
    self.assertIn('Skipping malformed line', '\n'.join(logs.output))
    self.assertIn('not-a-key-value-line', '\n'.join(logs.output))

  def test_valid_lines_still_work_after_malformed_line(self):
    with self.assertLogs('env', level='WARNING'):
      self._write_and_load('not-a-key-value-line\nFOO=bar')
    self.assertEqual(os.environ['FOO'], 'bar')


class EnvLoadingIntegrationTest(unittest.TestCase):
  """Verify that importing env.py loads a real .env file."""

  def setUp(self):
    self._saved = os.environ.copy()

  def tearDown(self):
    os.environ.clear()
    os.environ.update(self._saved)

  def test_load_from_project_root(self):
    """Write a .env to the project root and verify env.py picks it up."""
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / '.env'
    already_exists = dotenv_path.exists()
    backup = None

    try:
      if already_exists:
        backup = dotenv_path.read_text()
      dotenv_path.write_text('ERA_TEST_VAR_FROM_DOTENV=hello_from_file\n')
      # Force a re-load.
      os.environ.pop('ERA_TEST_VAR_FROM_DOTENV', None)
      env._load_dotenv()
      self.assertEqual(os.environ.get('ERA_TEST_VAR_FROM_DOTENV'),
                       'hello_from_file')
    finally:
      if backup is not None:
        dotenv_path.write_text(backup)
      elif not already_exists:
        dotenv_path.unlink(missing_ok=True)


class LLMEnvIntegrationTest(unittest.TestCase):
  """Verify that LLM backends read keys from the environment set by .env."""

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_openai_reads_key_from_env(self):
    os.environ['OPENAI_API_KEY'] = 'sk-from-env'
    from llm import OpenAILLM
    with mock.patch('openai.OpenAI') as mock_cls:
      OpenAILLM()
      mock_cls.assert_called_once_with(api_key='sk-from-env', timeout=120.0)

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_anthropic_reads_key_from_env(self):
    os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-from-env'
    from llm import AnthropicLLM
    with mock.patch('anthropic.Anthropic') as mock_cls:
      AnthropicLLM()
      mock_cls.assert_called_once_with(api_key='sk-ant-from-env', timeout=120.0)

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_explicit_key_wins_over_env(self):
    os.environ['OPENAI_API_KEY'] = 'sk-from-env'
    from llm import OpenAILLM
    with mock.patch('openai.OpenAI') as mock_cls:
      OpenAILLM(api_key='sk-explicit')
      mock_cls.assert_called_once_with(api_key='sk-explicit', timeout=120.0)


if __name__ == '__main__':
  unittest.main()
