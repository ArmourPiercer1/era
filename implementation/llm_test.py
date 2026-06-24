# Copyright 2026 Google LLC.
"""Tests for the LLM backends."""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import (
    LLM, _BaseLLM, GeminiLLM, OpenAILLM, AnthropicLLM,
)


# ---------------------------------------------------------------------------
# Prompt / response helpers (unit tests – no API calls)
# ---------------------------------------------------------------------------

class PromptFormattingTest(unittest.TestCase):
  """Verify that prompts are correctly split into system / user messages."""

  def test_system_prompt_is_not_empty(self):
    self.assertIn('Data Scientist', _BaseLLM.SYSTEM_PROMPT)

  def test_build_user_prompt_wraps_input(self):
    user = _BaseLLM._build_user_prompt('do the thing')
    self.assertIn('--- BEGIN PROMPT ---', user)
    self.assertIn('do the thing', user)
    self.assertIn('--- END PROMPT ---', user)

  def test_build_full_prompt_contains_system_and_user(self):
    full = _BaseLLM()._build_full_prompt('task')
    self.assertIn(_BaseLLM.SYSTEM_PROMPT, full)
    self.assertIn('--- BEGIN PROMPT ---', full)

  def test_clean_response_strips_python_fence(self):
    raw = '```python\nimport numpy\n```'
    self.assertEqual(_BaseLLM._clean_response(raw), 'import numpy')

  def test_clean_response_strips_bare_fence(self):
    raw = '```\ncode here\n```'
    self.assertEqual(_BaseLLM._clean_response(raw), 'code here')

  def test_clean_response_strips_leading_trailing_whitespace(self):
    raw = '\n\n  hello world  \n\n'
    self.assertEqual(_BaseLLM._clean_response(raw), 'hello world')

  def test_clean_response_no_fence_unchanged(self):
    raw = 'import numpy\nx = 1'
    self.assertEqual(_BaseLLM._clean_response(raw), raw)


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

class RateLimitTest(unittest.TestCase):

  def test_detects_429_in_message(self):
    exc = Exception('HTTP 429 Too Many Requests')
    self.assertTrue(_BaseLLM._is_rate_limit(exc))

  def test_no_false_positive_on_normal_error(self):
    exc = Exception('HTTP 500 Internal Server Error')
    self.assertFalse(_BaseLLM._is_rate_limit(exc))

  def test_openai_rate_limit_error_class(self):
    """openai.RateLimitError is caught by class-name heuristic."""
    # Simulate the error class without importing openai.
    FakeRateLimitError = type('RateLimitError', (Exception,), {})
    exc = FakeRateLimitError('429 rate limit')
    self.assertTrue(OpenAILLM._is_rate_limit(exc))

  def test_anthropic_rate_limit_error_class(self):
    FakeRateLimitError = type('RateLimitError', (Exception,), {})
    exc = FakeRateLimitError('429 rate limit')
    self.assertTrue(AnthropicLLM._is_rate_limit(exc))


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

class APIKeyResolutionTest(unittest.TestCase):

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_openai_missing_key_raises(self):
    with self.assertRaises(ValueError) as ctx:
      OpenAILLM()
    self.assertIn('OPENAI_API_KEY', str(ctx.exception))

  @mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'}, clear=True)
  @mock.patch('openai.OpenAI')
  def test_openai_key_from_env(self, mock_client_class):
    llm = OpenAILLM()
    self.assertEqual(llm.model_name, 'gpt-4.1')
    mock_client_class.assert_called_once_with(api_key='sk-test123', timeout=120.0)

  @mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-env'}, clear=True)
  @mock.patch('openai.OpenAI')
  def test_openai_explicit_key_wins_over_env(self, mock_client_class):
    llm = OpenAILLM(api_key='sk-explicit')
    mock_client_class.assert_called_once_with(api_key='sk-explicit', timeout=120.0)

  @mock.patch.dict(os.environ, {}, clear=True)
  def test_anthropic_missing_key_raises(self):
    with self.assertRaises(ValueError) as ctx:
      AnthropicLLM()
    self.assertIn('ANTHROPIC_API_KEY', str(ctx.exception))

  @mock.patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test'}, clear=True)
  @mock.patch('anthropic.Anthropic')
  def test_anthropic_key_from_env(self, mock_client_class):
    llm = AnthropicLLM()
    self.assertEqual(llm.model_name, 'claude-sonnet-4-6')
    mock_client_class.assert_called_once_with(api_key='sk-ant-test', timeout=120.0)


# ---------------------------------------------------------------------------
# OpenAI Chat Completions – integration-style with mocked client
# ---------------------------------------------------------------------------

class OpenAILLMMockedTest(unittest.TestCase):

  def setUp(self):
    # Prevent real network calls and real sleeps.
    self._sleep_patcher = mock.patch('time.sleep')
    self._sleep_patcher.start()
    self.addCleanup(self._sleep_patcher.stop)

    patcher = mock.patch('openai.OpenAI')
    self.addCleanup(patcher.stop)
    self.mock_client_class = patcher.start()
    self.mock_client = self.mock_client_class.return_value

  def _make_llm(self, **kwargs):
    kwargs.setdefault('api_key', 'sk-test')
    return OpenAILLM(**kwargs)

  def test_client_receives_default_timeout(self):
    llm = self._make_llm()
    self.assertIsInstance(llm, OpenAILLM)
    self.mock_client_class.assert_called_once_with(
        api_key='sk-test', timeout=120.0)

  def test_client_receives_custom_timeout(self):
    llm = self._make_llm(timeout_seconds=30.5)
    self.assertIsInstance(llm, OpenAILLM)
    self.mock_client_class.assert_called_once_with(
        api_key='sk-test', timeout=30.5)

  def test_draw_sample_basic(self):
    """A successful call returns the cleaned response text."""
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('import numpy\n'))
    llm = self._make_llm()
    result = llm.draw_sample('write a numpy import')
    self.assertEqual(result, 'import numpy')
    self.mock_client.chat.completions.create.assert_called_once()

  def test_temperature_top_p_passthrough(self):
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('ok'))
    llm = self._make_llm(temperature=0.7, top_p=0.9)
    llm.draw_sample('prompt')
    kwargs = self.mock_client.chat.completions.create.call_args[1]
    self.assertEqual(kwargs['temperature'], 0.7)
    self.assertEqual(kwargs['top_p'], 0.9)

  def test_per_call_temperature_overrides_instance(self):
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('ok'))
    llm = self._make_llm(temperature=0.2)
    llm.draw_sample('prompt', temperature=0.9)
    kwargs = self.mock_client.chat.completions.create.call_args[1]
    self.assertEqual(kwargs['temperature'], 0.9)

  def test_no_temperature_kwarg_when_unset(self):
    """Backward compat: no temperature/top_p kwargs by default."""
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('ok'))
    llm = self._make_llm()
    llm.draw_sample('prompt')
    kwargs = self.mock_client.chat.completions.create.call_args[1]
    self.assertNotIn('temperature', kwargs)
    self.assertNotIn('top_p', kwargs)

  def test_custom_system_prompt_used(self):
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('ok'))
    llm = self._make_llm(system_prompt='You are a PDE solver expert.')
    llm.draw_sample('prompt')
    messages = self.mock_client.chat.completions.create.call_args[1]['messages']
    self.assertEqual(messages[0]['content'], 'You are a PDE solver expert.')

  def test_draw_samples_returns_n(self):
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('x = 1'))
    llm = self._make_llm()
    out = llm.draw_samples('prompt', n=3)
    self.assertEqual(out, ['x = 1', 'x = 1', 'x = 1'])
    self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)

  def test_draw_sample_strips_fences(self):
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('```python\nx = 1\n```'))
    llm = self._make_llm()
    result = llm.draw_sample('prompt')
    self.assertEqual(result, 'x = 1')

  def test_draw_sample_passes_system_and_user_messages(self):
    self.mock_client.chat.completions.create.return_value = (
        self._make_completion('ok'))
    llm = self._make_llm(model_name='gpt-4.1-mini')
    llm.draw_sample('my task')

    call_kwargs = self.mock_client.chat.completions.create.call_args[1]
    self.assertEqual(call_kwargs['model'], 'gpt-4.1-mini')
    self.assertEqual(call_kwargs['max_tokens'], 4096)
    messages = call_kwargs['messages']
    self.assertEqual(messages[0]['role'], 'system')
    self.assertIn('Data Scientist', messages[0]['content'])
    self.assertEqual(messages[1]['role'], 'user')
    self.assertIn('my task', messages[1]['content'])

  def test_retry_on_rate_limit(self):
    self.mock_client.chat.completions.create.side_effect = [
        self._make_rate_limit_error(),
        self._make_completion('retried ok'),
    ]
    llm = self._make_llm()
    result = llm.draw_sample('prompt')
    self.assertEqual(result, 'retried ok')
    self.assertEqual(
        self.mock_client.chat.completions.create.call_count, 2)

  def test_retry_exhausted_raises(self):
    self.mock_client.chat.completions.create.side_effect = (
        self._make_rate_limit_error())
    llm = self._make_llm()
    with self.assertRaises(Exception):
      llm.draw_sample('prompt')
    # 5 attempts = MAX_RETRIES
    self.assertGreaterEqual(
        self.mock_client.chat.completions.create.call_count, 5)

  # ---- helpers ---------------------------------------------------------

  @staticmethod
  def _make_completion(text: str):
    """Build a minimal fake chat.completions response."""
    choice = mock.Mock()
    choice.message.content = text
    resp = mock.Mock()
    resp.choices = [choice]
    return resp

  @staticmethod
  def _make_rate_limit_error():
    return type('RateLimitError', (Exception,), {})(  # pylint: disable=too-many-function-args
        '429 rate limit exceeded')


# ---------------------------------------------------------------------------
# Anthropic Messages – integration-style with mocked client
# ---------------------------------------------------------------------------

class AnthropicLLMMockedTest(unittest.TestCase):

  def setUp(self):
    # Prevent real network calls and real sleeps.
    self._sleep_patcher = mock.patch('time.sleep')
    self._sleep_patcher.start()
    self.addCleanup(self._sleep_patcher.stop)

    patcher = mock.patch('anthropic.Anthropic')
    self.addCleanup(patcher.stop)
    self.mock_client_class = patcher.start()
    self.mock_client = self.mock_client_class.return_value

  def _make_llm(self, **kwargs):
    kwargs.setdefault('api_key', 'sk-ant-test')
    return AnthropicLLM(**kwargs)

  def test_client_receives_default_timeout(self):
    llm = self._make_llm()
    self.assertIsInstance(llm, AnthropicLLM)
    self.mock_client_class.assert_called_once_with(
        api_key='sk-ant-test', timeout=120.0)

  def test_client_receives_custom_timeout(self):
    llm = self._make_llm(timeout_seconds=30.5)
    self.assertIsInstance(llm, AnthropicLLM)
    self.mock_client_class.assert_called_once_with(
        api_key='sk-ant-test', timeout=30.5)

  def test_draw_sample_basic(self):
    self.mock_client.messages.create.return_value = (
        self._make_message('import numpy\n'))
    llm = self._make_llm()
    result = llm.draw_sample('write a numpy import')
    self.assertEqual(result, 'import numpy')
    self.mock_client.messages.create.assert_called_once()

  def test_temperature_top_p_passthrough(self):
    self.mock_client.messages.create.return_value = self._make_message('ok')
    llm = self._make_llm(temperature=0.5, top_p=0.8)
    llm.draw_sample('prompt')
    kwargs = self.mock_client.messages.create.call_args[1]
    self.assertEqual(kwargs['temperature'], 0.5)
    self.assertEqual(kwargs['top_p'], 0.8)

  def test_custom_system_prompt_used(self):
    self.mock_client.messages.create.return_value = self._make_message('ok')
    llm = self._make_llm(system_prompt='PDE expert')
    llm.draw_sample('prompt')
    kwargs = self.mock_client.messages.create.call_args[1]
    self.assertEqual(kwargs['system'], 'PDE expert')

  def test_draw_sample_strips_fences(self):
    self.mock_client.messages.create.return_value = (
        self._make_message('```python\nx = 1\n```'))
    llm = self._make_llm()
    result = llm.draw_sample('prompt')
    self.assertEqual(result, 'x = 1')

  def test_draw_sample_passes_system_param(self):
    """Anthropic uses a top-level ``system`` parameter, not a message."""
    self.mock_client.messages.create.return_value = (
        self._make_message('ok'))
    llm = self._make_llm(model_name='claude-sonnet-4-6')
    llm.draw_sample('my task')

    call_kwargs = self.mock_client.messages.create.call_args[1]
    self.assertEqual(call_kwargs['model'], 'claude-sonnet-4-6')
    self.assertEqual(call_kwargs['max_tokens'], 4096)
    # Anthropic separates ``system`` from ``messages``.
    self.assertIn('Data Scientist', call_kwargs['system'])
    messages = call_kwargs['messages']
    self.assertEqual(len(messages), 1)
    self.assertEqual(messages[0]['role'], 'user')
    self.assertIn('my task', messages[0]['content'])

  def test_retry_on_rate_limit(self):
    self.mock_client.messages.create.side_effect = [
        self._make_rate_limit_error(),
        self._make_message('retried ok'),
    ]
    llm = self._make_llm()
    result = llm.draw_sample('prompt')
    self.assertEqual(result, 'retried ok')
    self.assertEqual(self.mock_client.messages.create.call_count, 2)

  def test_retry_exhausted_raises(self):
    self.mock_client.messages.create.side_effect = (
        self._make_rate_limit_error())
    llm = self._make_llm()
    with self.assertRaises(Exception):
      llm.draw_sample('prompt')
    self.assertGreaterEqual(
        self.mock_client.messages.create.call_count, 5)

  # ---- helpers ---------------------------------------------------------

  @staticmethod
  def _make_message(text: str):
    """Build a minimal fake Messages response."""
    block = mock.Mock()
    block.text = text
    resp = mock.Mock()
    resp.content = [block]
    return resp

  @staticmethod
  def _make_rate_limit_error():
    return type('RateLimitError', (Exception,), {})(  # pylint: disable=too-many-function-args
        '429 rate limit exceeded')


# ---------------------------------------------------------------------------
# Gemini – constructor behavior with fake google.genai module
# ---------------------------------------------------------------------------

class GeminiLLMMockedTest(unittest.TestCase):

  def test_client_receives_timeout_http_options(self):
    fake_google = types.ModuleType('google')
    fake_genai = types.ModuleType('google.genai')
    fake_genai.Client = mock.Mock()
    fake_google.genai = fake_genai

    with mock.patch.dict(sys.modules, {
        'google': fake_google,
        'google.genai': fake_genai,
    }):
      llm = GeminiLLM(api_key='gemini-key', timeout_seconds=12.5)

    self.assertIsInstance(llm, GeminiLLM)
    fake_genai.Client.assert_called_once_with(
        api_key='gemini-key',
        vertexai=False,
        http_options={'timeout': 12500},
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class ProtocolComplianceTest(unittest.TestCase):

  def test_all_backends_implement_draw_sample(self):
    for cls in (GeminiLLM, OpenAILLM, AnthropicLLM):
      with self.subTest(cls=cls):
        self.assertTrue(hasattr(cls, 'draw_sample'))

  def test_gemini_signature(self):
    """GeminiLLM keeps its original constructor signature."""
    import inspect
    sig = inspect.signature(GeminiLLM.__init__)
    params = list(sig.parameters.keys())
    self.assertIn('api_key', params)
    self.assertIn('model_name', params)


if __name__ == '__main__':
  unittest.main()
