# Copyright 2026 Google LLC.
"""LLM backends for the ERA code-improvement pipeline.

Provides a common :class:`LLM` protocol and three concrete backends:

* :class:`GeminiLLM` – Google Gemini (``google-genai``)
* :class:`OpenAILLM` – OpenAI Chat Completions (``openai``)
* :class:`AnthropicLLM` – Anthropic Messages (``anthropic``)

All backends share the same prompting strategy (system prompt + user prompt)
and the same response-cleaning logic (stripping Markdown code fences).
"""

from __future__ import annotations

import os
import re
import time
import random
from typing import List, Optional, Protocol

# Side-effect: loads .env into os.environ before any LLM class reads its key.
import env  # noqa: F401  pylint: disable=unused-import


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class LLM(Protocol):
  """Protocol that every LLM backend must satisfy."""

  def draw_sample(self, prompt: str) -> str:
    """Generate a code sample for the given *prompt*."""
    ...

  def draw_samples(self, prompt: str, n: int) -> List[str]:
    """Generate *n* code samples for the given *prompt*."""
    ...


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _BaseLLM:
  """Shared prompting, response-cleaning, and retry logic.

  Subclasses need only implement ``_call_api(prompt, temperature) -> str``.
  """

  SYSTEM_PROMPT = (
      'You are an expert Data Scientist and Python programmer.\n'
      'Your task is to write Python code to solve a machine learning problem.\n'
      'Return ONLY the python code.'
  )

  MAX_RETRIES = 5
  BASE_DELAY = 5  # seconds – doubles each retry

  def __init__(
      self,
      model_name: str = '',
      system_prompt: Optional[str] = None,
      temperature: Optional[float] = None,
      top_p: Optional[float] = None,
  ) -> None:
    self.model_name = model_name
    # Fall back to the class default when no custom system prompt is given.
    self.system_prompt = system_prompt or self.SYSTEM_PROMPT
    self.temperature = temperature
    self.top_p = top_p

  # -------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------

  def draw_sample(self, prompt: str, temperature: Optional[float] = None) -> str:
    """Format the prompt, call the API, retry on rate-limit, and clean.

    Args:
        prompt: The user prompt.
        temperature: Per-call sampling temperature.  Falls back to the
            instance default (``self.temperature``) when ``None``.
    """
    temp = temperature if temperature is not None else self.temperature
    max_retries = self.MAX_RETRIES
    base_delay = self.BASE_DELAY
    for attempt in range(max_retries):
      try:
        raw = self._call_api(prompt, temp)
        return self._clean_response(raw)
      except Exception as exc:
        if self._is_rate_limit(exc) and attempt < max_retries - 1:
          delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
          print(f'  [!] Rate limited (429). Retrying in {delay:.1f}s…')
          time.sleep(delay)
        else:
          print(f'{self._backend_name()} API Error: {exc}')
          raise

  def draw_samples(
      self,
      prompt: str,
      n: int,
      temperature: Optional[float] = None,
  ) -> List[str]:
    """Generate *n* independent samples for *prompt*.

    Default implementation calls :meth:`draw_sample` ``n`` times, which works
    for every backend.  Subclasses may override with a native batch call.
    """
    return [self.draw_sample(prompt, temperature) for _ in range(n)]

  # -------------------------------------------------------------------
  # Hooks for subclasses
  # -------------------------------------------------------------------

  def _backend_name(self) -> str:
    return type(self).__name__

  def _call_api(self, prompt: str, temperature: Optional[float] = None) -> str:
    """Call the provider-specific API and return the raw text response."""
    raise NotImplementedError

  @staticmethod
  def _is_rate_limit(exc: Exception) -> bool:
    """Detect HTTP 429 / rate-limit errors heuristically."""
    return '429' in str(exc)

  # -------------------------------------------------------------------
  # Prompt helpers
  # -------------------------------------------------------------------

  def _build_full_prompt(self, prompt: str) -> str:
    """Combine system prompt and user prompt into a single string.

    Used by backends (like Gemini) that accept only a flat text prompt.
    """
    return (
        f'{self.system_prompt}\n\n'
        f'--- BEGIN PROMPT ---\n'
        f'{prompt}\n'
        f'--- END PROMPT ---'
    )

  @staticmethod
  def _build_user_prompt(prompt: str) -> str:
    """Format the user portion of the prompt."""
    return f'--- BEGIN PROMPT ---\n{prompt}\n--- END PROMPT ---'

  # -------------------------------------------------------------------
  # Response cleaning
  # -------------------------------------------------------------------

  @staticmethod
  def _clean_response(text: str) -> str:
    """Strip Markdown code fences from the model's output."""
    content = text.strip()
    # Leading ```python or ```
    content = re.sub(r"^```(?:python)?\s*\n?", "", content, flags=re.MULTILINE)
    # Trailing ```
    content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
    return content.strip()


# ---------------------------------------------------------------------------
# Gemini backend (existing – lightly refactored onto the base)
# ---------------------------------------------------------------------------

class GeminiLLM(_BaseLLM):
  """Google Gemini backend using the ``google-genai`` SDK."""

  def __init__(
      self,
      api_key: str,
      model_name: str = 'gemini-2.5-flash-image',
      timeout_seconds: float = 120.0,
      system_prompt: Optional[str] = None,
      temperature: Optional[float] = None,
      top_p: Optional[float] = None,
  ) -> None:
    super().__init__(
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    # Lazy import – google-genai may not be installed everywhere.
    from google import genai
    self._client = genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options={'timeout': int(timeout_seconds * 1000)},
    )

  def _backend_name(self) -> str:
    return 'Gemini'

  def _call_api(self, prompt: str, temperature: Optional[float] = None) -> str:
    full_prompt = self._build_full_prompt(prompt)
    config = {}
    if temperature is not None:
      config['temperature'] = temperature
    if self.top_p is not None:
      config['top_p'] = self.top_p
    kwargs = {'model': self.model_name, 'contents': full_prompt}
    if config:
      kwargs['config'] = config
    response = self._client.models.generate_content(**kwargs)
    return response.text


# ---------------------------------------------------------------------------
# OpenAI Chat Completions backend
# ---------------------------------------------------------------------------

class OpenAILLM(_BaseLLM):
  """OpenAI Chat Completions backend using the ``openai`` SDK.

  API key resolution order:
      1. Explicit *api_key* argument.
      2. ``OPENAI_API_KEY`` environment variable.

  Example::

      llm = OpenAILLM(model_name='gpt-4.1')
      code = llm.draw_sample('Write a function that sorts a list.')
  """

  def __init__(
      self,
      api_key: Optional[str] = None,
      model_name: str = 'gpt-4.1',
      max_tokens: int = 4096,
      timeout_seconds: float = 120.0,
      system_prompt: Optional[str] = None,
      temperature: Optional[float] = None,
      top_p: Optional[float] = None,
  ) -> None:
    super().__init__(
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
      raise ValueError(
          'OpenAI API key not found.  Provide *api_key* or set '
          'the OPENAI_API_KEY environment variable.')
    from openai import OpenAI
    self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)
    self._max_tokens = max_tokens

  def _backend_name(self) -> str:
    return 'OpenAI'

  @staticmethod
  def _is_rate_limit(exc: Exception) -> bool:
    # The openai SDK raises openai.RateLimitError for 429s.
    # Catch by class name to avoid a hard import dependency at module level.
    if type(exc).__name__ == 'RateLimitError':
      return True
    return _BaseLLM._is_rate_limit(exc)

  def _call_api(self, prompt: str, temperature: Optional[float] = None) -> str:
    kwargs = {
        'model': self.model_name,
        'max_tokens': self._max_tokens,
        'messages': [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self._build_user_prompt(prompt)},
        ],
    }
    if temperature is not None:
      kwargs['temperature'] = temperature
    if self.top_p is not None:
      kwargs['top_p'] = self.top_p
    response = self._client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Anthropic Messages backend
# ---------------------------------------------------------------------------

class AnthropicLLM(_BaseLLM):
  """Anthropic Messages backend using the ``anthropic`` SDK.

  API key resolution order:
      1. Explicit *api_key* argument.
      2. ``ANTHROPIC_API_KEY`` environment variable.

  Example::

      llm = AnthropicLLM(model_name='claude-sonnet-4-6')
      code = llm.draw_sample('Write a function that sorts a list.')
  """

  def __init__(
      self,
      api_key: Optional[str] = None,
      model_name: str = 'claude-sonnet-4-6',
      max_tokens: int = 4096,
      timeout_seconds: float = 120.0,
      system_prompt: Optional[str] = None,
      temperature: Optional[float] = None,
      top_p: Optional[float] = None,
  ) -> None:
    super().__init__(
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
      raise ValueError(
          'Anthropic API key not found.  Provide *api_key* or set '
          'the ANTHROPIC_API_KEY environment variable.')
    from anthropic import Anthropic
    self._client = Anthropic(api_key=api_key, timeout=timeout_seconds)
    self._max_tokens = max_tokens

  def _backend_name(self) -> str:
    return 'Anthropic'

  @staticmethod
  def _is_rate_limit(exc: Exception) -> bool:
    # The anthropic SDK raises anthropic.RateLimitError for 429s.
    if type(exc).__name__ == 'RateLimitError':
      return True
    return _BaseLLM._is_rate_limit(exc)

  def _call_api(self, prompt: str, temperature: Optional[float] = None) -> str:
    kwargs = {
        'model': self.model_name,
        'max_tokens': self._max_tokens,
        'system': self.system_prompt,
        'messages': [
            {'role': 'user', 'content': self._build_user_prompt(prompt)},
        ],
    }
    if temperature is not None:
      kwargs['temperature'] = temperature
    if self.top_p is not None:
      kwargs['top_p'] = self.top_p
    response = self._client.messages.create(**kwargs)
    # Anthropic returns a list of ContentBlock; the first one is text.
    return response.content[0].text
