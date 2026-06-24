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
"""Multi-parent generator: inspiration prompting, error feedback, diversity.

Design reference: OpenEvolve (Apache-2.0) ``prompt/sampler.py`` — render a
parent plus top-k elites and diverse-k non-elites, with SEARCH/REPLACE diff or
full-rewrite modes.  Reimplemented cleanly for ERA, adding error-traceback
feedback and method-diversity steering (lessons from the cosmic-strings ERA
paper: feed failures back and forbid already-found method families).
"""

from __future__ import annotations

import dataclasses
import re
from typing import List, Optional, Sequence


# SEARCH/REPLACE diff format (same shape as OpenEvolve's default).
_DIFF_RE = re.compile(
    r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
    re.DOTALL,
)


def apply_diff(parent_code: str, diff_text: str) -> Optional[str]:
  """Apply SEARCH/REPLACE blocks from *diff_text* to *parent_code*.

  Returns the patched code, or ``None`` if there are no valid blocks or a
  SEARCH segment is not found in the parent (so the caller can fall back to
  full-rewrite parsing).
  """
  blocks = _DIFF_RE.findall(diff_text)
  if not blocks:
    return None
  code = parent_code
  for search, replace in blocks:
    if search not in code:
      return None
    code = code.replace(search, replace, 1)
  return code


@dataclasses.dataclass
class GenerationResult:
  """A generated child program plus provenance."""

  code: str
  raw_response: str
  mode: str  # 'diff' or 'rewrite'
  parent_id: Optional[str] = None


class MultiParentGenerator:
  """Builds inspiration-rich prompts and parses LLM responses into code.

  Args:
      llm: an object exposing ``draw_sample(prompt, temperature=None)``.
      spec: the task specification text (problem + solver interface).
      method_families: the universe of algorithm families for diversity
          steering (e.g. ``['fft', 'multigrid', 'cg', 'direct']``).
      diff_based: prefer SEARCH/REPLACE diffs, falling back to full rewrite.
      max_code_length: reject generated code longer than this.
      temperature: per-call sampling temperature.
  """

  def __init__(
      self,
      llm,
      spec: str,
      method_families: Optional[Sequence[str]] = None,
      diff_based: bool = True,
      max_code_length: int = 10000,
      temperature: Optional[float] = None,
  ) -> None:
    self.llm = llm
    self.spec = spec
    self.method_families = list(method_families or [])
    self.diff_based = diff_based
    self.max_code_length = max_code_length
    self.temperature = temperature

  # -------------------------------------------------------------------
  # Prompt construction
  # -------------------------------------------------------------------

  def build_prompt(
      self,
      primary,
      inspirations: Sequence = (),
      discovered_families: Sequence[str] = (),
  ) -> str:
    parts = [self.spec]

    if primary is not None:
      parts.append(self._render_primary(primary))
      if primary.failure_mode or primary.metadata.get('traceback'):
        parts.append(self._render_error(primary))

    if inspirations:
      parts.append(self._render_inspirations(inspirations))

    diversity = self._render_diversity(discovered_families)
    if diversity:
      parts.append(diversity)

    parts.append(self._render_instructions(primary))
    return '\n\n'.join(parts)

  def _render_primary(self, primary) -> str:
    bits = [f'score={primary.score:.4g}']
    if primary.runtime is not None:
      bits.append(f'runtime={primary.runtime:.4g}s')
    if primary.failure_mode:
      bits.append(f'failure={primary.failure_mode}')
    meta = ', '.join(bits)
    return (
        f'## Current solution to improve ({meta})\n'
        f'```python\n{primary.code}\n```'
    )

  @staticmethod
  def _render_error(primary) -> str:
    tb = primary.metadata.get('traceback') or primary.failure_mode
    return (
        '## The current solution FAILED. Fix the cause:\n'
        f'```\n{tb}\n```'
    )

  def _render_inspirations(self, inspirations: Sequence) -> str:
    lines = ['## Other strong solutions to learn from and recombine']
    for i, p in enumerate(inspirations, 1):
      family = p.feature_descriptor.get('family', 'unknown')
      lines.append(
          f'### Inspiration {i} (score={p.score:.4g}, family={family})\n'
          f'```python\n{p.code}\n```')
    return '\n'.join(lines)

  def _render_diversity(self, discovered_families: Sequence[str]) -> str:
    discovered = sorted(set(discovered_families))
    if not discovered and not self.method_families:
      return ''
    untried = [f for f in self.method_families if f not in set(discovered)]
    lines = []
    if discovered:
      lines.append('## Method families already explored: '
                   + ', '.join(discovered))
    if untried:
      lines.append('Prefer trying a DIFFERENT, not-yet-explored family: '
                   + ', '.join(untried) + '.')
    return '\n'.join(lines)

  def _render_instructions(self, primary) -> str:
    if self.diff_based and primary is not None:
      return (
          '## Output format\n'
          'Return one or more SEARCH/REPLACE blocks editing the current '
          'solution:\n'
          '<<<<<<< SEARCH\n<exact lines to find>\n=======\n'
          '<replacement lines>\n>>>>>>> REPLACE')
    return (
        '## Output format\n'
        'Return ONLY the full, runnable Python code for the improved solution.')

  # -------------------------------------------------------------------
  # Generation
  # -------------------------------------------------------------------

  def generate(
      self,
      primary=None,
      inspirations: Sequence = (),
      discovered_families: Sequence[str] = (),
  ) -> Optional[GenerationResult]:
    """Build a prompt, call the LLM, and parse the response into code.

    Returns ``None`` when the response is empty or exceeds the length limit.
    """
    prompt = self.build_prompt(primary, inspirations, discovered_families)
    raw = self.llm.draw_sample(prompt, temperature=self.temperature)
    raw = raw or ''

    mode = 'rewrite'
    code = raw
    if self.diff_based and primary is not None:
      applied = apply_diff(primary.code, raw)
      if applied is not None:
        code, mode = applied, 'diff'

    if not code.strip() or len(code) > self.max_code_length:
      return None

    parent_id = getattr(primary, 'id', None) if primary is not None else None
    return GenerationResult(
        code=code, raw_response=raw, mode=mode, parent_id=parent_id)
