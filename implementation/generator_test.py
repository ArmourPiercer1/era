# Copyright 2026 Google LLC.
"""Tests for the multi-parent generator (Tier A, fake LLM)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import MultiParentGenerator, apply_diff, GenerationResult
from database import Program


class FakeLLM:
  """Records the last prompt and returns a canned response."""

  def __init__(self, response=''):
    self.response = response
    self.last_prompt = None
    self.last_temperature = None

  def draw_sample(self, prompt, temperature=None):
    self.last_prompt = prompt
    self.last_temperature = temperature
    return self.response


def _prog(code='def solve(rhs, grid, bc):\n    return rhs\n', score=1.0,
          family='fft', failure='', traceback=''):
  meta = {'traceback': traceback} if traceback else {}
  return Program(code=code, score=score, feature_descriptor={'family': family},
                 failure_mode=failure, metadata=meta)


class ApplyDiffTest(unittest.TestCase):

  def test_applies_search_replace(self):
    parent = 'a = 1\nb = 2\n'
    diff = '<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE'
    self.assertEqual(apply_diff(parent, diff), 'a = 1\nb = 3\n')

  def test_no_blocks_returns_none(self):
    self.assertIsNone(apply_diff('a = 1\n', 'just prose, no diff'))

  def test_unfound_search_returns_none(self):
    diff = '<<<<<<< SEARCH\nnope\n=======\nx\n>>>>>>> REPLACE'
    self.assertIsNone(apply_diff('a = 1\n', diff))

  def test_multiple_blocks(self):
    parent = 'x = 1\ny = 2\n'
    diff = ('<<<<<<< SEARCH\nx = 1\n=======\nx = 10\n>>>>>>> REPLACE\n'
            '<<<<<<< SEARCH\ny = 2\n=======\ny = 20\n>>>>>>> REPLACE')
    self.assertEqual(apply_diff(parent, diff), 'x = 10\ny = 20\n')


class PromptBuildingTest(unittest.TestCase):

  def setUp(self):
    self.llm = FakeLLM()
    self.gen = MultiParentGenerator(
        self.llm, spec='SOLVE POISSON SPEC',
        method_families=['fft', 'multigrid', 'cg', 'direct'])

  def test_prompt_contains_spec_and_parent(self):
    primary = _prog(code='PARENT_CODE', score=2.5)
    prompt = self.gen.build_prompt(primary, [])
    self.assertIn('SOLVE POISSON SPEC', prompt)
    self.assertIn('PARENT_CODE', prompt)
    self.assertIn('2.5', prompt)

  def test_prompt_contains_inspirations(self):
    primary = _prog(code='PARENT')
    insps = [_prog(code='INSP_A', score=9.0, family='cg'),
             _prog(code='INSP_B', score=8.0, family='multigrid')]
    prompt = self.gen.build_prompt(primary, insps)
    self.assertIn('INSP_A', prompt)
    self.assertIn('INSP_B', prompt)
    self.assertIn('Inspiration 1', prompt)
    self.assertIn('Inspiration 2', prompt)

  def test_prompt_contains_traceback_when_parent_failed(self):
    primary = _prog(failure='runtime_error', traceback='ZeroDivisionError: x')
    prompt = self.gen.build_prompt(primary, [])
    self.assertIn('FAILED', prompt)
    self.assertIn('ZeroDivisionError', prompt)

  def test_prompt_contains_method_diversity(self):
    primary = _prog()
    prompt = self.gen.build_prompt(primary, [], discovered_families=['fft'])
    self.assertIn('already explored', prompt)
    self.assertIn('fft', prompt)
    # Untried families should be suggested.
    self.assertIn('multigrid', prompt)

  def test_diversity_grows_with_archive(self):
    primary = _prog()
    p1 = self.gen.build_prompt(primary, [], discovered_families=['fft'])
    p2 = self.gen.build_prompt(
        primary, [], discovered_families=['fft', 'multigrid'])
    self.assertIn('multigrid', p2.split('already explored')[1].split('\n')[0])
    self.assertNotEqual(p1, p2)

  def test_diff_instructions_when_diff_based(self):
    prompt = self.gen.build_prompt(_prog(), [])
    self.assertIn('SEARCH', prompt)
    self.assertIn('REPLACE', prompt)

  def test_rewrite_instructions_when_no_parent(self):
    prompt = self.gen.build_prompt(None, [])
    self.assertIn('full, runnable Python', prompt)


class GenerateTest(unittest.TestCase):

  def test_diff_mode_applies_patch(self):
    parent = _prog(code='a = 1\nb = 2\n')
    llm = FakeLLM('<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE')
    gen = MultiParentGenerator(llm, spec='spec', diff_based=True)
    result = gen.generate(parent, [])
    self.assertEqual(result.mode, 'diff')
    self.assertEqual(result.code, 'a = 1\nb = 3\n')
    self.assertEqual(result.parent_id, parent.id)

  def test_falls_back_to_rewrite_when_no_diff(self):
    parent = _prog(code='old code')
    llm = FakeLLM('def solve(rhs, grid, bc):\n    return rhs * 2\n')
    gen = MultiParentGenerator(llm, spec='spec', diff_based=True)
    result = gen.generate(parent, [])
    self.assertEqual(result.mode, 'rewrite')
    self.assertIn('return rhs * 2', result.code)

  def test_rewrite_mode_when_no_parent(self):
    llm = FakeLLM('def solve(rhs, grid, bc):\n    return rhs\n')
    gen = MultiParentGenerator(llm, spec='spec', diff_based=True)
    result = gen.generate(None, [])
    self.assertEqual(result.mode, 'rewrite')
    self.assertIsNone(result.parent_id)

  def test_empty_response_returns_none(self):
    gen = MultiParentGenerator(FakeLLM(''), spec='spec')
    self.assertIsNone(gen.generate(None, []))

  def test_too_long_returns_none(self):
    llm = FakeLLM('x' * 50)
    gen = MultiParentGenerator(llm, spec='spec', diff_based=False,
                               max_code_length=10)
    self.assertIsNone(gen.generate(None, []))

  def test_temperature_passed_through(self):
    llm = FakeLLM('code')
    gen = MultiParentGenerator(llm, spec='spec', diff_based=False,
                               temperature=0.8)
    gen.generate(None, [])
    self.assertEqual(llm.last_temperature, 0.8)


if __name__ == '__main__':
  unittest.main()
