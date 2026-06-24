# Copyright 2026 Google LLC.
"""Tests for the cascade evaluation framework (Tier A, no sandbox)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import (
    StageOutcome, EvalReport, CascadeEvaluator,
    correctness_gated_speed, aggregate_runtime, RewardHackGuard,
    CORRECTNESS_FLOOR,
)


def _stage(passed=True, score=1.0, **kw):
  """Build a stage callable returning a fixed StageOutcome."""
  def stage(code, context):
    return StageOutcome(passed=passed, score=score, **kw)
  return stage


class CascadeEvaluatorTest(unittest.TestCase):

  def test_all_stages_pass(self):
    calls = []

    def s1(code, ctx):
      calls.append('s1')
      return StageOutcome(passed=True, score=1.0, metrics={'a': 1})

    def s2(code, ctx):
      calls.append('s2')
      return StageOutcome(passed=True, score=1.0, metrics={'b': 2},
                          timings=[0.1, 0.2])

    report = CascadeEvaluator([s1, s2]).evaluate('code')
    self.assertTrue(report.passed)
    self.assertEqual(report.stage_reached, 2)
    self.assertEqual(report.accuracy_metrics, {'a': 1, 'b': 2})
    self.assertEqual(report.timings, [0.1, 0.2])
    self.assertEqual(calls, ['s1', 's2'])

  def test_early_stop_skips_later_stages(self):
    """A stage1 failure must NOT invoke stage2 (the expensive timing stage)."""
    later_called = []

    def s1(code, ctx):
      return StageOutcome(passed=False, failure_mode='bad_shape')

    def s2(code, ctx):
      later_called.append(True)
      return StageOutcome(passed=True, timings=[1.0])

    report = CascadeEvaluator([s1, s2]).evaluate('code')
    self.assertFalse(report.passed)
    self.assertEqual(report.stage_reached, 0)
    self.assertEqual(report.failure_mode, 'bad_shape')
    self.assertEqual(later_called, [])

  def test_threshold_early_stop(self):
    s1 = _stage(passed=True, score=0.3)  # below threshold 0.5
    s2 = _stage(passed=True, score=1.0)
    report = CascadeEvaluator([s1, s2], thresholds=[0.5, 0.75]).evaluate('c')
    self.assertFalse(report.passed)
    self.assertEqual(report.stage_reached, 0)
    self.assertEqual(report.failure_mode, 'below_threshold')

  def test_stage_exception_recorded(self):
    def boom(code, ctx):
      raise ValueError('explode in stage')

    report = CascadeEvaluator([boom]).evaluate('code')
    self.assertFalse(report.passed)
    self.assertEqual(report.failure_mode, 'stage_exception')
    self.assertIn('ValueError', report.traceback)

  def test_dict_outcome_coerced(self):
    def s1(code, ctx):
      return {'passed': True, 'score': 1.0, 'metrics': {'x': 5}}

    report = CascadeEvaluator([s1]).evaluate('code')
    self.assertTrue(report.passed)
    self.assertEqual(report.accuracy_metrics, {'x': 5})

  def test_feature_descriptor_accumulates(self):
    s1 = _stage(passed=True, score=1.0, feature_descriptor={'family': 'fft'})
    s2 = _stage(passed=True, score=1.0, feature_descriptor={'tier': 'fast'})
    report = CascadeEvaluator([s1, s2]).evaluate('c')
    self.assertEqual(report.feature_descriptor,
                     {'family': 'fft', 'tier': 'fast'})

  def test_requires_at_least_one_stage(self):
    with self.assertRaises(ValueError):
      CascadeEvaluator([])


class CorrectnessGatedSpeedTest(unittest.TestCase):

  def test_passing_beats_failing(self):
    good = EvalReport(passed=True, stage_reached=3, timings=[1.0])
    bad = EvalReport(passed=False, stage_reached=2)
    self.assertGreater(correctness_gated_speed(good),
                       correctness_gated_speed(bad))

  def test_passing_above_floor(self):
    good = EvalReport(passed=True, stage_reached=3, timings=[1.0])
    self.assertGreaterEqual(correctness_gated_speed(good, baseline_time=1.0),
                            CORRECTNESS_FLOOR - 1e-9)

  def test_faster_scores_higher(self):
    fast = EvalReport(passed=True, stage_reached=3, timings=[0.1])
    slow = EvalReport(passed=True, stage_reached=3, timings=[1.0])
    self.assertGreater(correctness_gated_speed(fast),
                       correctness_gated_speed(slow))

  def test_more_accurate_scores_higher(self):
    acc = EvalReport(passed=True, timings=[1.0], accuracy_metrics={'accuracy': 5})
    less = EvalReport(passed=True, timings=[1.0], accuracy_metrics={'accuracy': 1})
    self.assertGreater(correctness_gated_speed(acc),
                       correctness_gated_speed(less))

  def test_failure_score_increases_with_stage_reached(self):
    """A deeper failure (got further) should score higher than a shallow one."""
    deep = EvalReport(passed=False, stage_reached=3)
    shallow = EvalReport(passed=False, stage_reached=1)
    self.assertGreater(correctness_gated_speed(deep),
                       correctness_gated_speed(shallow))

  def test_memory_penalty(self):
    light = EvalReport(passed=True, timings=[1.0], memory=10.0)
    heavy = EvalReport(passed=True, timings=[1.0], memory=1000.0)
    self.assertGreater(
        correctness_gated_speed(light, memory_weight=1.0),
        correctness_gated_speed(heavy, memory_weight=1.0))


class AggregateRuntimeTest(unittest.TestCase):

  def test_median_default(self):
    self.assertEqual(aggregate_runtime([3.0, 1.0, 2.0]), 2.0)

  def test_min(self):
    self.assertEqual(aggregate_runtime([3.0, 1.0, 2.0], how='min'), 1.0)

  def test_empty_is_inf(self):
    self.assertEqual(aggregate_runtime([]), float('inf'))


class RewardHackGuardTest(unittest.TestCase):

  def test_clean_code_passes(self):
    guard = RewardHackGuard(forbidden_names=['u_exact'], benchmark_dims=[513])
    code = "def solve(rhs, grid, bc):\n    return rhs * 0\n"
    self.assertIsNone(guard.check(code))

  def test_forbidden_ground_truth_reference_rejected(self):
    guard = RewardHackGuard(forbidden_names=['u_exact'])
    code = "def solve(rhs, grid, bc):\n    return u_exact\n"
    reason = guard.check(code)
    self.assertIsNotNone(reason)
    self.assertIn('u_exact', reason)

  def test_hardcoded_benchmark_dim_rejected(self):
    guard = RewardHackGuard(benchmark_dims=[513])
    code = "def solve(rhs, grid, bc):\n    if grid == 513:\n        return None\n"
    reason = guard.check(code)
    self.assertIsNotNone(reason)
    self.assertIn('513', reason)

  def test_substring_of_larger_number_not_flagged(self):
    guard = RewardHackGuard(benchmark_dims=[51])
    code = "x = 512\n"  # 51 is a substring but not the literal 51
    self.assertIsNone(guard.check(code))


if __name__ == '__main__':
  unittest.main()
