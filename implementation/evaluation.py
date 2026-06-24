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
"""Cascade evaluation, correctness-gated scoring, and reward-hack guards.

Design reference: OpenEvolve (Apache-2.0) ``evaluator.py`` / ``evaluation_result.py``
(staged cascade, threshold early-stop, metric/artifact merge).  Reimplemented
cleanly for ERA with a structured :class:`EvalReport` that carries timing and a
pluggable per-task score function.

The framework's job is to run a candidate through a sequence of increasingly
expensive stages (cheap correctness first, expensive timing last) and fill an
:class:`EvalReport` with accuracy metrics, timings, memory, and failure info.
The *task* supplies the stages and a ``score_fn(EvalReport) -> float``; a
reusable :func:`correctness_gated_speed` is provided for the common case.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StageOutcome:
  """Result of a single cascade stage.

  A stage callable receives ``(code, context)`` and returns one of these (or a
  plain ``dict`` with the same keys, which is coerced).
  """

  passed: bool
  score: float = 0.0
  metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
  timings: List[float] = dataclasses.field(default_factory=list)
  memory: Optional[float] = None
  failure_mode: str = ''
  traceback: str = ''
  feature_descriptor: Dict[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def coerce(cls, obj: Any) -> 'StageOutcome':
    """Accept a StageOutcome or a plain dict (OpenEvolve-style compatibility)."""
    if isinstance(obj, StageOutcome):
      return obj
    if isinstance(obj, dict):
      fields = {f.name for f in dataclasses.fields(cls)}
      return cls(**{k: v for k, v in obj.items() if k in fields})
    raise TypeError(
        f'Stage must return StageOutcome or dict, got {type(obj).__name__}')


@dataclasses.dataclass
class EvalReport:
  """Aggregated evaluation of a candidate program.

  ``passed`` is the correctness gate: ``True`` only if every cascade stage
  passed.  ``timings`` holds the measured runtimes of the timing stage (if
  reached).  ``feature_descriptor`` feeds the MAP-Elites grid in the database.
  """

  passed: bool
  stage_reached: int = 0
  accuracy_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
  timings: List[float] = dataclasses.field(default_factory=list)
  memory: Optional[float] = None
  failure_mode: str = ''
  traceback: str = ''
  feature_descriptor: Dict[str, Any] = dataclasses.field(default_factory=dict)


# A stage is any callable (code, context) -> StageOutcome | dict.
Stage = Callable[[str, Any], Any]


# ---------------------------------------------------------------------------
# Cascade evaluator
# ---------------------------------------------------------------------------

class CascadeEvaluator:
  """Runs a candidate through ordered stages, stopping at the first failure.

  Each stage's ``score`` is gated against ``thresholds[i]`` (when present): a
  score below the threshold halts the cascade just like ``passed=False``.  This
  mirrors OpenEvolve's ``cascade_thresholds`` early-stop.  Metrics and feature
  descriptors accumulate across stages; the deepest stage's timings/memory win.
  """

  def __init__(
      self,
      stages: List[Stage],
      thresholds: Optional[List[float]] = None,
  ) -> None:
    if not stages:
      raise ValueError('CascadeEvaluator requires at least one stage.')
    self.stages = stages
    self.thresholds = thresholds or []

  def evaluate(self, code: str, context: Any = None) -> EvalReport:
    metrics: Dict[str, float] = {}
    features: Dict[str, Any] = {}
    timings: List[float] = []
    memory: Optional[float] = None

    for i, stage in enumerate(self.stages):
      try:
        outcome = StageOutcome.coerce(stage(code, context))
      except Exception:  # pylint: disable=broad-except
        import traceback
        return EvalReport(
            passed=False,
            stage_reached=i,
            accuracy_metrics=dict(metrics),
            timings=timings,
            memory=memory,
            failure_mode='stage_exception',
            traceback=traceback.format_exc(),
            feature_descriptor=dict(features),
        )

      # Accumulate context from this stage.
      metrics.update(outcome.metrics)
      features.update(outcome.feature_descriptor)
      if outcome.timings:
        timings = list(outcome.timings)
      if outcome.memory is not None:
        memory = outcome.memory

      # Hard failure reported by the stage.
      if not outcome.passed:
        return EvalReport(
            passed=False,
            stage_reached=i,
            accuracy_metrics=dict(metrics),
            timings=timings,
            memory=memory,
            failure_mode=outcome.failure_mode or 'stage_failed',
            traceback=outcome.traceback,
            feature_descriptor=dict(features),
        )

      # Threshold early-stop.
      if i < len(self.thresholds) and outcome.score < self.thresholds[i]:
        return EvalReport(
            passed=False,
            stage_reached=i,
            accuracy_metrics=dict(metrics),
            timings=timings,
            memory=memory,
            failure_mode='below_threshold',
            traceback='',
            feature_descriptor=dict(features),
        )

    return EvalReport(
        passed=True,
        stage_reached=len(self.stages),
        accuracy_metrics=metrics,
        timings=timings,
        memory=memory,
        failure_mode='',
        traceback='',
        feature_descriptor=features,
    )


# ---------------------------------------------------------------------------
# Reusable scoring: correctness gate, then speed
# ---------------------------------------------------------------------------

# Any correct solver scores above this floor; any incorrect one scores below it.
CORRECTNESS_FLOOR = 1000.0
_FAILURE_BASE = -1.0e6
_FAILURE_STEP = 1.0e3
_EPS = 1e-12


def aggregate_runtime(timings: List[float], how: str = 'median') -> float:
  """Aggregate a list of timings; defaults to median, supports 'min'/'mean'."""
  if not timings:
    return float('inf')
  ordered = sorted(timings)
  if how == 'min':
    return ordered[0]
  if how == 'mean':
    return sum(ordered) / len(ordered)
  n = len(ordered)
  mid = n // 2
  if n % 2 == 1:
    return ordered[mid]
  return (ordered[mid - 1] + ordered[mid]) / 2.0


def correctness_gated_speed(
    report: EvalReport,
    baseline_time: float = 1.0,
    speed_weight: float = 100.0,
    accuracy_key: str = 'accuracy',
    accuracy_weight: float = 10.0,
    memory_weight: float = 0.0,
    runtime_agg: str = 'median',
) -> float:
  """Correctness-gated speed score (higher is better).

  - Failing candidates get a large negative score that *increases* with how
    far they reached in the cascade (a stage-3 failure beats a stage-1
    failure), so partial progress is rewarded for selection pressure.
  - Passing candidates score above :data:`CORRECTNESS_FLOOR`, with a speed term
    ``speed_weight * log(baseline_time / runtime)`` (monotonically increasing as
    runtime falls), an accuracy bonus, and an optional memory penalty.
  """
  if not report.passed:
    return _FAILURE_BASE + report.stage_reached * _FAILURE_STEP

  runtime = aggregate_runtime(report.timings, runtime_agg)
  speedup = baseline_time / max(runtime, _EPS)
  accuracy = float(report.accuracy_metrics.get(accuracy_key, 0.0))
  memory = report.memory or 0.0

  return (
      CORRECTNESS_FLOOR
      + speed_weight * math.log(max(speedup, _EPS))
      + accuracy_weight * accuracy
      - memory_weight * memory
  )


# ---------------------------------------------------------------------------
# Reward-hack guards (lesson from the solar-topography ERA paper)
# ---------------------------------------------------------------------------

class RewardHackGuard:
  """Static checks that reject obvious reward-hacking candidate code.

  - ``forbidden_names``: identifiers the solver must not reference (e.g. the
    ground-truth solution array names).  Matched as whole words.
  - ``benchmark_dims``: grid sizes used only in the hidden timing/accuracy
    tests; hard-coding them suggests the candidate is special-casing the
    benchmark rather than solving the problem.
  """

  def __init__(
      self,
      forbidden_names: Optional[List[str]] = None,
      benchmark_dims: Optional[List[int]] = None,
  ) -> None:
    self.forbidden_names = forbidden_names or []
    self.benchmark_dims = benchmark_dims or []

  def check(self, code: str) -> Optional[str]:
    """Return a failure reason if the code looks like a reward hack, else None."""
    for name in self.forbidden_names:
      if re.search(rf'\b{re.escape(name)}\b', code):
        return f'forbidden reference to ground-truth name {name!r}'
    for dim in self.benchmark_dims:
      # A bare integer literal equal to a hidden benchmark size is suspicious.
      if re.search(rf'(?<![\d.]){dim}(?![\d.])', code):
        return f'hard-coded benchmark dimension {dim}'
    return None
