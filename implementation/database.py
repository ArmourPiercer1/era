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
"""Evolutionary program database: islands, MAP-Elites, Pareto, persistence.

Design reference: OpenEvolve (Apache-2.0) ``database.py`` — per-island
populations, per-island MAP-Elites grids (keep-best-per-cell), ring migration
of elites, and checkpoint/resume.  Reimplemented cleanly for ERA, adding a
Pareto front over ``(accuracy, runtime, memory)`` which OpenEvolve does not
provide.
"""

from __future__ import annotations

import dataclasses
import json
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Program record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Program:
  """A candidate program plus its evaluation summary and lineage."""

  code: str
  score: float
  id: str = ''
  metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
  feature_descriptor: Dict[str, Any] = dataclasses.field(default_factory=dict)
  island_id: int = 0
  parent_ids: List[str] = dataclasses.field(default_factory=list)
  generation: int = 0
  created_iter: int = 0
  runtime: Optional[float] = None
  memory: Optional[float] = None
  failure_mode: str = ''
  metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if not self.id:
      self.id = uuid.uuid4().hex[:12]


def make_program(
    code: str,
    report,
    score: float,
    island_id: int = 0,
    parent_ids: Optional[List[str]] = None,
    generation: int = 0,
    created_iter: int = 0,
) -> Program:
  """Build a :class:`Program` from an :class:`evaluation.EvalReport`."""
  from evaluation import aggregate_runtime
  runtime = aggregate_runtime(report.timings) if report.timings else None
  if runtime == float('inf'):
    runtime = None
  return Program(
      code=code,
      score=score,
      metrics=dict(report.accuracy_metrics),
      feature_descriptor=dict(report.feature_descriptor),
      island_id=island_id,
      parent_ids=list(parent_ids or []),
      generation=generation,
      created_iter=created_iter,
      runtime=runtime,
      memory=report.memory,
      failure_mode=report.failure_mode,
  )


# ---------------------------------------------------------------------------
# Program database
# ---------------------------------------------------------------------------

class ProgramDatabase:
  """Islanded MAP-Elites archive with ring migration and a Pareto front."""

  def __init__(
      self,
      num_islands: int = 5,
      feature_dimensions: Optional[List[str]] = None,
      feature_bins: int = 10,
      migration_interval: int = 50,
      migration_rate: float = 0.1,
      num_top_programs: int = 3,
      num_diverse_programs: int = 2,
      population_size: int = 1000,
      seed: int = 42,
  ) -> None:
    self.num_islands = num_islands
    self.feature_dimensions = feature_dimensions or ['complexity', 'diversity']
    self.feature_bins = feature_bins
    self.migration_interval = migration_interval
    self.migration_rate = migration_rate
    self.num_top_programs = num_top_programs
    self.num_diverse_programs = num_diverse_programs
    self.population_size = population_size

    self.programs: Dict[str, Program] = {}
    self.islands: List[set] = [set() for _ in range(num_islands)]
    self.grids: List[Dict[str, str]] = [{} for _ in range(num_islands)]
    self.island_generations: List[int] = [0] * num_islands
    self.island_best: List[Optional[str]] = [None] * num_islands
    self.best_id: Optional[str] = None
    self.last_migration_generation: int = 0
    self.last_iteration: int = 0
    self.feature_stats: Dict[str, Dict[str, float]] = {}
    self._rng = random.Random(seed)

  # -------------------------------------------------------------------
  # Insertion
  # -------------------------------------------------------------------

  def add(
      self,
      program: Program,
      target_island: Optional[int] = None,
      iteration: Optional[int] = None,
  ) -> Program:
    """Insert *program*, updating island membership, MAP-Elites grid, best."""
    island = target_island if target_island is not None else program.island_id
    island %= self.num_islands
    program.island_id = island

    if iteration is not None:
      program.created_iter = iteration
      self.last_iteration = max(self.last_iteration, iteration)

    self.programs[program.id] = program
    self.islands[island].add(program.id)

    # MAP-Elites: keep the best program per feature cell.
    key = self._cell_key(program)
    occupant = self.grids[island].get(key)
    if (occupant is None or occupant not in self.programs
        or program.score > self.programs[occupant].score):
      self.grids[island][key] = program.id

    self._update_best(program)
    self._enforce_population_limit()
    return program

  def increment_island_generation(self, island_id: int) -> None:
    self.island_generations[island_id % self.num_islands] += 1

  # -------------------------------------------------------------------
  # MAP-Elites feature coordinates
  # -------------------------------------------------------------------

  def _cell_key(self, program: Program) -> str:
    return '-'.join(self._feature_coords(program))

  def _feature_coords(self, program: Program) -> List[str]:
    coords = []
    for dim in self.feature_dimensions:
      if dim not in program.feature_descriptor:
        raise ValueError(
            f'feature dimension {dim!r} missing from program descriptor')
      val = program.feature_descriptor[dim]
      if isinstance(val, str):
        coords.append(val)  # categorical coordinate (e.g. algorithm family)
      else:
        self._update_feature_stats(dim, val)
        scaled = self._scale(dim, val)
        b = min(int(scaled * self.feature_bins), self.feature_bins - 1)
        coords.append(str(b))
    return coords

  def _update_feature_stats(self, dim: str, value: float) -> None:
    st = self.feature_stats.setdefault(dim, {'min': value, 'max': value})
    st['min'] = min(st['min'], value)
    st['max'] = max(st['max'], value)

  def _scale(self, dim: str, value: float) -> float:
    st = self.feature_stats[dim]
    if st['max'] == st['min']:
      return 0.5
    scaled = (value - st['min']) / (st['max'] - st['min'])
    return max(0.0, min(1.0, scaled))

  # -------------------------------------------------------------------
  # Best / population bookkeeping
  # -------------------------------------------------------------------

  def _update_best(self, program: Program) -> None:
    if self.best_id is None or self.best_id not in self.programs or (
        program.score > self.programs[self.best_id].score):
      self.best_id = program.id
    ib = self.island_best[program.island_id]
    if ib is None or ib not in self.programs or (
        program.score > self.programs[ib].score):
      self.island_best[program.island_id] = program.id

  def _enforce_population_limit(self) -> None:
    if len(self.programs) <= self.population_size:
      return
    protected = {self.best_id}
    for grid in self.grids:
      protected.update(grid.values())
    removable = [p for pid, p in self.programs.items() if pid not in protected]
    removable.sort(key=lambda p: p.score)  # worst first
    n_remove = len(self.programs) - self.population_size
    for p in removable[:n_remove]:
      self._remove(p.id)

  def _remove(self, pid: str) -> None:
    self.programs.pop(pid, None)
    for s in self.islands:
      s.discard(pid)
    for grid in self.grids:
      for k, v in list(grid.items()):
        if v == pid:
          del grid[k]

  def best_program(self) -> Optional[Program]:
    if self.best_id and self.best_id in self.programs:
      return self.programs[self.best_id]
    return None

  # -------------------------------------------------------------------
  # Sampling parents + inspirations
  # -------------------------------------------------------------------

  def island_members(self, island_id: int) -> List[Program]:
    return [self.programs[pid] for pid in self.islands[island_id]
            if pid in self.programs]

  def top_programs(self, island_id: int, k: int) -> List[Program]:
    members = self.island_members(island_id)
    members.sort(key=lambda p: p.score, reverse=True)
    return members[:k]

  def inspirations(
      self,
      island_id: int,
      exclude_ids: Tuple[str, ...] = (),
      num_top: Optional[int] = None,
      num_diverse: Optional[int] = None,
  ) -> List[Program]:
    """Return top-k elites + diverse-k randomly sampled non-elites (deduped)."""
    num_top = self.num_top_programs if num_top is None else num_top
    num_diverse = self.num_diverse_programs if num_diverse is None else num_diverse
    exclude = set(exclude_ids)
    members = [p for p in self.island_members(island_id) if p.id not in exclude]
    members.sort(key=lambda p: p.score, reverse=True)
    top = members[:num_top]
    rest = members[num_top:]
    diverse = (self._rng.sample(rest, min(num_diverse, len(rest)))
               if rest else [])
    seen, out = set(), []
    for p in top + diverse:
      if p.id not in seen:
        seen.add(p.id)
        out.append(p)
    return out

  def sample_parents(
      self,
      island_id: int,
      num_top: Optional[int] = None,
      num_diverse: Optional[int] = None,
  ) -> Tuple[Optional[Program], List[Program]]:
    """Convenience: best-in-island primary + inspirations.

    The evolve orchestrator may instead pick the primary via futs PUCT and
    call :meth:`inspirations` directly.
    """
    members = self.island_members(island_id)
    if not members:
      return None, []
    primary = max(members, key=lambda p: p.score)
    insp = self.inspirations(island_id, exclude_ids=(primary.id,),
                             num_top=num_top, num_diverse=num_diverse)
    return primary, insp

  # -------------------------------------------------------------------
  # Migration (ring topology)
  # -------------------------------------------------------------------

  def should_migrate(self) -> bool:
    return (max(self.island_generations) - self.last_migration_generation
            >= self.migration_interval)

  def migrate(self) -> None:
    if self.num_islands < 2:
      return
    for i in range(self.num_islands):
      members = self.island_members(i)
      if not members:
        continue
      members.sort(key=lambda p: p.score, reverse=True)
      k = max(1, int(len(members) * self.migration_rate))
      migrants = [m for m in members[:k] if not m.metadata.get('migrant')]
      for target in ((i + 1) % self.num_islands, (i - 1) % self.num_islands):
        for m in migrants:
          # Skip if the target island already has identical code.
          if any(self.programs[pid].code == m.code
                 for pid in self.islands[target] if pid in self.programs):
            continue
          copy = Program(
              code=m.code,
              score=m.score,
              metrics=dict(m.metrics),
              feature_descriptor=dict(m.feature_descriptor),
              island_id=target,
              parent_ids=[m.id],
              generation=m.generation,
              created_iter=m.created_iter,
              runtime=m.runtime,
              memory=m.memory,
              failure_mode=m.failure_mode,
              metadata={**m.metadata, 'migrant': True},
          )
          self.add(copy, target_island=target)
    self.last_migration_generation = max(self.island_generations)

  # -------------------------------------------------------------------
  # Pareto front over (accuracy, runtime, memory)  [ERA-specific]
  # -------------------------------------------------------------------

  @staticmethod
  def _dominates(a: Program, b: Program) -> bool:
    a_acc = a.metrics.get('accuracy', 0.0)
    b_acc = b.metrics.get('accuracy', 0.0)
    a_rt, b_rt = a.runtime, b.runtime
    a_mem = a.memory or 0.0
    b_mem = b.memory or 0.0
    not_worse = (a_acc >= b_acc and a_rt <= b_rt and a_mem <= b_mem)
    strictly_better = (a_acc > b_acc or a_rt < b_rt or a_mem < b_mem)
    return not_worse and strictly_better

  def pareto_front(self) -> List[Program]:
    """Non-dominated programs over (accuracy↑, runtime↓, memory↓)."""
    candidates = [p for p in self.programs.values() if p.runtime is not None]
    front = []
    for p in candidates:
      if not any(self._dominates(q, p) for q in candidates if q is not p):
        front.append(p)
    return front

  # -------------------------------------------------------------------
  # Persistence (checkpoint / resume)
  # -------------------------------------------------------------------

  def save(self, path: str) -> None:
    data = {
        'config': {
            'num_islands': self.num_islands,
            'feature_dimensions': self.feature_dimensions,
            'feature_bins': self.feature_bins,
            'migration_interval': self.migration_interval,
            'migration_rate': self.migration_rate,
            'num_top_programs': self.num_top_programs,
            'num_diverse_programs': self.num_diverse_programs,
            'population_size': self.population_size,
        },
        'programs': {pid: dataclasses.asdict(p)
                     for pid, p in self.programs.items()},
        'islands': [sorted(s) for s in self.islands],
        'grids': self.grids,
        'island_generations': self.island_generations,
        'island_best': self.island_best,
        'best_id': self.best_id,
        'last_migration_generation': self.last_migration_generation,
        'last_iteration': self.last_iteration,
        'feature_stats': self.feature_stats,
    }
    with open(path, 'w', encoding='utf-8') as fh:
      json.dump(data, fh)

  @classmethod
  def load(cls, path: str) -> 'ProgramDatabase':
    with open(path, encoding='utf-8') as fh:
      data = json.load(fh)
    cfg = data['config']
    db = cls(
        num_islands=cfg['num_islands'],
        feature_dimensions=cfg['feature_dimensions'],
        feature_bins=cfg['feature_bins'],
        migration_interval=cfg['migration_interval'],
        migration_rate=cfg['migration_rate'],
        num_top_programs=cfg['num_top_programs'],
        num_diverse_programs=cfg['num_diverse_programs'],
        population_size=cfg['population_size'],
    )
    db.programs = {pid: Program(**d) for pid, d in data['programs'].items()}
    db.islands = [set(s) for s in data['islands']]
    db.grids = [dict(g) for g in data['grids']]
    db.island_generations = list(data['island_generations'])
    db.island_best = list(data['island_best'])
    db.best_id = data['best_id']
    db.last_migration_generation = data['last_migration_generation']
    db.last_iteration = data['last_iteration']
    db.feature_stats = data['feature_stats']
    return db
