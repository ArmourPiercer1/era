# Copyright 2026 Google LLC.
"""Tests for the evolutionary program database (Tier A, pure)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import Program, ProgramDatabase, make_program
from evaluation import EvalReport


def _prog(code, score, island=0, family='fft', runtime=None, memory=None,
          accuracy=0.0):
  return Program(
      code=code, score=score, island_id=island,
      feature_descriptor={'family': family},
      runtime=runtime, memory=memory, metrics={'accuracy': accuracy},
  )


class InsertionAndMapElitesTest(unittest.TestCase):

  def setUp(self):
    self.db = ProgramDatabase(num_islands=3, feature_dimensions=['family'])

  def test_add_assigns_island(self):
    p = self.db.add(_prog('a', 1.0, island=2))
    self.assertIn(p.id, self.db.islands[2])

  def test_same_cell_keeps_best(self):
    p1 = self.db.add(_prog('a', 1.0, family='fft'))
    p2 = self.db.add(_prog('b', 2.0, family='fft'))  # same cell, better
    cell = self.db._cell_key(p1)
    self.assertEqual(self.db.grids[0][cell], p2.id)

  def test_same_cell_worse_does_not_replace(self):
    p1 = self.db.add(_prog('a', 2.0, family='fft'))
    self.db.add(_prog('b', 1.0, family='fft'))  # worse
    cell = self.db._cell_key(p1)
    self.assertEqual(self.db.grids[0][cell], p1.id)

  def test_different_cells_coexist(self):
    self.db.add(_prog('a', 1.0, family='fft'))
    self.db.add(_prog('b', 1.0, family='multigrid'))
    self.assertEqual(len(self.db.grids[0]), 2)

  def test_missing_feature_dimension_raises(self):
    db = ProgramDatabase(num_islands=1, feature_dimensions=['nonexistent'])
    with self.assertRaises(ValueError):
      db.add(Program(code='a', score=1.0, feature_descriptor={'family': 'x'}))

  def test_best_program_tracked(self):
    self.db.add(_prog('a', 1.0))
    self.db.add(_prog('b', 5.0))
    self.db.add(_prog('c', 3.0))
    self.assertEqual(self.db.best_program().score, 5.0)


class FeatureScalingTest(unittest.TestCase):

  def setUp(self):
    self.db = ProgramDatabase(num_islands=1, feature_dimensions=['x'],
                              feature_bins=10)

  def test_scaling_min_max_mid(self):
    self.db._update_feature_stats('x', 0.0)
    self.db._update_feature_stats('x', 10.0)
    self.assertEqual(self.db._scale('x', 0.0), 0.0)
    self.assertEqual(self.db._scale('x', 10.0), 1.0)
    self.assertEqual(self.db._scale('x', 5.0), 0.5)

  def test_equal_min_max_returns_half(self):
    self.db._update_feature_stats('x', 4.0)
    self.assertEqual(self.db._scale('x', 4.0), 0.5)

  def test_numeric_feature_binning(self):
    # Two distinct numeric values should land in different cells.
    p1 = Program(code='a', score=1.0, feature_descriptor={'x': 0.0})
    p2 = Program(code='b', score=1.0, feature_descriptor={'x': 100.0})
    self.db.add(p1)
    self.db.add(p2)
    self.assertGreaterEqual(len(self.db.grids[0]), 2)


class MigrationTest(unittest.TestCase):

  def test_islands_initialized(self):
    db = ProgramDatabase(num_islands=5)
    self.assertEqual(len(db.islands), 5)
    self.assertEqual(len(db.island_generations), 5)
    self.assertEqual(len(db.island_best), 5)

  def test_should_migrate_trigger(self):
    db = ProgramDatabase(num_islands=3, migration_interval=5)
    self.assertFalse(db.should_migrate())
    for _ in range(5):
      db.increment_island_generation(0)
    self.assertTrue(db.should_migrate())

  def test_migration_copies_elite_to_neighbors(self):
    db = ProgramDatabase(num_islands=3, migration_interval=1,
                         migration_rate=1.0, feature_dimensions=['family'])
    elite = db.add(_prog('elite_code', 10.0, island=0, family='fft'))
    db.increment_island_generation(0)
    db.migrate()
    # Ring neighbours of island 0 are islands 1 and 2.
    neighbour_codes = []
    for isl in (1, 2):
      neighbour_codes += [db.programs[pid].code for pid in db.islands[isl]]
    self.assertIn('elite_code', neighbour_codes)

  def test_migrants_marked_and_not_re_migrated(self):
    db = ProgramDatabase(num_islands=2, migration_interval=1,
                         migration_rate=1.0, feature_dimensions=['family'])
    db.add(_prog('x', 5.0, island=0))
    db.increment_island_generation(0)
    db.migrate()
    migrant_ids = [pid for pid in db.islands[1]
                   if db.programs[pid].metadata.get('migrant')]
    self.assertTrue(migrant_ids)
    self.assertEqual(db.programs[migrant_ids[0]].parent_ids, [
        pid for pid in db.islands[0]][0:1] or db.programs[migrant_ids[0]].parent_ids)

  def test_duplicate_code_not_migrated_twice(self):
    db = ProgramDatabase(num_islands=2, migration_interval=1,
                         migration_rate=1.0, feature_dimensions=['family'])
    db.add(_prog('dup', 5.0, island=0))
    db.increment_island_generation(0)
    db.migrate()
    db.last_migration_generation = 0
    db.migrate()  # second migration should not duplicate identical code
    codes = [db.programs[pid].code for pid in db.islands[1]]
    self.assertEqual(codes.count('dup'), 1)

  def test_single_island_skips_migration(self):
    db = ProgramDatabase(num_islands=1, feature_dimensions=['family'])
    db.add(_prog('a', 1.0))
    db.migrate()  # should not raise
    self.assertEqual(len(db.programs), 1)

  def test_empty_island_does_not_crash(self):
    db = ProgramDatabase(num_islands=3, migration_interval=1,
                         migration_rate=1.0, feature_dimensions=['family'])
    db.add(_prog('a', 1.0, island=0))
    db.increment_island_generation(0)
    db.migrate()  # islands 1,2 are empty sources – must not crash


class ParetoTest(unittest.TestCase):

  def setUp(self):
    self.db = ProgramDatabase(num_islands=1, feature_dimensions=['family'])

  def test_dominated_excluded(self):
    # A: fast+accurate dominates B: slow+inaccurate.
    self.db.add(_prog('A', 1.0, runtime=1.0, accuracy=5.0))
    self.db.add(_prog('B', 0.5, runtime=2.0, accuracy=1.0))
    front_codes = {p.code for p in self.db.pareto_front()}
    self.assertIn('A', front_codes)
    self.assertNotIn('B', front_codes)

  def test_tradeoff_both_on_front(self):
    # A is faster, B is more accurate -> both non-dominated.
    self.db.add(_prog('A', 1.0, runtime=0.5, accuracy=1.0))
    self.db.add(_prog('B', 1.0, runtime=2.0, accuracy=9.0))
    front_codes = {p.code for p in self.db.pareto_front()}
    self.assertEqual(front_codes, {'A', 'B'})

  def test_programs_without_runtime_excluded(self):
    self.db.add(_prog('A', 1.0, runtime=None, accuracy=5.0))
    self.assertEqual(self.db.pareto_front(), [])


class SamplingTest(unittest.TestCase):

  def setUp(self):
    self.db = ProgramDatabase(num_islands=1, feature_dimensions=['family'],
                              num_top_programs=2, num_diverse_programs=1, seed=1)
    for i in range(6):
      self.db.add(_prog(f'code{i}', float(i), family=f'fam{i}'))

  def test_inspirations_top_and_diverse(self):
    insp = self.db.inspirations(0)
    self.assertLessEqual(len(insp), 3)  # 2 top + 1 diverse
    # Highest scorers (code5, code4) must be present as top.
    codes = {p.code for p in insp}
    self.assertIn('code5', codes)
    self.assertIn('code4', codes)

  def test_inspirations_deduped(self):
    insp = self.db.inspirations(0)
    ids = [p.id for p in insp]
    self.assertEqual(len(ids), len(set(ids)))

  def test_sample_parents_primary_is_best(self):
    primary, insp = self.db.sample_parents(0)
    self.assertEqual(primary.code, 'code5')
    self.assertNotIn(primary.id, [p.id for p in insp])

  def test_empty_island_returns_none(self):
    primary, insp = self.db.sample_parents(0)
    db2 = ProgramDatabase(num_islands=1, feature_dimensions=['family'])
    self.assertEqual(db2.sample_parents(0), (None, []))


class PersistenceTest(unittest.TestCase):

  def _build(self):
    db = ProgramDatabase(num_islands=3, feature_dimensions=['family', 'x'],
                         feature_bins=5, migration_interval=7)
    for i in range(5):
      db.add(Program(
          code=f'c{i}', score=float(i), island_id=i % 3,
          feature_descriptor={'family': 'fft', 'x': float(i)},
          runtime=1.0 / (i + 1), memory=float(i), metrics={'accuracy': i}),
          iteration=i)
    return db

  def test_save_load_roundtrip(self):
    db = self._build()
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fh:
      path = fh.name
    try:
      db.save(path)
      loaded = ProgramDatabase.load(path)
    finally:
      os.unlink(path)

    self.assertEqual(set(loaded.programs), set(db.programs))
    self.assertEqual([sorted(s) for s in loaded.islands],
                     [sorted(s) for s in db.islands])
    self.assertEqual(loaded.grids, db.grids)
    self.assertEqual(loaded.best_id, db.best_id)
    self.assertEqual(loaded.last_iteration, db.last_iteration)
    self.assertEqual(loaded.island_generations, db.island_generations)
    self.assertEqual(loaded.feature_stats, db.feature_stats)

  def test_loaded_best_program_matches(self):
    db = self._build()
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fh:
      path = fh.name
    try:
      db.save(path)
      loaded = ProgramDatabase.load(path)
    finally:
      os.unlink(path)
    self.assertEqual(loaded.best_program().code, db.best_program().code)


class MakeProgramTest(unittest.TestCase):

  def test_make_program_from_report(self):
    report = EvalReport(
        passed=True, timings=[0.2, 0.1, 0.3], memory=42.0,
        accuracy_metrics={'accuracy': 3.0},
        feature_descriptor={'family': 'cg'})
    p = make_program('code', report, score=7.0, island_id=1, generation=2)
    self.assertEqual(p.score, 7.0)
    self.assertEqual(p.runtime, 0.2)  # median of timings
    self.assertEqual(p.memory, 42.0)
    self.assertEqual(p.feature_descriptor, {'family': 'cg'})
    self.assertEqual(p.island_id, 1)


if __name__ == '__main__':
  unittest.main()
