"""Smoke test for the native PufferLib interface."""

import math

from absl.testing import absltest
import numpy as np

from gfootball.env import puffer_env
from gfootball.env import config
from gfootball.curriculum import (
    ATTACKER_ONLY_LEVELS, TOTAL_LEVELS, curriculum_episode, curriculum_state)


class PufferEnvTest(absltest.TestCase):

  def test_observations_are_normalized_and_egocentric(self):
    observations = np.zeros((1, 115), dtype=np.float32)
    own_positions = observations[:, :22].reshape(1, 11, 2)
    own_directions = observations[:, 22:44].reshape(1, 11, 2)
    opponent_positions = observations[:, 44:66].reshape(1, 11, 2)
    own_positions[0, 1] = (0.5, 0.21)
    own_positions[0, 2] = (-0.5, -0.21)
    own_directions[0, 1] = (0.02, -0.01)
    opponent_positions[0, 0] = (-1, -0.42)
    opponent_positions[0, 10] = -1
    observations[0, 88:91] = (1, 0.42, 3)
    observations[0, 91:94] = (0.04, -0.02, 2)
    observations[0, 98] = 1

    puffer_env.normalize_egocentric(observations)

    np.testing.assert_array_equal(own_positions[0, 1], (0, 0))
    np.testing.assert_allclose(
        own_positions[0, 2],
        np.array((-1, -0.42)) / puffer_env._RELATIVE_POSITION_MAX)
    np.testing.assert_allclose(own_directions[0, 1], (0, 0))
    np.testing.assert_allclose(
        observations[0, 88:90],
        np.array((0.5, 0.21)) / puffer_env._RELATIVE_POSITION_MAX)
    self.assertAlmostEqual(observations[0, 90], 3 / 5.5)
    np.testing.assert_array_equal(opponent_positions[0, 10], (-1, -1))
    self.assertGreaterEqual(observations.min(), -1)
    self.assertLessEqual(observations.max(), 1)

  def test_reset_and_step_use_fixed_buffers(self):
    env = puffer_env.FootballPufferEnv(
        env_name='tests.symmetric', seed=7, frame_stack=4)
    try:
      observations, infos = env.reset()
      self.assertEqual(observations.shape, (22, 460))
      self.assertGreaterEqual(observations.min(), -1)
      self.assertLessEqual(observations.max(), 1)
      frames = observations.reshape(22, 4, 115)
      active = frames[:, :, 97:108].argmax(axis=-1)
      own_positions = frames[:, :, :22].reshape(22, 4, 11, 2)
      rows, history = np.indices(active.shape)
      np.testing.assert_allclose(
          own_positions[rows, history, active], 0, atol=1e-6)
      np.testing.assert_array_equal(observations[:, :115],
                                    observations[:, 115:230])
      self.assertEqual(infos, [])
      observations, rewards, terminals, truncations, _ = env.step(
          np.zeros(22, dtype=np.int32))
      self.assertEqual(observations.shape, (22, 460))
      self.assertEqual(rewards.shape, (22,))
      self.assertEqual(terminals.shape, (22,))
      self.assertEqual(truncations.shape, (22,))
    finally:
      env.close()

  def test_curriculum_adds_players_before_moving_from_goal(self):
    cfg = config.Config({
        'level': '11_vs_11_curriculum',
        'curriculum_level': 0,
        'curriculum_levels': TOTAL_LEVELS,
        'game_engine_random_seed': 7,
        'players': ['agent:left_players=11,right_players=11'],
    })
    initial = cfg.ScenarioConfig()
    self.assertFalse(initial.use_magnet)
    self.assertLen(initial.left_team, 11)
    self.assertLen(initial.right_team, 11)
    self.assertGreater(abs(initial.ball_position[0]), 0.65)
    self.assertEqual(initial.game_duration, 119)
    ball = tuple(initial.ball_position[i] for i in range(2))
    if ball[0] > 0:
      attackers, attacker_side = initial.left_team[1:], 1
      defenders, defender_side = initial.right_team[1:], -1
    else:
      attackers, attacker_side = initial.right_team[1:], -1
      defenders, defender_side = initial.left_team[1:], 1
    attacker_distances = [
        math.hypot(attacker_side * player.position[0] - ball[0],
                   attacker_side * player.position[1] - ball[1])
        for player in attackers]
    defender_distances = [
        math.hypot(defender_side * player.position[0] - ball[0],
                   defender_side * player.position[1] - ball[1])
        for player in defenders]
    expected_attackers = curriculum_episode(0, 7, 1)[0]
    self.assertEqual(
        sum(distance < 0.25 for distance in attacker_distances),
        expected_attackers)
    self.assertEqual(sum(distance < 0.25 for distance in defender_distances), 0)
    self.assertGreater(np.median(defender_distances), 0.4)
    cfg['curriculum_level'] = 12
    cfg.NewScenario(0)
    all_attackers = cfg.ScenarioConfig()
    self.assertAlmostEqual(abs(all_attackers.ball_position[0]), 0.90)
    self.assertEqual(all_attackers.game_duration, 599)
    cfg['curriculum_level'] = 22
    cfg.NewScenario(0)
    full_near_goal = cfg.ScenarioConfig()
    self.assertAlmostEqual(abs(full_near_goal.ball_position[0]), 0.90)
    self.assertEqual(full_near_goal.game_duration, 599)
    cfg['curriculum_level'] = 27
    cfg.NewScenario(0)
    middle = cfg.ScenarioConfig()
    self.assertAlmostEqual(abs(middle.ball_position[0]), 0.45)
    self.assertEqual(middle.game_duration, 1799)
    cfg['curriculum_level'] = 32
    cfg.NewScenario(0)
    mature = cfg.ScenarioConfig()
    self.assertAlmostEqual(mature.ball_position[0], 0.0)
    self.assertEqual(mature.game_duration, 3000)

  def test_curriculum_player_counts(self):
    self.assertEqual(curriculum_state(0), (1, 0, 0.0))
    self.assertEqual(curriculum_state(3), (2, 0, 0.0))
    self.assertEqual(curriculum_state(12), (11, 0, 0.0))
    self.assertEqual(curriculum_state(22), (11, 10, 0.0))
    self.assertEqual(curriculum_state(27), (11, 10, 0.5))
    self.assertEqual(curriculum_state(32), (11, 10, 1.0))

  def test_first_transition_gradually_mixes_two_attackers(self):
    for level, expected in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
      attackers = [
          curriculum_episode(level, 7, episode)[0]
          for episode in range(1000)
      ]
      self.assertAlmostEqual(
          sum(count == 2 for count in attackers) / len(attackers),
          expected, delta=0.04)

  def test_spawn_templates_exclude_goal_aligned_shortcut(self):
    cfg = config.Config({
        'level': '11_vs_11_curriculum',
        'curriculum_level': 0,
        'curriculum_levels': TOTAL_LEVELS,
        'game_engine_random_seed': 0,
        'players': ['agent:left_players=11,right_players=11'],
    })
    offsets = []
    gaps = []
    for episode in range(8):
      cfg.NewScenario(episode)
      scenario = cfg.ScenarioConfig()
      attack_right = scenario.ball_position[0] > 0
      team = scenario.left_team if attack_right else scenario.right_team
      side = 1 if attack_right else -1
      carrier = team[2].position
      offsets.append(side * carrier[1] - scenario.ball_position[1])
      gaps.append(side * (scenario.ball_position[0] - side * carrier[0]))
      self.assertGreaterEqual(np.hypot(gaps[-1], offsets[-1]), 0.024)
    self.assertLess(min(gaps), -0.02)
    self.assertGreater(max(gaps), 0.01)
    self.assertLess(max(gaps), 0.02)
    self.assertTrue(any(offset < 0 for offset in offsets))
    self.assertTrue(any(offset > 0 for offset in offsets))

  def test_inactive_curriculum_players_are_hidden_and_forced_idle(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    try:
      observations, _ = env.reset()
      expected_active = env._episode_attackers + 1
      self.assertEqual(env._active_mask.sum(), expected_active)
      self.assertEqual(np.any(observations, axis=1).sum(), expected_active)
      raw_env = env._env.unwrapped._env
      initial = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      for _ in range(5):
        env.step(np.full(22, 5, dtype=np.int32))
      after = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      movement = np.linalg.norm(after - initial, axis=1)
      self.assertLess(movement[~env._active_mask].max(), 0.002)
      self.assertGreater(movement[env._active_mask].max(), 0.002)
    finally:
      env.close()

  def test_attacker_only_levels_keep_physical_goalkeeper_out_of_learning(self):
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=7, attacker_only_levels=11)
    try:
      observations, _ = env.reset()
      self.assertEqual(env._active_mask.sum(), env._episode_attackers)
      self.assertEqual(
          np.any(observations, axis=1).sum(), env._episode_attackers)
      raw = env._env.unwrapped._env.observation()
      self.assertLen(raw['left_team'], 11)
      self.assertLen(raw['right_team'], 11)
      defending_goalkeeper = 0 if env._attacking_left else 11
      self.assertFalse(env._active_mask[defending_goalkeeper])
      env.step(np.zeros(22, dtype=np.int32))
    finally:
      env.close()

  def test_no_magnet_requires_direction_to_move(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    try:
      env.reset()
      raw_env = env._env.unwrapped._env
      initial = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      for _ in range(5):
        env.step(np.full(22, 10, dtype=np.int32))
      after_pass = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      self.assertLess(np.median(np.linalg.norm(
          after_pass - initial, axis=1)[env._active_mask]), 0.002)

      env.reset()
      initial = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      for _ in range(5):
        env.step(np.full(22, 5, dtype=np.int32))
      after_move = np.concatenate(
          [raw_env.observation()['left_team'],
           raw_env.observation()['right_team']])
      self.assertGreater(
          np.median(np.linalg.norm(
              after_move - initial, axis=1)[env._active_mask]), 0.002)
    finally:
      env.close()

  def test_first_curriculum_attempt_uses_short_credit_horizon(self):
    env = puffer_env.FootballPufferEnv(frame_stack=1, seed=7)
    try:
      env.reset()
      for episode_length in range(1, 602):
        _, _, terminals, _, _ = env.step(np.zeros(22, dtype=np.int32))
        if terminals.all():
          break
      self.assertEqual(episode_length, 120)
    finally:
      env.close()

  def test_curriculum_advances_only_after_mastery_window(self):
    env = puffer_env.FootballPufferEnv(
        env_name='tests.symmetric', curriculum_levels=3,
        curriculum_window=3, curriculum_success_threshold=2 / 3)
    try:
      self.assertFalse(env._record_curriculum_result(True)[1])
      self.assertFalse(env._record_curriculum_result(False)[1])
      success_rate, advanced = env._record_curriculum_result(True)
      self.assertTrue(advanced)
      self.assertAlmostEqual(success_rate, 2 / 3)
      self.assertEqual(env._curriculum_level, 1)
      self.assertEmpty(env._curriculum_results)
    finally:
      env.close()

  def test_shared_curriculum_moves_all_matches_together(self):
    from multiprocessing import RawValue
    level = RawValue('i', 0)
    env = puffer_env.FootballPufferEnv(
        frame_stack=1, seed=7, attacker_only_levels=ATTACKER_ONLY_LEVELS,
        curriculum_level_value=level)
    try:
      env.reset()
      level.value = 3
      env.reset()
      self.assertEqual(env._episode_level, 3)
      self.assertEqual(env._active_mask.sum(), 2)
      level.value = 12
      env.reset()
      self.assertEqual(env._episode_level, 12)
      self.assertEqual(env._active_mask.sum(), 11)
    finally:
      env.close()

  def test_two_matches_run_in_parallel(self):
    env = puffer_env.make_vector_env(
        num_envs=2, num_workers=2, batch_size=2, reserved_cpus=0,
        env_name='tests.symmetric', frame_stack=4)
    try:
      observations, _ = env.reset(seed=7)
      self.assertEqual(observations.shape, (44, 460))
      observations, rewards, terminals, truncations, _ = env.step(
          np.zeros(44, dtype=np.int32))
      self.assertEqual(observations.shape, (44, 460))
      self.assertEqual(rewards.shape, (44,))
      self.assertEqual(terminals.shape, (44,))
      self.assertEqual(truncations.shape, (44,))
    finally:
      env.close()


if __name__ == '__main__':
  absltest.main()
