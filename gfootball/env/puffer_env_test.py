"""Smoke test for the native PufferLib interface."""

import math

from absl.testing import absltest
import numpy as np

from gfootball.env import puffer_env
from gfootball.env import config


class PufferEnvTest(absltest.TestCase):

  def test_reset_and_step_use_fixed_buffers(self):
    env = puffer_env.FootballPufferEnv(
        env_name='tests.symmetric', seed=7, frame_stack=4)
    try:
      observations, infos = env.reset()
      self.assertEqual(observations.shape, (22, 460))
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

  def test_curriculum_expands_to_standard_kickoff(self):
    cfg = config.Config({
        'level': '11_vs_11_curriculum',
        'curriculum_level': 0,
        'curriculum_levels': 11,
        'game_engine_random_seed': 7,
        'players': ['agent:left_players=11,right_players=11'],
    })
    initial = cfg.ScenarioConfig()
    self.assertLen(initial.left_team, 11)
    self.assertLen(initial.right_team, 11)
    self.assertGreater(abs(initial.ball_position[0]), 0.7)
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
    self.assertLess(max(attacker_distances), 0.25)
    self.assertEqual(sum(distance < 0.25 for distance in defender_distances), 1)
    self.assertGreater(np.median(defender_distances), 0.4)
    cfg['curriculum_level'] = 5
    cfg.NewScenario(0)
    middle = cfg.ScenarioConfig()
    self.assertLess(abs(middle.ball_position[0]), abs(initial.ball_position[0]))
    self.assertGreater(abs(middle.ball_position[0]), 0.3)
    self.assertEqual(middle.game_duration, 1800)
    cfg['curriculum_level'] = 10
    cfg.NewScenario(0)
    mature = cfg.ScenarioConfig()
    self.assertAlmostEqual(mature.ball_position[0], 0.0)
    self.assertEqual(mature.game_duration, 3000)

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
