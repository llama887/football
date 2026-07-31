"""Smoke test for the native PufferLib interface."""

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
        'curriculum_episodes': 256,
        'game_engine_random_seed': 7,
        'players': ['agent:left_players=11,right_players=11'],
    })
    initial = cfg.ScenarioConfig()
    self.assertLen(initial.left_team, 11)
    self.assertLen(initial.right_team, 11)
    self.assertGreater(abs(initial.ball_position[0]), 0.7)
    cfg['episode_number'] = 256
    cfg.NewScenario(0)
    mature = cfg.ScenarioConfig()
    self.assertAlmostEqual(mature.ball_position[0], 0.0)
    self.assertEqual(mature.game_duration, 3000)

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
