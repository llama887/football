"""Smoke test for the native PufferLib interface."""

from absl.testing import absltest
import numpy as np

from gfootball.env import puffer_env


class PufferEnvTest(absltest.TestCase):

  def test_reset_and_step_use_fixed_buffers(self):
    env = puffer_env.FootballPufferEnv(env_name='tests.symmetric', seed=7)
    try:
      observations, infos = env.reset()
      self.assertEqual(observations.shape, (22, 115))
      self.assertEqual(infos, [])
      observations, rewards, terminals, truncations, _ = env.step(
          np.zeros(22, dtype=np.int32))
      self.assertEqual(observations.shape, (22, 115))
      self.assertEqual(rewards.shape, (22,))
      self.assertEqual(terminals.shape, (22,))
      self.assertEqual(truncations.shape, (22,))
    finally:
      env.close()

  def test_two_matches_run_in_parallel(self):
    env = puffer_env.make_vector_env(
        num_envs=2, num_workers=2, batch_size=2, reserved_cpus=0,
        env_name='tests.symmetric')
    try:
      observations, _ = env.reset(seed=7)
      self.assertEqual(observations.shape, (44, 115))
      observations, rewards, terminals, truncations, _ = env.step(
          np.zeros(44, dtype=np.int32))
      self.assertEqual(observations.shape, (44, 115))
      self.assertEqual(rewards.shape, (44,))
      self.assertEqual(terminals.shape, (44,))
      self.assertEqual(truncations.shape, (44,))
    finally:
      env.close()


if __name__ == '__main__':
  absltest.main()
