"""Checks that policy evaluation uses the training observation contract."""

from absl.testing import absltest

from gfootball.env.puffer_env import FootballPufferEnv
from gfootball.examples import evaluate_puffer_policy


class EvaluatePufferPolicyTest(absltest.TestCase):

  def test_evaluation_uses_active_normalized_egocentric_observations(self):
    env = FootballPufferEnv(seed=7, frame_stack=4)
    try:
      observations, _ = env.reset()
      max_observation, max_ego_position, _ = (
          evaluate_puffer_policy._observation_contract(
              observations, env._active_mask))
      self.assertEqual(observations.shape, (22, 460))
      self.assertEqual(env._active_mask.sum(), 2)
      self.assertGreater(max_observation, 0)
      self.assertLessEqual(max_ego_position, 1e-6)
    finally:
      env.close()


if __name__ == '__main__':
  absltest.main()
