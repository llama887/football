"""Focused checks for football policy-health logging."""

import math
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

from gfootball.examples.train_puffer import FootballPolicy, policy_diagnostics


def test_policy_diagnostics_distinguish_uniform_and_collapsed_policies():
  uniform = policy_diagnostics(torch.zeros(8, 19))
  collapsed = policy_diagnostics(torch.tensor([[20.0] + [0.0] * 18]))

  assert math.isclose(uniform['policy_entropy_fraction'].item(), 1.0,
                      rel_tol=1e-6)
  assert math.isclose(uniform['policy_max_probability'].item(), 1 / 19,
                      rel_tol=1e-6)
  assert uniform['policy_max_abs_logit'].item() == 0
  assert collapsed['policy_entropy_fraction'].item() < 0.01
  assert collapsed['policy_max_abs_logit'].item() == 20
  assert collapsed['policy_max_probability'].item() > 0.99


def test_actor_logits_stay_centered_float32_under_autocast():
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  policy = FootballPolicy(env)
  with torch.autocast('cpu', dtype=torch.bfloat16):
    logits, _ = policy(torch.zeros(2, 460))
  assert logits.dtype == torch.float32
  assert torch.allclose(logits.mean(-1), torch.zeros(2), atol=1e-6)


if __name__ == '__main__':
  test_policy_diagnostics_distinguish_uniform_and_collapsed_policies()
  test_actor_logits_stay_centered_float32_under_autocast()
