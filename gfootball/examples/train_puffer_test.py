"""Focused checks for football policy-health logging."""

import math

import torch

from gfootball.examples.train_puffer import policy_diagnostics


def test_policy_diagnostics_distinguish_uniform_and_collapsed_policies():
  uniform = policy_diagnostics(torch.zeros(8, 19))
  collapsed = policy_diagnostics(torch.tensor([[20.0] + [0.0] * 18]))

  assert math.isclose(uniform['policy_entropy_fraction'].item(), 1.0,
                      rel_tol=1e-6)
  assert math.isclose(uniform['policy_max_probability'].item(), 1 / 19,
                      rel_tol=1e-6)
  assert collapsed['policy_entropy_fraction'].item() < 0.01
  assert collapsed['policy_max_probability'].item() > 0.99


if __name__ == '__main__':
  test_policy_diagnostics_distinguish_uniform_and_collapsed_policies()
