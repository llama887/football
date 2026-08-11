"""Focused checks for football policy-health logging."""

import math
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

from gfootball.examples.train_puffer import (
    FootballPolicy, active_minibatches, complete_episode_returns,
    gradient_norm, policy_diagnostics, policy_regularization_kls,
    priority_diagnostics, promotion_passes, promotion_statistics,
    sampleable_segments, valid_minibatch_size)


def test_inactive_segments_are_not_sampled_for_training():
  observations = torch.zeros(3, 4, 2)
  observations[1, 2, 0] = 1
  assert sampleable_segments(observations).tolist() == [False, True, False]


def test_minibatch_size_is_valid_for_any_worker_count():
  assert valid_minibatch_size(660, 320) == 10560
  assert valid_minibatch_size(308, 320) == 4800


def test_masked_agents_do_not_inflate_ppo_updates():
  active_transitions = 14 * 2 * 320
  assert active_minibatches(active_transitions, 4800, 2) == 4
  assert 2 <= 4 * 4800 / active_transitions < 2.2


def test_value_targets_use_only_complete_episode_outcomes():
  rewards = torch.zeros(1, 8)
  rewards[0, 3] = 1
  terminals = torch.zeros(1, 8)
  terminals[0, 3] = 1
  terminals[0, 6] = 1

  returns, valid = complete_episode_returns(rewards, terminals, gamma=0.5)

  assert returns.tolist() == [[0.25, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
  assert valid.tolist() == [[True, True, True, True, True, True,
                            False, False]]


def test_gradient_norm_does_not_consume_graph():
  parameter = torch.tensor([3.0, 4.0], requires_grad=True)
  loss = parameter.square().sum()
  assert gradient_norm(loss, (parameter,)).item() == 10
  loss.backward()
  assert parameter.grad.tolist() == [6.0, 8.0]


def test_priority_diagnostics_expose_goal_segment_oversampling():
  goal_segments = torch.tensor([True, True, False, False])
  sampleable = torch.ones(4, dtype=torch.bool)
  metrics = priority_diagnostics(
      torch.tensor([0.45, 0.45, 0.05, 0.05]),
      goal_segments, sampleable)

  assert math.isclose(
      metrics['goal_segment_fraction'].item(), 0.5, rel_tol=1e-6)
  assert math.isclose(
      metrics['goal_segment_priority_mass'].item(), 0.9, rel_tol=1e-6)
  assert math.isclose(
      metrics['goal_priority_amplification'].item(), 1.8, rel_tol=1e-6)
  assert metrics['priority_ess_fraction'].item() < 0.61
  assert math.isclose(
      metrics['priority_top_10pct_mass'].item(), 0.45, rel_tol=1e-6)

  uniform = priority_diagnostics(
      torch.full((4,), 0.25), goal_segments, sampleable)
  assert uniform['priority_ess_fraction'].item() == 1.0
  assert uniform['goal_segment_priority_mass'].item() == 0.5
  assert uniform['goal_priority_amplification'].item() == 1.0


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
  assert any(isinstance(layer, torch.nn.LayerNorm)
             for layer in policy.encoder)
  with torch.autocast('cpu', dtype=torch.bfloat16):
    logits, _ = policy(torch.zeros(2, 460))
  assert logits.dtype == torch.float32
  assert torch.allclose(logits.mean(-1), torch.zeros(2), atol=1e-6)


def test_inactive_zero_observation_has_no_policy_or_value_gradient():
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  policy = FootballPolicy(env)
  logits, values = policy(torch.zeros(2, 460))
  assert torch.count_nonzero(logits) == 0
  assert torch.count_nonzero(values) == 0


def test_regularization_retains_gradient_at_policy_collapse():
  logits = torch.tensor([[20.0] + [0.0] * 18], requires_grad=True)
  past_kl, uniform_kl = policy_regularization_kls(
      logits, torch.zeros_like(logits))
  (past_kl + uniform_kl).backward()

  assert past_kl.item() > 10
  assert uniform_kl.item() > 10
  assert logits.grad[0, 0].item() > 1


def test_promotion_requires_overall_and_every_heldout_template():
  episodes = []
  for template in range(8):
    episodes.extend({
        'curriculum_template': template,
        'curriculum_success': float(success),
    } for success in ([1] * 7 + [0] * 3))
  metrics = promotion_statistics(episodes)
  assert promotion_passes(metrics, 0.6, 0.4)
  assert metrics['promotion_template_0_success_rate'] == 0.7
  assert metrics['promotion_template_7_success_rate'] == 0.7

  episodes[-10:] = ({
      'curriculum_template': 7,
      'curriculum_success': 0.0,
  } for _ in range(10))
  metrics = promotion_statistics(episodes)
  assert not promotion_passes(metrics, 0.6, 0.4)
  assert metrics['promotion_template_7_success_rate'] == 0.0


if __name__ == '__main__':
  test_inactive_segments_are_not_sampled_for_training()
  test_minibatch_size_is_valid_for_any_worker_count()
  test_masked_agents_do_not_inflate_ppo_updates()
  test_value_targets_use_only_complete_episode_outcomes()
  test_gradient_norm_does_not_consume_graph()
  test_priority_diagnostics_expose_goal_segment_oversampling()
  test_policy_diagnostics_distinguish_uniform_and_collapsed_policies()
  test_actor_logits_stay_centered_float32_under_autocast()
  test_inactive_zero_observation_has_no_policy_or_value_gradient()
  test_regularization_retains_gradient_at_policy_collapse()
  test_promotion_requires_overall_and_every_heldout_template()
