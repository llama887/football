"""Focused checks for football policy-health logging."""

import math
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

from gfootball.examples.train_puffer import (
    FootballPolicy, active_minibatches,
    clip_optimizer_groups, configure_optimizer_groups,
    critic_diagnostics, generalized_advantages, gradient_comparison,
    gradient_norm, normalize_advantages, policy_diagnostics,
    promotion_passes, promotion_statistics,
    valid_minibatch_size, validate_agent_rows)


def test_minibatch_size_is_valid_for_any_worker_count():
  assert valid_minibatch_size(660, 320) == 10560
  assert valid_minibatch_size(308, 320) == 4800


def test_masked_agents_do_not_inflate_ppo_updates():
  active_transitions = 14 * 2 * 320
  assert active_minibatches(active_transitions, 4800, 2) == 4
  assert 2 <= 4 * 4800 / active_transitions < 2.2


def test_gae_uses_next_step_rewards_and_respects_episode_boundaries():
  values = torch.zeros(1, 8)
  rewards = torch.zeros(1, 8)
  rewards[0, 3] = 1
  rewards[0, 6] = 1
  terminals = torch.zeros(1, 8)
  terminals[0, 3] = 1
  terminals[0, 6] = 1

  advantages, returns, valid = generalized_advantages(
      values, rewards, terminals, gamma=0.5, gae_lambda=1)

  expected = [[0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 0.0, 0.0]]
  assert advantages.tolist() == expected
  assert returns.tolist() == expected
  assert valid.tolist() == [[True, True, True, True, True, True, True, False]]
  normalized = normalize_advantages(advantages[valid])
  assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
  assert torch.isclose(
      normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_replay_rows_match_controlled_player_identity():
  observations = torch.zeros(22, 2, 115)
  for row in range(22):
    observations[row, :, 97 + row % 11] = 1
  validate_agent_rows(observations)

  observations[3, 1, 100] = 0
  observations[3, 1, 101] = 1
  try:
    validate_agent_rows(observations)
  except RuntimeError:
    pass
  else:
    raise AssertionError('misaligned replay row was accepted')


def test_critic_diagnostics_are_exact_for_a_perfect_fit():
  metrics = critic_diagnostics(
      torch.tensor([0.0, 0.5, 1.0]), torch.tensor([0.0, 0.5, 1.0]))
  assert metrics['critic_mse'].item() == 0
  assert metrics['critic_residual_mean'].item() == 0
  assert metrics['critic_explained_variance'].item() == 1
  assert math.isclose(
      metrics['critic_positive_target_fraction'].item(), 2 / 3,
      rel_tol=1e-6)
  assert metrics['critic_prediction_on_positive_targets'].item() == 0.75
  assert metrics['critic_prediction_on_zero_targets'].item() == 0


def test_actor_and_critic_optimizer_groups_use_independent_rates():
  actor = torch.nn.Parameter(torch.ones(1))
  critic = torch.nn.Parameter(torch.ones(1))
  optimizer = torch.optim.Adam((actor, critic), lr=8e-5)
  configure_optimizer_groups(optimizer, (actor,), (critic,), 1e-5)
  assert optimizer.param_groups[0]['params'] == [actor]
  assert optimizer.param_groups[0]['lr'] == 8e-5
  assert optimizer.param_groups[1]['params'] == [critic]
  assert optimizer.param_groups[1]['lr'] == 1e-5


def test_actor_and_critic_gradients_are_clipped_independently():
  actor = torch.nn.Parameter(torch.zeros(1))
  critic = torch.nn.Parameter(torch.zeros(1))
  actor.grad = torch.tensor([0.25])
  critic.grad = torch.tensor([100.0])
  optimizer = torch.optim.Adam((actor, critic), lr=8e-5)
  configure_optimizer_groups(optimizer, (actor,), (critic,), 1e-5)

  norms = clip_optimizer_groups(optimizer, max_norm=0.5)

  assert torch.isclose(norms[0][0], torch.tensor(0.25))
  assert torch.isclose(norms[0][1], torch.tensor(0.25))
  assert torch.isclose(actor.grad, torch.tensor([0.25])).all()
  assert torch.isclose(norms[1][0], torch.tensor(100.0))
  assert torch.isclose(norms[1][1], torch.tensor(0.5))


def test_gradient_norm_does_not_consume_graph():
  parameter = torch.tensor([3.0, 4.0], requires_grad=True)
  loss = parameter.square().sum()
  assert gradient_norm(loss, (parameter,)).item() == 10
  loss.backward()
  assert parameter.grad.tolist() == [6.0, 8.0]


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


def test_critic_stays_float32_with_finite_gradients_under_autocast():
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  policy = FootballPolicy(env)
  with torch.autocast('cpu', dtype=torch.bfloat16):
    _, values = policy(torch.ones(2, 460))
  values.sum().backward()
  gradients = [
      parameter.grad for parameter in policy.critic_parameters()
      if parameter.grad is not None]
  assert values.dtype == torch.float32
  assert gradients
  assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_inactive_zero_observation_has_no_policy_or_value_gradient():
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  policy = FootballPolicy(env)
  logits, values = policy(torch.zeros(2, 460))
  assert torch.count_nonzero(logits) == 0
  assert torch.count_nonzero(values) == 0


def test_actor_and_critic_have_disjoint_parameters_and_gradients():
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  policy = FootballPolicy(env, hidden_size=32)
  actor_parameters = policy.actor_parameters()
  critic_parameters = policy.critic_parameters()
  assert not ({id(parameter) for parameter in actor_parameters} &
              {id(parameter) for parameter in critic_parameters})

  logits, values = policy(torch.randn(8, 460))
  actor_norm, critic_norm, cosine = gradient_comparison(
      logits[:, 0].mean(), values.square().mean(), policy.parameters())
  assert actor_norm.item() > 0
  assert critic_norm.item() > 0
  assert abs(cosine.item()) < 1e-8


def test_legacy_shared_encoder_checkpoint_initializes_split_critic():
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  source = FootballPolicy(env, hidden_size=32)
  legacy = {
      name: value for name, value in source.state_dict().items()
      if not name.startswith('critic_')
  }
  restored = FootballPolicy(env, hidden_size=32)
  restored.load_state_dict(legacy)
  assert all(torch.equal(actor, critic) for actor, critic in zip(
      restored.frame_encoder.parameters(),
      restored.critic_frame_encoder.parameters()))
  assert all(torch.equal(actor, critic) for actor, critic in zip(
      restored.encoder.parameters(), restored.critic_encoder.parameters()))


def test_rewardless_actor_skip_allows_critic_update_without_changing_actor():
  torch.manual_seed(0)
  env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(19))
  policy = FootballPolicy(env, hidden_size=32)
  observations = torch.randn(32, 460)
  targets = (observations[:, 0] > 0).float()
  actor_before = [parameter.detach().clone()
                  for parameter in policy.actor_parameters()]
  optimizer = torch.optim.Adam(policy.parameters(), lr=8e-5)
  configure_optimizer_groups(
      optimizer, policy.actor_parameters(), policy.critic_parameters(), 1e-3)
  with torch.no_grad():
    _, values = policy(observations)
    before = (values - targets).square().mean().item()
  for _ in range(5):
    logits, values = policy(observations)
    actor_loss = logits[:, 0].mean() * 0
    loss = actor_loss + (values - targets).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  with torch.no_grad():
    _, values = policy(observations)
    after = (values - targets).square().mean().item()
  assert after < before
  assert all(torch.equal(before_parameter, after_parameter)
             for before_parameter, after_parameter in zip(
                 actor_before, policy.actor_parameters()))


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
  test_minibatch_size_is_valid_for_any_worker_count()
  test_masked_agents_do_not_inflate_ppo_updates()
  test_gae_uses_next_step_rewards_and_respects_episode_boundaries()
  test_replay_rows_match_controlled_player_identity()
  test_critic_diagnostics_are_exact_for_a_perfect_fit()
  test_actor_and_critic_optimizer_groups_use_independent_rates()
  test_actor_and_critic_gradients_are_clipped_independently()
  test_gradient_norm_does_not_consume_graph()
  test_policy_diagnostics_distinguish_uniform_and_collapsed_policies()
  test_actor_logits_stay_centered_float32_under_autocast()
  test_critic_stays_float32_with_finite_gradients_under_autocast()
  test_inactive_zero_observation_has_no_policy_or_value_gradient()
  test_actor_and_critic_have_disjoint_parameters_and_gradients()
  test_legacy_shared_encoder_checkpoint_initializes_split_critic()
  test_rewardless_actor_skip_allows_critic_update_without_changing_actor()
  test_promotion_requires_overall_and_every_heldout_template()
