"""Train regularized 22-player self-play with PufferLib."""

import argparse
import copy
from collections import defaultdict
import json
import math
import os
import sys
import time

import torch

import pufferlib
from pufferlib import pufferl
import pufferlib.pytorch

from gfootball.env.puffer_env import make_vector_env
from gfootball.env import football_action_set
from gfootball.curriculum import (
    ATTACKER_ONLY_LEVELS, SPAWN_TEMPLATE_COUNT, TOTAL_LEVELS)


ACTION_NAMES = tuple(
    str(action) for action in football_action_set.action_set_dict['default'])


def valid_minibatch_size(num_agents, horizon):
  return horizon * max(1, round(num_agents * 16 / horizon))


def active_minibatches(active_transitions, minibatch_size, update_epochs):
  """Number of PPO minibatches needed to reuse active data as requested."""
  if active_transitions < 1:
    raise ValueError('rollout contains no active transitions')
  return max(1, math.ceil(
      update_epochs * active_transitions / minibatch_size))


def generalized_advantages(values, rewards, terminals, gamma, gae_lambda):
  """Standard GAE for Puffer's next-step reward storage convention."""
  advantages = torch.zeros_like(rewards)
  running = torch.zeros(rewards.shape[0], device=rewards.device)
  for timestep in range(rewards.shape[1] - 2, -1, -1):
    next_timestep = timestep + 1
    next_active = ~terminals[:, next_timestep].bool()
    delta = (rewards[:, next_timestep] +
             gamma * values[:, next_timestep] * next_active -
             values[:, timestep])
    running = delta + gamma * gae_lambda * next_active * running
    advantages[:, timestep] = running
  valid = torch.ones_like(terminals, dtype=torch.bool)
  valid[:, -1] = False
  return advantages, advantages + values, valid


def normalize_advantages(advantages):
  advantages = advantages.float()
  return ((advantages - advantages.mean()) /
          advantages.std(unbiased=False).clamp_min(1e-8))


def validate_agent_rows(observations):
  """Fail if a replay row contains another controlled player's observation."""
  frames = observations.reshape(*observations.shape[:2], -1, 115)
  active = observations.flatten(2).abs().sum(dim=-1) > 0
  observed = frames[:, :, -1, 97:108].argmax(dim=-1)
  expected = (torch.arange(observations.shape[0], device=observations.device) %
              11)[:, None]
  if (active & (observed != expected)).any():
    raise RuntimeError('replay rows are not aligned with controlled players')


def critic_diagnostics(predictions, targets):
  """Summarize critic calibration against fixed, outcome-grounded targets."""
  predictions = predictions.float()
  targets = targets.float()
  residuals = predictions - targets
  target_variance = targets.var(unbiased=False)
  explained_variance = (
      predictions.new_tensor(float('nan')) if target_variance == 0 else
      1 - (targets - predictions).var(unbiased=False) / target_variance)
  positive = targets > 0
  zero = targets == 0
  nan = predictions.new_tensor(float('nan'))
  return {
      'critic_prediction_mean': predictions.mean(),
      'critic_prediction_std': predictions.std(unbiased=False),
      'critic_prediction_min': predictions.min(),
      'critic_prediction_max': predictions.max(),
      'critic_target_mean': targets.mean(),
      'critic_target_std': targets.std(unbiased=False),
      'critic_target_min': targets.min(),
      'critic_target_max': targets.max(),
      'critic_residual_mean': residuals.mean(),
      'critic_mse': residuals.square().mean(),
      'critic_explained_variance': explained_variance,
      'critic_positive_target_fraction': positive.float().mean(),
      'critic_prediction_on_positive_targets': (
          predictions[positive].mean() if positive.any() else nan),
      'critic_prediction_on_zero_targets': (
          predictions[zero].mean() if zero.any() else nan),
  }


def gradient_norm(loss, parameters):
  """L2 norm of one loss component's gradient without consuming its graph."""
  gradients = torch.autograd.grad(
      loss, tuple(parameters), retain_graph=True, allow_unused=True)
  squared_norm = loss.new_zeros((), dtype=torch.float32)
  for gradient in gradients:
    if gradient is not None:
      squared_norm += gradient.float().square().sum()
  return squared_norm.sqrt()


def gradient_comparison(actor_loss, critic_loss, parameters):
  """Norms and cosine over one fixed parameter ordering."""
  parameters = tuple(parameters)
  actor_gradients = torch.autograd.grad(
      actor_loss, parameters, retain_graph=True, allow_unused=True)
  critic_gradients = torch.autograd.grad(
      critic_loss, parameters, retain_graph=True, allow_unused=True)
  dot = actor_loss.new_zeros((), dtype=torch.float32)
  actor_squared = dot.clone()
  critic_squared = dot.clone()
  for parameter, actor_gradient, critic_gradient in zip(
      parameters, actor_gradients, critic_gradients):
    actor_gradient = (
        torch.zeros_like(parameter) if actor_gradient is None else
        actor_gradient).float()
    critic_gradient = (
        torch.zeros_like(parameter) if critic_gradient is None else
        critic_gradient).float()
    dot += (actor_gradient * critic_gradient).sum()
    actor_squared += actor_gradient.square().sum()
    critic_squared += critic_gradient.square().sum()
  actor_norm = actor_squared.sqrt()
  critic_norm = critic_squared.sqrt()
  denominator = actor_norm * critic_norm
  cosine = torch.where(denominator > 0, dot / denominator, dot)
  return actor_norm, critic_norm, cosine


def configure_optimizer_groups(optimizer, actor_parameters,
                               critic_parameters, critic_learning_rate):
  """Give disjoint actor/critic branches independent Adam step sizes."""
  if len(optimizer.param_groups) != 1 or optimizer.state:
    raise ValueError('optimizer must be fresh with one parameter group')
  actor_parameters = list(actor_parameters)
  critic_parameters = list(critic_parameters)
  expected = {id(parameter)
              for parameter in optimizer.param_groups[0]['params']}
  actual = ({id(parameter) for parameter in actor_parameters} |
            {id(parameter) for parameter in critic_parameters})
  if expected != actual:
    raise ValueError('actor and critic parameters must partition the optimizer')
  optimizer.param_groups[0]['params'] = actor_parameters
  optimizer.add_param_group({
      'params': critic_parameters,
      'lr': float(critic_learning_rate),
  })


def clip_optimizer_groups(optimizer, max_norm):
  """Clip disjoint actor/critic gradients without cross-group rescaling."""
  norms = []
  for group in optimizer.param_groups:
    parameters = group['params']
    before = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    squared = before.new_zeros((), dtype=torch.float32)
    for parameter in parameters:
      if parameter.grad is not None:
        squared += parameter.grad.float().square().sum()
    norms.append((before, squared.sqrt()))
  return tuple(norms)


def policy_diagnostics(logits):
  """Small policy-health signals that expose uniform or collapsed behavior."""
  probabilities = torch.softmax(logits.float(), dim=-1)
  top_two = probabilities.topk(2, dim=-1).values
  entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(-1)
  return {
      'policy_entropy_fraction': entropy.mean() / math.log(logits.shape[-1]),
      'policy_max_abs_logit': logits.float().abs().max(),
      'policy_max_probability': top_two[:, 0].mean(),
      'policy_probability_margin': (top_two[:, 0] - top_two[:, 1]).mean(),
  }


def promotion_statistics(episodes):
  """Summarize held-out success overall and for the weakest spawn."""
  by_template = defaultdict(list)
  for episode in episodes:
    by_template[int(episode['curriculum_template'])].append(
        float(episode['curriculum_success']))
  success_rate = sum(map(float, (
      episode['curriculum_success'] for episode in episodes))) / len(episodes)
  template_rates = {
      template: sum(values) / len(values)
      for template, values in by_template.items()
  }
  metrics = {
      'promotion_success_rate': success_rate,
      'promotion_worst_template_success_rate': (
          min(template_rates.values()) if template_rates else 0.0),
      'promotion_templates_covered': float(len(template_rates)),
  }
  metrics.update({
      'promotion_template_{}_success_rate'.format(template):
      template_rates.get(template, 0.0)
      for template in range(SPAWN_TEMPLATE_COUNT)
  })
  return metrics


def promotion_passes(metrics, success_threshold, worst_template_threshold):
  return (
      metrics['promotion_templates_covered'] == SPAWN_TEMPLATE_COUNT and
      metrics['promotion_success_rate'] >= success_threshold and
      metrics['promotion_worst_template_success_rate'] >=
      worst_template_threshold)


def evaluate_promotion(policy, vecenv, episodes, seed, device):
  """Run policy-only episodes on spawn templates excluded from training."""
  observations, _ = vecenv.reset(seed=seed)
  generator = torch.Generator(device=device).manual_seed(seed)
  rows = []
  action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
  active_logits_rows = []
  decisions = 0
  was_training = policy.training
  policy.eval()
  try:
    while len(rows) < episodes:
      observation_tensor = torch.as_tensor(observations, device=device)
      active = observation_tensor.flatten(1).abs().sum(dim=-1) > 0
      with torch.inference_mode():
        logits, _ = policy(observation_tensor)
        active_logits_rows.append(logits[active].float().cpu())
        actions = torch.multinomial(
            torch.softmax(logits.float(), dim=-1), 1,
            generator=generator).squeeze(-1)
      active_actions = actions[active].cpu()
      action_counts += torch.bincount(
          active_actions, minlength=len(ACTION_NAMES))
      decisions += active_actions.numel()
      observations, _, _, _, infos = vecenv.step(actions.cpu().numpy())
      for info in infos:
        if 'curriculum_success' in info and len(rows) < episodes:
          rows.append(info)
  finally:
    policy.train(was_training)
  metrics = promotion_statistics(rows)
  diagnostics = policy_diagnostics(torch.cat(active_logits_rows))
  metrics.update({
      'promotion_episodes': float(len(rows)),
      'promotion_mean_episode_length': sum(
          row['episode_length'] for row in rows) / len(rows),
      'promotion_max_action_fraction': (
          action_counts.max().item() / decisions),
      'promotion_shot_fraction': action_counts[12].item() / decisions,
      **{
          'promotion_{}'.format(name): value.item()
          for name, value in diagnostics.items()
      },
      **{
          'promotion_action_{}_fraction'.format(name):
          action_counts[index].item() / decisions
          for index, name in enumerate(ACTION_NAMES)
      },
  })
  return metrics


class FootballPolicy(torch.nn.Module):
  """Independent actor and critic over four simple115v2 frames."""

  is_continuous = False

  def __init__(self, env, hidden_size=512):
    super().__init__()
    observation_size = env.single_observation_space.shape[0]
    if observation_size % 115:
      raise ValueError('observation size must be a multiple of 115')
    self.frame_stack = observation_size // 115
    self.frame_encoder = torch.nn.Sequential(
        pufferlib.pytorch.layer_init(torch.nn.Linear(115, 256)),
        torch.nn.ReLU(),
    )
    self.encoder = torch.nn.Sequential(
        pufferlib.pytorch.layer_init(
            torch.nn.Linear(256 * self.frame_stack, hidden_size)),
        torch.nn.ReLU(),
        pufferlib.pytorch.layer_init(
            torch.nn.Linear(hidden_size, hidden_size)),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(hidden_size),
    )
    self.critic_frame_encoder = copy.deepcopy(self.frame_encoder)
    self.critic_encoder = copy.deepcopy(self.encoder)
    self.action_head = pufferlib.pytorch.layer_init(
        torch.nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
    self.value_head = pufferlib.pytorch.layer_init(
        torch.nn.Linear(hidden_size, 1), std=1.0)

  def forward(self, observations, _state=None):
    active = observations.flatten(1).abs().sum(dim=-1) > 0
    active_indices = active.nonzero().flatten()
    frames = observations[active].reshape(-1, self.frame_stack, 115)
    actor_hidden = self.frame_encoder(frames).flatten(1)
    actor_hidden = self.encoder(actor_hidden)
    with torch.autocast(device_type=frames.device.type, enabled=False):
      critic_hidden = self.critic_frame_encoder(frames.float()).flatten(1)
      critic_hidden = self.critic_encoder(critic_hidden)
      active_values = self.value_head(critic_hidden).squeeze(-1)
    with torch.autocast(device_type=actor_hidden.device.type, enabled=False):
      active_logits = self.action_head(actor_hidden.float())
      active_logits -= active_logits.mean(dim=-1, keepdim=True)
    logits = active_logits.new_zeros(
        (observations.shape[0], self.action_head.out_features)).index_copy(
            0, active_indices, active_logits)
    values = active_values.new_zeros(observations.shape[0]).index_copy(
        0, active_indices, active_values)
    return logits, values

  def actor_parameters(self):
    return tuple(self.frame_encoder.parameters()) + tuple(
        self.encoder.parameters()) + tuple(self.action_head.parameters())

  def critic_parameters(self):
    return tuple(self.critic_frame_encoder.parameters()) + tuple(
        self.critic_encoder.parameters()) + tuple(self.value_head.parameters())

  def load_state_dict(self, state_dict, strict=True, assign=False):
    """Load pre-split checkpoints by cloning their shared critic features."""
    if not any(name.startswith('critic_') for name in state_dict):
      state_dict = dict(state_dict)
      for name, value in tuple(state_dict.items()):
        if name.startswith('frame_encoder.'):
          state_dict['critic_' + name] = value
        elif name.startswith('encoder.'):
          state_dict['critic_' + name] = value
    return super().load_state_dict(state_dict, strict=strict, assign=assign)

  def forward_eval(self, observations, state=None):
    return self.forward(observations, state)


class FootballPuffeRL(pufferl.PuffeRL):
  """Plain PPO with disjoint actor and critic optimization."""

  def __init__(self, config, vecenv, policy, gradient_audit=False,
               critic_learning_rate=1e-5, logger=None):
    super().__init__(config, vecenv, policy, logger=logger)
    configure_optimizer_groups(
        self.optimizer, self.uncompiled_policy.actor_parameters(),
        self.uncompiled_policy.critic_parameters(), critic_learning_rate)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer, T_max=self.total_epochs)
    self.gradient_audit = bool(gradient_audit)
    self.optimizer_steps = 0
    self.active_steps = 0
    self.last_active_steps = 0
    self.last_active_log_time = time.time()
    self.promotion_metrics = {}

  def record_promotion(self, level, metrics, advanced):
    self.promotion_metrics = {
        **metrics,
        'promotion_level': float(level),
        'promotion_advanced': float(advanced),
    }

  @pufferl.record
  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile('train', epoch)
    losses = defaultdict(float)
    audit_metrics = {}
    config = self.config
    device = config['device']
    clip_coef = config['clip_coef']
    vf_clip = config['vf_clip_coef']
    self.ratio[:] = 1
    validate_agent_rows(self.observations)
    rollout_active = self.observations.flatten(2).abs().sum(dim=-1) > 0
    rollout_values = self.values.clone()
    advantages, critic_returns, critic_valid = generalized_advantages(
        rollout_values, self.rewards, self.terminals,
        config['gamma'], config['gae_lambda'])
    trainable = rollout_active & critic_valid
    active_transitions = int(trainable.sum().item())
    sampleable = trainable.any(dim=1)
    active_segments = int(sampleable.sum().item())
    num_minibatches = active_minibatches(
        active_transitions, self.minibatch_size, config['update_epochs'])
    target_active_transitions = config['update_epochs'] * active_transitions
    sampled_active_transitions = 0
    minibatch = 0
    critic_active = trainable
    critic_observations = self.observations[critic_active][:8192]
    critic_targets = critic_returns[critic_active][:8192]
    critic_predictions = rollout_values[critic_active][:8192]
    if critic_targets.numel():
      for name, value in critic_diagnostics(
          critic_predictions, critic_targets).items():
        audit_metrics['pre_update_{}'.format(name)] = value.item()

    while (minibatch < num_minibatches or
           sampled_active_transitions < target_active_transitions):
      if minibatch >= 4 * self.total_minibatches:
        raise RuntimeError('could not sample enough active transitions')
      profile('train_misc', epoch, nest=True)
      self.amp_context.__enter__()
      profile('train_copy', epoch)
      probabilities = sampleable.float() / active_segments
      indices = torch.multinomial(probabilities, self.minibatch_segments)
      observations = self.observations[indices]
      actions = self.actions[indices]
      old_logprobs = self.logprobs[indices]
      values = rollout_values[indices]
      returns = critic_returns[indices]
      valid_returns = critic_valid[indices]
      old_advantages = advantages[indices]

      profile('train_forward', epoch)
      observations = observations.reshape(
          -1, *self.vecenv.single_observation_space.shape)
      state = {'action': actions, 'lstm_h': None, 'lstm_c': None}
      logits, new_values = self.policy(observations, state)
      _, new_logprobs, entropy = pufferlib.pytorch.sample_logits(
          logits, action=actions)
      active = observations.flatten(1).abs().sum(dim=-1) > 0
      active_steps = active.view(old_logprobs.shape) & valid_returns
      sampled_active_transitions += int(active_steps.sum().item())

      profile('train_misc', epoch)
      new_logprobs = new_logprobs.reshape(old_logprobs.shape)
      log_ratio = new_logprobs - old_logprobs
      ratio = log_ratio.exp()
      active_ratio = ratio[active_steps]
      self.ratio[indices] = ratio.detach()
      with torch.no_grad():
        active_log_ratio = log_ratio[active_steps]
        old_approx_kl = (-active_log_ratio).mean()
        approx_kl = ((active_ratio - 1) - active_log_ratio).mean()
        clip_fraction = (
            (active_ratio - 1.0).abs() > clip_coef).float().mean()

      active_advantages = old_advantages[active_steps]
      normalized_advantages = normalize_advantages(active_advantages)
      policy_loss = torch.max(
          -normalized_advantages * active_ratio,
          -normalized_advantages * torch.clamp(
              active_ratio, 1 - clip_coef, 1 + clip_coef)).mean()

      new_values = new_values.view(returns.shape)
      critic_steps = active_steps
      clipped_values = values + torch.clamp(
          new_values - values, -vf_clip, vf_clip)
      if critic_steps.any():
        value_loss = 0.5 * torch.max(
            (new_values[critic_steps] - returns[critic_steps]) ** 2,
            (clipped_values[critic_steps] - returns[critic_steps]) ** 2).mean()
      else:
        value_loss = new_values.sum() * 0
      entropy_loss = entropy.view(active_steps.shape)[active_steps].mean()

      active_logits = logits[active]
      actor_loss = policy_loss - config['ent_coef'] * entropy_loss
      critic_loss = config['vf_coef'] * value_loss
      loss = actor_loss + critic_loss
      if minibatch == 0:
        actor_gradient, critic_gradient, gradient_cosine = gradient_comparison(
            actor_loss, critic_loss, self.uncompiled_policy.parameters())
        audit_metrics.update({
            'actor_gradient_norm': actor_gradient.item(),
            'critic_gradient_norm': critic_gradient.item(),
            'actor_critic_gradient_cosine': gradient_cosine.item(),
        })
      if self.gradient_audit:
        actor_parameters = tuple(self.uncompiled_policy.action_head.parameters())
        gradient_components = {
            'policy': policy_loss,
            'entropy': -config['ent_coef'] * entropy_loss,
        }
        for name, component in gradient_components.items():
          losses['actor_{}_gradient_norm'.format(name)] += gradient_norm(
              component, actor_parameters).item()
      self.amp_context.__enter__()

      profile('train_misc', epoch)
      metrics = {
          'policy_loss': policy_loss,
          'value_loss': value_loss,
          'entropy': entropy_loss,
          'old_approx_kl': old_approx_kl,
          'approx_kl': approx_kl,
          'clipfrac': clip_fraction,
          'importance': active_ratio.mean(),
          **policy_diagnostics(active_logits),
      }
      for name, value in metrics.items():
        losses[name] += value.item()
      profile('learn', epoch)
      loss.backward()
      if (minibatch + 1) % self.accumulate_minibatches == 0:
        group_norms = clip_optimizer_groups(
            self.optimizer, config['max_grad_norm'])
        losses['pre_clip_actor_gradient_norm'] += group_norms[0][0].item()
        losses['post_clip_actor_gradient_norm'] += group_norms[0][1].item()
        losses['pre_clip_critic_gradient_norm'] += group_norms[1][0].item()
        losses['post_clip_critic_gradient_norm'] += group_norms[1][1].item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.optimizer_steps += 1
      minibatch += 1

    num_minibatches = minibatch
    for name in tuple(losses):
      losses[name] /= num_minibatches
    losses.update(audit_metrics)

    active_actions = self.actions[rollout_active]
    action_fractions = []
    for action_index, action_name in enumerate(ACTION_NAMES):
      action_fraction = (
          (active_actions == action_index).float().mean().item())
      action_fractions.append(action_fraction)
      losses['action_{}_fraction'.format(action_name)] = action_fraction
    losses['action_head_weight_norm'] = (
        self.uncompiled_policy.action_head.weight.norm().item())
    losses['active_agent_fraction'] = rollout_active.float().mean().item()
    losses['active_transitions'] = float(active_transitions)
    losses['active_segments'] = float(active_segments)
    losses['ppo_minibatches'] = float(num_minibatches)
    losses['sampled_active_transitions'] = float(sampled_active_transitions)
    losses['advantage_mean'] = advantages[trainable].mean().item()
    losses['advantage_std'] = (
        advantages[trainable].std(unbiased=False).item())
    outcome_rewards = self.rewards[rollout_active]
    losses['positive_reward_fraction'] = (
        (outcome_rewards > 0).float().mean().item())
    losses['negative_reward_fraction'] = (
        (outcome_rewards < 0).float().mean().item())
    losses['effective_update_epochs'] = (
        sampled_active_transitions / active_transitions)
    losses['optimizer_steps'] = float(self.optimizer_steps)
    losses['actor_learning_rate'] = self.optimizer.param_groups[0]['lr']
    losses['critic_learning_rate'] = self.optimizer.param_groups[1]['lr']
    self.active_steps += active_transitions
    now = time.time()
    losses['active_SPS'] = (
        (self.active_steps - self.last_active_steps) /
        max(1e-6, now - self.last_active_log_time))
    self.last_active_steps = self.active_steps
    self.last_active_log_time = now

    frames = self.observations.reshape(
        *self.observations.shape[:2], self.uncompiled_policy.frame_stack, 115)
    controlled_players = frames[:, :, -1, 97:108].argmax(dim=-1)
    for role, role_mask in (
        ('goalkeeper', rollout_active & (controlled_players == 0)),
        ('field_player', rollout_active & (controlled_players != 0))):
      role_actions = self.actions[role_mask]
      if role_actions.numel():
        for action_index, action_name in enumerate(ACTION_NAMES):
          losses['{}_action_{}_fraction'.format(role, action_name)] = (
              (role_actions == action_index).float().mean().item())

    with torch.no_grad():
      if critic_targets.numel():
        _, post_critic_predictions = self.policy(critic_observations)
        for name, value in critic_diagnostics(
            post_critic_predictions, critic_targets).items():
          losses['post_update_{}'.format(name)] = value.item()
      post_observations = self.observations[rollout_active][:2048]
      post_logits, _ = self.policy(post_observations)
      post_diagnostics = policy_diagnostics(post_logits)
      for name, value in post_diagnostics.items():
        losses['post_update_{}'.format(name)] = value.item()
      post_actions = post_logits.argmax(dim=-1)
      post_max_action_fraction = max(
          (post_actions == action).float().mean().item()
          for action in range(len(ACTION_NAMES)))
      losses['post_update_max_action_fraction'] = post_max_action_fraction
    for name, parameter in self.uncompiled_policy.named_parameters():
      if name.endswith('weight'):
        metric_name = 'weight_norm_{}'.format(name.replace('.', '_'))
        losses[metric_name] = parameter.float().norm().item()
    losses['max_action_fraction'] = max(action_fractions)
    losses.update(self.promotion_metrics)

    profile('train_misc', epoch)
    if config['anneal_lr']:
      self.scheduler.step()
    predictions = rollout_values[critic_active]
    targets = critic_returns[critic_active]
    target_variance = targets.var(unbiased=False)
    losses['explained_variance'] = (
        float('nan') if target_variance == 0 else
        (1 - (targets - predictions).var(unbiased=False) /
         target_variance).item())

    profile.end()
    logs = None
    self.epoch += 1
    done_training = self.global_step >= config['total_timesteps']
    if (done_training or self.global_step == 0 or
        time.time() > self.last_log_time + 0.25):
      self.losses = losses
      logs = self.mean_and_log()
      self.print_dashboard()
      self.stats = defaultdict(list)
      self.last_log_time = time.time()
      self.last_log_step = self.global_step
      profile.clear()
    if self.epoch % config['checkpoint_interval'] == 0 or done_training:
      self.save_checkpoint()
    return logs


def _base_config():
  original_argv = sys.argv
  sys.argv = [original_argv[0]]
  try:
    return dict(pufferl.load_config('default')['train'])
  finally:
    sys.argv = original_argv


def _make_promotion_env(args, curriculum_level_value):
  return make_vector_env(
      num_envs=args.promotion_workers, num_workers=args.promotion_workers,
      batch_size=args.promotion_workers, reserved_cpus=0,
      seed=args.seed + 1000000, env_name='11_vs_11_curriculum',
      frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.promotion_episodes + 1,
      curriculum_success_threshold=args.curriculum_success_threshold,
      attacker_only_levels=args.attacker_only_levels,
      curriculum_level_value=curriculum_level_value,
      curriculum_evaluation=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-workers', type=int, default=30)
  parser.add_argument('--total-timesteps', type=int, default=1_000_000_000)
  parser.add_argument('--curriculum-levels', type=int, default=TOTAL_LEVELS)
  parser.add_argument('--curriculum-window', type=int, default=20)
  parser.add_argument('--curriculum-success-threshold', type=float, default=0.6)
  parser.add_argument('--attacker-only-levels', type=int,
                      default=ATTACKER_ONLY_LEVELS)
  parser.add_argument('--promotion-interval', type=int, default=50)
  parser.add_argument('--promotion-episodes', type=int, default=256)
  parser.add_argument('--promotion-workers', type=int, default=30)
  parser.add_argument('--promotion-worst-template-threshold', type=float,
                      default=0.4)
  parser.add_argument('--gradient-audit', action='store_true')
  parser.add_argument('--anneal-lr', action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument('--critic-learning-rate', type=float, default=1e-5)
  parser.add_argument('--learning-rate', type=float, default=8e-5)
  parser.add_argument('--ent-coef', type=float, default=0.01)
  parser.add_argument('--frame-stack', type=int, default=4, choices=(1, 4))
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
  parser.add_argument('--data-dir', default='experiments/football-ppo')
  parser.add_argument('--wandb', action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument('--wandb-project', default='google-football-fast-rl')
  parser.add_argument('--wandb-group', default='simple-marl-ppo')
  parser.add_argument('--wandb-tag', default=None)
  args = parser.parse_args()
  if args.critic_learning_rate <= 0 or args.learning_rate <= 0:
    raise ValueError('learning rates must be positive')
  if args.ent_coef < 0:
    raise ValueError('ent-coef must be nonnegative')
  if args.promotion_interval < 1 or args.promotion_episodes < 1:
    raise ValueError('promotion interval and episodes must be positive')
  if args.promotion_workers < 1:
    raise ValueError('promotion-workers must be positive')
  if not 0 <= args.promotion_worst_template_threshold <= 1:
    raise ValueError('promotion worst-template threshold must be in [0, 1]')
  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA training requested but no GPU is visible')

  os.makedirs(args.data_dir, exist_ok=True)
  env = make_vector_env(
      num_envs=args.num_workers, num_workers=args.num_workers,
      batch_size=args.num_workers, reserved_cpus=0, seed=args.seed,
      env_name='11_vs_11_curriculum', frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.curriculum_window,
      curriculum_success_threshold=args.curriculum_success_threshold,
      attacker_only_levels=args.attacker_only_levels,
      centralized_curriculum=True)
  horizon = 320
  config = _base_config()
  config.update({
      'anneal_lr': args.anneal_lr,
      'critic_learning_rate': args.critic_learning_rate,
      'adam_eps': 1e-8,
      'batch_size': env.num_agents * horizon,
      'bptt_horizon': horizon,
      'checkpoint_interval': 100,
      'compile': False,
      'cpu_offload': False,
      'data_dir': os.path.abspath(args.data_dir),
      'device': args.device,
      'ent_coef': args.ent_coef,
      'env': 'gfootball',
      'gae_lambda': 0.95,
      'gamma': 0.997,
      'learning_rate': args.learning_rate,
      'max_grad_norm': 0.5,
      'minibatch_size': valid_minibatch_size(env.num_agents, horizon),
      'optimizer': 'adam',
      'precision': 'bfloat16' if args.device == 'cuda' else 'float32',
      'seed': args.seed,
      'torch_deterministic': False,
      'total_timesteps': args.total_timesteps,
      'update_epochs': 2,
      'use_rnn': False,
      'vf_coef': 2.0,
      'vf_clip_coef': 0.2,
      'clip_coef': 0.27,
  })
  for unused in ('prio_alpha', 'prio_beta0',
                 'vtrace_c_clip', 'vtrace_rho_clip'):
    config.pop(unused, None)
  if config['total_timesteps'] < config['batch_size']:
    raise ValueError('total_timesteps must cover at least one rollout batch')
  print(json.dumps({
      'config': config,
      'curriculum_levels': args.curriculum_levels,
      'curriculum_success_threshold': args.curriculum_success_threshold,
      'curriculum_window': args.curriculum_window,
      'attacker_only_levels': args.attacker_only_levels,
      'frame_stack': args.frame_stack,
      'num_workers': args.num_workers,
      'promotion_interval': args.promotion_interval,
      'promotion_episodes': args.promotion_episodes,
      'promotion_workers': args.promotion_workers,
      'promotion_worst_template_threshold': (
          args.promotion_worst_template_threshold),
      'gradient_audit': args.gradient_audit,
      'anneal_lr': args.anneal_lr,
  }, sort_keys=True), flush=True)

  policy = FootballPolicy(env).to(args.device)
  logger = None
  if args.wandb:
    logger = pufferl.WandbLogger({
        'wandb_project': args.wandb_project,
        'wandb_group': args.wandb_group,
        'tag': args.wandb_tag,
    })
  trainer = FootballPuffeRL(
      config, env, policy,
      gradient_audit=args.gradient_audit,
      critic_learning_rate=args.critic_learning_rate, logger=logger)
  try:
    while trainer.global_step < config['total_timesteps']:
      if trainer.epoch % args.promotion_interval == 0:
        level = env.curriculum_level_value.value
        promotion_env = _make_promotion_env(
            args, env.curriculum_level_value)
        try:
          metrics = evaluate_promotion(
              trainer.uncompiled_policy, promotion_env,
              args.promotion_episodes,
              args.seed + 1000000 + 10000 * level, args.device)
        finally:
          promotion_env.close()
        advanced = (
            level < args.curriculum_levels - 1 and
            promotion_passes(
                metrics, args.curriculum_success_threshold,
                args.promotion_worst_template_threshold))
        if advanced:
          env.curriculum_level_value.value = level + 1
        trainer.record_promotion(level, metrics, advanced)
        print('PROMOTION {}'.format(json.dumps({
            'level': level, 'advanced': advanced, **metrics,
        }, sort_keys=True)), flush=True)
      trainer.evaluate()
      trainer.train()
  finally:
    model_path = trainer.close()
    if logger is not None:
      logger.close(model_path)
    print('Saved model: {}'.format(model_path), flush=True)


if __name__ == '__main__':
  main()
