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
from gfootball.curriculum import TOTAL_LEVELS


ACTION_NAMES = tuple(
    str(action) for action in football_action_set.action_set_dict['default'])


def sampleable_segments(observations):
  return observations.flatten(1).abs().sum(dim=-1) > 0


def valid_minibatch_size(num_agents, horizon):
  return horizon * max(1, round(num_agents * 16 / horizon))


def active_minibatches(active_transitions, minibatch_size, update_epochs):
  """Number of PPO minibatches needed to reuse active data as requested."""
  if active_transitions < 1:
    raise ValueError('rollout contains no active transitions')
  return max(1, math.ceil(
      update_epochs * active_transitions / minibatch_size))


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


def policy_regularization_kls(logits, old_logits):
  """Reverse KL barriers retain a corrective gradient near policy collapse."""
  new_log_probs = torch.log_softmax(logits.float(), dim=-1)
  old_log_probs = torch.log_softmax(old_logits.float(), dim=-1)
  old_probs = torch.softmax(old_logits.float(), dim=-1)
  past_kl = torch.sum(
      old_probs * (old_log_probs - new_log_probs), dim=-1).mean()
  uniform_kl = (
      -new_log_probs.mean(dim=-1) - math.log(logits.shape[-1])).mean()
  return past_kl, uniform_kl


class FootballPolicy(torch.nn.Module):
  """Shared actor-critic over four simple115v2 frames."""

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
    )
    self.action_head = pufferlib.pytorch.layer_init(
        torch.nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
    self.value_head = pufferlib.pytorch.layer_init(
        torch.nn.Linear(hidden_size, 1), std=1.0)

  def forward(self, observations, _state=None):
    active = observations.flatten(1).abs().sum(dim=-1) > 0
    active_indices = active.nonzero().flatten()
    frames = observations[active].reshape(-1, self.frame_stack, 115)
    hidden = self.frame_encoder(frames).flatten(1)
    hidden = self.encoder(hidden)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
      active_logits = self.action_head(hidden.float())
      active_logits -= active_logits.mean(dim=-1, keepdim=True)
    logits = active_logits.new_zeros(
        (observations.shape[0], self.action_head.out_features)).index_copy(
            0, active_indices, active_logits)
    active_values = self.value_head(hidden).squeeze(-1)
    values = active_values.new_zeros(observations.shape[0]).index_copy(
        0, active_indices, active_values)
    return logits, values

  def forward_eval(self, observations, state=None):
    return self.forward(observations, state)


class RegularizedPuffeRL(pufferl.PuffeRL):
  """PuffeRL PPO with Puffer-Soccer's two KL penalties."""

  def __init__(self, config, vecenv, policy, past_kl_coef=0.1,
               uniform_kl_base_coef=0.05, uniform_kl_power=0.0,
               logit_l2_coef=1e-4, collapse_threshold=0.95,
               collapse_patience=3, logger=None):
    super().__init__(config, vecenv, policy, logger=logger)
    self.past_kl_coef = float(past_kl_coef)
    self.uniform_kl_base_coef = float(uniform_kl_base_coef)
    self.uniform_kl_power = float(uniform_kl_power)
    self.logit_l2_coef = float(logit_l2_coef)
    self.collapse_threshold = float(collapse_threshold)
    self.collapse_patience = int(collapse_patience)
    self.collapse_epochs = 0
    self.optimizer_steps = 0
    self.active_steps = 0
    self.last_active_steps = 0
    self.last_active_log_time = time.time()
    self.past_policy = copy.deepcopy(self.uncompiled_policy).to(config['device'])
    self.past_policy.eval()
    for parameter in self.past_policy.parameters():
      parameter.requires_grad_(False)

  @pufferl.record
  def train(self):
    profile = self.profile
    epoch = self.epoch
    profile('train', epoch)
    losses = defaultdict(float)
    config = self.config
    device = config['device']
    self.past_policy.load_state_dict(self.uncompiled_policy.state_dict())

    beta0 = config['prio_beta0']
    alpha = config['prio_alpha']
    clip_coef = config['clip_coef']
    vf_clip = config['vf_clip_coef']
    anneal_beta = beta0 + (1 - beta0) * alpha * epoch / self.total_epochs
    self.ratio[:] = 1
    rollout_active = self.observations.flatten(2).abs().sum(dim=-1) > 0
    active_transitions = int(rollout_active.sum().item())
    sampleable = sampleable_segments(self.observations)
    active_segments = int(sampleable.sum().item())
    num_minibatches = active_minibatches(
        active_transitions, self.minibatch_size, config['update_epochs'])
    target_active_transitions = config['update_epochs'] * active_transitions
    sampled_active_transitions = 0
    minibatch = 0

    while (minibatch < num_minibatches or
           sampled_active_transitions < target_active_transitions):
      if minibatch >= 4 * self.total_minibatches:
        raise RuntimeError('could not sample enough active transitions')
      profile('train_misc', epoch, nest=True)
      self.amp_context.__enter__()
      advantages = torch.zeros(self.values.shape, device=device)
      advantages = pufferl.compute_puff_advantage(
          self.values, self.rewards, self.terminals, self.ratio, advantages,
          config['gamma'], config['gae_lambda'], config['vtrace_rho_clip'],
          config['vtrace_c_clip'])

      profile('train_copy', epoch)
      priority = advantages.abs().sum(axis=1)
      priority_weights = torch.nan_to_num(priority**alpha, 0, 0, 0)
      priority_weights = torch.where(
          sampleable,
          priority_weights + 1e-6, 0)
      priority_probs = priority_weights / priority_weights.sum()
      indices = torch.multinomial(priority_probs, self.minibatch_segments)
      minibatch_priority = (
          active_segments * priority_probs[indices, None]) ** -anneal_beta
      observations = self.observations[indices]
      actions = self.actions[indices]
      old_logprobs = self.logprobs[indices]
      rewards = self.rewards[indices]
      terminals = self.terminals[indices]
      values = self.values[indices]
      returns = advantages[indices] + values
      old_advantages = advantages[indices]

      profile('train_forward', epoch)
      observations = observations.reshape(
          -1, *self.vecenv.single_observation_space.shape)
      state = {'action': actions, 'lstm_h': None, 'lstm_c': None}
      logits, new_values = self.policy(observations, state)
      _, new_logprobs, entropy = pufferlib.pytorch.sample_logits(
          logits, action=actions)
      active = observations.flatten(1).abs().sum(dim=-1) > 0
      active_steps = active.view(old_logprobs.shape)
      sampled_active_transitions += int(active.sum().item())

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

      updated_advantages = pufferl.compute_puff_advantage(
          values, rewards, terminals, ratio, advantages[indices],
          config['gamma'], config['gae_lambda'], config['vtrace_rho_clip'],
          config['vtrace_c_clip'])
      del updated_advantages
      active_advantages = old_advantages[active_steps]
      normalized_advantages = (
          minibatch_priority.expand_as(old_advantages)[active_steps] *
          (active_advantages - active_advantages.mean()) /
          (active_advantages.std() + 1e-8))
      policy_loss = torch.max(
          -normalized_advantages * active_ratio,
          -normalized_advantages * torch.clamp(
              active_ratio, 1 - clip_coef, 1 + clip_coef)).mean()

      new_values = new_values.view(returns.shape)
      clipped_values = values + torch.clamp(
          new_values - values, -vf_clip, vf_clip)
      value_loss = 0.5 * torch.max(
          (new_values[active_steps] - returns[active_steps]) ** 2,
          (clipped_values[active_steps] - returns[active_steps]) ** 2).mean()
      entropy_loss = entropy.view(active_steps.shape)[active_steps].mean()

      with torch.no_grad():
        old_logits, _ = self.past_policy(observations, state)
      active_logits = logits[active]
      past_kl, uniform_kl = policy_regularization_kls(
          active_logits, old_logits[active])
      uniform_kl_coef = self.uniform_kl_base_coef / (
          max(1, epoch + 1) ** self.uniform_kl_power)
      regularization = (self.past_kl_coef * past_kl +
                        uniform_kl_coef * uniform_kl)
      logit_l2 = active_logits.float().square().mean()
      loss = (policy_loss + config['vf_coef'] * value_loss -
              config['ent_coef'] * entropy_loss + regularization +
              self.logit_l2_coef * logit_l2)
      self.amp_context.__enter__()
      self.values[indices] = new_values.detach().float()

      profile('train_misc', epoch)
      metrics = {
          'policy_loss': policy_loss,
          'value_loss': value_loss,
          'entropy': entropy_loss,
          'old_approx_kl': old_approx_kl,
          'approx_kl': approx_kl,
          'clipfrac': clip_fraction,
          'importance': active_ratio.mean(),
          'past_kl': past_kl,
          'past_kl_term': self.past_kl_coef * past_kl,
          'uniform_kl': uniform_kl,
          'uniform_kl_term': uniform_kl_coef * uniform_kl,
          'regularization_term': regularization,
          'logit_l2': logit_l2,
          'logit_l2_term': self.logit_l2_coef * logit_l2,
          **policy_diagnostics(active_logits),
      }
      for name, value in metrics.items():
        losses[name] += value.item()
      losses['past_kl_coef'] += self.past_kl_coef
      losses['uniform_kl_coef'] += uniform_kl_coef

      profile('learn', epoch)
      loss.backward()
      if (minibatch + 1) % self.accumulate_minibatches == 0:
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), config['max_grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.optimizer_steps += 1
      minibatch += 1

    num_minibatches = minibatch
    for name in tuple(losses):
      losses[name] /= num_minibatches

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
    losses['effective_update_epochs'] = (
        sampled_active_transitions / active_transitions)
    losses['optimizer_steps'] = float(self.optimizer_steps)
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
    max_action_fraction = max(action_fractions)
    post_update_collapsed = (
        post_max_action_fraction >= self.collapse_threshold and
        post_diagnostics['policy_max_probability'].item() >=
        self.collapse_threshold)
    collapse_fraction = max(
        max_action_fraction,
        post_max_action_fraction if post_update_collapsed else 0)
    self.collapse_epochs = (
        self.collapse_epochs + 1
        if collapse_fraction >= self.collapse_threshold else 0)
    collapsed = self.collapse_epochs >= self.collapse_patience
    losses['max_action_fraction'] = max_action_fraction
    losses['collapse_epochs'] = float(self.collapse_epochs)

    profile('train_misc', epoch)
    if config['anneal_lr']:
      self.scheduler.step()
    predictions = self.values[rollout_active]
    targets = advantages[rollout_active] + predictions
    target_variance = targets.var()
    losses['explained_variance'] = (
        torch.nan if target_variance == 0 else
        1 - (targets - predictions).var() / target_variance).item()

    profile.end()
    logs = None
    self.epoch += 1
    done_training = self.global_step >= config['total_timesteps']
    if (done_training or collapsed or self.global_step == 0 or
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
    if collapsed:
      self.save_checkpoint()
      raise RuntimeError(
          'Policy collapse: one action occupied {:.1%} of decisions for {} '
          'epochs'.format(collapse_fraction, self.collapse_epochs))
    return logs


def _base_config():
  original_argv = sys.argv
  sys.argv = [original_argv[0]]
  try:
    return dict(pufferl.load_config('default')['train'])
  finally:
    sys.argv = original_argv


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-workers', type=int, default=30)
  parser.add_argument('--total-timesteps', type=int, default=1_000_000_000)
  parser.add_argument('--curriculum-levels', type=int, default=TOTAL_LEVELS)
  parser.add_argument('--curriculum-window', type=int, default=20)
  parser.add_argument('--curriculum-success-threshold', type=float, default=0.6)
  parser.add_argument('--frame-stack', type=int, default=4, choices=(1, 4))
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
  parser.add_argument('--data-dir', default='experiments/football-regularized')
  parser.add_argument('--past-kl-coef', type=float, default=0.1)
  parser.add_argument('--uniform-kl-base-coef', type=float, default=0.05)
  parser.add_argument('--uniform-kl-power', type=float, default=0.0)
  parser.add_argument('--logit-l2-coef', type=float, default=1e-4)
  parser.add_argument('--collapse-threshold', type=float, default=0.95)
  parser.add_argument('--collapse-patience', type=int, default=3)
  parser.add_argument('--wandb', action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument('--wandb-project', default='google-football-fast-rl')
  parser.add_argument('--wandb-group', default='regularized-self-play')
  parser.add_argument('--wandb-tag', default=None)
  args = parser.parse_args()
  if args.logit_l2_coef < 0:
    raise ValueError('logit-l2-coef must be nonnegative')
  if not 0 < args.collapse_threshold <= 1:
    raise ValueError('collapse-threshold must be in (0, 1]')
  if args.collapse_patience < 1:
    raise ValueError('collapse-patience must be positive')
  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA training requested but no GPU is visible')

  os.makedirs(args.data_dir, exist_ok=True)
  env = make_vector_env(
      num_envs=args.num_workers, num_workers=args.num_workers,
      batch_size=args.num_workers, reserved_cpus=0, seed=args.seed,
      env_name='11_vs_11_curriculum', frame_stack=args.frame_stack,
      curriculum_levels=args.curriculum_levels,
      curriculum_window=args.curriculum_window,
      curriculum_success_threshold=args.curriculum_success_threshold)
  horizon = 320
  config = _base_config()
  config.update({
      'anneal_lr': True,
      'adam_eps': 1e-8,
      'batch_size': env.num_agents * horizon,
      'bptt_horizon': horizon,
      'checkpoint_interval': 100,
      'compile': False,
      'cpu_offload': False,
      'data_dir': os.path.abspath(args.data_dir),
      'device': args.device,
      'ent_coef': 0.01,
      'env': 'gfootball',
      'gae_lambda': 0.95,
      'gamma': 0.997,
      'learning_rate': 8e-5,
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
  if config['total_timesteps'] < config['batch_size']:
    raise ValueError('total_timesteps must cover at least one rollout batch')
  print(json.dumps({
      'config': config,
      'curriculum_levels': args.curriculum_levels,
      'curriculum_success_threshold': args.curriculum_success_threshold,
      'curriculum_window': args.curriculum_window,
      'frame_stack': args.frame_stack,
      'num_workers': args.num_workers,
      'past_kl_coef': args.past_kl_coef,
      'uniform_kl_base_coef': args.uniform_kl_base_coef,
      'uniform_kl_power': args.uniform_kl_power,
      'logit_l2_coef': args.logit_l2_coef,
      'collapse_threshold': args.collapse_threshold,
      'collapse_patience': args.collapse_patience,
  }, sort_keys=True), flush=True)

  policy = FootballPolicy(env).to(args.device)
  logger = None
  if args.wandb:
    logger = pufferl.WandbLogger({
        'wandb_project': args.wandb_project,
        'wandb_group': args.wandb_group,
        'tag': args.wandb_tag,
    })
  trainer = RegularizedPuffeRL(
      config, env, policy, past_kl_coef=args.past_kl_coef,
      uniform_kl_base_coef=args.uniform_kl_base_coef,
      uniform_kl_power=args.uniform_kl_power,
      logit_l2_coef=args.logit_l2_coef,
      collapse_threshold=args.collapse_threshold,
      collapse_patience=args.collapse_patience, logger=logger)
  try:
    while trainer.global_step < config['total_timesteps']:
      trainer.evaluate()
      trainer.train()
  finally:
    model_path = trainer.close()
    if logger is not None:
      logger.close(model_path)
    print('Saved model: {}'.format(model_path), flush=True)


if __name__ == '__main__':
  main()
