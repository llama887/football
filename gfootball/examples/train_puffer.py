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
    frames = observations.reshape(-1, self.frame_stack, 115)
    hidden = self.frame_encoder(frames).flatten(1)
    hidden = self.encoder(hidden)
    return self.action_head(hidden), self.value_head(hidden).squeeze(-1)

  def forward_eval(self, observations, state=None):
    return self.forward(observations, state)


class RegularizedPuffeRL(pufferl.PuffeRL):
  """PuffeRL PPO with Puffer-Soccer's two KL penalties."""

  def __init__(self, config, vecenv, policy, past_kl_coef=0.1,
               uniform_kl_base_coef=0.05, uniform_kl_power=0.3):
    super().__init__(config, vecenv, policy)
    self.past_kl_coef = float(past_kl_coef)
    self.uniform_kl_base_coef = float(uniform_kl_base_coef)
    self.uniform_kl_power = float(uniform_kl_power)
    self.uniform_log_prob = -math.log(float(vecenv.single_action_space.n))
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

    for minibatch in range(self.total_minibatches):
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
      priority_probs = ((priority_weights + 1e-6) /
                        (priority_weights.sum() + 1e-6))
      indices = torch.multinomial(priority_probs, self.minibatch_segments)
      minibatch_priority = (
          self.segments * priority_probs[indices, None]) ** -anneal_beta
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

      profile('train_misc', epoch)
      new_logprobs = new_logprobs.reshape(old_logprobs.shape)
      log_ratio = new_logprobs - old_logprobs
      ratio = log_ratio.exp()
      self.ratio[indices] = ratio.detach()
      with torch.no_grad():
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()

      updated_advantages = pufferl.compute_puff_advantage(
          values, rewards, terminals, ratio, advantages[indices],
          config['gamma'], config['gae_lambda'], config['vtrace_rho_clip'],
          config['vtrace_c_clip'])
      del updated_advantages
      normalized_advantages = minibatch_priority * (
          old_advantages - old_advantages.mean()) / (
              old_advantages.std() + 1e-8)
      policy_loss = torch.max(
          -normalized_advantages * ratio,
          -normalized_advantages * torch.clamp(
              ratio, 1 - clip_coef, 1 + clip_coef)).mean()

      new_values = new_values.view(returns.shape)
      clipped_values = values + torch.clamp(
          new_values - values, -vf_clip, vf_clip)
      value_loss = 0.5 * torch.max(
          (new_values - returns) ** 2,
          (clipped_values - returns) ** 2).mean()
      entropy_loss = entropy.mean()

      with torch.no_grad():
        old_logits, _ = self.past_policy(observations, state)
      new_log_probs = torch.log_softmax(logits, dim=-1)
      new_probs = torch.softmax(logits, dim=-1)
      old_log_probs = torch.log_softmax(old_logits, dim=-1)
      past_kl = torch.sum(
          new_probs * (new_log_probs - old_log_probs), dim=-1).mean()
      uniform_kl = torch.sum(
          new_probs * (new_log_probs - self.uniform_log_prob), dim=-1).mean()
      uniform_kl_coef = self.uniform_kl_base_coef / (
          max(1, epoch + 1) ** self.uniform_kl_power)
      regularization = (self.past_kl_coef * past_kl +
                        uniform_kl_coef * uniform_kl)
      loss = (policy_loss + config['vf_coef'] * value_loss -
              config['ent_coef'] * entropy_loss + regularization)
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
          'importance': ratio.mean(),
          'past_kl': past_kl,
          'uniform_kl': uniform_kl,
          'regularization_term': regularization,
      }
      for name, value in metrics.items():
        losses[name] += value.item() / self.total_minibatches
      losses['past_kl_coef'] += self.past_kl_coef / self.total_minibatches
      losses['uniform_kl_coef'] += uniform_kl_coef / self.total_minibatches

      profile('learn', epoch)
      loss.backward()
      if (minibatch + 1) % self.accumulate_minibatches == 0:
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), config['max_grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()

    profile('train_misc', epoch)
    if config['anneal_lr']:
      self.scheduler.step()
    predictions = self.values.flatten()
    targets = advantages.flatten() + predictions
    target_variance = targets.var()
    losses['explained_variance'] = (
        torch.nan if target_variance == 0 else
        1 - (targets - predictions).var() / target_variance).item()

    profile.end()
    logs = None
    self.epoch += 1
    done_training = self.global_step >= config['total_timesteps']
    if (done_training or self.global_step == 0 or
        time.time() > self.last_log_time + 0.25):
      logs = self.mean_and_log()
      self.losses = losses
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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-workers', type=int, default=30)
  parser.add_argument('--total-timesteps', type=int, default=1_000_000_000)
  parser.add_argument('--curriculum-levels', type=int, default=11)
  parser.add_argument('--curriculum-window', type=int, default=20)
  parser.add_argument('--curriculum-success-threshold', type=float, default=0.6)
  parser.add_argument('--frame-stack', type=int, default=4, choices=(1, 4))
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
  parser.add_argument('--data-dir', default='experiments/football-regularized')
  parser.add_argument('--past-kl-coef', type=float, default=0.1)
  parser.add_argument('--uniform-kl-base-coef', type=float, default=0.05)
  parser.add_argument('--uniform-kl-power', type=float, default=0.3)
  args = parser.parse_args()
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
  horizon = 64
  config = _base_config()
  config.update({
      'anneal_lr': True,
      'batch_size': env.num_agents * horizon,
      'bptt_horizon': horizon,
      'checkpoint_interval': 100,
      'compile': False,
      'cpu_offload': False,
      'data_dir': os.path.abspath(args.data_dir),
      'device': args.device,
      'ent_coef': 0.01,
      'env': 'gfootball',
      'gae_lambda': 0.90,
      'gamma': 0.993,
      'learning_rate': 8e-5,
      'max_grad_norm': 0.5,
      'minibatch_size': env.num_agents * 16,
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
  }, sort_keys=True), flush=True)

  policy = FootballPolicy(env).to(args.device)
  trainer = RegularizedPuffeRL(
      config, env, policy, past_kl_coef=args.past_kl_coef,
      uniform_kl_base_coef=args.uniform_kl_base_coef,
      uniform_kl_power=args.uniform_kl_power)
  try:
    while trainer.global_step < config['total_timesteps']:
      trainer.evaluate()
      trainer.train()
  finally:
    model_path = trainer.close()
    print('Saved model: {}'.format(model_path), flush=True)


if __name__ == '__main__':
  main()
