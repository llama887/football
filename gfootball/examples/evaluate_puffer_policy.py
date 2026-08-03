"""Measure what a trained shared football policy actually does."""

import argparse
from collections import Counter
import json
import math
import os
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

import gfootball.env as football_env
from gfootball.env import football_action_set
from gfootball.examples.train_puffer import FootballPolicy


ACTION_NAMES = tuple(
    str(action) for action in football_action_set.action_set_dict['default'])


def evaluate(checkpoint, episodes, greedy, seed):
  torch.manual_seed(seed)
  policy_env = SimpleNamespace(
      single_observation_space=gymnasium.spaces.Box(
          low=-np.inf, high=np.inf, shape=(460,), dtype=np.float32),
      single_action_space=gymnasium.spaces.Discrete(len(ACTION_NAMES)))
  policy = FootballPolicy(policy_env)
  policy.load_state_dict(torch.load(
      checkpoint, map_location='cpu', weights_only=True))
  policy.eval()

  env = football_env.create_environment(
      env_name='11_vs_11_curriculum', representation='simple115v2',
      rewards='scoring', render=False, write_goal_dumps=False,
      write_full_episode_dumps=False, write_video=False, stacked=True,
      number_of_left_players_agent_controls=11,
      number_of_right_players_agent_controls=11, extra_players=None,
      other_config_options={
          'action_set': 'default',
          'curriculum_level': 0,
          'curriculum_levels': 11,
          'fast_mode': True,
          'game_engine_random_seed': seed,
          'real_time': False,
      })

  action_counts = Counter()
  totals = Counter()
  episode_rows = []
  try:
    for episode in range(episodes):
      observations = env.reset()
      ball_start = float(observations[0, -27])
      attack_sign = 1.0 if ball_start > 0 else -1.0
      attacking_slice = slice(0, 11) if attack_sign > 0 else slice(11, 22)
      previous_actions = None
      max_ball_progress = 0.0
      first_shot_step = None
      done = False
      step = 0
      while not done:
        with torch.inference_mode():
          logits, _ = policy(torch.as_tensor(observations))
          probabilities = torch.softmax(logits.float(), dim=-1)
          actions = (probabilities.argmax(-1) if greedy else
                     torch.multinomial(probabilities, 1).squeeze(-1))
        action_array = actions.numpy()
        attack_actions = action_array[attacking_slice]
        action_counts.update(action_array.tolist())
        totals['decisions'] += len(action_array)
        totals['attacking_decisions'] += len(attack_actions)
        totals['attacking_shots'] += int(np.sum(attack_actions == 12))
        totals['attacking_kicks'] += int(np.sum(
            (attack_actions >= 9) & (attack_actions <= 12)))
        totals['movement'] += int(np.sum(
            (action_array >= 1) & (action_array <= 8)))
        if previous_actions is not None:
          totals['changed_actions'] += int(np.sum(
              action_array != previous_actions))
          totals['change_opportunities'] += len(action_array)
        if first_shot_step is None and np.any(attack_actions == 12):
          first_shot_step = step
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        top_two = probabilities.topk(2, dim=-1).values
        totals['entropy'] += float(entropy.sum())
        totals['max_probability'] += float(top_two[:, 0].sum())
        totals['probability_margin'] += float(
            (top_two[:, 0] - top_two[:, 1]).sum())

        observations, _rewards, done, info = env.step(action_array)
        ball_x = float(observations[0, -27])
        max_ball_progress = max(
            max_ball_progress, attack_sign * (ball_x - ball_start))
        previous_actions = action_array
        step += 1

      score = float(info['score_reward'])
      success = score * attack_sign > 0
      episode_rows.append({
          'episode': episode,
          'success': float(success),
          'score_reward': score,
          'length': step,
          'max_ball_progress': max_ball_progress,
          'first_shot_step': first_shot_step if first_shot_step is not None else step,
      })
  finally:
    env.close()

  decisions = totals['decisions']
  metrics = {
      'episodes': episodes,
      'success_rate': float(np.mean([row['success'] for row in episode_rows])),
      'mean_episode_length': float(np.mean(
          [row['length'] for row in episode_rows])),
      'mean_max_ball_progress': float(np.mean(
          [row['max_ball_progress'] for row in episode_rows])),
      'mean_first_shot_step': float(np.mean(
          [row['first_shot_step'] for row in episode_rows])),
      'policy_entropy': totals['entropy'] / decisions,
      'policy_entropy_fraction': totals['entropy'] / decisions / math.log(19),
      'policy_max_probability': totals['max_probability'] / decisions,
      'policy_probability_margin': totals['probability_margin'] / decisions,
      'action_change_rate': totals['changed_actions'] /
                            max(1, totals['change_opportunities']),
      'movement_fraction': totals['movement'] / decisions,
      'attacking_kick_fraction': totals['attacking_kicks'] /
                                 totals['attacking_decisions'],
      'attacking_shot_fraction': totals['attacking_shots'] /
                                 totals['attacking_decisions'],
  }
  action_fractions = {
      name: action_counts[index] / decisions
      for index, name in enumerate(ACTION_NAMES)
  }
  return metrics, action_fractions, episode_rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--episodes', type=int, default=20)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--wandb-project', default='google-football-fast-rl')
  parser.add_argument('--wandb-group', default='policy-postmortem')
  parser.add_argument('--no-wandb', action='store_true')
  args = parser.parse_args()

  results = {}
  rows = []
  for mode, greedy in (('sampled', False), ('greedy', True)):
    metrics, actions, episodes = evaluate(
        args.checkpoint, args.episodes, greedy, args.seed)
    results[mode] = {'metrics': metrics, 'action_fractions': actions}
    rows.extend([{'mode': mode, **row} for row in episodes])
  print(json.dumps(results, indent=2, sort_keys=True), flush=True)

  if not args.no_wandb:
    import wandb
    run = wandb.init(
        project=args.wandb_project, group=args.wandb_group,
        job_type='evaluation', config=vars(args))
    payload = {}
    for mode, result in results.items():
      payload.update({
          '{}/{}'.format(mode, key): value
          for key, value in result['metrics'].items()
      })
      payload.update({
          '{}/actions/{}'.format(mode, key): value
          for key, value in result['action_fractions'].items()
      })
    payload['episodes'] = wandb.Table(
        columns=list(rows[0]), data=[list(row.values()) for row in rows])
    run.log(payload)
    run.finish()


if __name__ == '__main__':
  main()
