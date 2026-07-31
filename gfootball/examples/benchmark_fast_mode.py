"""Measure environment-only throughput without starting RL training."""

import argparse
import json
import statistics
import time

import gfootball.env as football_env


def measure(fast_mode, steps, repeats):
  results = []
  for repeat in range(repeats):
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',
        representation='simple115v2',
        render=False,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False,
        stacked=False,
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
        other_config_options={
            'action_set': 'default',
            'fast_mode': fast_mode,
            'game_engine_random_seed': repeat,
            'real_time': False,
        })
    try:
      env.reset()
      actions = [0] * 22
      for _ in range(20):
        _, _, done, _ = env.step(actions)
        if done:
          env.reset()
      start = time.perf_counter()
      for _ in range(steps):
        _, _, done, _ = env.step(actions)
        if done:
          env.reset()
      results.append(steps / (time.perf_counter() - start))
    finally:
      env.close()
  return statistics.median(results)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=500)
  parser.add_argument('--repeats', type=int, default=3)
  args = parser.parse_args()
  baseline = measure(False, args.steps, args.repeats)
  fast = measure(True, args.steps, args.repeats)
  print(json.dumps({
      'baseline_steps_per_second': baseline,
      'fast_steps_per_second': fast,
      'speedup': fast / baseline,
  }, sort_keys=True))


if __name__ == '__main__':
  main()
