"""PufferLib-native interface for headless 22-player self-play."""

from functools import partial

import gymnasium
import numpy as np
import psutil
import pufferlib
import pufferlib.vector

import gfootball.env as football_env
from gfootball.env import football_action_set


class FootballPufferEnv(pufferlib.PufferEnv):
  """One GRF match exposed as 22 PufferLib agents."""

  def __init__(self, env_name='11_vs_11_curriculum', render=False, buf=None,
               seed=0, frame_stack=4, curriculum_episodes=256):
    if frame_stack not in (1, 4):
      raise ValueError('frame_stack must be 1 or 4')
    self.num_envs = 1
    self.num_agents = 22
    self.agents_per_batch = self.num_agents
    self.single_observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(115 * frame_stack,), dtype=np.float32)
    self.single_action_space = gymnasium.spaces.Discrete(
        len(football_action_set.action_set_dict['default']))
    super().__init__(buf)

    self._env_name = env_name
    self._render = render
    self._seed = int(seed)
    self._frame_stack = frame_stack
    self._curriculum_episodes = int(curriculum_episodes)
    self._env = self._make_env()
    self._episode_return = np.zeros(2, dtype=np.float32)
    self._episode_length = 0

  def _make_env(self):
    return football_env.create_environment(
        env_name=self._env_name,
        representation='simple115v2',
        rewards='scoring',
        render=self._render,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False,
        stacked=self._frame_stack == 4,
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
        extra_players=None,
        other_config_options={
            'action_set': 'default',
            'curriculum_episodes': self._curriculum_episodes,
            'fast_mode': not self._render,
            'game_engine_random_seed': self._seed,
            'real_time': False,
        })

  def _write_observations(self, observations):
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape != self.observations.shape:
      raise ValueError('Expected observations with shape {}, got {}'.format(
          self.observations.shape, observations.shape))
    self.observations[:] = observations

  def reset(self, seed=None):
    if seed is not None and int(seed) != self._seed:
      self._env.close()
      self._seed = int(seed)
      self._env = self._make_env()
    self._write_observations(self._env.reset())
    self.rewards.fill(0)
    self.terminals.fill(False)
    self.truncations.fill(False)
    self._episode_return.fill(0)
    self._episode_length = 0
    return self.observations, []

  def step(self, actions):
    observations, rewards, done, info = self._env.step(
        np.asarray(actions).reshape(self.num_agents))
    rewards = np.asarray(rewards, dtype=np.float32)
    self._episode_return += [rewards[:11].mean(), rewards[11:].mean()]
    self._episode_length += 1
    self.rewards[:] = rewards
    self.terminals.fill(done)
    self.truncations.fill(False)

    infos = []
    if done:
      infos.append({
          'episode_length': self._episode_length,
          'left_episode_return': float(self._episode_return[0]),
          'right_episode_return': float(self._episode_return[1]),
          'score_reward': float(info['score_reward']),
      })
      observations = self._env.reset()
      self._episode_return.fill(0)
      self._episode_length = 0
    self._write_observations(observations)
    return (self.observations, self.rewards, self.terminals,
            self.truncations, infos)

  def close(self):
    if self._env is not None:
      self._env.close()
      self._env = None


def make_vector_env(num_envs=None, num_workers=None, batch_size=None,
                    reserved_cpus=2, seed=0, **env_kwargs):
  """Create one headless match per PufferLib multiprocessing worker."""
  if num_workers is None:
    available_cpus = (len(psutil.Process().cpu_affinity())
                      if hasattr(psutil.Process(), 'cpu_affinity') else
                      psutil.cpu_count(logical=False) or 1)
    num_workers = max(1, available_cpus - reserved_cpus)
  num_envs = num_workers if num_envs is None else num_envs
  batch_size = num_workers if batch_size is None else batch_size
  return pufferlib.vector.make(
      partial(FootballPufferEnv, **env_kwargs),
      backend=pufferlib.vector.Multiprocessing,
      num_envs=num_envs,
      num_workers=num_workers,
      batch_size=batch_size,
      zero_copy=True,
      seed=seed)
