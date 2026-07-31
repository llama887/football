# coding=utf-8
"""11v11 self-play that expands from a near-goal scoring curriculum."""

import random

from . import *


_FORMATION = (
    (-1.0, 0.0, e_PlayerRole_GK),
    (0.0, 0.02, e_PlayerRole_RM),
    (0.0, -0.02, e_PlayerRole_CF),
    (-0.422, -0.19576, e_PlayerRole_LB),
    (-0.5, -0.06356, e_PlayerRole_CB),
    (-0.5, 0.063559, e_PlayerRole_CB),
    (-0.422, 0.19576, e_PlayerRole_RB),
    (-0.184212, -0.10568, e_PlayerRole_CM),
    (-0.267574, 0.0, e_PlayerRole_CM),
    (-0.184212, 0.10568, e_PlayerRole_CM),
    (-0.01, -0.21610, e_PlayerRole_LM),
)


def _add_team(builder, team, progress, ball_x, rng):
  builder.SetTeam(team)
  for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
    if index == 0:
      guided_x, guided_y = -1.0, 0.0
    else:
      row = (index - 1) // 3
      side_of_ball = -1.0 if rng.random() < 0.5 else 1.0
      absolute_x = ball_x + side_of_ball * (0.04 + 0.05 * row)
      absolute_y = rng.uniform(-0.20, 0.20)
      side = 1.0 if team == Team.e_Left else -1.0
      guided_x, guided_y = side * absolute_x, side * absolute_y
    x = progress * standard_x + (1.0 - progress) * guided_x
    y = progress * standard_y + (1.0 - progress) * guided_y
    builder.AddPlayer(x, y, role)


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  curriculum_episodes = max(1, int(builder._config['curriculum_episodes']))
  progress = min(1.0, episode / curriculum_episodes)
  seed = int(builder._config._values.get('game_engine_random_seed', 0))
  rng = random.Random(seed + episode)
  attack_right = (seed + episode) % 2 == 0
  direction = 1.0 if attack_right else -1.0
  ball_x = direction * 0.78 * (1.0 - progress)
  ball_y = rng.uniform(-0.04 - 0.18 * progress, 0.04 + 0.18 * progress)

  builder.config().game_duration = 600 if progress < 1.0 else 3000
  builder.config().deterministic = False
  builder.config().offsides = progress >= 0.75
  builder.config().end_episode_on_score = progress < 1.0
  builder.SetBallPosition(ball_x, ball_y)

  _add_team(builder, Team.e_Left, progress, ball_x, rng)
  _add_team(builder, Team.e_Right, progress, ball_x, rng)
