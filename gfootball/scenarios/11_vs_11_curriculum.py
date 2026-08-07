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


_DEFENDER_JOIN_PROGRESS = (0.0, 0.18, 0.32, 0.46, 0.58,
                           0.68, 0.76, 0.84, 0.90, 0.95)
_DEFENDER_ORDER = (4, 5, 3, 6, 8, 7, 9, 1, 10, 2)


def _to_team_coordinates(team, x, y):
  side = 1.0 if team == Team.e_Left else -1.0
  return side * x, side * y


def _add_team(builder, team, attacking, progress, ball_x, ball_y, direction,
              rng):
  builder.SetTeam(team)
  for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
    if index == 0:
      guided_x, guided_y = -1.0, 0.0
    elif attacking:
      row, lane = divmod(index - 1, 5)
      world_x = ball_x - direction * (0.04 + 0.04 * row)
      world_y = ball_y + (lane - 2) * 0.07 + rng.uniform(-0.01, 0.01)
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36, world_y)))
    else:
      defender_rank = _DEFENDER_ORDER.index(index)
      if progress >= _DEFENDER_JOIN_PROGRESS[defender_rank]:
        world_x = ball_x + direction * (0.06 + 0.02 * (defender_rank // 2))
        world_y = ball_y + (defender_rank // 2 + 1) * 0.05 * (
            -1.0 if defender_rank % 2 else 1.0)
      else:
        world_x = direction * (0.12 + 0.04 * (defender_rank // 3))
        world_y = (defender_rank % 5 - 2) * 0.15
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36, world_y)))
    x = progress * standard_x + (1.0 - progress) * guided_x
    y = progress * standard_y + (1.0 - progress) * guided_y
    builder.AddPlayer(x, y, role)


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  curriculum_levels = max(2, int(builder._config['curriculum_levels']))
  curriculum_level = max(0, min(
      curriculum_levels - 1, int(builder._config['curriculum_level'])))
  progress = curriculum_level / (curriculum_levels - 1)
  seed = int(builder._config._values.get('game_engine_random_seed', 0))
  rng = random.Random(seed + episode)
  attack_right = (seed + episode) % 2 == 0
  direction = 1.0 if attack_right else -1.0
  ball_x = direction * 0.78 * (1.0 - progress)
  ball_y = rng.uniform(-0.04 - 0.18 * progress, 0.04 + 0.18 * progress)

  builder.config().game_duration = int(319 + 2681 * progress)
  builder.config().deterministic = False
  builder.config().use_magnet = False
  builder.config().offsides = progress >= 0.75
  builder.config().end_episode_on_score = progress < 1.0
  builder.SetBallPosition(ball_x, ball_y)

  attacking_team = Team.e_Left if attack_right else Team.e_Right
  _add_team(builder, Team.e_Left, attacking_team == Team.e_Left, progress,
            ball_x, ball_y, direction, rng)
  _add_team(builder, Team.e_Right, attacking_team == Team.e_Right, progress,
            ball_x, ball_y, direction, rng)
