# coding=utf-8
"""11v11 self-play that expands from a near-goal scoring curriculum."""

import random

from . import *
from gfootball.curriculum import (
    ATTACKER_ORDER, DEFENDER_ORDER, TOTAL_LEVELS, curriculum_state)


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


def _to_team_coordinates(team, x, y):
  side = 1.0 if team == Team.e_Left else -1.0
  return side * x, side * y


def _add_team(builder, team, attacking, active_count, progress, ball_x, ball_y,
              direction, rng):
  builder.SetTeam(team)
  for index, (standard_x, standard_y, role) in enumerate(_FORMATION):
    order = ATTACKER_ORDER if attacking else DEFENDER_ORDER
    rank = order.index(index) if index in order else -1
    if attacking and rank < active_count:
      row, lane = divmod(rank, 5)
      row_count = min(5, active_count - 5 * row)
      world_x = ball_x - direction * (0.04 + 0.04 * row)
      world_y = ball_y + (lane - (row_count - 1) / 2) * 0.055
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36,
                                       world_y + rng.uniform(-0.006, 0.006))))
    elif attacking:
      world_x = -direction * (0.30 + 0.04 * (rank // 3))
      world_y = (rank % 5 - 2) * 0.15
      guided_x, guided_y = _to_team_coordinates(team, world_x, world_y)
    elif index == 0:
      guided_x, guided_y = -1.0, 0.0
    elif rank < active_count:
      world_x = ball_x + direction * (0.07 + 0.025 * (rank // 2))
      world_y = ball_y + (rank // 2 + 1) * 0.055 * (
          -1.0 if rank % 2 else 1.0)
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36, world_y)))
    else:
      world_x = direction * (0.12 + 0.04 * (rank // 3))
      world_y = (rank % 5 - 2) * 0.15
      guided_x, guided_y = _to_team_coordinates(
          team, world_x, max(-0.36, min(0.36, world_y)))
    x = progress * standard_x + (1.0 - progress) * guided_x
    y = progress * standard_y + (1.0 - progress) * guided_y
    builder.AddPlayer(x, y, role)


def build_scenario(builder):
  episode = builder.EpisodeNumber()
  curriculum_level = max(0, min(
      TOTAL_LEVELS - 1, int(builder._config['curriculum_level'])))
  active_attackers, active_defenders, progress = curriculum_state(
      curriculum_level)
  seed = int(builder._config._values.get('game_engine_random_seed', 0))
  rng = random.Random(seed + episode)
  attack_right = (seed + episode) % 2 == 0
  direction = 1.0 if attack_right else -1.0
  ball_x = direction * 0.78 * (1.0 - progress)
  ball_y = rng.uniform(-0.04 - 0.18 * progress, 0.04 + 0.18 * progress)

  builder.config().game_duration = int(599 + 2401 * progress)
  builder.config().deterministic = False
  builder.config().use_magnet = False
  builder.config().offsides = progress >= 0.75
  builder.config().end_episode_on_score = progress < 1.0
  builder.SetBallPosition(ball_x, ball_y)

  attacking_team = Team.e_Left if attack_right else Team.e_Right
  _add_team(builder, Team.e_Left, attacking_team == Team.e_Left,
            active_attackers if attacking_team == Team.e_Left
            else active_defenders, progress, ball_x, ball_y, direction, rng)
  _add_team(builder, Team.e_Right, attacking_team == Team.e_Right,
            active_attackers if attacking_team == Team.e_Right
            else active_defenders, progress, ball_x, ball_y, direction, rng)
