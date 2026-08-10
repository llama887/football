"""Shared schedule for the near-goal player-count curriculum."""

import random

ATTACKER_ORDER = (2, 1, 10, 7, 9, 8, 3, 6, 4, 5, 0)
DEFENDER_ORDER = (4, 5, 3, 6, 8, 7, 9, 1, 10, 2)
ATTACKER_ONLY_LEVELS = 13
SPAWN_TEMPLATE_COUNT = 8
TOTAL_LEVELS = 33


def curriculum_state(level):
  """Return active attackers, field defenders, and distance progress."""
  level = max(0, min(TOTAL_LEVELS - 1, int(level)))
  if level < 4:
    return min(2, level + 1), 0, 0.0
  if level < ATTACKER_ONLY_LEVELS:
    return level - 1, 0, 0.0
  if level < 23:
    return 11, level - 12, 0.0
  return 11, 10, (level - 22) / 10.0


def curriculum_episode(level, seed, episode):
  """Return episode attacker count, attack direction, and spawn template."""
  attackers, _, _ = curriculum_state(level)
  if level < 4:
    rng = random.Random((int(seed) + 1) * 1000003 + int(episode))
    attackers = 2 if rng.randrange(3) < level else 1
  cycle = int(seed) + int(episode)
  return attackers, (cycle // SPAWN_TEMPLATE_COUNT) % 2 == 0, (
      cycle % SPAWN_TEMPLATE_COUNT)
