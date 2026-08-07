"""Shared schedule for the near-goal player-count curriculum."""

ATTACKER_ORDER = (2, 1, 10, 7, 9, 8, 3, 6, 4, 5, 0)
DEFENDER_ORDER = (4, 5, 3, 6, 8, 7, 9, 1, 10, 2)
TOTAL_LEVELS = 31


def curriculum_state(level):
  """Return active attackers, field defenders, and distance progress."""
  level = max(0, min(TOTAL_LEVELS - 1, int(level)))
  if level < 11:
    return level + 1, 0, 0.0
  if level < 21:
    return 11, level - 10, 0.0
  return 11, 10, (level - 20) / 10.0
