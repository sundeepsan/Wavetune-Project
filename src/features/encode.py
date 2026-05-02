"""
Symbol encoding for HMM consumption.

Stage 4 of the WaveTune pipeline: turn a continuous (x, y) centroid trajectory
into a sequence of discrete integer symbols that hmmlearn.CategoricalHMM can
consume.

Symbol vocabulary: 24 symbols = 8 directions x 3 speed levels.

  Direction bins (8): E, NE, N, NW, W, SW, S, SE
    Index 0 = East  (positive x, moving right)
    Index 1 = North-East
    Index 2 = North (negative y, moving up; image y axis is flipped)
    Index 3 = North-West
    Index 4 = West  (negative x, moving left)
    Index 5 = South-West
    Index 6 = South (positive y, moving down)
    Index 7 = South-East

  Speed bins (3): slow, medium, fast
    Thresholds are in normalized image coordinates (frame width = 1.0).

  Final symbol = direction_bin * 3 + speed_bin   in range 0..23

Frames whose motion magnitude is below MOTION_MIN are treated as "still" and
dropped from the symbol sequence (no symbol emitted). This keeps idle drift
from polluting the gesture pattern.
"""

from __future__ import annotations

import numpy as np

NUM_DIRECTIONS = 8
NUM_SPEEDS = 3
NUM_SYMBOLS = NUM_DIRECTIONS * NUM_SPEEDS  # = 24

# Thresholds in normalized image coordinates (units = fraction of frame).
# Tuned for ~30 fps capture; revisit if you change frame rate.
MOTION_MIN = 0.001       # below this, frame is "still" and dropped
SPEED_SLOW_MAX = 0.008   # below this, motion is "slow"
SPEED_FAST_MIN = 0.025   # above this, motion is "fast"


def direction_bin(dx: float, dy: float) -> int:
    """
    Quantize a velocity vector to one of 8 compass directions.

    Note: image y axis grows downward, so we negate dy so that "up" in the
    visual sense maps to "north" in compass terms.
    """
    angle = np.degrees(np.arctan2(-dy, dx))   # range -180..180, 0 = East, 90 = North
    angle = (angle + 360.0) % 360.0           # range 0..360
    # Center each 45-degree bin on its compass point: shift by half-bin then floor.
    return int(((angle + 22.5) % 360.0) // 45)


def speed_bin(speed: float) -> int:
    """Quantize speed to slow (0), medium (1), or fast (2)."""
    if speed < SPEED_SLOW_MAX:
        return 0
    if speed > SPEED_FAST_MIN:
        return 2
    return 1


def encode_trajectory(points: np.ndarray) -> np.ndarray:
    """
    Convert a (T, 2) trajectory of (x, y) centroids to a sequence of integer
    symbols in [0, 24).

    Frames with motion below MOTION_MIN are dropped, so the output may be
    shorter than T-1.
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (T, 2) array, got shape {arr.shape}")
    if len(arr) < 2:
        return np.empty(0, dtype=np.int32)

    deltas = arr[1:] - arr[:-1]                              # shape (T-1, 2)
    speeds = np.linalg.norm(deltas, axis=1)                  # shape (T-1,)

    symbols = []
    for (dx, dy), s in zip(deltas, speeds):
        if s < MOTION_MIN:
            continue
        symbols.append(direction_bin(dx, dy) * NUM_SPEEDS + speed_bin(s))
    return np.asarray(symbols, dtype=np.int32)


def symbol_to_label(symbol: int) -> str:
    """Human-readable label for a symbol, e.g. 'E-fast' or 'NW-slow'."""
    if not 0 <= symbol < NUM_SYMBOLS:
        return f"?{symbol}"
    direction_names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    speed_names = ["slow", "medium", "fast"]
    d = symbol // NUM_SPEEDS
    s = symbol % NUM_SPEEDS
    return f"{direction_names[d]}-{speed_names[s]}"
