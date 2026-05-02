"""
Gesture segmentation: trim a trajectory to its most active portion.

When users record gestures with the SPACE-START / SPACE-STOP convention, the
recording often includes setup motion (raising the hand into position) and
withdrawal motion (lowering the hand) bracketing the actual gesture. These
brackets confuse the HMM because they look the same across all gesture
classes (every class includes some "hand goes up", "hand goes down" motion).

`trim_to_motion()` finds the longest contiguous run of frames whose velocity
exceeds a fraction of the recording's peak velocity, and returns just that
slice of the trajectory. The remaining trajectory is mostly the pure gesture.

This is also the building block for real-time gesture detection: the same
energy-threshold logic can decide when a gesture starts and ends in a live
webcam stream (see future src/app/realtime.py).
"""

from __future__ import annotations

import numpy as np

from src.features.smooth import moving_average


def trim_to_motion(
    points: np.ndarray,
    smooth_window: int = 5,
    energy_frac: float = 0.3,
    min_length: int = 5,
    pad: int = 1,
) -> np.ndarray:
    """
    Return the slice of `points` covering the longest run of high-velocity frames.

    Parameters
    ----------
    points : ndarray of shape (T, 2)
        Raw centroid trajectory.
    smooth_window : int
        Window for smoothing before velocity computation. Removes jitter.
    energy_frac : float
        Frames whose smoothed velocity magnitude exceeds
        `energy_frac * max_velocity` are considered "active".
    min_length : int
        If the longest active run is shorter than this, return the original
        trajectory unchanged (we'd rather have noisy data than no data).
    pad : int
        Include this many extra frames on each side of the active window.

    Returns
    -------
    ndarray of shape (T', 2) where T' <= T
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (T, 2) array, got shape {arr.shape}")
    T = len(arr)
    if T < min_length + 1:
        return arr.copy()

    smoothed = moving_average(arr, window=smooth_window)
    deltas = smoothed[1:] - smoothed[:-1]
    speeds = np.linalg.norm(deltas, axis=1)

    if speeds.size == 0 or speeds.max() < 1e-6:
        return arr.copy()  # no motion detected at all

    threshold = float(speeds.max()) * float(energy_frac)
    active = speeds >= threshold

    # Find the longest contiguous run of active frames.
    best_start = 0
    best_len = 0
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start = i
            i = j
        else:
            i += 1

    if best_len < min_length:
        return arr.copy()

    # speeds[k] is the motion between arr[k] and arr[k+1], so the trajectory
    # window covering active velocities [best_start, best_start+best_len) is
    # arr[best_start : best_start + best_len + 1].
    lo = max(0, best_start - pad)
    hi = min(T, best_start + best_len + 1 + pad)
    return arr[lo:hi].copy()
