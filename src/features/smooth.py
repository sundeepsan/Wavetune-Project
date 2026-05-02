"""
Trajectory smoothing.

Raw landmark centroids jitter several pixels per frame even when the hand or
head is still. Smoothing reduces jitter so velocity/direction features stay
meaningful.

We use a simple centered moving average. The output has the same length as
the input (edge values are repeat-padded). This is fast (O(T)), introduces no
phase lag, and is sufficient given the ~30 fps capture rate.
"""

from __future__ import annotations

import numpy as np


def moving_average(points: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth a (T, D) trajectory with a centered moving average.

    Parameters
    ----------
    points : ndarray of shape (T, D)
        Trajectory in arbitrary D-dimensional coords (typically D=2 for x, y).
    window : int
        Smoothing window size. Must be odd; will be coerced to odd if even.

    Returns
    -------
    ndarray of shape (T, D)
        Smoothed trajectory, same length as input.
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {arr.shape}")

    T = arr.shape[0]
    if T == 0:
        return arr.copy()

    # Coerce to odd window so the average is centered.
    if window % 2 == 0:
        window += 1
    if window < 3:
        return arr.copy()
    if T < window:
        return arr.copy()

    half = window // 2
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window

    smoothed = np.empty_like(arr)
    for d in range(arr.shape[1]):
        smoothed[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return smoothed
