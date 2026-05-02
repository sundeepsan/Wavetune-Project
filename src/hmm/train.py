"""
Train one HMM per gesture class.

Loads every .npy trajectory under data/raw/<gesture>/, smooths each one,
encodes it as a sequence of 24 symbols, then fits one CategoricalHMM per
gesture using the Baum-Welch (EM) algorithm. Trained models are pickled to
models/<gesture>.pkl.

Run:
    python src/hmm/train.py
    python src/hmm/train.py --states 5 --iter 100
    python src/hmm/train.py --data-root data/raw --models-dir models
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from hmmlearn.hmm import CategoricalHMM

# Allow running as a script: add project root to sys.path.
import sys
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))

from src.features.encode import NUM_SYMBOLS, encode_trajectory
from src.features.segment import trim_to_motion
from src.features.smooth import moving_average


MIN_SAMPLES_PER_GESTURE = 5
MIN_SEQUENCE_LENGTH = 3

# Structured smoothing for emission probabilities.
#
# The 24-symbol vocabulary has known structure (8 directions on a circle x
# 3 speeds on a line), and a swipe_left that drifts a bit toward south-west
# is much more likely than one that suddenly emits a north-fast symbol.
# We encode that with three stages:
#
#   1. DIRECTION smoothing (beta): each direction's prob bleeds into its
#      circular neighbors at the same speed. Run twice so 2-step neighbors
#      get a meaningful tail too (W -> SW -> S). This is what lets a
#      swipe_left model still score plausibly on a sequence with some
#      south symbols mixed in.
#   2. SPEED smoothing (gamma): each speed bleeds into its linear neighbor.
#      Lets the model tolerate "the same gesture, slightly faster/slower".
#   3. LAPLACE floor (alpha): no symbol is exactly impossible. Catches the
#      edge case where direction/speed smoothing didn't reach that symbol.
EMISSION_SMOOTHING_ALPHA = 1e-3
DIRECTION_SMOOTHING_BETA = 0.15
SPEED_SMOOTHING_GAMMA = 0.10
DIRECTION_SMOOTHING_PASSES = 2


def smooth_emissions(
    model,
    alpha: float = EMISSION_SMOOTHING_ALPHA,
    beta: float = DIRECTION_SMOOTHING_BETA,
    gamma: float = SPEED_SMOOTHING_GAMMA,
    direction_passes: int = DIRECTION_SMOOTHING_PASSES,
) -> None:
    """Smooth a fitted CategoricalHMM's emissions in place using vocab structure."""
    # Reshape (n_states, 24) -> (n_states, 8 directions, 3 speeds)
    e = model.emissionprob_.reshape(-1, 8, 3)

    # 1. Direction smoothing on the circle. roll axis=1 cycles directions.
    for _ in range(direction_passes):
        ccw = np.roll(e, -1, axis=1)
        cw = np.roll(e, 1, axis=1)
        e = (1.0 - beta) * e + (beta / 2.0) * (ccw + cw)

    # 2. Speed smoothing along the line slow(0) - medium(1) - fast(2).
    new_e = e.copy()
    new_e[:, :, 0] = (1.0 - gamma) * e[:, :, 0] + gamma * e[:, :, 1]
    new_e[:, :, 2] = (1.0 - gamma) * e[:, :, 2] + gamma * e[:, :, 1]
    new_e[:, :, 1] = (1.0 - gamma) * e[:, :, 1] + (gamma / 2.0) * (e[:, :, 0] + e[:, :, 2])
    e = new_e

    # 3. Flatten back, add Laplace floor, renormalize.
    e = e.reshape(-1, 24) + alpha
    model.emissionprob_ = e / e.sum(axis=1, keepdims=True)


def load_and_encode(
    gesture_dir: Path,
    smooth_window: int = 5,
    trim: bool = True,
) -> list[np.ndarray]:
    """Load every .npy in gesture_dir, optionally trim to motion, smooth + encode.

    Returns a list of symbol sequences (one per usable file).
    """
    sequences: list[np.ndarray] = []
    for npy_path in sorted(gesture_dir.glob("*.npy")):
        traj = np.load(npy_path)
        raw_len = len(traj)
        if trim:
            traj = trim_to_motion(traj, smooth_window=smooth_window)
        smoothed = moving_average(traj, window=smooth_window)
        symbols = encode_trajectory(smoothed)
        if len(symbols) >= MIN_SEQUENCE_LENGTH:
            sequences.append(symbols)
            if trim and len(traj) < raw_len:
                print(f"  trimmed {npy_path.name}: {raw_len} -> {len(traj)} frames "
                      f"({len(symbols)} symbols)")
        else:
            print(f"  skip (too short after encoding, {len(symbols)} symbols): {npy_path.name}")
    return sequences


def train_gesture_hmm(
    sequences: list[np.ndarray],
    n_states: int = 5,
    n_iter: int = 100,
    random_state: int = 42,
) -> CategoricalHMM:
    """Fit a left-to-right-ish CategoricalHMM on a list of symbol sequences."""
    if not sequences:
        raise ValueError("no sequences to train on")

    # Concatenate sequences and track per-sequence lengths (hmmlearn convention).
    lengths = [len(s) for s in sequences]
    X = np.concatenate(sequences).reshape(-1, 1).astype(np.int64)

    model = CategoricalHMM(
        n_components=n_states,
        n_features=NUM_SYMBOLS,        # fix alphabet size = 24
        n_iter=n_iter,
        tol=1e-3,
        init_params="ste",              # init startprob, transmat, emissionprob
        random_state=random_state,
    )
    model.fit(X, lengths)
    smooth_emissions(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one HMM per gesture class.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--states", type=int, default=5,
                        help="Number of hidden states per HMM (default: 5).")
    parser.add_argument("--iter", type=int, default=100,
                        help="Max Baum-Welch iterations (default: 100).")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Centroid smoothing window in frames (default: 5).")
    parser.add_argument("--no-trim", action="store_true",
                        help="Disable automatic motion trimming (useful for A/B comparison).")
    args = parser.parse_args()
    trim = not args.no_trim
    if not trim:
        print("(motion trimming disabled)")

    if not args.data_root.exists():
        raise SystemExit(f"data root not found: {args.data_root}")
    args.models_dir.mkdir(parents=True, exist_ok=True)

    gesture_dirs = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if not gesture_dirs:
        raise SystemExit(f"no gesture sub-folders found under {args.data_root}")

    summary: list[tuple[str, int, int, float]] = []  # (gesture, n_samples, total_symbols, log_lik)

    for gdir in gesture_dirs:
        gesture = gdir.name
        print(f"\n=== {gesture} ===")
        sequences = load_and_encode(gdir, smooth_window=args.smooth_window, trim=trim)

        if len(sequences) < MIN_SAMPLES_PER_GESTURE:
            print(f"  SKIP: only {len(sequences)} usable samples "
                  f"(need >= {MIN_SAMPLES_PER_GESTURE})")
            continue

        total_symbols = sum(len(s) for s in sequences)
        avg_len = total_symbols / len(sequences)
        print(f"  samples: {len(sequences)}   total symbols: {total_symbols}   "
              f"avg seq len: {avg_len:.1f}")

        model = train_gesture_hmm(
            sequences,
            n_states=args.states,
            n_iter=args.iter,
        )

        # Mean log-likelihood per sample on the training set (sanity check; not a true score).
        train_ll = sum(model.score(s.reshape(-1, 1)) for s in sequences) / len(sequences)
        print(f"  trained {args.states}-state HMM   mean log-lik/sample: {train_ll:.2f}")

        out_path = args.models_dir / f"{gesture}.pkl"
        with out_path.open("wb") as f:
            pickle.dump({"gesture": gesture, "model": model}, f)
        print(f"  saved -> {out_path}")
        summary.append((gesture, len(sequences), total_symbols, train_ll))

    print("\n=== summary ===")
    if not summary:
        print("no models trained.")
        return
    print(f"{'gesture':<14} {'samples':>8} {'symbols':>8} {'log-lik/sample':>16}")
    for g, n, t, ll in summary:
        print(f"{g:<14} {n:>8d} {t:>8d} {ll:>16.2f}")


if __name__ == "__main__":
    main()
