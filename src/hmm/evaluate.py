"""
Train/test evaluation for the HMM gesture classifier.

Splits each gesture's samples into a train/test partition (default 80/20),
trains one CategoricalHMM per gesture on the train portion, predicts on the
test portion, and prints overall accuracy, per-class accuracy, and a
confusion matrix.

Run:
    python src/hmm/evaluate.py
    python src/hmm/evaluate.py --test-frac 0.25 --states 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import sys
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))

from src.features.encode import encode_trajectory
from src.features.segment import trim_to_motion
from src.features.smooth import moving_average
from src.hmm.train import (
    MIN_SAMPLES_PER_GESTURE,
    MIN_SEQUENCE_LENGTH,
    train_gesture_hmm,
)


def load_gesture_files(data_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for gdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = sorted(gdir.glob("*.npy"))
        if files:
            out[gdir.name] = files
    return out


def encode_file(path: Path, smooth_window: int, trim: bool = True) -> np.ndarray:
    traj = np.load(path)
    if trim:
        traj = trim_to_motion(traj, smooth_window=smooth_window)
    smoothed = moving_average(traj, window=smooth_window)
    return encode_trajectory(smoothed)


def split_train_test(
    files: list[Path], test_frac: float, rng: random.Random
) -> tuple[list[Path], list[Path]]:
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_frac))) if len(shuffled) >= 5 else 1
    if n_test >= len(shuffled):
        n_test = len(shuffled) - 1
    return shuffled[n_test:], shuffled[:n_test]


def print_confusion(matrix: dict[tuple[str, str], int], labels: list[str]) -> None:
    width = max(8, max(len(l) for l in labels))
    header = " " * (width + 2) + " ".join(f"{l[:width]:>{width}}" for l in labels)
    print(header)
    for true_label in labels:
        row = " ".join(
            f"{matrix.get((true_label, pred), 0):>{width}}" for pred in labels
        )
        print(f"{true_label:<{width}}  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the HMM classifier with a train/test split.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-trim", action="store_true",
                        help="Disable automatic motion trimming (A/B comparison).")
    args = parser.parse_args()
    trim = not args.no_trim
    if not trim:
        print("(motion trimming disabled)")

    rng = random.Random(args.seed)

    gesture_files = load_gesture_files(args.data_root)
    if not gesture_files:
        raise SystemExit(f"no gesture sub-folders found under {args.data_root}")

    # 1. Split files into train/test for each gesture.
    train_files: dict[str, list[Path]] = {}
    test_files: dict[str, list[Path]] = {}
    print(f"\n{'gesture':<14} {'total':>6} {'train':>6} {'test':>6}")
    for gesture, files in gesture_files.items():
        tr, te = split_train_test(files, args.test_frac, rng)
        train_files[gesture] = tr
        test_files[gesture] = te
        print(f"{gesture:<14} {len(files):>6d} {len(tr):>6d} {len(te):>6d}")

    # 2. Train one HMM per gesture using only the train partition.
    print("\nTraining...")
    trained: dict[str, object] = {}
    for gesture, paths in train_files.items():
        seqs = [encode_file(p, args.smooth_window, trim=trim) for p in paths]
        seqs = [s for s in seqs if len(s) >= MIN_SEQUENCE_LENGTH]
        if len(seqs) < MIN_SAMPLES_PER_GESTURE:
            print(f"  SKIP {gesture}: only {len(seqs)} usable train samples")
            continue
        model = train_gesture_hmm(seqs, n_states=args.states, n_iter=args.iter)
        trained[gesture] = model
        print(f"  {gesture}: trained on {len(seqs)} samples")

    if not trained:
        raise SystemExit("no models could be trained.")

    labels = sorted(trained.keys())

    # 3. Predict on the test set.
    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {l: 0 for l in labels}
    per_class_total: dict[str, int] = {l: 0 for l in labels}
    confusion: dict[tuple[str, str], int] = {}

    for true_gesture, paths in test_files.items():
        if true_gesture not in labels:
            continue
        for p in paths:
            symbols = encode_file(p, args.smooth_window, trim=trim)
            if len(symbols) < MIN_SEQUENCE_LENGTH:
                continue
            X = np.asarray(symbols, dtype=np.int64).reshape(-1, 1)
            scores = {g: float(m.score(X)) for g, m in trained.items()}
            pred = max(scores, key=scores.get)

            total += 1
            per_class_total[true_gesture] += 1
            confusion[(true_gesture, pred)] = confusion.get((true_gesture, pred), 0) + 1
            if pred == true_gesture:
                correct += 1
                per_class_correct[true_gesture] += 1

    if total == 0:
        raise SystemExit("no test samples could be evaluated.")

    print(f"\n=== Results ===")
    print(f"overall accuracy: {correct}/{total} = {100.0 * correct / total:.1f}%")
    print("\nper-class accuracy:")
    for l in labels:
        if per_class_total[l] == 0:
            print(f"  {l:<14} (no test samples)")
        else:
            pct = 100.0 * per_class_correct[l] / per_class_total[l]
            print(f"  {l:<14} {per_class_correct[l]}/{per_class_total[l]}  =  {pct:.1f}%")

    print("\nconfusion matrix (rows = true label, cols = predicted):")
    print_confusion(confusion, labels)


if __name__ == "__main__":
    main()
