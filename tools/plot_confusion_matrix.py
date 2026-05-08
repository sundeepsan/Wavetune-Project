"""
This program will be used to generate a publication-quality confusion-matrix heatmap from an 80/20 data split


"""
"""
Simple script to generate a publication-quality confusion matrix 
for gesture recognition (80/20 train/test split).
"""

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from src.features.encode import encode_trajectory
from src.features.segment import trim_to_motion
from src.features.smooth import moving_average
from src.hmm.train import MIN_SAMPLES_PER_GESTURE, MIN_SEQUENCE_LENGTH, train_gesture_hmm


def load_and_encode(path: Path, smooth_window: int):
    """Load trajectory, clean it, and encode it."""
    traj = np.load(path)
    traj = trim_to_motion(traj, smooth_window=smooth_window)
    smoothed = moving_average(traj, window=smooth_window)
    return encode_trajectory(smoothed)


def split_data(files, test_frac=0.2, seed=0):
    """Shuffle and split files into train/test."""
    files = files[:]
    random.Random(seed).shuffle(files)
    
    n_test = max(1, int(round(len(files) * test_frac)))
    n_test = min(n_test, len(files) - 1)  # leave at least one for training
    
    return files[n_test:], files[:n_test]  # train, test


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix for gestures")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("figures/confusion_matrix.png"))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gestures", nargs="*", help="Only use these gestures")
    parser.add_argument("--title", type=str, help="Custom plot title")
    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if not args.data_root.exists():
        raise SystemExit(f"Data folder not found: {args.data_root}")

    # Discover gesture folders
    gesture_files = {}
    for folder in sorted(args.data_root.iterdir()):
        if folder.is_dir():
            files = sorted(folder.glob("*.npy"))
            if files:
                gesture_files[folder.name] = files

    # Filter gestures if requested
    if args.gestures:
        gesture_files = {g: gesture_files[g] for g in args.gestures
                        if g in gesture_files}

    if not gesture_files:
        raise SystemExit("No gesture data found!")

    # Split into train/test
    train_files = {}
    test_files = {}
    for gesture, files in gesture_files.items():
        train, test = split_data(files, args.test_frac, args.seed)
        train_files[gesture] = train
        test_files[gesture] = test

    # Train one HMM per gesture
    models = {}
    for gesture, paths in train_files.items():
        sequences = [load_and_encode(p, args.smooth) for p in paths]
        sequences = [s for s in sequences if len(s) >= MIN_SEQUENCE_LENGTH]
        
        if len(sequences) < MIN_SAMPLES_PER_GESTURE:
            print(f"Skipping {gesture} — only {len(sequences)} usable samples")
            continue
            
        models[gesture] = train_gesture_hmm(
            sequences,
            n_states=args.states,
            n_iter=args.iterations
        )

    if not models:
        raise SystemExit("No models could be trained.")

    labels = sorted(models.keys())
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Evaluate on test set
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    correct = total = 0

    for true_label, paths in test_files.items():
        if true_label not in models:
            continue
            
        for path in paths:
            sequence = load_and_encode(path, args.smooth)
            if len(sequence) < MIN_SEQUENCE_LENGTH:
                continue
                
            X = np.asarray(sequence, dtype=np.int64).reshape(-1, 1)
            
            # Predict using the model with highest score
            scores = {g: m.score(X) for g, m in models.items()}
            predicted = max(scores, key=scores.get)
            
            cm[label_to_idx[true_label], label_to_idx[predicted]] += 1
            total += 1
            if predicted == true_label:
                correct += 1

    if total == 0:
        raise SystemExit("No valid test samples found.")

    accuracy = 100 * correct / total
    print(f"\nOverall accuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Plotting
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Row-normalized for coloring
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(1.6 * n + 1, 1.3 * n + 1), dpi=160)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    title = args.title or f"Gesture Confusion Matrix — {n} classes (acc: {accuracy:.1f}%)"
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add counts in cells
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(count), ha="center", va="center",
                   color=color, fontsize=11, fontweight=weight)

    plt.colorbar(im, ax=ax, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=200)
    
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
