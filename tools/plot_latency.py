
"""
Plot latency statistics from realtime gesture logs.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_csv(csv_path: Path):
    import csv
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def get_column(name):
        vals = []
        for r in rows:
            v = r.get(name, "").strip()
            if v and v.lower() != "none":
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return np.array(vals, dtype=float)

    return {
        "infer_ms": get_column("infer_ms"),
        "total_ms": get_column("total_ms"),
        "duration_ms": get_column("duration_ms"),
        "key_ms": get_column("key_ms"),
        "n": len(rows)
    }


def print_stats(name, values):
    if len(values) == 0:
        print(f"  {name:<20} (no data)")
        return
    print(f"  {name:<20} n={len(values):>3}  "
          f"mean={values.mean():>6.1f}  median={np.median(values):>6.1f}  "
          f"p95={np.percentile(values, 95):>6.1f} ms")


def main():
    p = argparse.ArgumentParser(description="Plot gesture latency stats")
    p.add_argument("csv", type=Path, help="Path to latency CSV from realtime.py")
    p.add_argument("--out", type=Path, default=Path("figures/latency.png"))
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"File not found: {args.csv}")

    data = load_csv(args.csv)

    print(f"\nLatency stats ({data['n']} gestures):")
    print_stats("HMM Inference", data["infer_ms"])
    print_stats("Total Response", data["total_ms"])
    print_stats("Gesture Duration", data["duration_ms"])
    if len(data["key_ms"]) > 0:
        print_stats("Key Press", data["key_ms"])

    # Plot
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)

    panels = [
        ("HMM Inference (ms)", data["infer_ms"], "#3a6ea5"),
        ("Total Response (ms)", data["total_ms"], "#4a8e62"),
        ("Gesture Duration (ms)", data["duration_ms"], "#a35434"),
    ]

    for ax, (title, values, color) in zip(axes, panels):
        if len(values) == 0:
            continue
        ax.hist(values, bins=15, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.median(values), color="black", linestyle="--", label=f"median {np.median(values):.1f}")
        ax.set_title(title)
        ax.set_xlabel("milliseconds")
        ax.set_ylabel("count")
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
