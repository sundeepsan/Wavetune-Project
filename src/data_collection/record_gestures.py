"""
Record labeled gesture samples for HMM training.

Runs the webcam with MediaPipe (hand or face mode) and records a stream of
landmark centroids while you perform a gesture. Each recording is saved as
a NumPy array of shape (T, 2) holding (x, y) centroid coordinates per frame.

Controls:
    SPACE  - start / stop a recording
    q      - quit

Files are written to: data/raw/<gesture>/<gesture>_<timestamp>.npy

Recommended: collect ~20 samples per gesture per user, varying speed slightly.

Usage:
    python src/data_collection/record_gestures.py --gesture swipe_right --mode hand
    python src/data_collection/record_gestures.py --gesture head_up    --mode head
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


FACE_LANDMARK_INDICES = [1, 33, 263, 152]  # nose tip, eye corners, chin


def hand_centroid(hand_landmarks) -> tuple[float, float]:
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def face_centroid(face_landmarks) -> tuple[float, float]:
    xs = [face_landmarks.landmark[i].x for i in FACE_LANDMARK_INDICES]
    ys = [face_landmarks.landmark[i].y for i in FACE_LANDMARK_INDICES]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def save_recording(out_dir: Path, gesture: str, points: list[tuple[float, float]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    arr = np.asarray(points, dtype=np.float32)
    out_path = out_dir / f"{gesture}_{ts}.npy"
    np.save(out_path, arr)
    return out_path


def run(gesture: str, mode: str, camera_index: int, out_root: Path) -> None:
    if mode not in {"hand", "head"}:
        raise ValueError(f"mode must be 'hand' or 'head', got {mode!r}")

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh

    # On macOS, force AVFoundation backend; on other OSes use default.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    # Warm up: macOS often returns empty frames for ~200ms after opening.
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    out_dir = out_root / gesture

    detector_ctx = (
        mp_hands.Hands(model_complexity=0, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
        if mode == "hand"
        else mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
    )

    recording = False
    points: list[tuple[float, float]] = []
    n_saved = 0

    print(f"Recording '{gesture}' in '{mode}' mode -> {out_dir}")
    print("SPACE: start/stop  |  q: quit")

    with detector_ctx as detector:
        empty_streak = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                empty_streak += 1
                if empty_streak < 30:
                    time.sleep(0.03)
                    continue
                print("Failed to read frame from webcam (no frames for ~1s).")
                break
            empty_streak = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = detector.process(rgb)
            rgb.flags.writeable = True

            cx = cy = None
            if mode == "hand":
                if results.multi_hand_landmarks:
                    cx, cy = hand_centroid(results.multi_hand_landmarks[0])
            else:
                if results.multi_face_landmarks:
                    cx, cy = face_centroid(results.multi_face_landmarks[0])

            if cx is not None:
                px, py = int(cx * w), int(cy * h)
                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
                if recording:
                    points.append((cx, cy))

            color = (0, 0, 255) if recording else (200, 200, 200)
            status = "REC" if recording else "idle"
            cv2.putText(frame, f"[{status}] gesture={gesture}  saved={n_saved}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "SPACE: start/stop   q: quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

            cv2.imshow("WaveTune - record gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not recording:
                    points = []
                    recording = True
                    print("[REC] started")
                else:
                    recording = False
                    if len(points) >= 5:
                        path = save_recording(out_dir, gesture, points)
                        n_saved += 1
                        print(f"[REC] saved {len(points)} frames -> {path}")
                    else:
                        print(f"[REC] discarded - too short ({len(points)} frames)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {n_saved} sample(s) saved under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled gesture samples for WaveTune.")
    parser.add_argument("--gesture", required=True,
                        help="Gesture label, e.g. swipe_right, swipe_left, hand_up, head_down.")
    parser.add_argument("--mode", choices=["hand", "head"], default="hand")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    run(args.gesture, args.mode, args.camera, args.out_root)


if __name__ == "__main__":
    main()
