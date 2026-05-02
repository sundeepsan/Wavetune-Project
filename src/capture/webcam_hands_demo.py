"""
Webcam + MediaPipe Hands demo.

Opens the default webcam, detects hand landmarks at ~30 fps, draws them on the
frame, and prints the wrist (landmark 0) coordinates to the console.

Press 'q' to quit.

This is Stage 1 + Stage 2 of the WaveTune pipeline. Run this first to confirm
your webcam and MediaPipe install both work before building anything else.

Usage:
    python src/capture/webcam_hands_demo.py
    python src/capture/webcam_hands_demo.py --camera 1     # secondary camera
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp


def run(camera_index: int = 0, max_hands: int = 1) -> None:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # On macOS, force AVFoundation backend; on other OSes use default.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)  # fallback to default backend
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    # Warm up: discard the first few frames; macOS often returns empty frames
    # for ~200ms after opening the device.
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    prev_t = time.time()
    fps = 0.0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        empty_streak = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                empty_streak += 1
                # Tolerate brief blips, but bail if it's persistent.
                if empty_streak < 30:
                    time.sleep(0.03)
                    continue
                print("Failed to read frame from webcam (no frames for ~1s).")
                break
            empty_streak = 0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            h, w = frame.shape[:2]

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    wrist = hand_landmarks.landmark[0]
                    wx, wy = int(wrist.x * w), int(wrist.y * h)
                    cv2.circle(frame, (wx, wy), 8, (0, 255, 0), -1)
                    print(f"wrist=({wrist.x:.3f}, {wrist.y:.3f}, {wrist.z:.3f})")

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            cv2.putText(
                frame,
                f"FPS: {fps:5.1f}   press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("WaveTune - Hands demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="WaveTune webcam + MediaPipe Hands demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--max-hands", type=int, default=1, help="Max hands to track.")
    args = parser.parse_args()
    run(camera_index=args.camera, max_hands=args.max_hands)


if __name__ == "__main__":
    main()
