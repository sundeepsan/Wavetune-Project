"""
Webcam + MediaPipe Face Mesh demo.

Opens the default webcam, runs Face Mesh, draws the 468-landmark tessellation,
and prints the nose-tip (landmark 1) coordinates to the console. The nose tip
is a useful single-point proxy for head position when you don't yet need full
pitch/yaw/roll estimation.

Press 'q' to quit.

This is Stage 2 of the WaveTune pipeline for the head-gesture mode. Run after
webcam_hands_demo.py works, to confirm Face Mesh also runs on your machine.

Usage:
    python src/capture/webcam_face_demo.py
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp


NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
CHIN = 152


def run(camera_index: int = 0) -> None:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # On macOS, force AVFoundation backend; on other OSes use default.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    # Warm up: discard the first few frames; macOS often returns empty frames
    # for ~200ms after opening the device.
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    prev_t = time.time()
    fps = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            h, w = frame.shape[:2]

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles
                        .get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles
                        .get_default_face_mesh_contours_style(),
                    )
                    nose = face_landmarks.landmark[NOSE_TIP]
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    cv2.circle(frame, (nx, ny), 6, (0, 255, 255), -1)
                    print(f"nose=({nose.x:.3f}, {nose.y:.3f}, {nose.z:.3f})")

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
                (0, 255, 255),
                2,
            )

            cv2.imshow("WaveTune - Face Mesh demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="WaveTune webcam + MediaPipe Face Mesh demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    args = parser.parse_args()
    run(camera_index=args.camera)


if __name__ == "__main__":
    main()
