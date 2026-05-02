"""
Real-time gesture recognition: webcam -> MediaPipe -> motion-energy gate ->
HMM classifier -> optional media-key emission.

Architecture
------------
Per frame:
  1. Read frame, run MediaPipe Hands, extract centroid (avg of 21 landmarks).
  2. Feed the centroid into a small state machine (`GestureSegmenter`):
       - idle      : watch for several consecutive high-velocity frames.
       - recording : append frames to an in-progress gesture buffer.
       - finished  : after several consecutive low-velocity frames, the gesture
                     buffer is closed off and handed to the classifier.
  3. On gesture-end: trim_to_motion -> moving_average -> encode_trajectory ->
     GestureClassifier.predict(). If the prediction is confident enough,
     emit a media key (next/prev track, vol up/down) and enter cooldown.

Constants/preproc must match record_gestures.py exactly so the live data
distribution matches the training distribution.

Run:
    python src/app/realtime.py
    python src/app/realtime.py --camera 1
    python src/app/realtime.py --no-keys             # detect, but don't fire media keys
    python src/app/realtime.py --min-margin 3.0      # stricter confidence gate
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Allow running as a script: add project root to sys.path.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))

from src.features.encode import encode_trajectory, symbol_to_label
from src.features.segment import trim_to_motion
from src.features.smooth import moving_average
from src.hmm.classify import GestureClassifier


# Default mapping: gesture label -> macOS media-key name (pyautogui).
# Adjust freely; these are the natural defaults for music control.
DEFAULT_KEY_MAP = {
    "swipe_left":  "prevtrack",
    "swipe_right": "nexttrack",
    "hand_up":     "volumeup",
    "hand_down":   "volumedown",
    # Head gestures (if/when trained):
    "head_up":     "volumeup",
    "head_down":   "volumedown",
    "head_left":   "prevtrack",
    "head_right":  "nexttrack",
}


# -------- centroid extraction (must match record_gestures.py) ----------------

# For head mode we average a small set of stable face landmarks: nose tip,
# eye outer corners, chin. This same 4-point centroid is what the recorder
# saves to disk, so live-time and training-time inputs agree.
FACE_LANDMARK_INDICES = [1, 33, 263, 152]   # nose tip, eye corners, chin


def hand_centroid(hand_landmarks) -> tuple[float, float]:
    """Average x, y of all 21 hand landmarks. Normalized to [0, 1]."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def face_centroid(face_landmarks) -> tuple[float, float]:
    """Average x, y of FACE_LANDMARK_INDICES. Normalized to [0, 1]."""
    xs = [face_landmarks.landmark[i].x for i in FACE_LANDMARK_INDICES]
    ys = [face_landmarks.landmark[i].y for i in FACE_LANDMARK_INDICES]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# -------- state machine: idle <-> recording <-> finished ---------------------

class GestureSegmenter:
    """
    Detects gesture START / END from rolling centroid velocity.

    The gate ends a gesture in three ways, in priority order:

      1. STOP-VELOCITY: speed falls below `stop_threshold` for `stop_frames`
         consecutive frames. This is the natural "user paused" trigger.

      2. DIRECTION-REVERSAL: after `reversal_min_frames` of recording, if the
         most recent motion direction opposes the dominant direction so far
         by more than `reversal_cos_threshold`, the user has started returning
         to neutral and we cut the buffer here. This catches fast snappy
         gestures (like a quick head-turn back-and-forth) that don't have a
         still moment at the apex - without this, the buffer captures both
         the outbound and return motion as one curve, and trim_to_motion
         picks whichever direction had more total displacement, frequently
         producing the WRONG label.

      3. MAX-BUFFER: hit `max_buffer` frames - safety valve.

    Thresholds are in normalized image coordinates per frame (units = fraction
    of frame width).
    """

    def __init__(
        self,
        start_threshold: float = 0.008,
        stop_threshold: float = 0.002,
        start_frames: int = 3,
        stop_frames: int = 8,
        max_buffer: int = 120,
        reversal_min_frames: int = 8,
        reversal_cos_threshold: float = -0.3,
    ):
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.max_buffer = max_buffer
        # Direction-reversal config:
        #  reversal_min_frames - require this much outbound motion before we
        #    even consider checking for reversal (otherwise initial jitter
        #    triggers false positives).
        #  reversal_cos_threshold - cosine of the angle between the dominant
        #    outbound direction and the recent direction. Range -1 (opposite)
        #    to +1 (same). -0.3 ~ 107 degrees apart, which catches U-turns
        #    while still tolerating natural ~60-90 degree curving gestures.
        self.reversal_min_frames = reversal_min_frames
        self.reversal_cos_threshold = reversal_cos_threshold

        # Always keep the last few raw points so we can include a small lead-in
        # when a gesture starts (the first few high-velocity frames).
        self.lookback: deque = deque(maxlen=start_frames + 3)
        self.recording = False
        self.gesture_points: list[tuple[float, float]] = []
        self.high_streak = 0
        self.low_streak = 0

    def reset(self) -> None:
        self.lookback.clear()
        self.recording = False
        self.gesture_points = []
        self.high_streak = 0
        self.low_streak = 0

    def _reversal_detected(self) -> bool:
        """Return True if recent motion opposes the gesture's dominant direction."""
        n = len(self.gesture_points)
        if n < self.reversal_min_frames:
            return False
        # Outbound direction: net displacement over the first half of the buffer.
        half = n // 2
        sx, sy = self.gesture_points[0]
        mx, my = self.gesture_points[half]
        out_dx, out_dy = mx - sx, my - sy
        # Recent direction: net displacement over the last 3 samples.
        rx0, ry0 = self.gesture_points[max(0, n - 4)]
        rx1, ry1 = self.gesture_points[n - 1]
        rec_dx, rec_dy = rx1 - rx0, ry1 - ry0
        # Magnitude check - skip if either vector is too small to have a
        # reliable direction.
        out_mag = (out_dx * out_dx + out_dy * out_dy) ** 0.5
        rec_mag = (rec_dx * rec_dx + rec_dy * rec_dy) ** 0.5
        if out_mag < 1e-6 or rec_mag < 1e-6:
            return False
        cos_angle = (out_dx * rec_dx + out_dy * rec_dy) / (out_mag * rec_mag)
        return cos_angle < self.reversal_cos_threshold

    def update(self, x: float, y: float) -> str:
        """
        Add a new centroid sample and advance the state machine.

        Returns one of:
            'idle'        - no gesture, none in progress
            'recording'   - a gesture is being collected
            'gesture-end' - a gesture just finished; call pop_buffer()
        """
        # Speed against the previous point. Empty buffer => speed = 0.
        if self.lookback:
            px, py = self.lookback[-1]
            dx, dy = x - px, y - py
            speed = (dx * dx + dy * dy) ** 0.5
        else:
            speed = 0.0
        self.lookback.append((x, y))

        if not self.recording:
            # Watch for start.
            if speed > self.start_threshold:
                self.high_streak += 1
            else:
                self.high_streak = 0
            if self.high_streak >= self.start_frames:
                # Begin recording: include the recent lookback as lead-in,
                # so the very first symbols of the gesture aren't lost.
                self.gesture_points = list(self.lookback)
                self.recording = True
                self.high_streak = 0
                self.low_streak = 0
                return "recording"
            return "idle"

        # Already recording.
        self.gesture_points.append((x, y))

        if speed < self.stop_threshold:
            self.low_streak += 1
        else:
            self.low_streak = 0

        if self.low_streak >= self.stop_frames:
            self.recording = False
            self.low_streak = 0
            return "gesture-end"

        # Direction-reversal: catches fast snappy gestures with no apex pause.
        if self._reversal_detected():
            self.recording = False
            self.low_streak = 0
            return "gesture-end"

        if len(self.gesture_points) >= self.max_buffer:
            self.recording = False
            self.low_streak = 0
            return "gesture-end"

        return "recording"

    def pop_buffer(self) -> np.ndarray:
        """Return the just-finished gesture as a (T, 2) float32 array."""
        arr = np.asarray(self.gesture_points, dtype=np.float32)
        self.gesture_points = []
        return arr


# -------- media-key emission --------------------------------------------------

def emit_key(key: str) -> bool:
    """Press a single media key via pyautogui. Returns True on success."""
    try:
        import pyautogui  # imported lazily; not all envs have it
        pyautogui.press(key)
        return True
    except Exception as e:
        print(f"  (failed to press {key!r}: {e})", file=sys.stderr)
        return False


# -------- main loop -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time HMM gesture recognition.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=["hand", "head"], default="hand",
                        help="Which gesture modality to recognize. 'hand' uses MediaPipe "
                             "Hands and the hand_/swipe_ gesture set; 'head' uses Face Mesh "
                             "and the head_ gesture set. Models for both modalities can "
                             "coexist in --models-dir; this flag picks which to score against.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--min-margin", type=float, default=2.0,
                        help="Minimum log-prob margin best-vs-second to accept a prediction "
                             "(default: 2.0 - stricter than the offline classifier default of 1.0).")
    parser.add_argument("--start-threshold", type=float, default=None,
                        help="Frame-to-frame speed (in normalized coords) above which we "
                             "consider motion is starting. Default: 0.008 hand / 0.002 head.")
    parser.add_argument("--stop-threshold", type=float, default=None,
                        help="Speed below which we consider motion has stopped. "
                             "Default: 0.002 hand / 0.0005 head.")
    parser.add_argument("--start-frames", type=int, default=3,
                        help="Consecutive high-speed frames needed to enter RECORDING.")
    parser.add_argument("--stop-frames", type=int, default=None,
                        help="Consecutive low-speed frames needed to leave RECORDING. "
                             "Default: 8 hand / 4 head (head closes the gate faster so the "
                             "return-to-neutral motion gets locked out by the cooldown).")
    parser.add_argument("--cooldown", type=float, default=None,
                        help="Seconds to ignore input after firing a prediction. "
                             "Default: 1.0 hand / 2.0 head (head needs longer to swallow "
                             "the slower return motion).")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--no-keys", action="store_true",
                        help="Don't actually press media keys. Just print predictions.")
    parser.add_argument("--debug", action="store_true",
                        help="Print full symbol sequence and per-class log-probs on every gesture.")
    parser.add_argument("--debug-dump", type=Path, default=None,
                        help="Save each captured trajectory as .npy under this dir for replay.")
    parser.add_argument("--latency-log", type=Path, default=None,
                        help="If set, write per-gesture latency rows to this CSV file. "
                             "Useful for the report's response-time table.")
    args = parser.parse_args()
    if args.debug_dump:
        args.debug_dump.mkdir(parents=True, exist_ok=True)
        print(f"Debug-dumping trajectories to {args.debug_dump}")

    # Mode-specific gate-threshold defaults. Heads move ~4x less in the frame
    # than hands during a typical gesture, so we need much tighter thresholds
    # to avoid the gate either never starting (if too high) or never stopping
    # (if equal to hand-mode values, the slow head residual motion never dips
    # below). Override either via CLI if these don't fit your setup.
    if args.start_threshold is None:
        args.start_threshold = 0.008 if args.mode == "hand" else 0.002
    if args.stop_threshold is None:
        args.stop_threshold = 0.002 if args.mode == "hand" else 0.0005
    if args.stop_frames is None:
        args.stop_frames = 8 if args.mode == "hand" else 4
    if args.cooldown is None:
        args.cooldown = 1.0 if args.mode == "hand" else 2.0

    # Which gesture prefixes to score against. Keeps a head HMM from ever
    # winning a hand-mode prediction (and vice versa) even though all .pkl
    # files live together in models/.
    mode_prefixes = ("hand_", "swipe_") if args.mode == "hand" else ("head_",)

    classifier = GestureClassifier(args.models_dir, min_margin=args.min_margin)
    # Restrict to the active mode's gesture vocabulary.
    classifier.models = {
        g: m for g, m in classifier.models.items() if g.startswith(mode_prefixes)
    }
    if not classifier.models:
        raise SystemExit(
            f"no {args.mode}-mode models found in {args.models_dir} "
            f"(looking for prefixes: {mode_prefixes}). "
            f"Record some {args.mode} gestures and re-run train.py."
        )
    print(f"Mode: {args.mode}.  Loaded {args.mode} models for: {classifier.gestures()}")
    if not args.no_keys:
        unmapped = [g for g in classifier.gestures() if g not in DEFAULT_KEY_MAP]
        if unmapped:
            print(f"  warning: no media-key mapping for: {unmapped}")

    segmenter = GestureSegmenter(
        start_threshold=args.start_threshold,
        stop_threshold=args.stop_threshold,
        start_frames=args.start_frames,
        stop_frames=args.stop_frames,
    )

    # ---- detector setup (modality-dependent) -----------------------------
    # Both branches expose a single `detector` object with .process(rgb), and
    # set up `extract_centroid(results)` which returns (cx, cy) or None.
    mp_drawing = mp.solutions.drawing_utils
    if args.mode == "hand":
        mp_hands = mp.solutions.hands
        detector = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        def extract_centroid(results):
            if not results.multi_hand_landmarks:
                return None, None
            lms = results.multi_hand_landmarks[0]
            cx, cy = hand_centroid(lms)
            return (cx, cy), lms
    else:  # head
        mp_face_mesh = mp.solutions.face_mesh
        detector = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        def extract_centroid(results):
            if not results.multi_face_landmarks:
                return None, None
            lms = results.multi_face_landmarks[0]
            cx, cy = face_centroid(lms)
            return (cx, cy), lms

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera at index {args.camera}.")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    last_prediction_text = ""
    last_prediction_at = 0.0
    cooldown_until = 0.0
    no_hand_streak = 0
    empty_streak = 0
    state = "idle"

    # ---- response-time instrumentation -------------------------------------
    # We track four timestamps per gesture so we can decompose total latency:
    #   gesture_started_at : when motion first crossed start_threshold
    #   t_end_detected     : when the segmenter declared the gesture finished
    #                        (lags real motion-end by stop_frames / fps seconds)
    #   t_predict_done     : after classifier.predict() returns
    #   t_key_done         : after pyautogui.press() returns
    # All durations get appended to `latencies` and summarized at exit.
    gesture_started_at: float | None = None
    latencies: list[dict] = []

    print("\n[realtime] running. Press q in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                empty_streak += 1
                if empty_streak > 30:
                    print("Camera stopped delivering frames.")
                    break
                time.sleep(0.03)
                continue
            empty_streak = 0

            frame = cv2.flip(frame, 1)            # mirror; matches recorder
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = detector.process(rgb)
            rgb.flags.writeable = True

            now = time.time()
            in_cooldown = now < cooldown_until

            centroid, lms = extract_centroid(results)
            cx = cy = None
            if centroid is not None:
                cx, cy = centroid
                # Lightweight overlay: hand mode draws the skeleton, head mode
                # just marks the centroid (drawing the full 468-point face mesh
                # every frame is wasteful and noisy on screen).
                if args.mode == "hand":
                    mp_drawing.draw_landmarks(
                        frame, lms, mp.solutions.hands.HAND_CONNECTIONS
                    )
                no_hand_streak = 0
            else:
                no_hand_streak += 1
                # If the subject disappears mid-gesture for too long, force-end
                # so we classify what we have rather than wait forever.
                if segmenter.recording and no_hand_streak > 12:
                    state = "gesture-end"
                    no_hand_streak = 0

            if cx is not None and not in_cooldown:
                prev_state = state
                state = segmenter.update(cx, cy)
                if state == "recording" and prev_state != "recording":
                    gesture_started_at = now

            if state == "gesture-end":
                t_end_detected = now            # set above as time.time()
                traj = segmenter.pop_buffer()
                segmenter.reset()
                pred_label, pred_lp, pred_margin, n_syms = (None, float("-inf"), 0.0, 0)
                trimmed_len = 0
                symbols = np.empty(0, dtype=np.int32)
                if len(traj) >= 5:
                    trimmed = trim_to_motion(traj, smooth_window=args.smooth_window)
                    trimmed_len = len(trimmed)
                    smoothed = moving_average(trimmed, window=args.smooth_window)
                    symbols = encode_trajectory(smoothed)
                    n_syms = len(symbols)
                    if n_syms >= 3:
                        pred_label, pred_lp, pred_margin = classifier.predict(symbols)
                t_predict_done = time.time()

                tstamp = time.strftime("%H:%M:%S")
                key_fired = False
                if pred_label is not None:
                    last_prediction_text = f"{pred_label}  margin={pred_margin:.1f}"
                    last_prediction_at = now
                    cooldown_until = now + args.cooldown
                    key = DEFAULT_KEY_MAP.get(pred_label)
                    if key and not args.no_keys:
                        emit_key(key)
                        key_fired = True
                t_key_done = time.time()

                # --- decompose the latency ---
                duration_ms = (
                    (t_end_detected - gesture_started_at) * 1000.0
                    if gesture_started_at is not None
                    else None
                )
                infer_ms = (t_predict_done - t_end_detected) * 1000.0
                key_ms = (t_key_done - t_predict_done) * 1000.0 if key_fired else None
                total_ms = (t_key_done - t_end_detected) * 1000.0

                # --- one consolidated print line per gesture ---
                if pred_label is not None:
                    head = (f"[{tstamp}] {pred_label:<12}  margin={pred_margin:>5.1f}   "
                            f"raw={len(traj):>3d}  trim={trimmed_len:>3d}  symbols={n_syms:>3d}")
                    timing = f"  infer={infer_ms:>4.1f}ms  total={total_ms:>4.1f}ms"
                    if key_fired:
                        timing += f"  -> {key}"
                    print(head + timing)
                else:
                    last_prediction_text = f"(uncertain margin={pred_margin:.1f})"
                    last_prediction_at = now
                    print(f"[{tstamp}] uncertain    margin={pred_margin:>5.1f}   "
                          f"raw={len(traj):>3d}  trim={trimmed_len:>3d}  symbols={n_syms:>3d}"
                          f"  infer={infer_ms:>4.1f}ms  total={total_ms:>4.1f}ms")

                latencies.append({
                    "timestamp": tstamp,
                    "predicted": pred_label if pred_label is not None else "uncertain",
                    "margin": round(pred_margin, 2) if pred_margin not in (float("inf"), float("-inf")) else None,
                    "raw_frames": len(traj),
                    "trim_frames": trimmed_len,
                    "n_symbols": n_syms,
                    "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
                    "infer_ms": round(infer_ms, 2),
                    "key_ms": round(key_ms, 2) if key_ms is not None else None,
                    "total_ms": round(total_ms, 2),
                    "key_fired": key_fired,
                })
                gesture_started_at = None

                if args.debug and n_syms > 0:
                    sym_preview = " ".join(symbol_to_label(int(s)) for s in symbols[:16])
                    if n_syms > 16:
                        sym_preview += "  ..."
                    print(f"           symbols: {sym_preview}")
                    if n_syms >= 3:
                        scores = classifier.score_all(symbols)
                        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                        score_line = "   ".join(f"{g}={lp:>6.1f}" for g, lp in ranked)
                        print(f"           scores : {score_line}")

                if args.debug_dump and len(traj) >= 5:
                    fname = f"realtime_{time.strftime('%Y%m%d_%H%M%S')}_{n_syms}sym.npy"
                    np.save(args.debug_dump / fname, traj)

                state = "idle"

            # ---------- HUD overlay ----------
            if in_cooldown:
                hud_state = "COOLDOWN"
                color = (0, 200, 255)
            elif state == "recording":
                hud_state = "RECORDING"
                color = (0, 0, 255)
            else:
                hud_state = "IDLE"
                color = (200, 200, 200)
            cv2.putText(frame, hud_state, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            buf_len = len(segmenter.gesture_points) if segmenter.recording else 0
            if buf_len:
                cv2.putText(frame, f"buf={buf_len}", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

            if last_prediction_text and now - last_prediction_at < 2.0:
                cv2.putText(frame, last_prediction_text, (15, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if cx is not None:
                cv2.circle(frame, (int(cx * w), int(cy * h)), 6, (0, 255, 0), -1)

            cv2.putText(frame, f"mode: {args.mode}", (w - 180, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "q: quit", (w - 110, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(f"WaveTune realtime [{args.mode}]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        # ---- response-time summary ----------------------------------------
        if latencies:
            print(f"\n=== response-time summary ({len(latencies)} gestures) ===")

            def _stats(name: str, values: list[float], unit: str = "ms") -> None:
                if not values:
                    return
                vals = sorted(values)
                mean = statistics.mean(vals)
                median = statistics.median(vals)
                p95 = vals[int(0.95 * (len(vals) - 1))]
                print(f"  {name:<20} mean={mean:>6.1f}{unit}  median={median:>6.1f}{unit}  "
                      f"p95={p95:>6.1f}{unit}  min={vals[0]:>6.1f}{unit}  max={vals[-1]:>6.1f}{unit}")

            _stats("inference",
                   [r["infer_ms"] for r in latencies if r["infer_ms"] is not None])
            _stats("total response",
                   [r["total_ms"] for r in latencies if r["total_ms"] is not None])
            _stats("gesture duration",
                   [r["duration_ms"] for r in latencies if r["duration_ms"] is not None])
            keyed = [r["key_ms"] for r in latencies if r["key_ms"] is not None]
            if keyed:
                _stats("key emit", keyed)

            confident = sum(1 for r in latencies if r["predicted"] != "uncertain")
            print(f"  confident:           {confident}/{len(latencies)} "
                  f"({100.0 * confident / len(latencies):.1f}%)")
            note = ("  note: response time excludes the gate's stop-detection wait "
                    f"(stop-frames * 1/fps ~ {args.stop_frames * 33}ms at 30fps).")
            print(note)

        # ---- optional CSV log ---------------------------------------------
        if args.latency_log and latencies:
            args.latency_log.parent.mkdir(parents=True, exist_ok=True)
            fields = ["timestamp", "predicted", "margin", "raw_frames", "trim_frames",
                      "n_symbols", "duration_ms", "infer_ms", "key_ms", "total_ms",
                      "key_fired"]
            with args.latency_log.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in latencies:
                    w.writerow(row)
            print(f"\nWrote {len(latencies)} latency rows to {args.latency_log}")


if __name__ == "__main__":
    main()
