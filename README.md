# WaveTune

An adaptive gesture-recognition controlled music player using Hidden Markov Models.

**COMS 5750 — Team 8**
Tejas Gosula, Deepak Sundaresan, Chandrashekar
Instructor: Professor Alexander Stoytchev

## Overview

WaveTune lets users control music playback (next/previous track, volume up/down) through hand or head gestures captured by a standard webcam. Motion is tracked with MediaPipe, encoded as a sequence of discrete direction/speed symbols, and classified with one Hidden Markov Model per gesture class.

The system supports two interaction modes:

- **Hand mode** — `hand_up`, `hand_down`, `swipe_left`, `swipe_right`. Default modality.
- **Head mode** — `head_up`, `head_down`, `head_left`, `head_right`. Designed as the accessibility modality for users who cannot use hand gestures.

Either mode can be used standalone via a single CLI flag (`--mode hand|head`).

## Pipeline

```
Webcam  ->  Landmark detection  ->  Centroid + velocity  ->  Symbol encoding  ->  HMM bank  ->  Media key
(OpenCV)    (MediaPipe Hands /     (avg landmark, frame    (24 symbols:        (CategoricalHMM,  (pyautogui)
            Face Mesh)              -to-frame delta)        8 dirs x 3 speeds)  smoothed,
                                                                                with margin gate)
```

Six stages, all live: capture -> landmarks -> features -> symbols -> classification -> command.

The realtime app adds a state machine on top that detects gesture START / END from rolling velocity, plus a direction-reversal trigger that closes the buffer when the user begins returning to neutral. This is what makes fast snappy gestures classify correctly without a long apex pause.

## Setup

Requires Python 3.11+ (for newer mediapipe).

```bash
git clone https://github.com/sundeepsan/Wavetune---Gesture-Controlled-Music-Player-using-HMMs.git
cd Wavetune---Gesture-Controlled-Music-Player-using-HMMs

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`mediapipe==0.10.18` is pinned because newer versions removed the `mp.solutions` namespace this project uses.

On macOS the first run will trigger camera permission and (if you skip `--no-keys`) accessibility permission. Enable both under **System Settings -> Privacy & Security**.

## Quickstart

Verify webcam + MediaPipe:

```bash
python src/capture/webcam_hands_demo.py     # 21 hand landmarks
python src/capture/webcam_face_demo.py      # 468-point face mesh
```

Record a few gesture samples (10+ per gesture is the working minimum):

```bash
python src/data_collection/record_gestures.py --gesture swipe_left  --mode hand
python src/data_collection/record_gestures.py --gesture swipe_right --mode hand
python src/data_collection/record_gestures.py --gesture hand_up     --mode hand
python src/data_collection/record_gestures.py --gesture hand_down   --mode hand
```

In each session, press **SPACE** to start a recording, do the gesture, press **SPACE** again to save, **q** to exit.

Train one HMM per gesture:

```bash
python src/hmm/train.py --states 3
```

Evaluate offline with an 80/20 train/test split:

```bash
python src/hmm/evaluate.py --states 3
```

Run the live recognizer:

```bash
python src/app/realtime.py --mode hand                    # default; fires media keys
python src/app/realtime.py --mode hand --no-keys --debug  # detect-only, prints symbols & scores
python src/app/realtime.py --mode head                    # head-gesture modality
python src/app/realtime.py --latency-log data/results/run.csv   # log per-gesture latencies
```

## What's in the boxes

`src/features/`

- `smooth.py` — moving-average smoothing of centroid trajectory
- `segment.py` — `trim_to_motion()` clips a recording to its longest contiguous high-velocity span; gets rid of bracket-the-gesture setup/withdrawal motion
- `encode.py` — converts a smoothed (T, 2) trajectory into a sequence of integer symbols 0..23 (8 directions x 3 speeds)

`src/hmm/`

- `train.py` — fits one CategoricalHMM per gesture, applies structured emission smoothing (direction-circular + speed-linear + Laplace floor) so unseen symbols at inference don't make scores -inf
- `classify.py` — `GestureClassifier` loads all `.pkl` models, scores a sequence under each, returns the top label only if its log-likelihood margin over the runner-up exceeds `min_margin`
- `evaluate.py` — train/test split harness with overall accuracy, per-class accuracy, and confusion matrix

`src/app/`

- `realtime.py` — the live application. Mode-aware (hand vs head) detector, motion-energy gate, direction-reversal close, classifier filter by mode, optional media-key emission, response-time instrumentation with optional CSV export

`src/data_collection/`

- `record_gestures.py` — SPACE-START / SPACE-STOP recording loop, saves `(T, 2)` trajectories under `data/raw/<gesture>/`

## Current performance (single-user, Deepak)

Trained on ~10–20 samples per gesture, 3 hidden states, structured emission smoothing.

**Hand mode (offline 80/20, averaged over 5 random seeds):**

- Overall accuracy: 88.3%
- Per-class: hand_up 100%, hand_down 100%, swipe_right 90%, swipe_left 75%
- Inference latency: ~1 ms per gesture

**Head mode (live test, 19 gestures):**

- 19/19 confident, 19/19 correct
- Margins range 2.8–90.8 (median ~30)
- Inference latency: ~0.8 ms per gesture
- Median gesture duration (start->end detected): 380 ms with reversal-detection

Multi-user evaluation and the proposal's full test-condition sweep (distance / lighting / background / speed) are still pending — see `CONTRIBUTING.md`.

## Project layout

```
.
├── data/
│   ├── raw/<gesture>/    # recorded .npy trajectories  (gitignored)
│   ├── debug/            # realtime --debug-dump output (gitignored)
│   └── results/          # response-time CSVs          (gitignored)
├── models/<gesture>.pkl  # trained HMMs                (gitignored)
├── src/
│   ├── capture/          # webcam + MediaPipe demos
│   ├── data_collection/  # record_gestures.py
│   ├── features/         # smooth, segment, encode
│   ├── hmm/              # train, classify, evaluate
│   └── app/              # realtime.py
├── tests/
├── requirements.txt
├── README.md
└── CONTRIBUTING.md
```

## Roadmap

- [x] Repo scaffold
- [x] Webcam + MediaPipe Hands demo
- [x] Webcam + MediaPipe Face Mesh demo
- [x] Gesture recording tool (hand + head modes)
- [x] Feature extraction (centroid, velocity, smoothing)
- [x] Symbol encoding (24 symbols)
- [x] HMM training (one model per gesture, with structured smoothing)
- [x] Train/test evaluation harness (accuracy, confusion matrix)
- [x] Real-time recognizer (motion-energy gate + direction-reversal close)
- [x] Music control hookup (pyautogui media keys)
- [x] Response-time instrumentation
- [ ] Multi-user data collection (Tejas, Chandrashekar)
- [ ] Test-condition sweep (distance, lighting, background, speed)
- [ ] Final report

## Team workflow

- `main` is protected; no direct pushes.
- Work on feature branches: `<name>/<feature>`, e.g. `tejas/head-gesture-data`.
- Open a pull request and ask a teammate to review before merging.
- For onboarding and recording instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
