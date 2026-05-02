# Contributing to WaveTune

This is the team-internal onboarding doc. Read it before recording gesture samples or pushing changes.

## First-time setup

```bash
git clone https://github.com/sundeepsan/Wavetune---Gesture-Controlled-Music-Player-using-HMMs.git
cd Wavetune---Gesture-Controlled-Music-Player-using-HMMs

# Use Python 3.11. On macOS:  brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick sanity check:

```bash
python src/capture/webcam_hands_demo.py
```

If the camera window opens and shows your hand outlined with 21 dots, you're good. Press `q` to quit.

If you get **camera not authorized** on macOS: **System Settings -> Privacy & Security -> Camera -> enable Terminal** (or whichever app you're running Python in). Quit and relaunch the terminal afterwards.

## Recording gesture samples (your main job)

Each teammate needs to record both hand and head gestures so we can evaluate cross-user generalization. We're aiming for **~10 samples per gesture per person**, total of 8 gestures.

### Hand gestures (4 classes)

```bash
python src/data_collection/record_gestures.py --gesture swipe_left  --mode hand
python src/data_collection/record_gestures.py --gesture swipe_right --mode hand
python src/data_collection/record_gestures.py --gesture hand_up     --mode hand
python src/data_collection/record_gestures.py --gesture hand_down   --mode hand
```

### Head gestures (4 classes)

```bash
python src/data_collection/record_gestures.py --gesture head_left  --mode head
python src/data_collection/record_gestures.py --gesture head_right --mode head
python src/data_collection/record_gestures.py --gesture head_up    --mode head
python src/data_collection/record_gestures.py --gesture head_down  --mode head
```

### Recording technique (read this — it matters)

For each session:

1. Position yourself ~50 cm from the webcam, well-lit, plain background ideally.
2. Hand gestures: get your hand into starting position **before** pressing SPACE.
3. Press **SPACE** to start a recording.
4. Do the gesture cleanly:
   - **Swipes**: straight horizontal motion, no arc, no winding up
   - **Hand up/down**: straight vertical motion
   - **Head turns/nods**: decisive, not subtle
5. **Hold the endpoint position for ~0.5 second.** This is the most important rule. The recorder ends when you press SPACE; if your hand or head is already moving back to neutral when you press SPACE, the return motion gets bundled into the gesture and the encoder will misread it.
6. Press **SPACE** again to save.
7. Repeat ~10 times.
8. Press **q** to quit when done.

Tip: vary the speed slightly across reps (some slow, some fast). The HMM benefits from a bit of within-class variance.

Tip: if a recording got messed up (wrong gesture, hand left frame, sneeze) just don't worry — the recorder doesn't save discarded ones, and we can clean up obvious bad files manually before training. Just keep going.

## Sharing your recorded data

`data/raw/` is **gitignored** — your `.npy` files don't go in commits. Instead:

1. Zip your `data/raw/` folder
2. Send it to Deepak (Slack DM or email) with the subject `WaveTune samples - <your name>`
3. Deepak will merge into the team dataset and retrain

If we end up with too much friction, we'll set up a shared Drive folder.

## Test condition recordings (do this *after* the basic 10/gesture set)

The proposal commits us to evaluating across 5 conditions. After your initial 10/gesture, please record an additional 3-5 samples of each gesture under each of these conditions:

- **Distance**: stand ~1.5 meters from the camera (further than usual)
- **Dim lighting**: turn down the room lights to roughly half normal brightness
- **Cluttered background**: do the gestures in front of a busy background (bookshelf, kitchen, etc.)
- **Fast speed**: do each gesture noticeably faster than your "normal" pace
- **Slow speed**: do each gesture noticeably slower than your "normal" pace

Save these as separate folders so we can evaluate per-condition: e.g. `data/raw_distance/swipe_left/...`, `data/raw_dimlight/...`. Use the recorder's `--out-root` flag:

```bash
python src/data_collection/record_gestures.py --gesture swipe_left --mode hand --out-root data/raw_distance
```

## Code changes

If you want to modify the codebase (not just record data):

1. Branch off main: `git checkout -b <yourname>/<short-feature-name>`
2. Make your changes. Keep them focused — one logical change per PR.
3. Push and open a PR. Tag a teammate to review.

Don't push:
- Anything in `data/`, `models/`, `data/debug/`, `data/results/`
- Your `.venv/`
- IDE-specific files (`.vscode/`, `.idea/`)

These are all gitignored, but double-check with `git status` before committing.

## Getting unblocked

Common issues:

- **`ModuleNotFoundError: No module named 'cv2'` (or mediapipe, hmmlearn, etc.)** — your venv isn't activated, or `pip install -r requirements.txt` didn't finish. Re-activate and re-install.
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`** — wrong mediapipe version. The pin in `requirements.txt` is `0.10.18` because newer versions deleted the API we use. Run `pip install mediapipe==0.10.18`.
- **Camera opens on iPhone instead of Mac webcam** — that's macOS Continuity Camera. Lock your iPhone, or pass `--camera 1` to use the next camera index, or disable Continuity Camera in System Settings.
- **Failed to read frame from webcam** — camera permission isn't granted yet, or another app is using the camera. Quit any video-call app and re-run.

If something else breaks, post in the team chat with the full error message and the command you ran. Don't suffer in silence.
