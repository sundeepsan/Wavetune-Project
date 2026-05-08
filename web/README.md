# WaveTune Web Player

A self-contained web music player that responds to gestures.



1. Drop one or more MP3 files into `web/tracks/` and name them `01.mp3`,
   `02.mp3`, `03.mp3` (or edit the `tracks` array in `index.html` to point at
   different paths). You can also click **+ Add tracks** in the UI to load
   audio from anywhere on your computer.

2. Open `web/index.html` in Chrome or any modern browser. Click the play
   button once to start playback.

3. In a separate terminal:

   ```
   python src/app/realtime.py --controller wavetune
   ```

   This starts a small local HTTP bridge at `localhost:7531`. The page polls
   it every 250 ms and applies whatever action comes back. The status
   indicator at the top of the player will switch from "listening" to
   "wavetune connected" when the bridge is up.

4. Perform a gesture. swipe_right skips to the next track, swipe_left goes
   back, hand_up / hand_down nudge volume up or down on the page itself.

## Controls

| Source | Action |
| --- | --- |
| Click play / pause button | Toggle |
| Click prev / next | Change track |
| Volume slider | Set page volume |
| Drag progress bar | Seek |
| Spacebar | Play / pause |
| Arrow Left / Right | Prev / next |
| Arrow Up / Down | Volume |

## How WaveTune connects

The `wavetune` controller runs a small HTTP server on `localhost:7531`. When
a gesture fires, it pushes a JSON action (`{"action": "next"}`,
`{"action": "volume_up"}`, etc.) into a queue. The web player polls
`/next` every 250 ms and dispatches each action it receives. Volume changes
adjust the on-page slider directly, not system volume.

For other media apps, the original backends still work:

| Controller | What it drives |
| --- | --- |
| `mediakey` | OS media keys via pyautogui (Spotify Web, YouTube Music, etc.) |
| `applemusic` | Apple Music desktop app via AppleScript |
| `spotify` | Spotify desktop app via AppleScript |
| `wavetune` | This web player via local HTTP bridge |
