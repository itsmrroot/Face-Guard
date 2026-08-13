# 🛡️ Face Guard

**Watches your webcam and locks your laptop automatically if it sees a face that isn't yours.**

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-contrib--python-5C3EE8)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

Uses OpenCV's Haar cascade for face detection and an LBPH recognizer for face
matching — no heavy dependencies like `dlib`, so it installs cleanly
everywhere.

> [!NOTE]
> Three Haar cascade files are bundled in `assets/` — frontal face, profile
> face, and upper body — since some `opencv-contrib-python` wheels (e.g. on
> newer Python versions) ship without their `cv2.data.haarcascades` data
> files. `face_guard.py` uses the bundled copies automatically and only
> falls back to `cv2.data.haarcascades` if one is missing.

## Contents

- [Features](#-features)
- [Getting started](#-getting-started)
- [Tuning](#-tuning)
- [Run it automatically at startup](#-run-it-automatically-at-startup)
- [Troubleshooting](#-troubleshooting)
- [Notes & limits](#-notes--limits)

## ✨ Features

- 👤 **Face enrollment** — capture training photos straight from your webcam
- 🧠 **Local recognition** — LBPH face matching, nothing sent over the network
- 🙈 **Hidden-face detection** — locks fast if a person is detected but their face isn't (covered, turned away, dodging the camera)
- 🫥 **Covered-lens detection** — locks almost instantly if the camera itself is physically blocked (a hand or object over the lens from behind, etc.)
- 🔒 **Automatic lock** — locks the screen the moment an unrecognized face, a hidden face, or a blocked lens is confirmed
- 🎥 **Live preview** — color-coded boxes and a real-time status line show exactly what the guard sees
- 📸 **Intruder snapshots** — auto-saved and organized by date
- 🖥️ **Cross-platform locking** — macOS, Linux, and Windows all supported

## 🚀 Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment, which
is recommended: `python -m venv venv && source venv/bin/activate` first.)

### 2. Enroll your face

```bash
python face_guard.py enroll --name yourname
```

A window opens showing your webcam feed. Move your head slightly (angles,
expressions, glasses on/off if you sometimes wear them) while it captures
~40 photos. Press `q` to stop early if needed.

You can enroll more than one person (e.g. yourself under multiple lighting
conditions, or a family member) by running `enroll` again with a different
`--name`.

### 3. Train the recognizer

```bash
python face_guard.py train
```

This builds a model from everyone you enrolled and saves it to
`trainer/trainer.yml`.

### 4. Run the guard

```bash
python face_guard.py run --owner yourname
```

A preview window opens showing the webcam feed with color-coded boxes:
**green** with your name for a recognized face, **red** with "STRANGER"
for an unrecognized one, **yellow** "FACE (side)" for a face detected in
profile, and **orange** "PERSON (no face)" when a person is detected but
no face at all. If the lens itself looks physically blocked, a red
**"CAMERA COVERED"** warning appears on the frame. A status line in the
top-left corner (also printed to the terminal) shows the live detection
state and every streak counter, e.g.
`[PERSON, NO FACE] face=False side=False person=True covered=False stranger=0/8 person_no_face=3/5 camera_covered=0/2 no_face=3/50`
— use it to see exactly what the guard is seeing while you test.

Each check runs a small pipeline:

0. **Covered-lens check** — the frame's pixel variance (`np.std`) is
   checked first, before any cascade runs. A physically blocked lens
   produces an almost flat, textureless image, unlike even an empty room
   (which still has plenty of visual detail) — so this can safely lock
   almost instantly without being confused for "the owner walked away".
   If triggered, none of the steps below run for that frame.
1. **Frontal face** — detected and run through the LBPH recognizer to
   decide owner vs. stranger.
2. **Profile face** — if no frontal face is found, checks for a face
   turned to the side (checked in both directions), so a head turn isn't
   mistaken for "no face at all". Not run through the recognizer (LBPH is
   trained on frontal photos), it just counts as "a face is present".
3. **Upper body** — if no face at all (front or side) is found, checks
   whether a person is still in frame. This is what tells "the owner
   walked away" (nothing detected) apart from "someone's sitting right
   there with their face hidden, covered, or turned away".

It will:
- Check the webcam every ~0.3 seconds
- If the lens looks physically covered/blocked for
  `CAMERA_COVERED_LOCK_FRAMES` checks in a row, it **locks the screen
  almost instantly**
- If it sees a face that doesn't match `--owner` (or doesn't match well
  enough) for `CONSEC_STRANGER_FRAMES` checks in a row, it **locks the
  screen**
- If it detects a **person with no face at all** (front or side) for
  `PERSON_NO_FACE_LOCK_FRAMES` checks in a row, it **locks the screen too**
  — this is what catches someone deliberately hiding their face
- Walking away and leaving an empty, person-free desk **never** locks the
  screen by default — see `LOCK_ON_NO_FACE` in [Tuning](#-tuning) if you
  want an extra, stricter fallback that locks after a long period with
  nothing detected at all (this also locks when you simply walk away,
  since that looks identical to the camera)
- Save a snapshot of the stranger to
  `intruder_snapshots/<YYYY-MM-DD>/stranger_<HH-MM-SS>.jpg` (one subfolder
  per day, so it's easy to browse who showed up when)
- Log lock events to `face_guard.log`

Stop it any time with `Ctrl+C`, or press `q` in the preview window.

## 🎛️ Tuning

Open `face_guard.py` and adjust the constants near the top if needed:

| Setting | What it does |
|---|---|
| `CONFIDENCE_THRESHOLD` | Lower = stricter matching (fewer false "it's you"s, but more false lockouts). Default `65`. |
| `CONSEC_STRANGER_FRAMES` | How many bad frames in a row before locking. Higher = fewer accidental locks, slower to react. |
| `PERSON_NO_FACE_LOCK_FRAMES` | How many checks in a row with a person detected but no face (front or side) before locking. Default `5` (~1.5s). This is what catches a hidden face — always on, not gated by `LOCK_ON_NO_FACE`. |
| `CAMERA_COVERED_STD_THRESHOLD` | Grayscale pixel std-dev below which the lens is considered physically covered/blocked. Default `12`. Lower = requires a more totally featureless frame (fewer false positives, may miss a translucent cover); higher = more sensitive (may false-trigger in a very dark, low-detail room). |
| `CAMERA_COVERED_LOCK_FRAMES` | How many checks in a row the lens must look covered before locking. Default `2` (near-instant) — always on, not gated by `LOCK_ON_NO_FACE`, since this signal can't mean "owner walked away". |
| `LOCK_ON_NO_FACE` | Default `False`. Set `True` to *also* lock when literally nothing (no face, no person, no covered-lens signal) is detected for too long (see `NO_FACE_LOCK_FRAMES`) — catches edge cases the two checks above miss, but **also locks when you simply walk away**, since an empty frame looks the same either way. |
| `NO_FACE_LOCK_FRAMES` | (only matters if `LOCK_ON_NO_FACE = True`) checks in a row with nothing detected at all before locking anyway. Default `50` (~15s). |
| `LOCK_COOLDOWN_SECONDS` | Minimum time between lock triggers. |
| `CHECK_INTERVAL` | Seconds between webcam checks. Lower = more CPU use, faster reaction. |

## 🔁 Run it automatically at startup

| OS | How |
|---|---|
| **Linux** | Add a `systemd --user` service or an autostart `.desktop` entry that runs `python /path/to/face_guard.py run --owner yourname`. |
| **macOS** | Create a LaunchAgent plist in `~/Library/LaunchAgents/`. |
| **Windows** | Add a shortcut to the script in your Startup folder, or use Task Scheduler to run it at logon. |

## 🩺 Troubleshooting

Watch the status line in the preview window / terminal (see
[Run the guard](#4-run-the-guard)) while testing — it tells you exactly
which of these is happening.

<details>
<summary><strong>[NOTHING] even when someone's clearly in frame</strong></summary>

None of the three cascades (frontal face, profile face, upper body) are
detecting anything — too far from the camera, bad lighting, an odd angle,
or too little of the upper body in frame. With the default
`LOCK_ON_NO_FACE = False` this never locks by itself, and it also won't
reach `[PERSON, NO FACE]` since no person was detected either. If you want
detection from further away, lower the `minSize=(80, 80)` arguments passed
to `detectMultiScale` for the face cascades in `enroll()`/`run()`, and/or
the upper-body cascade's `minSize=(80, 80)` in `run()` — smaller values
detect smaller/farther subjects but raise the risk of false detections.
</details>

<details>
<summary><strong>[PERSON, NO FACE] never appears / never reaches the lock</strong></summary>

The upper-body cascade is less reliable than the face ones — it's tuned
for a certain distance and framing, so sitting very close to (or far from)
the webcam can make it miss a person entirely. Watch the `person=` field
in the status line while testing; if it stays `False` even with someone
clearly in frame, try adjusting the `scaleFactor`/`minNeighbors`/`minSize`
passed to `upperbody_cascade.detectMultiScale(...)` in `run()`, or sit so
more of your upper body is visible to the camera.
</details>

<details>
<summary><strong>[CAMERA COVERED] doesn't trigger when the lens is blocked, or false-triggers when it isn't</strong></summary>

Watch the `covered=` field in the status line while testing:
- **Doesn't trigger when actually covered**: some covering materials (thin
  cloth, a translucent object) still let enough light/detail through to
  keep the frame's pixel variance above the threshold. Raise
  `CAMERA_COVERED_STD_THRESHOLD` slightly (e.g. `15`–`20`) to make the
  check more sensitive.
- **False-triggers in a dim room**: a very dark but otherwise normal room
  can occasionally dip close to the threshold. Lower
  `CAMERA_COVERED_STD_THRESHOLD` slightly (e.g. `8`–`10`) to require a
  more totally featureless frame before it counts as covered.
</details>

<details>
<summary><strong>[OWNER OK] for someone who isn't the owner</strong></summary>

The LBPH recognizer is confidently (mis)matching a stranger to the owner.
This is a training problem, not a detection problem: 40 photos from one
sitting/one lighting condition can be too easy to match. Fix by lowering
`CONFIDENCE_THRESHOLD` (stricter matching) and/or re-running `enroll` for
more samples across different angles/lighting, then `train` again.
</details>

<details>
<summary><strong>[STRANGER] appears but never reaches the lock</strong></summary>

Streaks reset to 0 the moment a frame doesn't detect any face (flaky
detection interrupts the streak) or the moment the owner is confirmed.
Make sure the stranger's face stays detected (green/red box visible)
continuously for `CONSEC_STRANGER_FRAMES` (default 8, ~2.4s) — a
flickering box means detection is intermittent, see the point above.
</details>

<details>
<summary><strong>Log says "Screen locked" on macOS but nothing visibly happens</strong></summary>

This means `lock_screen()` ran a command successfully, not that it
definitely locked the display. Check `face_guard.log` for *which* method
it used (`Screen locked via 'open'` / `'osascript'` / `'pmset'`):

- `open` (launches `ScreenSaverEngine` directly) needs no special
  permission and should always show the lock screen — this is tried first.
- `osascript` (sends the ⌃⌘Q shortcut) needs Accessibility permission
  granted to whatever runs the script (Terminal/iTerm/your IDE) under
  System Settings → Privacy & Security → Accessibility. It's only reached
  if `open` fails.
- `pmset displaysleepnow` is the last resort — it only *looks* like a lock
  if your Mac is set to require a password immediately after sleep (System
  Settings → Lock Screen → "Require password after screen saver begins or
  display is turned off" → Immediately).
</details>

## ⚠️ Notes & limits

- This is a convenience layer on top of your OS's normal password/PIN lock —
  it locks the screen, it doesn't replace your login credentials. Someone
  can still unlock with your password once it's locked, same as always.
- Face recognition (especially LBPH) isn't foolproof — photos, low light,
  similar-looking people, etc. can occasionally fool it. Don't treat this as
  your *only* line of defense; keep a strong OS password/PIN as the real
  security boundary.
- The upper-body detector (used for the "person present, face hidden" lock)
  is a coarser, less reliable Haar cascade than the face ones — it may miss
  a person depending on distance/framing, or occasionally false-trigger on
  background shapes. Watch the status line and tune `scaleFactor`/
  `minNeighbors`/`minSize` in `run()` for your setup.
- The webcam light will turn on whenever the guard checks the camera.