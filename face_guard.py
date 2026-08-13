#!/usr/bin/env python3
"""
Face Guard - webcam-based laptop lock.

Modes:
    enroll  - capture face photos of the owner
    train   - train the recognizer on enrolled photos
    run     - watch the webcam and lock the screen if a stranger is detected

Usage:
    python face_guard.py enroll --name yourname
    python face_guard.py train
    python face_guard.py run

Requires: opencv-contrib-python  (pip install opencv-contrib-python --break-system-packages)
"""

import argparse
import os
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trainer")
MODEL_PATH = os.path.join(MODEL_DIR, "trainer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "intruder_snapshots")
LOG_PATH = os.path.join(BASE_DIR, "face_guard.log")

_CASCADE_DIR = os.path.join(BASE_DIR, "assets")


def _cascade_path(filename):
    bundled = os.path.join(_CASCADE_DIR, filename)
    return bundled if os.path.exists(bundled) else os.path.join(cv2.data.haarcascades, filename)


FACE_CASCADE_PATH = _cascade_path("haarcascade_frontalface_default.xml")
PROFILE_CASCADE_PATH = _cascade_path("haarcascade_profileface.xml")
UPPERBODY_CASCADE_PATH = _cascade_path("haarcascade_upperbody.xml")
EYE_CASCADE_PATH = _cascade_path("haarcascade_eye.xml")

# Recognition tuning
CONFIDENCE_THRESHOLD = 65      # LBPH: LOWER distance = better match. Below this = "you".
ENROLL_CAPTURE_INTERVAL = 0.35 # minimum seconds between saved enrollment photos -- without this,
                                # a continuously-detected face gets all ~40 samples captured in about
                                # a second, producing near-duplicate photos with zero real pose/
                                # lighting variation, which is a common cause of poor recognition later
CONSEC_STRANGER_FRAMES = 8     # how many bad (frontal-face) frames in a row before locking
CONSEC_NO_FACE_OK = True       # if True, a brief empty frame (nobody there) does NOT count as a stranger
PERSON_NO_FACE_LOCK_FRAMES = 5 # if a PERSON (upper body) is detected but NO face (front or side) for
                                # this many checks in a row, lock quickly -- this is what catches someone
                                # sitting right there with their face hidden/covered/turned away
LOCK_ON_NO_FACE = False        # if True, ALSO lock after NO_FACE_LOCK_FRAMES checks with nothing
                                # detected at all (no face, no person) -- this ALSO locks when the owner
                                # just walks away, since an empty frame looks the same either way.
                                # Default False: walking away never locks; only an unrecognized face, or
                                # a detected person with a hidden face, does.
NO_FACE_LOCK_FRAMES = 50       # (only used if LOCK_ON_NO_FACE) checks with nothing detected before locking
CAMERA_COVERED_STD_THRESHOLD = 12   # grayscale std-dev below this means the lens is almost certainly
                                     # physically covered/blocked -- a real scene (even an empty room)
                                     # has far more visual texture than a hand/object pressed on the lens
CAMERA_COVERED_LOCK_FRAMES = 2      # consecutive covered checks before locking -- near-instant, since
                                     # this signal is unambiguous and can't mean "owner just walked away"
LOCK_COOLDOWN_SECONDS = 30     # don't re-trigger a lock spam within this window
CHECK_INTERVAL = 0.3           # seconds between full detection/recognition checks (lower = more
                                # responsive, more CPU)
DISPLAY_REFRESH_MS = 50        # how often the preview window redraws between checks -- a CSRT tracker
                                # glides the box smoothly across these frames so it visibly follows
                                # the face/body instead of only updating once per CHECK_INTERVAL
MAX_TRACKED_OBJECTS = 5        # cap on how many faces/bodies get their own tracker at once -- each
                                # CSRT tracker is fairly CPU-heavy, so this bounds worst-case load

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def load_cascade(path):
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        filename = os.path.basename(path)
        sys.exit(
            f"ERROR: failed to load Haar cascade from '{path}'. "
            "The file is missing or corrupt. Re-download it from "
            f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename} "
            f"and save it to '{os.path.join(_CASCADE_DIR, filename)}'."
        )
    return cascade


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face(gray_frame, box, eye_cascade=None):
    """Crops a face region and normalizes it for LBPH: eye-based rotation alignment (when two
    eyes can be found), resize to a fixed size, then CLAHE for lighting-robust contrast. Used
    identically at both enrollment and recognition time, since LBPH needs consistent preprocessing
    on both sides to compare fairly."""
    x, y, w, h = box
    face = gray_frame[y:y + h, x:x + w]
    if face.size == 0:
        return None

    if eye_cascade is not None:
        eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        if len(eyes) >= 2:
            # Use the two largest detections (most likely the real eyes, not eyebrows/glasses
            # glare), ordered left-to-right, and rotate so the line between them is level.
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = sorted(eyes, key=lambda e: e[0])
            c1 = (ex1 + ew1 / 2, ey1 + eh1 / 2)
            c2 = (ex2 + ew2 / 2, ey2 + eh2 / 2)
            angle = np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0]))
            center = (face.shape[1] / 2, face.shape[0] / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            face = cv2.warpAffine(face, rot_mat, (face.shape[1], face.shape[0]))

    face = cv2.resize(face, (200, 200))
    return _clahe.apply(face)


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Screen locking (cross platform)
# ---------------------------------------------------------------------------
def lock_screen():
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            ctypes.windll.user32.LockWorkStation()
        elif system == "Darwin":  # macOS
            # CGSession -suspend was removed in newer macOS versions.
            candidates = [
                # Launches the screen-saver engine directly -- this needs no
                # special permission and immediately shows the lock screen
                # if "require password after screen saver begins" is on
                # (System Settings -> Lock Screen), which is the default.
                ["open", "-a", "ScreenSaverEngine"],
                # Standard "Lock Screen" keyboard shortcut; requires
                # Accessibility permission for whatever runs this script.
                ["osascript", "-e",
                 'tell application "System Events" to keystroke "q" using {control down, command down}'],
                # Last resort: just sleep the display.
                ["pmset", "displaysleepnow"],
            ]
            locked_with = None
            for cmd in candidates:
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    locked_with = cmd[0]
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            if locked_with is None:
                log("WARNING: could not lock screen with any known macOS method.")
                return
            log(f"Screen locked via '{locked_with}'.")
            return
        elif system == "Linux":
            # Try a few common lock commands until one works
            candidates = [
                ["loginctl", "lock-session"],
                ["xdg-screensaver", "lock"],
                ["gnome-screensaver-command", "--lock"],
                ["dm-tool", "lock"],
            ]
            for cmd in candidates:
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        else:
            log(f"WARNING: unsupported OS '{system}', cannot lock screen.")
            return
        log("Screen locked.")
    except Exception as e:
        log(f"ERROR locking screen: {e}")


# ---------------------------------------------------------------------------
# Enroll
# ---------------------------------------------------------------------------
def enroll(name, num_samples=40):
    face_cascade = load_cascade(FACE_CASCADE_PATH)
    eye_cascade = load_cascade(EYE_CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("ERROR: could not open webcam.")
        sys.exit(1)

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Continue numbering from whatever's already there instead of overwriting it -- this lets you
    # run enroll again on a different day/lighting/angle to build a richer, more varied dataset
    # instead of each session destroying the last one.
    existing_indices = [
        int(os.path.splitext(f)[0]) for f in os.listdir(person_dir)
        if os.path.splitext(f)[0].isdigit() and f.lower().endswith(".jpg")
    ]
    count = max(existing_indices) if existing_indices else 0
    target = count + num_samples
    last_capture_time = 0.0
    prompts = ["look straight ahead", "turn slightly left", "turn slightly right",
               "tilt your head up a bit", "tilt your head down a bit", "normal / relaxed"]
    print(f"Look at the camera. Capturing {num_samples} more samples for '{name}' -- "
          f"move your head slowly through different angles as it captures. Press 'q' to stop early.")

    while count < target:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        now = time.time()
        can_capture = (now - last_capture_time) >= ENROLL_CAPTURE_INTERVAL

        for (x, y, w, h) in faces:
            color = (0, 255, 0) if can_capture else (0, 200, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if can_capture:
                face_img = preprocess_face(gray, (x, y, w, h), eye_cascade)
                if face_img is not None:
                    count += 1
                    last_capture_time = now
                    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)
                    cv2.putText(frame, f"{count}/{target}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            break  # one face per frame is enough

        prompt = prompts[count % len(prompts)]
        cv2.putText(frame, f"Now: {prompt}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Enroll - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log(f"'{name}' now has {count} enrolled samples total in {person_dir}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train():
    if not hasattr(cv2, "face"):
        log("ERROR: cv2.face not found. Install opencv-contrib-python (not plain opencv-python).")
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    images = []
    labels = []
    label_map = {}
    next_label_id = 0

    people = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not people:
        log("ERROR: no enrolled people found. Run 'enroll' first.")
        sys.exit(1)

    for person in people:
        label_map[next_label_id] = person
        person_dir = os.path.join(DATASET_DIR, person)
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(next_label_id)
        next_label_id += 1

    if not images:
        log("ERROR: no training images found.")
        sys.exit(1)

    recognizer.train(images, np.array(labels))
    recognizer.save(MODEL_PATH)

    # Calibrate the match threshold from the data itself instead of relying purely on the fixed
    # CONFIDENCE_THRESHOLD constant: re-predict every training image (LBPH: lower = better match)
    # and set the cutoff a bit above the worst self-match, so it's tailored to how distinguishable
    # this particular dataset actually is instead of a one-size-fits-all guess.
    self_confidences = []
    for img, true_label in zip(images, labels):
        predicted_label, confidence = recognizer.predict(img)
        if predicted_label == true_label:
            self_confidences.append(confidence)

    if self_confidences:
        calibrated_threshold = float(np.mean(self_confidences) + 2 * np.std(self_confidences))
        calibrated_threshold = max(40.0, min(calibrated_threshold, 100.0))
    else:
        calibrated_threshold = CONFIDENCE_THRESHOLD

    with open(LABELS_PATH, "wb") as f:
        pickle.dump({"label_map": label_map, "confidence_threshold": calibrated_threshold}, f)

    log(f"Trained on {len(images)} images across {len(people)} people: {list(label_map.values())} "
        f"(calibrated confidence threshold: {calibrated_threshold:.1f})")


# ---------------------------------------------------------------------------
# Run (the guard loop)
# ---------------------------------------------------------------------------
def run(owner_name=None):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        log("ERROR: no trained model found. Run 'enroll' then 'train' first.")
        sys.exit(1)

    face_cascade = load_cascade(FACE_CASCADE_PATH)
    profile_cascade = load_cascade(PROFILE_CASCADE_PATH)
    upperbody_cascade = load_cascade(UPPERBODY_CASCADE_PATH)
    eye_cascade = load_cascade(EYE_CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        saved = pickle.load(f)
    if isinstance(saved, dict) and "label_map" in saved:
        label_map = saved["label_map"]
        confidence_threshold = saved.get("confidence_threshold", CONFIDENCE_THRESHOLD)
    else:
        label_map = saved  # labels.pickle from before threshold calibration existed
        confidence_threshold = CONFIDENCE_THRESHOLD
    log(f"Using confidence threshold: {confidence_threshold:.1f}")

    # figure out which label id is "the owner"
    if owner_name:
        owner_ids = [lid for lid, name in label_map.items() if name == owner_name]
        if not owner_ids:
            log(f"ERROR: '{owner_name}' not found in trained labels: {list(label_map.values())}")
            sys.exit(1)
        owner_id = owner_ids[0]
    else:
        # default: first enrolled person
        owner_id = sorted(label_map.keys())[0]
        owner_name = label_map[owner_id]

    log(f"Guard started. Owner = '{owner_name}'. Watching webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("ERROR: could not open webcam.")
        sys.exit(1)

    stranger_streak = 0
    no_face_streak = 0
    person_no_face_streak = 0
    camera_covered_streak = 0
    last_lock_time = 0

    # Trackers keep boxes smoothly gliding across every detected face/body between the (slower)
    # full detection+recognition checks below, instead of boxes that only appear on the exact
    # frames a cascade happens to fire on -- one tracker per detected object, so multiple people
    # in frame each keep their own box instead of only the first one being followed.
    trackers = []
    debug_line = ""
    camera_covered = False
    last_check_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cv2.waitKey(DISPLAY_REFRESH_MS)
                continue

            now = time.time()
            if now - last_check_time >= CHECK_INTERVAL:
                last_check_time = now

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # A physically covered/blocked lens produces a near-featureless image (very low
                # pixel variance) -- unlike an empty room, which still has plenty of visual texture.
                camera_covered = float(np.std(gray)) < CAMERA_COVERED_STD_THRESHOLD

                faces = [] if camera_covered else list(face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)))

                frame_is_stranger = False
                frame_has_face = len(faces) > 0
                detections = []  # (box, color, label) for every face/body found this check

                for (x, y, w, h) in faces:
                    face_img = preprocess_face(gray, (x, y, w, h), eye_cascade)
                    if face_img is None:
                        continue
                    label_id, confidence = recognizer.predict(face_img)
                    # LBPH: LOWER confidence value = better match
                    is_owner = (label_id == owner_id) and (confidence < confidence_threshold)

                    if not is_owner:
                        frame_is_stranger = True
                        now_dt = datetime.now()
                        day_dir = os.path.join(SNAPSHOT_DIR, now_dt.strftime("%Y-%m-%d"))
                        os.makedirs(day_dir, exist_ok=True)
                        snap_path = os.path.join(day_dir, f"stranger_{now_dt.strftime('%H-%M-%S')}.jpg")
                        cv2.imwrite(snap_path, frame)

                    color = (0, 255, 0) if is_owner else (0, 0, 255)
                    label = owner_name if is_owner else "STRANGER"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    detections.append(((x, y, w, h), color, label))

                # No frontal face -> check for a face turned to the side before giving up on
                # "is there a face at all" (avoids false stranger/person-hiding flags on a head turn).
                profile_detected = False
                if not frame_has_face and not camera_covered:
                    profile_faces = list(profile_cascade.detectMultiScale(
                        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)))
                    if not profile_faces:
                        # the profile cascade only looks one way -- flip the frame to catch the other side
                        flipped = cv2.flip(gray, 1)
                        frame_width = gray.shape[1]
                        for (fx, fy, fw, fh) in profile_cascade.detectMultiScale(
                                flipped, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)):
                            profile_faces.append((frame_width - fx - fw, fy, fw, fh))
                    profile_detected = len(profile_faces) > 0
                    for (x, y, w, h) in profile_faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(frame, "FACE (side)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        detections.append(((x, y, w, h), (0, 255, 255), "FACE (side)"))

                frame_has_any_face = frame_has_face or profile_detected

                # No face at all (front or side) -> check whether a PERSON is still there.
                # This is what tells "owner walked away" (nothing detected) apart from
                # "someone's sitting right there with their face hidden/covered/turned away".
                person_detected = False
                if not frame_has_any_face and not camera_covered:
                    bodies = upperbody_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80))
                    person_detected = len(bodies) > 0
                    for (x, y, w, h) in bodies:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
                        cv2.putText(frame, "PERSON (no face)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                        detections.append(((x, y, w, h), (0, 140, 255), "PERSON (no face)"))

                camera_covered_streak = camera_covered_streak + 1 if camera_covered else 0

                if frame_is_stranger:
                    stranger_streak += 1
                    no_face_streak = 0
                    person_no_face_streak = 0
                elif frame_has_any_face:
                    stranger_streak = 0  # owner's face confirmed, reset
                    no_face_streak = 0
                    person_no_face_streak = 0
                else:
                    no_face_streak += 1
                    person_no_face_streak = person_no_face_streak + 1 if person_detected else 0
                    if not CONSEC_NO_FACE_OK:
                        stranger_streak += 1
                    # else: empty desk (briefly) doesn't count against the short-term streak -- but a
                    # PERSON detected with no face trips person_no_face_streak fast (someone hiding),
                    # a covered lens trips camera_covered_streak almost instantly, and a prolonged
                    # total absence of anything still trips no_face_streak eventually if
                    # LOCK_ON_NO_FACE is on

                should_lock = (
                    stranger_streak >= CONSEC_STRANGER_FRAMES
                    or person_no_face_streak >= PERSON_NO_FACE_LOCK_FRAMES
                    or camera_covered_streak >= CAMERA_COVERED_LOCK_FRAMES
                )
                if LOCK_ON_NO_FACE:
                    should_lock = should_lock or no_face_streak >= NO_FACE_LOCK_FRAMES
                if should_lock:
                    lock_now = time.time()
                    if lock_now - last_lock_time > LOCK_COOLDOWN_SECONDS:
                        if stranger_streak >= CONSEC_STRANGER_FRAMES:
                            reason = "Unrecognized face"
                        elif camera_covered_streak >= CAMERA_COVERED_LOCK_FRAMES:
                            reason = "Camera appears physically covered/blocked"
                        elif person_no_face_streak >= PERSON_NO_FACE_LOCK_FRAMES:
                            reason = "Person present with no recognizable face (hiding?)"
                        else:
                            reason = "No face visible for a long time"
                        log(f"{reason} detected repeatedly -> locking screen.")
                        lock_screen()
                        last_lock_time = lock_now
                    stranger_streak = 0
                    no_face_streak = 0
                    person_no_face_streak = 0
                    camera_covered_streak = 0

                if camera_covered:
                    status = "CAMERA COVERED"
                elif frame_is_stranger:
                    status = "STRANGER"
                elif frame_has_face:
                    status = "OWNER OK"
                elif profile_detected:
                    status = "SIDE FACE"
                elif person_detected:
                    status = "PERSON, NO FACE"
                else:
                    status = "NOTHING"
                debug_line = (
                    f"[{status}] face={frame_has_face} side={profile_detected} person={person_detected} "
                    f"covered={camera_covered} stranger={stranger_streak}/{CONSEC_STRANGER_FRAMES} "
                    f"person_no_face={person_no_face_streak}/{PERSON_NO_FACE_LOCK_FRAMES} "
                    f"camera_covered={camera_covered_streak}/{CAMERA_COVERED_LOCK_FRAMES} "
                    f"no_face={no_face_streak}/{NO_FACE_LOCK_FRAMES}"
                )
                print(debug_line)

                # (Re)anchor a tracker on every detection we just found (up to the cap), so each
                # one keeps gliding smoothly across the faster-refreshing frames until the next
                # check -- not just the first face/body in the frame.
                trackers = []
                for box, color, label in detections[:MAX_TRACKED_OBJECTS]:
                    t = cv2.TrackerCSRT_create()
                    t.init(frame, tuple(int(v) for v in box))
                    trackers.append({"tracker": t, "color": color, "label": label})

            else:
                # Between checks: glide every tracked box smoothly instead of leaving it frozen
                # or vanished. Drop any tracker that loses its target until the next check.
                still_tracking = []
                for entry in trackers:
                    ok, box = entry["tracker"].update(frame)
                    if ok:
                        x, y, w, h = (int(v) for v in box)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), entry["color"], 2)
                        cv2.putText(frame, entry["label"], (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, entry["color"], 2)
                        still_tracking.append(entry)
                trackers = still_tracking

            cv2.putText(frame, debug_line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if camera_covered:
                cv2.putText(frame, "CAMERA COVERED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Face Guard - press q to quit", frame)
            if cv2.waitKey(DISPLAY_REFRESH_MS) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        log("Guard stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Face Guard - webcam laptop lock")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_enroll = sub.add_parser("enroll", help="capture face samples")
    p_enroll.add_argument("--name", required=True, help="your name / label")
    p_enroll.add_argument("--samples", type=int, default=40)

    sub.add_parser("train", help="train the recognizer on enrolled samples")

    p_run = sub.add_parser("run", help="start watching the webcam")
    p_run.add_argument("--owner", default=None, help="which enrolled name counts as 'you' (default: first enrolled)")

    args = parser.parse_args()

    if args.mode == "enroll":
        enroll(args.name, args.samples)
    elif args.mode == "train":
        train()
    elif args.mode == "run":
        run(args.owner)


if __name__ == "__main__":
    main()
