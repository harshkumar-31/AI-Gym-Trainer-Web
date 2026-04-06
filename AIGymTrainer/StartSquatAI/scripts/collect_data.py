"""
collect_data.py — Squat AI Data Collection Tool (v2: Multi-Exercise)
=====================================================================
Uses the NEW mediapipe.tasks API (compatible with mediapipe 0.10+ / Python 3.14).

USAGE:
  python collect_data.py --exercise squat
  python collect_data.py --exercise pushup
  python collect_data.py --exercise plank
  python collect_data.py --exercise lunge

On first run, downloads the PoseLandmarker model (~6 MB) automatically.
All exercises append to the SAME exercise_data.csv file.
"""

import argparse
import cv2
import csv
import os
import sys
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── CLI argument ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--exercise', required=True,
                    choices=['squat', 'pushup', 'plank', 'lunge'],
                    help='Which exercise to collect data for')
args = parser.parse_args()
EXERCISE = args.exercise.upper()

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
CSV_PATH   = os.path.join(DATA_DIR, 'exercise_data.csv')       # unified CSV (spec §2.3)
MODEL_PATH = os.path.join(MODELS_DIR, 'pose_landmarker_full.task')
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
)

# ─── Per-exercise label keybindings (spec §2.2) ───────────────────────────────
EXERCISE_LABELS: dict[str, dict[int, str]] = {
    'SQUAT': {
        ord('g'): 'SQUAT_GOOD',
        ord('s'): 'SQUAT_SHALLOW',
        ord('f'): 'SQUAT_FORWARD_LEAN',
        ord(' '): 'SQUAT_STANDING',
    },
    'PUSHUP': {
        ord('g'): 'PUSHUP_GOOD',
        ord('s'): 'PUSHUP_SAGGING_HIPS',
        ord('p'): 'PUSHUP_PIKED_HIPS',
        ord('r'): 'PUSHUP_PARTIAL_REP',
        ord(' '): 'PUSHUP_FLOOR',
    },
    'PLANK': {
        ord('g'): 'PLANK_GOOD',
        ord('s'): 'PLANK_SAGGING_HIPS',
        ord('p'): 'PLANK_PIKED_HIPS',
        ord('h'): 'PLANK_HEAD_DROP',
        ord(' '): 'PLANK_HOLDING',
    },
    'LUNGE': {
        ord('g'): 'LUNGE_GOOD',
        ord('k'): 'LUNGE_FRONT_KNEE_CAVE',
        ord('f'): 'LUNGE_FORWARD_LEAN',
        ord('s'): 'LUNGE_SHORT_STEP',
        ord(' '): 'LUNGE_STANDING',
    },
}

LABELS = EXERCISE_LABELS[EXERCISE]

# ─── Per-exercise camera tips (spec §2.4) ────────────────────────────────────
CAMERA_TIPS: dict[str, str] = {
    'SQUAT':  'RIGHT SIDE to camera | hip height | ~6 ft back',
    'PUSHUP': 'RIGHT SIDE to camera | floor level | ~5 ft back',
    'PLANK':  'RIGHT SIDE to camera | floor level | ~5 ft back',
    'LUNGE':  'FACE camera directly | waist height | ~7 ft back',
}

# ─── Colours ──────────────────────────────────────────────────────────────────
# Map label suffix → BGR colour for display
LABEL_COLORS: dict[str, tuple] = {
    # Squat
    'SQUAT_GOOD':         (0,   220,  80),
    'SQUAT_SHALLOW':      (0,   165, 255),
    'SQUAT_FORWARD_LEAN': (0,    60, 255),
    'SQUAT_STANDING':     (200, 200, 200),
    # Push-up
    'PUSHUP_GOOD':        (0,   220,  80),
    'PUSHUP_SAGGING_HIPS':(0,    60, 255),
    'PUSHUP_PIKED_HIPS':  (0,   165, 255),
    'PUSHUP_PARTIAL_REP': (0,   200, 255),
    'PUSHUP_FLOOR':       (200, 200, 200),
    # Plank
    'PLANK_GOOD':         (0,   220,  80),
    'PLANK_SAGGING_HIPS': (0,    60, 255),
    'PLANK_PIKED_HIPS':   (0,   165, 255),
    'PLANK_HEAD_DROP':    (0,   200, 255),
    'PLANK_HOLDING':      (200, 200, 200),
    # Lunge
    'LUNGE_GOOD':         (0,   220,  80),
    'LUNGE_FRONT_KNEE_CAVE':(0,  60, 255),
    'LUNGE_FORWARD_LEAN': (0,   165, 255),
    'LUNGE_SHORT_STEP':   (0,   200, 255),
    'LUNGE_STANDING':     (200, 200, 200),
}

# Skeleton connections to draw
CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,31),(28,32),(13,14),(14,16),
    (11,13),(12,14),
]


# ─── Model download ───────────────────────────────────────────────────────────
def ensure_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return
    print(f'📥 Downloading PoseLandmarker model (~6 MB)…')
    def _progress(count, block, total):
        pct = min(count * block / total * 100, 100)
        bar = '█' * int(pct / 4)
        print(f'\r   [{bar:<25}] {pct:.0f}%', end='', flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
    print(f'\n✅ Model saved.')


# ─── CSV helpers ──────────────────────────────────────────────────────────────
def csv_header() -> list[str]:
    header = ['frame_id', 'label', 'timestamp']
    for i in range(33):
        for coord in ['x', 'y', 'z', 'vis']:
            header.append(f'lm{i}_{coord}')
    return header


def landmarks_to_row(frame_id: int, label: str, landmarks: list) -> list:
    row: list = [frame_id, label, round(time.time(), 4)]
    for lm in landmarks:
        row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6),
                    round(getattr(lm, 'visibility', 1.0), 4)])
    return row


# ─── Drawing helpers ──────────────────────────────────────────────────────────
def draw_skeleton(frame, landmarks, h, w, color=(0, 200, 255)):
    pts = {i: (int(lm.x * w), int(lm.y * h))
           for i, lm in enumerate(landmarks)
           if getattr(lm, 'visibility', 1.0) > 0.3}
    for (a, b) in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for idx, pt in pts.items():
        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_panel(img, x, y, w, h, alpha=0.60):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, pos, scale=0.5, color=(220, 220, 220), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ─── Keybinding labels for current exercise ───────────────────────────────────
def get_control_lines() -> list[tuple[str, tuple]]:
    """Return (description, color) pairs for the controls panel."""
    lines = []
    key_names = {ord(' '): 'SPC', ord('g'): 'G', ord('s'): 'S',
                 ord('f'): 'F', ord('p'): 'P', ord('r'): 'R',
                 ord('h'): 'H', ord('k'): 'K'}
    for key_code, label in LABELS.items():
        key_name = key_names.get(key_code, chr(key_code).upper())
        color = LABEL_COLORS.get(label, (200, 200, 200))
        short = label.replace(f'{EXERCISE}_', '')
        lines.append((f'[{key_name}]  {short}', color))
    lines.append(('[Q]   Quit & save', (100, 100, 100)))
    return lines


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ensure_model()

    # Resume: count existing data
    file_exists = os.path.exists(CSV_PATH)
    label_counts: dict[str, int] = {}
    frame_id = 0

    if file_exists:
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                frame_id += 1
                lbl = row[1] if len(row) > 1 else ''
                if lbl:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f'📂 Resuming — {frame_id} total frames in {CSV_PATH}')
    else:
        print('📄 Starting fresh exercise_data.csv')

    # ── MediaPipe PoseLandmarker ───────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Could not open webcam.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    current_label: str | None = None
    recording = False
    frames_this_session = 0
    ts_ms = 0
    control_lines = get_control_lines()
    camera_tip = CAMERA_TIPS[EXERCISE]

    print(f'\n🎥 Recording exercise: {EXERCISE}')
    print(f'📐 Camera: {camera_tip}')
    print('Controls shown on screen. Press Q or ESC to quit.\n')

    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_header())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            ts_ms += 33

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            pose_detected = bool(result.pose_landmarks)
            current_landmarks = result.pose_landmarks[0] if pose_detected else None

            if current_landmarks:
                skel_color = LABEL_COLORS.get(current_label, (0, 200, 255)) \
                             if (recording and current_label) else (0, 200, 255)
                draw_skeleton(frame, current_landmarks, h, w, color=skel_color)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('x'):       # explicit release
                recording = False
                current_label = None
            elif key in LABELS:
                current_label = LABELS[key]
                recording = True

            # ── Save frame ────────────────────────────────────────────────
            if recording and current_label and current_landmarks:
                row = landmarks_to_row(frame_id, current_label, current_landmarks)
                writer.writerow(row)
                label_counts[current_label] = label_counts.get(current_label, 0) + 1
                frame_id += 1
                frames_this_session += 1

            # ── Controls panel ────────────────────────────────────────────
            panel_h = 30 + len(control_lines) * 24 + 20
            draw_panel(frame, 8, 8, 268, panel_h)
            put_text(frame, f'  {EXERCISE} DATA COLLECTOR',
                     (12, 32), scale=0.55, color=(255, 210, 50), thickness=2)
            for i, (text, color) in enumerate(control_lines):
                put_text(frame, text, (16, 58 + i * 24), color=color, scale=0.45)

            # Release key hint
            put_text(frame, '[X]   Release label', (16, 58 + len(control_lines) * 24),
                     color=(100, 100, 100), scale=0.45)

            # ── Camera tip (top centre) ───────────────────────────────────
            draw_panel(frame, w // 2 - 220, 8, 440, 34)
            put_text(frame, f'📐 {camera_tip}',
                     (w // 2 - 215, 30), scale=0.42, color=(180, 230, 255))

            # ── Recording indicator ───────────────────────────────────────
            if recording and current_label:
                c = LABEL_COLORS.get(current_label, (255, 255, 255))
                draw_panel(frame, w - 270, 8, 262, 60)
                put_text(frame, '● RECORDING', (w - 260, 32),
                         scale=0.65, color=(0, 0, 255), thickness=2)
                put_text(frame, current_label.replace(f'{EXERCISE}_', ''),
                         (w - 260, 56), scale=0.55, color=c)
            else:
                draw_panel(frame, w - 200, 8, 192, 36)
                put_text(frame, 'HOLD KEY TO RECORD',
                         (w - 196, 30), scale=0.42, color=(130, 130, 130))

            # ── No pose warning ───────────────────────────────────────────
            if not pose_detected:
                put_text(frame, 'No pose detected — adjust camera',
                         (w // 2 - 190, h // 2),
                         scale=0.6, color=(0, 80, 255), thickness=2)

            # ── Frame counts for this exercise (bottom left) ───────────────
            ex_counts = {k: v for k, v in label_counts.items()
                         if k.startswith(EXERCISE + '_')}
            n = max(len(ex_counts), 1)
            ph = 20 + n * 24 + 10
            by = h - ph - 8
            draw_panel(frame, 8, by - 4, 268, ph + 8)
            put_text(frame, f'{EXERCISE} COUNTS', (16, by + 16),
                     scale=0.48, color=(160, 160, 160))
            for i, (lbl, cnt) in enumerate(ex_counts.items()):
                bar_w = min(int(cnt / 4), 120)
                bx, bby = 16, by + 24 + i * 24
                cv2.rectangle(frame, (bx, bby+4), (bx+120, bby+14), (40, 40, 40), -1)
                if bar_w > 0:
                    cv2.rectangle(frame, (bx, bby+4), (bx+bar_w, bby+14),
                                  LABEL_COLORS.get(lbl, (200, 200, 200)), -1)
                short = lbl.replace(f'{EXERCISE}_', '')
                put_text(frame, f'{short}: {cnt}', (bx+125, bby+14),
                         scale=0.38, color=LABEL_COLORS.get(lbl, (200, 200, 200)))

            # Session counter
            draw_panel(frame, w - 230, h - 40, 222, 32)
            put_text(frame, f'Session frames: {frames_this_session}',
                     (w - 226, h - 18), scale=0.47, color=(200, 200, 200))

            cv2.imshow(f'Squat AI — {EXERCISE} Data Collector', frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # ── Summary ───────────────────────────────────────────────────────────
    ex_counts = {k: v for k, v in label_counts.items()
                 if k.startswith(EXERCISE + '_')}
    print(f'\n✅ Done! Data saved to: {CSV_PATH}')
    print(f'📦 Frames this session: {frames_this_session}')
    print(f'📊 {EXERCISE} frame counts:')
    for lbl, cnt in ex_counts.items():
        bar = '█' * (cnt // 10)
        status = '✅' if cnt >= 200 else ('⚠️ ' if cnt >= 100 else '❌')
        print(f'   {status} {lbl:35s}: {cnt:4d}  {bar}')


if __name__ == '__main__':
    main()
