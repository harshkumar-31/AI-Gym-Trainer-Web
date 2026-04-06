"""
serve_model.py — Squat AI Flask-SocketIO Inference Server
==========================================================
Receives MediaPipe landmarks from the browser via WebSocket,
runs the trained classifier, and returns real-time predictions.

The browser (StartSquatWeb) connects with socket.io-client and
emits 'landmarks' payloads; this server emits back 'prediction'.

USAGE:
  python serve_model.py

Server runs on: http://localhost:5001
Then open StartSquatWeb in a browser (npm run dev → http://localhost:5173)
"""

import os
import sys
import json
import time
import numpy as np
import joblib
from collections import deque
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins='*')
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    async_mode='threading',
    logger=False,
    engineio_logger=False,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# ─── Globals (loaded once at startup) ────────────────────────────────────────
model          = None
label_encoder  = None
model_metadata = None

# ─── MediaPipe landmark indices ───────────────────────────────────────────────
class LM:
    RIGHT_EAR      = 8
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW    = 14
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28
    RIGHT_FOOT     = 32

# ─── Feedback map ─────────────────────────────────────────────────────────────
FEEDBACK = {
    # Squat
    'SQUAT_GOOD':          {'message': '✅ Perfect squat!',         'tip': 'Great depth and alignment.',                          'color': '#00e676', 'severity': 'good'},
    'SQUAT_SHALLOW':       {'message': '⬇ Go deeper!',             'tip': 'Aim for thighs parallel to floor (~90° knee angle).', 'color': '#ffab00', 'severity': 'warning'},
    'SQUAT_FORWARD_LEAN':  {'message': '🔄 Chest up.',              'tip': 'Drive hips back, keep torso upright.',                'color': '#ff3d00', 'severity': 'error'},
    'SQUAT_STANDING':      {'message': '🧍 Ready to squat.',        'tip': 'Right side to camera, step back ~6 ft.',              'color': '#90a4ae', 'severity': 'neutral'},
    # Push-up
    'PUSHUP_GOOD':         {'message': '✅ Perfect push-up!',       'tip': 'Full range, straight body line.',                     'color': '#00e676', 'severity': 'good'},
    'PUSHUP_SAGGING_HIPS': {'message': '⬆ Hips dropping!',         'tip': 'Squeeze your core and glutes.',                       'color': '#ff3d00', 'severity': 'error'},
    'PUSHUP_PIKED_HIPS':   {'message': '⬇ Hips too high.',          'tip': 'Lower your hips to form a straight line.',            'color': '#ffab00', 'severity': 'warning'},
    'PUSHUP_PARTIAL_REP':  {'message': '⬇ Go lower.',              'tip': 'Chest closer to floor — elbows to ~90°.',             'color': '#ffab00', 'severity': 'warning'},
    'PUSHUP_FLOOR':        {'message': '🧍 Bottom position.',       'tip': 'Push back up with control.',                          'color': '#90a4ae', 'severity': 'neutral'},
    # Plank
    'PLANK_GOOD':          {'message': '✅ Solid plank!',           'tip': 'Keep breathing — hold that line.',                    'color': '#00e676', 'severity': 'good'},
    'PLANK_SAGGING_HIPS':  {'message': '⬆ Hips sinking!',          'tip': 'Brace your core — imagine a straight line.',          'color': '#ff3d00', 'severity': 'error'},
    'PLANK_PIKED_HIPS':    {'message': '⬇ Hips too high.',          'tip': 'Push hips down to align with shoulders.',             'color': '#ffab00', 'severity': 'warning'},
    'PLANK_HEAD_DROP':     {'message': '👀 Head up.',               'tip': 'Keep your neck neutral — look at the floor.',         'color': '#ffab00', 'severity': 'warning'},
    'PLANK_HOLDING':       {'message': '⏱ Holding…',               'tip': 'Breathe steadily.',                                   'color': '#90a4ae', 'severity': 'neutral'},
    # Lunge
    'LUNGE_GOOD':          {'message': '✅ Perfect lunge!',         'tip': 'Front knee at 90°, back knee near floor.',            'color': '#00e676', 'severity': 'good'},
    'LUNGE_FRONT_KNEE_CAVE': {'message': '➡ Knee caving in!',      'tip': 'Push your front knee out over your pinky toe.',       'color': '#ff3d00', 'severity': 'error'},
    'LUNGE_FORWARD_LEAN':  {'message': '🔄 Chest up.',              'tip': 'Keep torso upright — engage your core.',              'color': '#ffab00', 'severity': 'warning'},
    'LUNGE_SHORT_STEP':    {'message': '⬇ Stride wider.',          'tip': 'Step out further — front knee should reach 90°.',     'color': '#ffab00', 'severity': 'warning'},
    'LUNGE_STANDING':      {'message': '🧍 Ready.',                 'tip': 'Step forward into your next lunge.',                  'color': '#90a4ae', 'severity': 'neutral'},
}

# ─── Accuracy rolling window (per-client would need sessions; shared is fine for 1-person MVP) ──
ACCURACY_WINDOW = 20
_accuracy_buffer: deque = deque(maxlen=ACCURACY_WINDOW)
_rep_count    = 0
_prev_label   = ''
_hold_seconds = 0.0
_plank_start  = None


# ─── Geometry ─────────────────────────────────────────────────────────────────
def get_lm(landmarks: list, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm['x'], lm['y']], dtype=np.float64)


def angle_at_b(a, b, c) -> float:
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))))


def vector_angle_from_vertical(p_top, p_bottom) -> float:
    vec = p_bottom - p_top
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(vec, [0, 1]) / norm, -1.0, 1.0))))


def signed_body_line_deviation(shoulder: np.ndarray,
                               hip: np.ndarray,
                               ankle: np.ndarray) -> float:
    """
    Signed perpendicular distance of hip from the shoulder-ankle line,
    normalised by body length.
    Positive = hip above the line (piked). Negative = hip below (sagging).
    """
    body_vec = ankle - shoulder
    body_len = np.linalg.norm(body_vec) + 1e-8
    hip_vec  = hip - shoulder
    t = np.dot(hip_vec, body_vec) / (body_len ** 2)
    closest = shoulder + t * body_vec
    deviation = hip - closest
    sign = -1.0 if deviation[1] < 0 else 1.0
    return sign * float(np.linalg.norm(deviation)) / body_len


def extract_features(landmarks: list, exercise: str) -> list[float]:
    def lm(idx): return np.array([landmarks[idx]['x'], landmarks[idx]['y']], dtype=np.float64)

    rs   = lm(LM.RIGHT_SHOULDER); rh  = lm(LM.RIGHT_HIP)
    rk   = lm(LM.RIGHT_KNEE);     ra  = lm(LM.RIGHT_ANKLE)
    rf   = lm(LM.RIGHT_FOOT);     lh  = lm(LM.LEFT_HIP)
    lk   = lm(LM.LEFT_KNEE);      la  = lm(LM.LEFT_ANKLE)
    re   = lm(LM.RIGHT_ELBOW);    rw  = lm(LM.RIGHT_WRIST)
    rear = lm(LM.RIGHT_EAR)

    body_len = float(np.linalg.norm(rs - ra)) + 1e-8

    knee_angle        = angle_at_b(rh, rk, ra)
    hip_angle         = angle_at_b(rs, rh, rk)
    ankle_angle       = angle_at_b(rk, ra, rf)
    trunk_lean_deg    = vector_angle_from_vertical(rs, rh)
    hip_knee_y_diff   = float(rh[1] - rk[1])
    knee_ankle_x_diff = float(rk[0] - ra[0])
    hip_height_norm   = float((ra[1] - rh[1]) / body_len)
    torso_len         = float(np.linalg.norm(rs - rh))
    thigh_len         = float(np.linalg.norm(rh - rk))
    torso_thigh_ratio = torso_len / (thigh_len + 1e-8)
    hip_symmetry      = float(abs(rh[1] - lh[1]))

    elbow_angle           = angle_at_b(rs, re, rw)
    body_line_deviation   = signed_body_line_deviation(rs, rh, ra)
    shoulder_wrist_x_diff = float(rw[0] - rs[0])
    neck_vec   = rs - rear
    neck_angle = float(np.degrees(np.arctan2(abs(neck_vec[1]), abs(neck_vec[0]) + 1e-8)))
    knee_bend_angle = angle_at_b(rh, rk, ra)

    front_knee_angle   = angle_at_b(rh, rk, ra)
    back_knee_angle    = angle_at_b(lh, lk, la)
    front_knee_x_drift = float(rk[0] - ra[0])
    stance_width_norm  = float(abs(ra[0] - la[0]) / body_len)
    hip_level_diff     = float(rh[1] - lh[1])

    exercise_map     = {'squat': 0, 'pushup': 1, 'plank': 2, 'lunge': 3}
    exercise_encoded = exercise_map.get(exercise.lower(), 0)

    return [
        knee_angle, hip_angle, ankle_angle, trunk_lean_deg,
        hip_knee_y_diff, knee_ankle_x_diff, hip_height_norm,
        torso_thigh_ratio, hip_symmetry,
        elbow_angle, body_line_deviation, shoulder_wrist_x_diff,
        neck_angle, knee_bend_angle,
        front_knee_angle, back_knee_angle, front_knee_x_drift,
        stance_width_norm, hip_level_diff,
        exercise_encoded,
    ]


# ─── Socket events ────────────────────────────────────────────────────────────
def _emit_status(reset=False):
    """Emit current server status to the connected client."""
    emit('status', {
        'connected':    True,
        'model_loaded': model is not None,
        'classes':      list(model_metadata['classes']) if model_metadata else [],
        'reset':        reset,
    })


@socketio.on('connect')
def on_connect():
    print('🔗 Client connected')
    _emit_status()   # best-effort; client also calls request_status as backup


@socketio.on('request_status')
def on_request_status():
    """Client explicitly asks for status — reliable, called right after connect."""
    print('📡 Status requested by client')
    _emit_status()


@socketio.on('disconnect')
def on_disconnect():
    print('🔌 Client disconnected')



@socketio.on('landmarks')
def on_landmarks(data: dict):
    global _accuracy_buffer, _rep_count, _prev_label

    if model is None:
        emit('prediction', {'error': 'Model not loaded. Run train_model.py first.'})
        return

    try:
        landmarks = data['landmarks']   # list[{x, y, z, visibility}]
        exercise  = data.get('exercise', 'squat').lower()   # new field; defaults to squat

        # Visibility gate: skip if key joints are occluded (exercise-aware)
        VIS_GATES = {
            'squat':  [LM.RIGHT_KNEE, LM.RIGHT_HIP],
            'pushup': [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW],
            'plank':  [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
            'lunge':  [LM.RIGHT_HIP, LM.RIGHT_KNEE],
        }
        for idx in VIS_GATES.get(exercise, [LM.RIGHT_KNEE, LM.RIGHT_HIP]):
            if landmarks[idx]['visibility'] < 0.35:
                emit('prediction', {
                    'label':            'OCCLUDED',
                    'confidence':       0.0,
                    'rolling_accuracy': _rolling_accuracy(),
                    'feedback': {
                        'message':  "👁 Can't see your joints clearly.",
                        'tip':      'Make sure your full right side is visible.',
                        'color':    '#607d8b',
                        'severity': 'neutral',
                    },
                })
                return

        features = extract_features(landmarks, exercise)
        X = np.array([features])

        proba = model.predict_proba(X)[0]

        # ── Strict Exercise Masking ──
        # Ensure no cross-pollination of exercise states by zeroing out non-relevant classes
        mask = np.zeros_like(proba, dtype=bool)
        for i, cls_name in enumerate(label_encoder.classes_):
            if exercise == 'squat':
                if cls_name in ('GOOD', 'SHALLOW', 'FORWARD_LEAN', 'STANDING'):
                    mask[i] = True
            else:
                if cls_name.lower().startswith(f"{exercise}_"):
                    mask[i] = True
                    
        proba[~mask] = 0.0
        sum_proba = np.sum(proba)
        if sum_proba > 0:
            proba = proba / sum_proba

        pred_idx   = int(np.argmax(proba))
        raw_label  = label_encoder.classes_[pred_idx]
        
        # Legacy squat label handling
        pred_label = f"SQUAT_{raw_label}" if raw_label in ('GOOD', 'SHALLOW', 'FORWARD_LEAN', 'STANDING') else raw_label
        confidence = float(proba[pred_idx])

        _accuracy_buffer.append(1 if pred_label.endswith('_GOOD') else 0)

        _update_rep_count(pred_label, exercise)

        all_proba = {
            (f"SQUAT_{lbl}" if lbl in ('GOOD', 'SHALLOW', 'FORWARD_LEAN', 'STANDING') else lbl): round(float(p), 4)
            for lbl, p, m in zip(label_encoder.classes_, proba, mask) if m
        }
        feedback = FEEDBACK.get(pred_label, {
            'message': pred_label.replace('_', ' ').title(),
            'tip': '', 'color': '#ffffff', 'severity': 'neutral'
        })

        emit('prediction', {
            'label':            pred_label,
            'confidence':       round(confidence, 4),
            'rolling_accuracy': round(_rolling_accuracy(), 4),
            'all_proba':        all_proba,
            'rep_count':        _rep_count,
            'hold_seconds':     _hold_seconds if exercise == 'plank' else None,
            'feedback':         feedback,
            'debug': {
                'knee_angle':  round(features[0], 1),
                'trunk_lean':  round(features[3], 1),
                'hip_height':  round(features[6], 3),
                'elbow_angle': round(features[9], 1),
                'body_line':   round(features[10], 3),
            },
        })

    except Exception as exc:
        emit('prediction', {'error': str(exc)})
        print(f'⚠️  Error processing landmarks: {exc}')


@socketio.on('reset')
def on_reset():
    global _accuracy_buffer, _rep_count, _prev_label, _hold_seconds, _plank_start
    _accuracy_buffer.clear()
    _rep_count    = 0
    _prev_label   = ''
    _hold_seconds = 0.0
    _plank_start  = None
    _emit_status(reset=True)


# ─── Rep counter ───────────────────────────────────────────────────────────
def _update_rep_count(label: str, exercise: str):
    global _rep_count, _prev_label, _hold_seconds, _plank_start

    if exercise == 'plank':
        if label in ('PLANK_GOOD', 'PLANK_HOLDING'):
            if _plank_start is None:
                _plank_start = time.time()
            _hold_seconds = round(time.time() - _plank_start, 1)
        else:
            _plank_start  = None
            _hold_seconds = 0.0
    elif exercise == 'pushup':
        if _prev_label == 'PUSHUP_GOOD' and label == 'PUSHUP_FLOOR':
            _rep_count += 1
    elif exercise == 'lunge':
        if _prev_label == 'LUNGE_GOOD' and label == 'LUNGE_STANDING':
            _rep_count += 1
    else:  # squat (existing logic)
        if _prev_label == 'SQUAT_GOOD' and label == 'SQUAT_STANDING':
            _rep_count += 1

    _prev_label = label


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _rolling_accuracy() -> float:
    if not _accuracy_buffer:
        return 0.0
    return sum(_accuracy_buffer) / len(_accuracy_buffer)


def load_model() -> bool:
    global model, label_encoder, model_metadata
    model_path = os.path.join(MODELS_DIR, 'exercise_model.joblib')
    le_path    = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    meta_path  = os.path.join(MODELS_DIR, 'model_metadata.json')

    if not os.path.exists(model_path):
        # Backward compat: try old squat_model.joblib
        old_path = os.path.join(MODELS_DIR, 'squat_model.joblib')
        if os.path.exists(old_path):
            print(f'⚠️  exercise_model.joblib not found; falling back to squat_model.joblib')
            print('   Re-train after collecting push-up/plank/lunge data.')
            model_path = old_path
        else:
            print(f'⚠️  Model not found at: {model_path}')
            print('   Run train_model.py first, then restart this server.')
            return False

    model         = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    with open(meta_path) as f:
        model_metadata = json.load(f)

    print(f'✅ Model loaded: {model_metadata["model_name"]}')
    print(f'   Test accuracy: {model_metadata["test_accuracy"]:.2%}')
    print(f'   Classes: {model_metadata["classes"]}')
    return True


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('🚀 Squat AI Inference Server starting …')
    load_model()
    print('🌐 Listening on http://localhost:5001')
    print('   Start the frontend with: cd StartSquatWeb && npm run dev\n')
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, use_reloader=False)
