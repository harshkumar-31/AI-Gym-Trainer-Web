"""
feature_engineering.py — StartSquat AI Feature Extraction (v2: Multi-Exercise)
===============================================================================
Transforms raw MediaPipe landmark CSV into biomechanical features
(joint angles, ratios) that a combined multi-exercise classifier can learn from.

Input:  data/exercise_data.csv     (from collect_data.py)
Output: data/exercise_features.csv

FEATURES EXTRACTED (21 total — superset across all exercises):
  Squat/shared:  knee_angle, hip_angle, ankle_angle, trunk_lean_deg,
                 hip_knee_y_diff, knee_ankle_x_diff, hip_height_norm,
                 torso_thigh_ratio, hip_symmetry
  Push-up/Plank: elbow_angle, body_line_deviation, shoulder_wrist_x_diff,
                 neck_angle, knee_bend_angle
  Lunge:         front_knee_angle, back_knee_angle, front_knee_x_drift,
                 stance_width_norm, hip_level_diff
  Context:       exercise_encoded  (0=squat, 1=pushup, 2=plank, 3=lunge)

USAGE:
  python feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
INPUT_CSV  = os.path.join(DATA_DIR, 'exercise_data.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'exercise_features.csv')

# ─── MediaPipe landmark indices ───────────────────────────────────────────────
class LM:
    LEFT_EAR       = 7
    RIGHT_EAR      = 8
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28
    LEFT_HEEL      = 29
    RIGHT_HEEL     = 30
    LEFT_FOOT      = 31
    RIGHT_FOOT     = 32


# ─── Geometry helpers ─────────────────────────────────────────────────────────
def get_lm(row: pd.Series, idx: int) -> np.ndarray:
    """Return (x, y) normalized coords for landmark idx."""
    return np.array([row[f'lm{idx}_x'], row[f'lm{idx}_y']], dtype=np.float64)


def angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b, formed by a–b–c."""
    ba = a - b
    bc = c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return 180.0
    cos_val = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def vector_angle_from_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    """
    Angle (degrees) between the vector p_top→p_bottom and the downward vertical.
    0° = perfectly upright torso; >45° = significant forward lean.
    """
    vec = p_bottom - p_top
    vertical = np.array([0.0, 1.0])            # positive Y goes downward in image coords
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 0.0
    cos_val = np.dot(vec, vertical) / norm
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def signed_body_line_deviation(shoulder: np.ndarray,
                               hip: np.ndarray,
                               ankle: np.ndarray) -> float:
    """
    Signed perpendicular distance of hip from the shoulder-ankle line,
    normalised by body length.
    Positive = hip above the line (piked).
    Negative = hip below the line (sagging).
    """
    body_vec = ankle - shoulder
    body_len = np.linalg.norm(body_vec) + 1e-8
    hip_vec  = hip - shoulder
    t = np.dot(hip_vec, body_vec) / (body_len ** 2)
    closest = shoulder + t * body_vec
    deviation = hip - closest
    sign = -1.0 if deviation[1] < 0 else 1.0
    return sign * float(np.linalg.norm(deviation)) / body_len


# ─── Feature extraction ───────────────────────────────────────────────────────
def compute_features(row: pd.Series) -> dict | None:
    """
    Compute all biomechanical features for one frame row (v2: all 4 exercises).
    Returns None if key landmarks for the detected exercise have low visibility.
    """
    label    = row.get('label', '')
    exercise = label.split('_')[0] if '_' in label else 'SQUAT'

    # Visibility gate — key joints vary per exercise
    VISIBILITY_GATES = {
        'SQUAT':  [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
        'PUSHUP': [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_HIP],
        'PLANK':  [LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_ANKLE],
        'LUNGE':  [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.LEFT_KNEE],
    }
    for idx in VISIBILITY_GATES.get(exercise, []):
        if row.get(f'lm{idx}_vis', 1.0) < 0.3:
            return None

    # ── Shared landmarks ────────────────────────────────────────────────
    r_shoulder = get_lm(row, LM.RIGHT_SHOULDER)
    r_hip      = get_lm(row, LM.RIGHT_HIP)
    r_knee     = get_lm(row, LM.RIGHT_KNEE)
    r_ankle    = get_lm(row, LM.RIGHT_ANKLE)
    r_foot     = get_lm(row, LM.RIGHT_FOOT)
    l_shoulder = get_lm(row, LM.LEFT_SHOULDER)
    l_hip      = get_lm(row, LM.LEFT_HIP)
    l_knee     = get_lm(row, LM.LEFT_KNEE)

    # ── Squat / Lunge features (existing) ──────────────────────────────
    knee_angle        = angle_at_b(r_hip, r_knee, r_ankle)
    hip_angle         = angle_at_b(r_shoulder, r_hip, r_knee)
    ankle_angle       = angle_at_b(r_knee, r_ankle, r_foot)
    trunk_lean_deg    = vector_angle_from_vertical(r_shoulder, r_hip)
    hip_knee_y_diff   = float(r_hip[1] - r_knee[1])
    knee_ankle_x_diff = float(r_knee[0] - r_ankle[0])
    body_len          = float(np.linalg.norm(r_shoulder - r_ankle)) + 1e-8
    hip_height_norm   = float((r_ankle[1] - r_hip[1]) / body_len)
    torso_len         = float(np.linalg.norm(r_shoulder - r_hip))
    thigh_len         = float(np.linalg.norm(r_hip - r_knee))
    torso_thigh_ratio = torso_len / (thigh_len + 1e-8)
    hip_symmetry      = float(abs(r_hip[1] - l_hip[1]))

    # ── Push-up / Plank features (new) ─────────────────────────────────
    r_elbow = get_lm(row, LM.RIGHT_ELBOW)
    r_wrist = get_lm(row, LM.RIGHT_WRIST)
    r_ear   = get_lm(row, LM.RIGHT_EAR)

    elbow_angle           = angle_at_b(r_shoulder, r_elbow, r_wrist)
    body_line_deviation   = signed_body_line_deviation(r_shoulder, r_hip, r_ankle)
    shoulder_wrist_x_diff = float(r_wrist[0] - r_shoulder[0])
    neck_vec   = r_shoulder - r_ear
    neck_angle = float(np.degrees(np.arctan2(abs(neck_vec[1]), abs(neck_vec[0]) + 1e-8)))
    knee_bend_angle = angle_at_b(r_hip, r_knee, r_ankle)  # alias for plank

    # ── Lunge-specific features (new) ──────────────────────────────────
    l_ankle           = get_lm(row, LM.LEFT_ANKLE)
    front_knee_angle  = angle_at_b(r_hip,  r_knee,  r_ankle)   # right = front leg
    back_knee_angle   = angle_at_b(l_hip,  l_knee,  l_ankle)   # left  = back leg
    front_knee_x_drift = float(r_knee[0] - r_ankle[0])
    stance_width_norm  = float(abs(r_ankle[0] - l_ankle[0]) / body_len)
    hip_level_diff     = float(r_hip[1] - l_hip[1])

    # ── Exercise encoding ───────────────────────────────────────────────
    exercise_map     = {'SQUAT': 0, 'PUSHUP': 1, 'PLANK': 2, 'LUNGE': 3}
    exercise_encoded = exercise_map.get(exercise, 0)

    return {
        # Squat / shared
        'knee_angle':          round(knee_angle, 4),
        'hip_angle':           round(hip_angle, 4),
        'ankle_angle':         round(ankle_angle, 4),
        'trunk_lean_deg':      round(trunk_lean_deg, 4),
        'hip_knee_y_diff':     round(hip_knee_y_diff, 6),
        'knee_ankle_x_diff':   round(knee_ankle_x_diff, 6),
        'hip_height_norm':     round(hip_height_norm, 6),
        'torso_thigh_ratio':   round(torso_thigh_ratio, 4),
        'hip_symmetry':        round(hip_symmetry, 6),
        # Push-up / Plank
        'elbow_angle':           round(elbow_angle, 4),
        'body_line_deviation':   round(body_line_deviation, 6),
        'shoulder_wrist_x_diff': round(shoulder_wrist_x_diff, 6),
        'neck_angle':            round(neck_angle, 4),
        'knee_bend_angle':       round(knee_bend_angle, 4),
        # Lunge
        'front_knee_angle':    round(front_knee_angle, 4),
        'back_knee_angle':     round(back_knee_angle, 4),
        'front_knee_x_drift':  round(front_knee_x_drift, 6),
        'stance_width_norm':   round(stance_width_norm, 6),
        'hip_level_diff':      round(hip_level_diff, 6),
        # Exercise context
        'exercise_encoded':    exercise_encoded,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Backward compat: if old squat_data.csv exists but exercise_data.csv doesn't, remind user
    if not os.path.exists(INPUT_CSV):
        old_csv = os.path.join(DATA_DIR, 'squat_data.csv')
        if os.path.exists(old_csv):
            print(f'⚠️  Found old squat_data.csv — rename it to exercise_data.csv first:')
            print(f'   ren "{old_csv}" exercise_data.csv')
        else:
            print(f'❌ Input file not found: {INPUT_CSV}')
            print('   Run: python collect_data.py --exercise squat  (and other exercises)')
        sys.exit(1)

    print(f'📂 Loading raw data from {INPUT_CSV} …')
    df = pd.read_csv(INPUT_CSV)
    print(f'   {len(df):,} rows, {df["label"].nunique()} classes')
    print(df['label'].value_counts().to_string())
    print()

    print('⚙️  Extracting features …')
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        features = compute_features(row)
        if features is None:
            skipped += 1
            continue
        features['label'] = row['label']
        rows.append(features)

    if not rows:
        print('❌ No valid rows produced. Check that visibility scores are present in your data.')
        sys.exit(1)

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_CSV, index=False)

    feature_cols = [c for c in features_df.columns if c != 'label']
    print(f'✅ Features saved to: {OUTPUT_CSV}')
    print(f'   Rows: {len(features_df):,}  (skipped {skipped} low-visibility frames)')
    print(f'   Features ({len(feature_cols)}): {feature_cols}')
    print()
    print('📊 Class distribution after filtering:')
    print(features_df['label'].value_counts().to_string())
    print()

    # Quick sanity: check for NaN
    nan_count = features_df[feature_cols].isnull().sum().sum()
    if nan_count:
        print(f'⚠️  {nan_count} NaN values found — they will be dropped during training.')
    else:
        print('✅ No NaN values — data looks clean!')

    print('\nNext step:  python train_model.py')
    print('   (trains combined model on all exercise data → models/exercise_model.joblib)')


if __name__ == '__main__':
    main()
