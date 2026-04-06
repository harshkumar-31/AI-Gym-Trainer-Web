"""
train_model.py — StartSquat AI Model Training (v2: Multi-Exercise)
===================================================================
Trains a Random Forest + Gradient Boosting classifier on the
combined multi-exercise feature set and exports the best model.

Input:  data/exercise_features.csv   (from feature_engineering.py)
Output: models/exercise_model.joblib
        models/label_encoder.joblib
        models/model_metadata.json
        models/confusion_matrix.png
        models/feature_importance.png

USAGE:
  python train_model.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR  = os.path.join(SCRIPT_DIR, '..', 'models')
FEATURES_CSV = os.path.join(DATA_DIR, 'exercise_features.csv')

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Feature columns ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Squat / shared
    'knee_angle', 'hip_angle', 'ankle_angle', 'trunk_lean_deg',
    'hip_knee_y_diff', 'knee_ankle_x_diff', 'hip_height_norm',
    'torso_thigh_ratio', 'hip_symmetry',
    # Push-up / Plank
    'elbow_angle', 'body_line_deviation', 'shoulder_wrist_x_diff',
    'neck_angle', 'knee_bend_angle',
    # Lunge
    'front_knee_angle', 'back_knee_angle', 'front_knee_x_drift',
    'stance_width_norm', 'hip_level_diff',
    # Exercise context
    'exercise_encoded',
]


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names: list, path: str):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('Confusion Matrix', fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'   📊 Confusion matrix → {path}')


def plot_feature_importance(clf, feature_names: list, path: str):
    if not hasattr(clf, 'feature_importances_'):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Feature Importances', fontsize=14)
    bars = ax.bar(range(len(importances)), importances[idx], color='steelblue', edgecolor='navy')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Importance')
    ax.grid(axis='y', alpha=0.3)

    for bar, imp in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{imp:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'   📊 Feature importance → {path}')


def print_section(title: str):
    print(f'\n{"─"*55}')
    print(f'  {title}')
    print(f'{"─"*55}')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(FEATURES_CSV):
        print(f'❌ Features file not found: {FEATURES_CSV}')
        print('   Run feature_engineering.py first.')
        sys.exit(1)

    print_section('Loading Data')
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=FEATURE_COLS + ['label'])

    print(f'   Samples:  {len(df):,}')
    print(f'   Features: {len(FEATURE_COLS)}')
    print(f'\n   Class distribution:')
    print(df['label'].value_counts().to_string())

    # Warn if any class has < 100 samples
    for lbl, cnt in df['label'].value_counts().items():
        if cnt < 100:
            print(f'   ⚠️  {lbl} has only {cnt} samples — consider collecting more data.')

    # ── Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X = df[FEATURE_COLS].values

    print(f'\n   Classes: {list(le.classes_)}')

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f'\n   Train: {len(X_train):,}  |  Test: {len(X_test):,}')

    # ── Random Forest ──────────────────────────────────────────────────────
    print_section('Training Random Forest')
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_test_acc = accuracy_score(y_test, rf_pipeline.predict(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'   Test Accuracy:   {rf_test_acc:.2%}')
    print(f'   5-Fold CV:       {rf_cv.mean():.2%} ± {rf_cv.std():.2%}')

    # ── Gradient Boosting (Disabled for v2 due to too many classes) ────────
    # GradientBoosting scales linearly with the number of classes. With 19 classes
    # it takes 10+ minutes to train. Random Forest gives 96%+ accuracy instantly!
    
    # ── Choose best ────────────────────────────────────────────────────────
    print_section('Results')
    best_pipeline = rf_pipeline
    best_name = 'Random Forest'
    best_acc  = rf_test_acc
    best_cv   = rf_cv

    print(f'   🏆 Training complete: {best_name}  ({best_acc:.2%})')

    y_pred = best_pipeline.predict(X_test)
    print(f'\n   Classification Report:\n')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Plots ──────────────────────────────────────────────────────────────
    print_section('Saving Plots')
    plot_confusion_matrix(
        y_test, y_pred, list(le.classes_),
        os.path.join(MODELS_DIR, 'confusion_matrix.png')
    )
    plot_feature_importance(
        best_pipeline.named_steps['clf'], FEATURE_COLS,
        os.path.join(MODELS_DIR, 'feature_importance.png')
    )

    # ── Export model ───────────────────────────────────────────────────────
    print_section('Exporting Model')

    model_path = os.path.join(MODELS_DIR, 'exercise_model.joblib')
    le_path    = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    meta_path  = os.path.join(MODELS_DIR, 'model_metadata.json')

    joblib.dump(best_pipeline, model_path)
    joblib.dump(le, le_path)

    metadata = {
        'model_name':           best_name,
        'test_accuracy':        float(best_acc),
        'cv_accuracy_mean':     float(best_cv.mean()),
        'cv_accuracy_std':      float(best_cv.std()),
        'classes':              list(le.classes_),
        'features':             FEATURE_COLS,
        'n_training_samples':   int(len(X_train)),
        'n_test_samples':       int(len(X_test)),
        'n_total_samples':      int(len(X)),
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'   ✅ Model          → {model_path}')
    print(f'   ✅ Label encoder  → {le_path}')
    print(f'   ✅ Metadata       → {meta_path}')

    if best_acc < 0.80:
        print('\n   ⚠️  Accuracy is below 80%. Consider collecting more data (200+ per class)')
        print('      and ensuring consistent camera angle (right side facing camera).')
    elif best_acc < 0.90:
        print('\n   🟡 Good accuracy! Collecting 400+ frames per class may push this to 90%+.')
    else:
        print('\n   🎉 Excellent accuracy! Model is ready for deployment.')

    print('\nNext step:  python serve_model.py')


if __name__ == '__main__':
    main()
