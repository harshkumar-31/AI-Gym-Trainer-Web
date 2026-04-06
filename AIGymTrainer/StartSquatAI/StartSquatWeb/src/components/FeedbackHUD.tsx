/**
 * FeedbackHUD.tsx
 * ───────────────
 * Full-screen heads-up display rendered over the camera.
 * Shows:
 *   • Top-left:  REPS counter
 *   • Top-right: AI ACCURACY % (rolling over last 20 frames)
 *   • Centre:    Feedback banner (colour-coded) + corrective tip
 *   • Bottom:    Probability bars for each class + debug angles
 *   • Corner:    Server connection status badge
 */

import React, { useEffect, useRef } from 'react';
import type { AIPrediction, SquatState } from '../types';
import type { UseSquatAIResult } from '../hooks/useSquatAI';
import './FeedbackHUD.css';

interface Props {
    prediction:       AIPrediction | null;
    serverStatus:     UseSquatAIResult['serverStatus'];
    squatState:       SquatState;        // rule-based fallback — always available
    selectedExercise: string;            // 'squat' | 'pushup' | 'plank' | 'lunge'
    onReset:          () => void;
    onExerciseChange: (ex: 'squat' | 'pushup' | 'plank' | 'lunge') => void;
}

// ─── Colour map: maps label suffix (or full namespaced label) → hex colour ───────
const suffixColor = (label: string): string => {
    if (label.endsWith('_GOOD'))    return '#00e676';
    if (label.includes('SHALLOW') || label.includes('PARTIAL_REP') || label.includes('SHORT_STEP') || label.includes('HEAD_DROP')) return '#ffab00';
    if (label.includes('FORWARD_LEAN') || label.includes('SAGGING') || label.includes('PIKED') || label.includes('CAVE')) return '#ff3d00';
    if (label.includes('STANDING') || label.includes('FLOOR') || label.includes('HOLDING')) return '#90a4ae';
    if (label === 'OCCLUDED') return '#607d8b';
    return '#90a4ae';
};

const EXERCISE_LABELS: Record<string, { label: string; icon: string }> = {
    squat:  { label: 'Squat',   icon: '🏋️' },
    pushup: { label: 'Push-up', icon: '💪' },
    plank:  { label: 'Plank',   icon: '🪨' },
    lunge:  { label: 'Lunge',   icon: '🦵' },
};

const STATUS_LABEL: Record<UseSquatAIResult['serverStatus'], { text: string; cls: string }> = {
    connected:    { text: '● AI Connected',         cls: 'status-connected'    },
    connecting:   { text: '◌ Connecting to AI…',    cls: 'status-connecting'   },
    disconnected: { text: '✕ AI Server Offline',    cls: 'status-disconnected' },
    no_model:     { text: '⚠ Model not loaded',     cls: 'status-no-model'     },
};

// ─── Accuracy ring component ───────────────────────────────────────────────
const AccuracyRing: React.FC<{ value: number }> = ({ value }) => {
    const pct   = Math.round(value * 100);
    const r     = 38;
    const circ  = 2 * Math.PI * r;
    const dash  = (pct / 100) * circ;

    const color =
        pct >= 80 ? '#00e676' :
        pct >= 50 ? '#ffab00' : '#ff3d00';

    return (
        <div className="accuracy-ring-wrapper">
            <svg width="100" height="100" className="accuracy-svg">
                <circle cx="50" cy="50" r={r} fill="none" stroke="#1a1a2e" strokeWidth="8" />
                <circle
                    cx="50" cy="50" r={r}
                    fill="none"
                    stroke={color}
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray={`${dash} ${circ - dash}`}
                    strokeDashoffset={circ / 4}   /* start at top */
                    style={{ transition: 'stroke-dasharray 0.4s ease, stroke 0.4s ease' }}
                />
            </svg>
            <div className="accuracy-label">
                <span className="accuracy-pct" style={{ color }}>{pct}%</span>
                <span className="accuracy-sub">accuracy</span>
            </div>
        </div>
    );
};

// ─── Probability bar ──────────────────────────────────────────────
// Strip prefix for display: "PUSHUP_SAGGING_HIPS" → "SAGGING HIPS"
const stripExPrefix = (label: string) =>
    label.replace(/^[A-Z]+_/, '').replace(/_/g, ' ');

const ProbBar: React.FC<{ label: string; prob: number; active: boolean }> = ({ label, prob, active }) => {
    const pct   = Math.round(prob * 100);
    const color = suffixColor(label);
    return (
        <div className={`prob-row ${active ? 'prob-active' : ''}`}>
            <span className="prob-label">{stripExPrefix(label)}</span>
            <div className="prob-track">
                <div
                    className="prob-fill"
                    style={{
                        width: `${pct}%`,
                        background: color,
                        transition: 'width 0.25s ease',
                    }}
                />
            </div>
            <span className="prob-pct" style={{ color }}>{pct}%</span>
        </div>
    );
};

// ─── Main HUD ─────────────────────────────────────────────────────────────
export const FeedbackHUD: React.FC<Props> = ({ prediction, serverStatus, squatState, selectedExercise, onReset, onExerciseChange }) => {
    const bannerRef = useRef<HTMLDivElement>(null);

    // Flash the banner in whenever label changes
    const prevLabel = useRef<string | null>(null);
    useEffect(() => {
        if (!bannerRef.current || !prediction) return;
        if (prediction.label !== prevLabel.current) {
            bannerRef.current.classList.remove('banner-flash');
            void bannerRef.current.offsetWidth;   // reflow
            bannerRef.current.classList.add('banner-flash');
            prevLabel.current = prediction.label;
        }
    }, [prediction?.label]);     // eslint-disable-line

    const { text: stText, cls: stCls } = STATUS_LABEL[serverStatus];

    const fb    = prediction?.feedback;
    const label = prediction?.label ?? 'STANDING';

    // Rep count or hold timer depending on exercise
    const isPlank      = selectedExercise === 'plank';
    const metricLabel  = isPlank ? 'HOLD' : 'REPS';
    const metricValue  = isPlank
        ? `${(prediction?.hold_seconds ?? 0).toFixed(1)}s`
        : String(serverStatus === 'connected' && prediction ? prediction.rep_count : squatState.reps);

    const accVal  = prediction?.rolling_accuracy ?? 0;
    const allProb = prediction?.all_proba ?? {};
    const dbg     = prediction?.debug;

    const isAiConnected = serverStatus === 'connected';
    const isAiLive      = isAiConnected && prediction !== null;

    const bannerColor = fb?.color ?? suffixColor(label);

    return (
        <div className="hud-root">

            {/* ── Top row ─────────────────────────────────────────── */}
            <div className="hud-top">
                {/* Reps / Hold counter */}
                <div className="hud-card reps-card">
                    <span className="card-label">{metricLabel}</span>
                    <span className="card-value">{metricValue}</span>
                </div>

                {/* Server status */}
                <span className={`status-badge ${stCls}`}>{stText}</span>

                {/* Accuracy ring */}
                <div className="hud-card acc-card">
                    <AccuracyRing value={accVal} />
                </div>
            </div>

            {/* ── Centre feedback banner ───────────────────────────────── */}
            <div className="hud-centre">
                {/* AI feedback when server has predictions */}
                {isAiLive && fb && !label.includes('_STANDING') && !label.includes('_FLOOR') && !label.includes('_HOLDING') && (
                    <div
                        ref={bannerRef}
                        className="feedback-banner"
                        style={{ borderColor: bannerColor, boxShadow: `0 0 28px ${bannerColor}44` }}
                    >
                        <p className="fb-message" style={{ color: bannerColor }}>
                            {fb.message}
                        </p>
                        {fb.tip && (
                            <p className="fb-tip">{fb.tip}</p>
                        )}
                    </div>
                )}

                {/* Plank hold timer (also shown as inline timer in centre) */}
                {isPlank && prediction?.hold_seconds != null && label !== 'PLANK_HOLDING' && (
                    <div className="plank-timer">
                        {prediction.hold_seconds.toFixed(1)}s
                    </div>
                )}

                {/* Waiting for first pose — connected but no prediction yet */}
                {isAiConnected && !prediction && (
                    <div className="offline-hint">
                        <p className="offline-title">🤖 AI Ready</p>
                        <p className="offline-sub">Stand in front of the camera with your right side facing it</p>
                    </div>
                )}

                {/* Offline hint — only when server is genuinely not running */}
                {!isAiConnected && serverStatus !== 'connecting' && (
                    <div className="offline-hint">
                        <p className="offline-title">🏋️ Squat counting active</p>
                        <p className="offline-sub">
                            {serverStatus === 'no_model'
                                ? 'Train the model first, then restart serve_model.py for AI form feedback'
                                : 'Run serve_model.py to enable AI form correction & accuracy %'
                            }
                        </p>
                    </div>
                )}
            </div>

            {/* ── Bottom panel ─────────────────────────────────────── */}
            <div className="hud-bottom">
                {/* Probability bars */}
                {Object.keys(allProb).length > 0 && (
                    <div className="hud-card prob-panel">
                        <p className="panel-heading">Class Probabilities</p>
                        {Object.entries(allProb)
                            .sort((a, b) => b[1] - a[1])
                            .map(([lbl, prob]) => (
                                <ProbBar
                                    key={lbl}
                                    label={lbl}
                                    prob={prob}
                                    active={lbl === label}
                                />
                            ))}
                    </div>
                )}

                {/* Debug angles */}
                {dbg && (
                    <div className="hud-card debug-panel">
                        <p className="panel-heading">Joint Angles</p>
                        <div className="debug-row">
                            <span>Knee</span>
                            <span className="debug-val">{dbg.knee_angle}°</span>
                        </div>
                        <div className="debug-row">
                            <span>Trunk lean</span>
                            <span className="debug-val">{dbg.trunk_lean}°</span>
                        </div>
                        <div className="debug-row">
                            <span>Hip height</span>
                            <span className="debug-val">{dbg.hip_height.toFixed(3)}</span>
                        </div>
                        {dbg.elbow_angle !== undefined && (
                            <div className="debug-row">
                                <span>Elbow</span>
                                <span className="debug-val">{dbg.elbow_angle}°</span>
                            </div>
                        )}
                        {dbg.body_line !== undefined && (
                            <div className="debug-row">
                                <span>Body line</span>
                                <span className="debug-val">{dbg.body_line.toFixed(3)}</span>
                            </div>
                        )}
                    </div>
                )}

                {/* Exercise Switcher pill (in bottom panel for mid-session switching) */}
                <div className="exercise-pill">
                    {Object.entries(EXERCISE_LABELS).map(([key, { icon, label: exLabel }]) => (
                        <button
                            key={key}
                            title={exLabel}
                            className={selectedExercise === key ? 'active' : ''}
                            onClick={() => onExerciseChange(key as 'squat' | 'pushup' | 'plank' | 'lunge')}
                        >
                            {icon}
                        </button>
                    ))}
                </div>

                {/* Reset button */}
                <button className="reset-btn" onClick={onReset} title="Reset session">
                    ↺ Reset
                </button>
            </div>

            {/* ── Footer hint ──────────────────────────────────────────────── */}
            <div className="hud-footer">
                {selectedExercise === 'lunge'
                    ? <>Face the camera directly · full body visible</>
                    : <>Stand with your <strong>right side</strong> facing the camera · Full body visible</>
                }
            </div>

        </div>
    );
};
