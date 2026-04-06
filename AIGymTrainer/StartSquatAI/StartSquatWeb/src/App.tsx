import { useEffect, useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { Point } from './types';
import { SquatAnalyzer } from './logic/SquatAnalyzer';
import type { SquatState } from './types';
import { SkeletonOverlay } from './components/SkeletonOverlay';
import { FeedbackHUD }    from './components/FeedbackHUD';
import { useSquatAI }     from './hooks/useSquatAI';
import './index.css';

// ─── Constants ────────────────────────────────────────────────────────────────
const AI_SEND_INTERVAL_MS = 80;   // throttle landmarks to AI server (~12 fps)

// Exercises available in v2
type ExerciseId = 'squat' | 'pushup' | 'plank' | 'lunge';
const EXERCISES: { id: ExerciseId; label: string; icon: string; tip: string }[] = [
    { id: 'squat',  label: 'Squat',   icon: '🏋️', tip: 'Right side to camera · hip height · ~6 ft back' },
    { id: 'pushup', label: 'Push-up', icon: '💪', tip: 'Right side to camera · floor level · ~5 ft back' },
    { id: 'plank',  label: 'Plank',   icon: '🪨', tip: 'Right side to camera · floor level · ~5 ft back' },
    { id: 'lunge',  label: 'Lunge',   icon: '🦵', tip: 'Face the camera · waist height · ~7 ft back' },
];

// WASM served locally from public/mediapipe-wasm/ — no CDN needed
const WASM_PATH = '/mediapipe-wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';

function App() {
    const webcamRef        = useRef<Webcam>(null);
    const landmarkerRef    = useRef<PoseLandmarker | null>(null);
    const requestRef       = useRef<number>(0);
    const lastSendRef      = useRef<number>(0);
    const lastDetectRef    = useRef<number>(-1);

    // ── Rule-based rep counter (always on) ───────────────────────────────────
    const analyzerRef = useRef<SquatAnalyzer>(new SquatAnalyzer());
    const [squatState, setSquatState] = useState<SquatState>({
        currentDepth: 180, isSquatting: false,
        reps: 0, feedback: 'NONE', lastFeedbackTime: 0,
    });

    const [appPose,     setAppPose]     = useState<{ keypoints: Point[] } | null>(null);
    const [cameraReady, setCameraReady] = useState(false);
    const [videoSize,   setVideoSize]   = useState({ width: 640, height: 480 });
    const [modelLoaded, setModelLoaded] = useState(false);
    const [loadError,   setLoadError]   = useState<string | null>(null);
    const [selectedExercise, setSelectedExercise] = useState<ExerciseId | null>(null);

    // ── AI hook ──────────────────────────────────────────────────────────────
    const { prediction, serverStatus, sendLandmarks, resetSession } = useSquatAI();

    // ── Load PoseLandmarker (MediaPipe Tasks Vision) ─────────────────────────
    useEffect(() => {
        let cancelled = false;

        async function init() {
            try {
                console.log('[Pose] Loading MediaPipe Tasks Vision model…');
                const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
                const landmarker = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: MODEL_URL,
                        delegate: 'GPU',
                    },
                    runningMode:                  'VIDEO',
                    numPoses:                     1,
                    minPoseDetectionConfidence:   0.5,
                    minPosePresenceConfidence:    0.5,
                    minTrackingConfidence:        0.5,
                });

                if (!cancelled) {
                    landmarkerRef.current = landmarker;
                    setModelLoaded(true);
                    console.log('[Pose] PoseLandmarker ready ✅');
                }
            } catch (err) {
                console.error('[Pose] Failed to load:', err);
                if (!cancelled) setLoadError(String(err));
            }
        }

        init();
        return () => { cancelled = true; };
    }, []);

    // ── Render loop ───────────────────────────────────────────────────────────
    const loop = useCallback(() => {
        const video     = webcamRef.current?.video;
        const landmarker = landmarkerRef.current;

        if (video && video.readyState === 4 && landmarker) {
            if (!cameraReady) {
                setCameraReady(true);
                if (video.videoWidth > 0) {
                    setVideoSize({ width: video.videoWidth, height: video.videoHeight });
                }
            }

            const nowMs = performance.now();
            // Avoid sending the same timestamp twice (Tasks API requires strictly increasing ts)
            if (nowMs > lastDetectRef.current) {
                lastDetectRef.current = nowMs;
                const result = landmarker.detectForVideo(video, nowMs);

                if (result.landmarks.length > 0) {
                    const lms = result.landmarks[0];

                    const keypoints: Point[] = lms.map(lm => ({
                        x:          lm.x,
                        y:          lm.y,
                        z:          lm.z,
                        visibility: lm.visibility ?? 1,
                        confidence: lm.visibility ?? 1,
                    }));

                    setAppPose({ keypoints });

                    // Rule-based counter — always runs regardless of AI server
                    const newState = analyzerRef.current.process({ keypoints }, Date.now());
                    setSquatState(newState);

                    // Send to AI server (throttled)
                    if (nowMs - lastSendRef.current >= AI_SEND_INTERVAL_MS) {
                        lastSendRef.current = nowMs;
                        sendLandmarks(keypoints, selectedExercise ?? 'squat');
                    }
                }
            }
        }

        requestRef.current = requestAnimationFrame(loop);
    }, [cameraReady, sendLandmarks, selectedExercise]);

    useEffect(() => {
        requestRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(requestRef.current);
    }, [loop]);

    // ── Skeleton colour driven by AI when available, rule-based otherwise ─────
    const skeletonColor =
        serverStatus === 'connected' && prediction
            ? prediction.label.endsWith('_GOOD')         ? '#00e676'
            : prediction.label.includes('SHALLOW') || prediction.label.includes('PARTIAL_REP') || prediction.label.includes('SHORT_STEP') ? '#ffab00'
            : prediction.label.includes('FORWARD_LEAN') || prediction.label.includes('SAGGING') || prediction.label.includes('PIKED') || prediction.label.includes('CAVE') ? '#ff3d00'
            : '#4fc3f7'
        : squatState.isSquatting ? '#00e676' : '#4fc3f7';

    const handleReset = useCallback(() => {
        resetSession();
        analyzerRef.current = new SquatAnalyzer();
        setSquatState({ currentDepth: 180, isSquatting: false, reps: 0, feedback: 'NONE', lastFeedbackTime: 0 });
    }, [resetSession]);

    const handleExerciseSelect = useCallback((ex: ExerciseId) => {
        setSelectedExercise(ex);
        resetSession();   // reset server-side rep counter & hold timer on exercise switch
    }, [resetSession]);

    const isLoading = !cameraReady || !modelLoaded;
    const loadingMsg = loadError
        ? `Model load failed: ${loadError}`
        : !modelLoaded ? 'Loading pose model… (first run downloads ~7 MB)' : 'Starting camera…';

    return (
        <div style={{ position: 'relative', width: '100vw', height: '100vh', background: '#000', overflow: 'hidden' }}>

            {/* ── Exercise selector (shown before camera starts) ─── */}
            {!selectedExercise && (
                <div style={{
                    position: 'absolute', inset: 0, zIndex: 100,
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    background: 'linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%)',
                    gap: 24,
                }}>
                    <div style={{ fontSize: 48 }}>🏃</div>
                    <h1 style={{ color: '#fff', fontSize: 28, fontWeight: 800, margin: 0, letterSpacing: 1 }}>Choose Exercise</h1>
                    <p style={{ color: '#90a4ae', fontSize: 14, margin: 0 }}>Select an exercise to begin coaching</p>
                    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center', padding: '0 24px' }}>
                        {EXERCISES.map(ex => (
                            <button
                                key={ex.id}
                                id={`exercise-btn-${ex.id}`}
                                onClick={() => handleExerciseSelect(ex.id)}
                                style={{
                                    background: 'rgba(255,255,255,0.06)',
                                    border: '1.5px solid rgba(255,255,255,0.12)',
                                    borderRadius: 16,
                                    padding: '20px 28px',
                                    cursor: 'pointer',
                                    color: '#fff',
                                    textAlign: 'center',
                                    minWidth: 120,
                                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                                    transition: 'background 0.2s, border-color 0.2s, transform 0.15s',
                                }}
                                onMouseEnter={e => {
                                    (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.14)';
                                    (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(-3px)';
                                }}
                                onMouseLeave={e => {
                                    (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.06)';
                                    (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
                                }}
                            >
                                <span style={{ fontSize: 36 }}>{ex.icon}</span>
                                <span style={{ fontWeight: 700, fontSize: 16 }}>{ex.label}</span>
                                <span style={{ fontSize: 11, color: '#90a4ae', lineHeight: 1.4, whiteSpace: 'nowrap' }}>{ex.tip}</span>
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* ── Camera ─────────────────────────────────────────── */}
            <Webcam
                ref={webcamRef}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }}
                mirrored={false}
                videoConstraints={{ facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }}
            />

            {/* ── Skeleton overlay ────────────────────────────────── */}
            <div style={{ position: 'absolute', inset: 0, display: 'flex', justifyContent: 'center', alignItems: 'center', pointerEvents: 'none' }}>
                <SkeletonOverlay
                    pose={appPose}
                    width={videoSize.width}
                    height={videoSize.height}
                    color={skeletonColor}
                />
            </div>

            {/* ── HUD ─────────────────────────────────────────────── */}
            <FeedbackHUD
                prediction={prediction}
                serverStatus={serverStatus}
                squatState={squatState}
                selectedExercise={selectedExercise ?? 'squat'}
                onReset={handleReset}
                onExerciseChange={handleExerciseSelect}
            />

            {/* ── Loading / error screen ──────────────────────────── */}
            {selectedExercise && isLoading && (
                <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    background: '#000', zIndex: 50, gap: 16,
                }}>
                    <div style={{ fontSize: 48 }}>{loadError ? '❌' : '🏋️'}</div>
                    <p style={{ color: loadError ? '#ff3d00' : '#fff', fontSize: 18, fontWeight: 700, margin: 0, textAlign: 'center', maxWidth: 500, padding: '0 24px' }}>
                        {loadingMsg}
                    </p>
                    {!loadError && (
                        <p style={{ color: '#90a4ae', fontSize: 13, margin: 0 }}>
                            Squat counting + AI correction ready after load
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
