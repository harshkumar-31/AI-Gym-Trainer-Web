export type Point = {
    x: number;
    y: number;
    z?: number;
    visibility?: number;
    confidence?: number; // mapped from visibility or score
};

export type Pose = {
    keypoints: Point[]; // 33 MediaPipe keypoints
};

export type FeedbackType = 'NONE' | 'GOOD_REP' | 'BAD_REP_DEPTH' | 'BAD_REP_BACK' | 'WARNING_OCCLUSION';

export type SquatState = {
    currentDepth: number; // Degrees
    isSquatting: boolean; // Are we effectively in a rep?
    reps: number;
    feedback: FeedbackType;
    lastFeedbackTime: number;
    debug?: {
        hipY: number;
        kneeY: number;
        angle: number;
        stage: string;
    };
};

// Index mapping for MediaPipe BlazePose
export const KP = {
    NOSE: 0,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_HIP: 23,
    RIGHT_HIP: 24,
    LEFT_KNEE: 25,
    RIGHT_KNEE: 26,
    LEFT_ANKLE: 27,
    RIGHT_ANKLE: 28,
};

// ─── AI Prediction types (from Flask-SocketIO server) ───────────────────────

export type AIFeedback = {
    message: string;
    tip: string;
    color: string;         // hex colour for the banner
    severity: 'good' | 'warning' | 'error' | 'neutral';
};

export type AIDebug = {
    knee_angle: number;
    trunk_lean: number;
    hip_height: number;
    elbow_angle?: number;
    body_line?: number;
};

export type AIPrediction = {
    label: string;              // e.g. 'SQUAT_GOOD', 'PUSHUP_SAGGING_HIPS', etc.
    confidence: number;         // 0–1 model confidence for this label
    rolling_accuracy: number;   // 0–1 rolling accuracy over last 20 frames
    all_proba: Record<string, number>;
    rep_count: number;
    hold_seconds: number | null; // plank hold duration (null for rep-based exercises)
    feedback: AIFeedback;
    debug?: AIDebug;
    error?: string;
};

export type AIStatus = {
    connected: boolean;
    model_loaded: boolean;
    classes: string[];
    reset?: boolean;
};

