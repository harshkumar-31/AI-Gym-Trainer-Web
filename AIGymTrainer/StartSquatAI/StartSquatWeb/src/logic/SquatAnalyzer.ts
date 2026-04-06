import type { Pose, SquatState, Point } from '../types';
import { KP } from '../types';

// Constants
const MIN_CONFIDENCE = 0.5;

// Simplified Configuration
// Thresholds for Normalized Y movement (0 = top, 1 = bottom)
// We assume the user is standing initially.
const SQUAT_DEPTH_Y_THRESHOLD = 0.15; // Hip needs to move down by 0.15 units approx
const STANDING_RESET_THRESHOLD = 0.05; // Hip needs to return within 0.05 of start

type SquatStage = 'STANDING' | 'DOWN';

export class SquatAnalyzer {
    private state: SquatState = {
        currentDepth: 180,
        isSquatting: false,
        reps: 0,
        feedback: 'NONE',
        lastFeedbackTime: 0,
        debug: { hipY: 0, kneeY: 0, angle: 0, stage: 'STANDING' }
    };

    private stage: SquatStage = 'STANDING';

    // Baseline for "Standing" Hip Height
    // We calibrate this constantly when in STANDING state to account for camera movement?
    // Or just take the first stable reading? 
    // Let's use a rolling average for standing height or just the current reading if we are "high".
    private baselineHipY: number | null = null;

    process(pose: Pose, timestamp: number): SquatState {
        const rHip = pose.keypoints[KP.RIGHT_HIP];
        const rKnee = pose.keypoints[KP.RIGHT_KNEE];
        const rAnkle = pose.keypoints[KP.RIGHT_ANKLE];

        // Safety check needed for type safety, though logic handles undefined check below
        if (!rHip || !rKnee) {
            return this.state;
        }

        // 1. Confidence Check
        if ((rHip.confidence ?? 0) < MIN_CONFIDENCE || (rKnee.confidence ?? 0) < MIN_CONFIDENCE) {
            this.state.feedback = 'WARNING_OCCLUSION';
            return this.state;
        }

        const hipY = rHip.y;
        const kneeY = rKnee.y;
        // Calculate angle just for display/debug
        const angle = rAnkle ? this.calculateAngle(rHip, rKnee, rAnkle) : 0;
        this.state.currentDepth = angle;

        // Auto-Calibration of "Standing" Height
        // If we haven't set a baseline, or we are in STANDING state and the hip is "higher" (smaller Y) than baseline,
        // update baseline "up". This handles if user stands up straighter.
        if (this.baselineHipY === null || (this.stage === 'STANDING' && hipY < this.baselineHipY)) {
            this.baselineHipY = hipY;
        }

        // Logic relies on baseline
        if (this.baselineHipY !== null) {
            const deltaY = hipY - this.baselineHipY; // Positive = went down

            switch (this.stage) {
                case 'STANDING':
                    // If we moved down significantly
                    if (deltaY > SQUAT_DEPTH_Y_THRESHOLD) {
                        this.stage = 'DOWN';
                        this.state.isSquatting = true;
                        this.state.feedback = 'NONE';
                    }
                    break;

                case 'DOWN':
                    // If we returned up close to baseline
                    if (deltaY < STANDING_RESET_THRESHOLD) {
                        this.stage = 'STANDING';
                        this.state.isSquatting = false;
                        this.state.reps += 1;
                        this.state.feedback = 'GOOD_REP';
                        this.state.lastFeedbackTime = timestamp;
                    }
                    break;
            }
        }

        // Update Debug Info
        this.state.debug = {
            hipY: parseFloat(hipY.toFixed(4)),
            kneeY: parseFloat(kneeY.toFixed(4)),
            angle: Math.round(angle),
            stage: this.stage + (this.baselineHipY ? ` (Base: ${this.baselineHipY.toFixed(2)})` : '')
        };

        return { ...this.state };
    }

    private calculateAngle(a: Point, b: Point, c: Point): number {
        const AB = Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
        const BC = Math.sqrt(Math.pow(b.x - c.x, 2) + Math.pow(b.y - c.y, 2));
        const AC = Math.sqrt(Math.pow(c.x - a.x, 2) + Math.pow(c.y - a.y, 2));
        if (AB === 0 || BC === 0) return 180;
        let rad = Math.acos((AB * AB + BC * BC - AC * AC) / (2 * AB * BC));
        if (isNaN(rad)) return 180;
        return (rad * 180) / Math.PI;
    }
}
