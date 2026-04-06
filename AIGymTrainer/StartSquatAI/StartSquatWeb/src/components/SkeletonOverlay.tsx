import React, { useEffect, useRef } from 'react';
import { KP } from '../types';
import type { Pose } from '../types';

interface Props {
    pose: Pose | null;
    width: number;
    height: number;
    color?: string;   // e.g. '#00e676' for good, '#ff3d00' for error
}

const CONNECTIONS = [
    [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER],
    [KP.LEFT_SHOULDER, KP.LEFT_HIP],
    [KP.RIGHT_SHOULDER, KP.RIGHT_HIP],
    [KP.LEFT_HIP, KP.RIGHT_HIP],
    [KP.LEFT_HIP, KP.LEFT_KNEE],
    [KP.RIGHT_HIP, KP.RIGHT_KNEE],
    [KP.LEFT_KNEE, KP.LEFT_ANKLE],
    [KP.RIGHT_KNEE, KP.RIGHT_ANKLE],
];

export const SkeletonOverlay: React.FC<Props> = ({ pose, width, height, color = '#4fc3f7' }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, width, height);

        if (!pose) return;

        // Glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur  = 10;

        // Draw connections
        ctx.lineWidth   = 3;
        ctx.strokeStyle = color;
        CONNECTIONS.forEach(([startIdx, endIdx]) => {
            const start = pose.keypoints[startIdx];
            const end   = pose.keypoints[endIdx];
            if (start && end && (start.confidence ?? 0) > 0.4 && (end.confidence ?? 0) > 0.4) {
                ctx.beginPath();
                ctx.moveTo(start.x * width, start.y * height);
                ctx.lineTo(end.x * width,   end.y * height);
                ctx.stroke();
            }
        });

        // Draw keypoints
        ctx.fillStyle = '#fff';
        pose.keypoints.forEach((kp, idx) => {
            if ((kp.confidence ?? 0) > 0.4 && Object.values(KP).includes(idx)) {
                ctx.beginPath();
                ctx.arc(kp.x * width, kp.y * height, 5, 0, 2 * Math.PI);
                ctx.fill();
            }
        });

        ctx.shadowBlur = 0;

    }, [pose, width, height, color]);

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
    );
};
