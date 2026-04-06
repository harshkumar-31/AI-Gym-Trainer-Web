/**
 * useSquatAI.ts
 * ─────────────
 * React hook that connects to the Flask-SocketIO inference server,
 * sends MediaPipe landmarks on each pose frame, and returns the
 * latest AI prediction (label, accuracy %, rep count, feedback).
 *
 * Usage:
 *   const { prediction, serverStatus, sendLandmarks, resetSession } = useSquatAI();
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { io, type Socket }  from 'socket.io-client';
import type { AIPrediction, AIStatus, Point } from '../types';

const SERVER_URL = 'http://localhost:5001';

export interface UseSquatAIResult {
    prediction:   AIPrediction | null;
    serverStatus: 'connecting' | 'connected' | 'disconnected' | 'no_model';
    sendLandmarks: (keypoints: Point[], exercise: string) => void;
    resetSession:  () => void;
}

export function useSquatAI(): UseSquatAIResult {
    const socketRef  = useRef<Socket | null>(null);
    const [prediction,   setPrediction]   = useState<AIPrediction | null>(null);
    const [serverStatus, setServerStatus] = useState<UseSquatAIResult['serverStatus']>('connecting');

    // ── Connect once on mount ──────────────────────────────────────────────
    useEffect(() => {
        const socket = io(SERVER_URL, {
            // Start with polling (needed for socket.io HTTP upgrade handshake),
            // then upgrade to websocket. Using websocket-only breaks on many setups.
            transports:        ['polling', 'websocket'],
            reconnectionDelay: 2000,
            timeout:           8000,
        });
        socketRef.current = socket;

        let statusTimeoutId: ReturnType<typeof setTimeout> | null = null;

        socket.on('connect', () => {
            console.log('[SquatAI] Connected to inference server');
            setServerStatus('connecting');   // wait for status response

            // Actively request status — more reliable than waiting for server push.
            // The server emitting in on_connect can be dropped mid-handshake.
            socket.emit('request_status');

            // Safety timeout: if status never arrives, mark offline
            statusTimeoutId = setTimeout(() => {
                console.warn('[SquatAI] Status timeout — re-requesting...');
                socket.emit('request_status');   // one retry
                statusTimeoutId = setTimeout(() => {
                    setServerStatus('disconnected');
                }, 4000);
            }, 5000);
        });

        socket.on('status', (data: AIStatus) => {
            if (statusTimeoutId) clearTimeout(statusTimeoutId);
            if (!data.model_loaded) {
                setServerStatus('no_model');
            } else {
                setServerStatus('connected');
            }
        });

        socket.on('prediction', (data: AIPrediction) => {
            if (data.error) {
                console.warn('[SquatAI] Prediction error:', data.error);
            } else {
                setPrediction(data);
            }
        });

        socket.on('disconnect', () => {
            console.warn('[SquatAI] Disconnected from inference server');
            setServerStatus('disconnected');
        });

        socket.on('connect_error', (err) => {
            console.warn('[SquatAI] Connection error:', err.message);
            setServerStatus('disconnected');
        });

        return () => {
            if (statusTimeoutId) clearTimeout(statusTimeoutId);
            socket.disconnect();
        };
    }, []);

    // ── Send a frame's landmarks to the server ────────────────────────────
    const sendLandmarks = useCallback((keypoints: Point[], exercise: string) => {
        const socket = socketRef.current;
        if (!socket || !socket.connected) return;

        // Convert Point[] → [{x, y, z, visibility}] expected by Python
        const landmarks = keypoints.map(kp => ({
            x:          kp.x,
            y:          kp.y,
            z:          kp.z          ?? 0,
            visibility: kp.visibility ?? kp.confidence ?? 0,
        }));

        socket.emit('landmarks', { exercise, landmarks });   // include exercise field (spec §6.2)
    }, []);

    // ── Reset session accuracy & rep counter ──────────────────────────────
    const resetSession = useCallback(() => {
        socketRef.current?.emit('reset');
        setPrediction(null);
    }, []);

    return { prediction, serverStatus, sendLandmarks, resetSession };
}
