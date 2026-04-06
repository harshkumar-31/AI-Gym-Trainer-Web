# StartSquatWeb - MVP

Web-based Bodyweight Squat Analysis using MediaPipe Pose.

## Run Instructions

### 1. Prerequisites
- Node.js installed (v16+ recommended).
- A webcam.

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Dev Server
```bash
npm run dev
```

### 4. Test Camera Flow
- Open the URL shown in terminal (usually `http://localhost:5173/`).
- Grant Camera Permissions when prompted.
- Stand back ~6-8 feet to ensure full body visibility.
- Turn sideways to the camera.
- Perform a squat.

## Performance Guardrails
- **FPS Expectations**: 
    - Laptop (Intel/M1/M2): **20-30 FPS** (High smoothness).
    - Mobile (High-end): **20-30 FPS**.
    - Lower end devices: **10-15 FPS**.
- **Degradation**: The app uses `requestAnimationFrame`. If the device is slow, the frame rate will naturally drop. The logic is time-independent (state machine checks angles per frame), so it remains accurate even at lower FPS, provided the motion isn't too fast (motion blur might affect accuracy).
- **Optimization**: Currently set to `modelComplexity: 1` (Balanced). If performance is poor, change to `0` in `src/App.tsx`.
