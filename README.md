# AI Gym Trainer v2

A fully unified, real-time AI gym coach powered by **MediaPipe Pose**, a custom **Random Forest Classifier**, and an interactive **React WebSocket Frontend**.

## Features

- **Multi-Exercise Support:** Track form for Squats, Push-Ups, Planks, and Lunges.
- **Strict Exercise Masking:** Select a mode and the AI strictly limits and normalizes probability scoring to only output labels for your active exercise.
- **Dynamic Rep Counting:** Accurately counts repetitions specifically attuned to the active exercise's form-transitions. 
- **Hold Timer:** Custom timer logic that triggers automatically for static exercises like Planks.
- **Namespaced Labels & Colorful Alerts:** The frontend automatically maps ML labels (like `PUSHUP_PIKED_HIPS` or `SQUAT_SHALLOW`) to actionable dynamic CSS styling and tips.

## Architecture

- **Backend (`SquatAI/`)**
  - **`collect_data.py`**: Gathers structured webcam data and extracts skeletal landmarks, tagged via namespaces.
  - **`feature_engineering.py`**: Computes 20 joint angles and relative ratios (knee, hip, body-deviation) and outputs them as CSV.
  - **`train_model.py`**: Trains the single master unified ensemble Random Forest on the 20 features to hit >96% cross-validated accuracy.
  - **`serve_model.py`**: Flask-SocketIO streaming server that maps the live web stream directly into the Random Forest.

- **Frontend (`StartSquatWeb/`)**
  - **React + Vite** architecture with robust componentized UI.
  - WebSockets provide ~30fps real-time inference streaming to overlay skeletons, accuracy charts, and feedback alerts in the HUD inline with the camera feed.

## Setup Instructions

### 1. Training AI Model
```bash
cd SquatAI
pip install -r requirements.txt
python scripts/collect_data.py --exercise [squat|pushup|plank|lunge]
python scripts/feature_engineering.py
python scripts/train_model.py
```

### 2. Start Inference Server
```bash
cd SquatAI
python scripts/serve_model.py
```

### 3. Start Frontend 
```bash
cd StartSquatWeb
npm install
npm run dev
```
