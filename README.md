<div align="center">

# ✋ GestureFlow

### Control Your PC with Hand Gestures — Completely Touch-Free

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

**GestureFlow** is an AI-powered, real-time hand gesture recognition system that lets you control your entire PC — cursor movement, clicking, scrolling, volume, window switching, and screenshots — using only your webcam and hand gestures, all through a sleek browser-based dashboard.

[🌐 Live Demo](https://gesture-flask-control.onrender.com) · [📸 Gallery](#-screenshot-gallery) · [🚀 Quick Start](#-quick-start)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖱️ **Cursor Control** | Move the mouse with your index finger in real-time |
| 🖱️ **Click / Double Click** | Two-finger gesture triggers single or double click |
| 🔼 **Scroll Up / Down** | Three or four fingers to scroll pages |
| 🔊 **Volume Control** | Pinch gesture (thumb + index distance) adjusts system volume |
| 📸 **Screenshot** | Open palm captures and saves a screenshot |
| 🔀 **Alt+Tab** | Pinky gesture switches between windows |
| 🖥️ **Show Desktop** | Phone gesture (thumb + pinky) minimises all windows |
| 🌐 **Web Dashboard** | Live MJPEG video stream, status, settings, and gallery in your browser |
| ⚙️ **Adjustable Settings** | Scroll sensitivity, mouse sensitivity, and cooldown tunable from the UI |
| 📷 **Screenshot Gallery** | Browse all captured screenshots in the web UI |
| ⚡ **Demo Mode** | Works on Render (cloud) as a demo; full control requires local run |

---

## 🎯 Gesture Reference

| Gesture | Fingers Up | Action |
|---|---|---|
| ☝️ One Finger | Index only | **Move Cursor** |
| ✌️ Two Fingers | Index + Middle | **Click** (double-click if rapid) |
| 🤟 Three Fingers | Index + Middle + Ring | **Scroll Up** |
| 🖐️ Four Fingers | Index + Middle + Ring + Pinky | **Scroll Down** |
| ✋ Open Palm | All five | **Take Screenshot** |
| 🤏 Pinch | Thumb ↔ Index distance | **Volume Control** |
| 🤙 Pinky | Pinky only | **Alt + Tab** |
| 📞 Phone | Thumb + Pinky | **Show Desktop (Win+D)** |

> **Tip:** Hold gestures steady for best recognition. Adjust the **Cooldown** slider if actions fire too fast.

---

## 🏗️ How It Works

```
Webcam → OpenCV frames → MediaPipe (21 hand landmarks) → Gesture Recognition → OS Action
                                                                       ↓
                                              Flask web server ← MJPEG stream → Browser Dashboard
```

1. **Camera Capture** — OpenCV reads frames from your webcam at 1280×720
2. **Hand Landmark Detection** — Google MediaPipe identifies 21 landmarks on your hand in real-time
3. **Gesture Recognition** — Finger states (up/down) are mapped to one of 8 gestures
4. **Action Execution** — PyAutoGUI sends the corresponding mouse/keyboard event to your OS
5. **Web Dashboard** — Flask streams annotated frames as MJPEG and exposes a REST API for start/stop/config

---

## 🗂️ Project Structure

```
gesture_flask/
├── app.py                  # Flask routes & MJPEG video stream
├── gesture_control.py      # Core gesture engine (GestureController + DemoController)
├── requirements.txt        # Python dependencies (pinned)
├── runtime.txt             # Python version for Render
├── render.yaml             # Render deployment configuration
├── Procfile                # Gunicorn start command
├── .gitignore
├── templates/
│   ├── base.html           # Shared layout (navbar, fonts)
│   ├── index.html          # Main dashboard (video, status, settings)
│   ├── gallery.html        # Screenshot gallery
│   └── about.html          # About & How-it-works page
└── static/
    ├── style.css           # Full UI styling (dark, glassmorphism)
    ├── app.js              # Frontend JS (polling, settings, controls)
    └── screenshots/        # Saved gesture screenshots (gitignored)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Webcam** connected to your PC
- **Windows OS** (for full gesture control — PyAutoGUI + pycaw)

### 1. Clone the Repository

```bash
git clone https://github.com/Parthx-06/gesture-flask-control.git
cd gesture-flask-control
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Windows-only extras for full gesture control:
pip install pyautogui pycaw comtypes
```

### 4. Run the App

```bash
python app.py
```

Open your browser at **http://localhost:5000** and click **▶ Start**.

---

## ⚙️ Configuration

All settings are adjustable live from the web dashboard without restarting:

| Setting | Default | Description |
|---|---|---|
| Scroll Sensitivity | `40` | Pixels scrolled per gesture trigger |
| Mouse Sensitivity | `1.5` | Multiplier for cursor speed |
| Cooldown | `0.2s` | Minimum time between repeated gesture actions |

Settings are sent to `POST /api/config` and take effect immediately.

---

## 🌐 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main dashboard |
| `GET` | `/gallery` | Screenshot gallery |
| `GET` | `/about` | About page |
| `GET` | `/video` | MJPEG camera stream |
| `POST` | `/api/start` | Start the gesture controller |
| `POST` | `/api/stop` | Stop the gesture controller |
| `GET` | `/api/status` | Current status (running, gesture, volume) |
| `GET/POST` | `/api/config` | Get or update settings |
| `GET` | `/api/health` | Health check (used by Render) |

---

## ☁️ Deploying to Render

This project is ready to deploy on [Render](https://render.com) with zero config changes.

### One-Click Deploy

1. Fork this repository
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — hit **Deploy**

### What Happens on Render

- The `RENDER=true` environment variable is set automatically via `render.yaml`
- This activates **Demo Mode** — the app runs without a camera or OS access
- The UI shows a demo stream and all routes work; gesture actions are simulated
- Full gesture control requires running locally on a Windows machine with a webcam

### Render Configuration (`render.yaml`)

```yaml
services:
  - type: web
    name: gesture-flask-control
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
    envVars:
      - key: RENDER
        value: "true"
      - key: PYTHON_VERSION
        value: "3.11.9"
    healthCheckPath: /api/health
```

> **Why `--workers 1`?** The gesture controller is a module-level singleton. Multiple workers would each create their own isolated instance, breaking shared state.

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.11** | Runtime |
| **Flask 3.0** | Web framework & REST API |
| **OpenCV 4.10** (headless) | Webcam capture & frame processing |
| **MediaPipe 0.10** | Real-time hand landmark detection |
| **NumPy** | Numerical operations (volume interpolation) |
| **PyAutoGUI** | Mouse/keyboard OS control *(local only)* |
| **pycaw + comtypes** | Windows audio volume control *(local only)* |
| **Gunicorn** | Production WSGI server |
| **HTML / CSS / JS** | Frontend dashboard (vanilla, no frameworks) |

---

## 🔒 Environment Modes

| Mode | When | Camera | OS Control | Volume |
|---|---|---|---|---|
| **Full Mode** | Running locally on Windows | ✅ | ✅ | ✅ |
| **Demo Mode** | `RENDER=true` env var set | ❌ | ❌ | Simulated |

The app automatically selects the right mode at startup — no manual changes needed.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Built with ❤️ using Python, Flask, and MediaPipe
</div>
