# gesture_control.py
import os
import sys
import time
import math
import numpy as np
from datetime import datetime
from threading import Thread, Event, Lock

# --- Detect environment ---
IS_WINDOWS = sys.platform == "win32"
HAS_CAMERA = False
HAS_DISPLAY = os.environ.get("DISPLAY") or IS_WINDOWS  # X11 or Windows

# --- Conditional imports (these FAIL on headless Linux / Render) ---
cv2 = None
mp = None
pyautogui = None
volume_available = False

try:
    import cv2 as _cv2
    cv2 = _cv2
except ImportError:
    print("[INFO] opencv-python not available")

try:
    import mediapipe as _mp
    mp = _mp
except ImportError:
    print("[INFO] mediapipe not available")

try:
    import pyautogui as _pyautogui
    pyautogui = _pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    print("[INFO] pyautogui not available")

# Windows-only audio
audio_volume_iface = None
if IS_WINDOWS:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        audio_volume_iface = cast(interface, POINTER(IAudioEndpointVolume))
        volume_available = True
    except Exception as e:
        print(f"[INFO] Audio setup skipped: {e}")


class GestureController:
    """Full gesture controller — only works on Windows with a camera."""

    def __init__(self):
        self.volume_iface = audio_volume_iface

        # MediaPipe
        self.mp_hands = mp.solutions.hands if mp else None
        self.hands = (
            self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.7,
            )
            if self.mp_hands
            else None
        )
        self.mp_drawing = mp.solutions.drawing_utils if mp else None

        # Screen / camera
        self.screen_w, self.screen_h = (
            pyautogui.size() if pyautogui else (1920, 1080)
        )
        self.cam_w, self.cam_h = 1280, 720

        # Config (can be updated from UI)
        self.config = {
            "scroll_sensitivity": 40,
            "mouse_sensitivity": 1.5,
            "cooldown": 0.2,
            "screenshot_dir": "static/screenshots",
            "vol_min_distance": 0.02,
            "vol_max_distance": 0.20,
            "double_click_threshold": 0.30,
        }
        os.makedirs(self.config["screenshot_dir"], exist_ok=True)

        # State
        self._thread = None
        self._stop = Event()
        self._lock = Lock()
        self._last_action_time = 0.0
        self._last_click_time = 0.0
        self._current_gesture = "None"
        self._current_volume = 0
        self._last_frame = None

    # ---------- public API ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self._lock:
            return {
                "running": self.is_running(),
                "gesture": self._current_gesture,
                "volume": int(self._current_volume * 100),
            }

    def update_config(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k in self.config and v is not None:
                    if k == "cooldown":
                        v = max(0.05, float(v))
                    elif k in ("scroll_sensitivity",):
                        v = int(v)
                    elif k in ("mouse_sensitivity",):
                        v = float(v)
                    self.config[k] = v
        return self.config

    def get_jpeg_frame(self):
        """Return the most recent frame as JPEG bytes for MJPEG streaming."""
        with self._lock:
            frame = self._last_frame.copy() if self._last_frame is not None else None
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if cv2:
                cv2.putText(
                    frame,
                    "Waiting for camera...",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
        if cv2:
            ret, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()
        # Fallback: return a minimal 1x1 JPEG
        return b""

    # ---------- internals ----------
    def _calculate_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _map(self, value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _take_screenshot(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.config["screenshot_dir"], f"screenshot_{timestamp}.png"
        )
        if pyautogui:
            pyautogui.screenshot(filename)
        return filename

    def _get_finger_state(self, hand):
        fingers = []
        # Thumb (right-hand heuristic)
        fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(1 if hand.landmark[tip].y < hand.landmark[pip].y else 0)
        return fingers

    def _recognize_gesture(self, f):
        if f == [0, 1, 0, 0, 0]:
            return "one_finger"
        if f == [0, 1, 1, 0, 0]:
            return "two_fingers"
        if f == [0, 1, 1, 1, 0]:
            return "three_fingers"
        if f == [0, 1, 1, 1, 1]:
            return "four_fingers"
        if f == [1, 1, 1, 1, 1]:
            return "five_fingers"
        if f == [0, 0, 0, 0, 1]:
            return "pinky"
        if f == [1, 0, 0, 0, 1]:
            return "phone"
        return None

    def _run(self):
        if not cv2:
            print("[ERROR] OpenCV not available, cannot start camera loop")
            self._stop.set()
            return

        cap = cv2.VideoCapture(0)
        cap.set(3, self.cam_w)
        cap.set(4, self.cam_h)
        if not cap.isOpened():
            print("Camera open failed")
            self._stop.set()
            return

        try:
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    continue
                img = cv2.flip(img, 1)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_img) if self.hands else None

                current_gesture = "None"
                current_time = time.time()
                vol_scalar = 0.0

                if results and results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    if self.mp_drawing and self.mp_hands:
                        self.mp_drawing.draw_landmarks(
                            img, hand, self.mp_hands.HAND_CONNECTIONS
                        )

                    fingers = self._get_finger_state(hand)
                    gesture = self._recognize_gesture(fingers)

                    thumb_tip = hand.landmark[4]
                    index_tip = hand.landmark[8]

                    # Volume control via pinch
                    vol_dist = self._calculate_distance(thumb_tip, index_tip)
                    vol_scalar = np.interp(
                        vol_dist,
                        [
                            self.config["vol_min_distance"],
                            self.config["vol_max_distance"],
                        ],
                        [0, 1],
                    )
                    vol_scalar = float(np.clip(vol_scalar, 0, 1))
                    if self.volume_iface:
                        try:
                            self.volume_iface.SetMasterVolumeLevelScalar(
                                vol_scalar, None
                            )
                        except Exception:
                            pass

                    # Actions
                    if gesture == "one_finger" and pyautogui:
                        mx = int(
                            self._map(index_tip.x, 0.1, 0.9, 0, self.screen_w)
                        )
                        my = int(
                            self._map(index_tip.y, 0.1, 0.9, 0, self.screen_h)
                        )
                        pyautogui.moveTo(
                            mx,
                            my,
                            duration=max(
                                0.01, 0.07 / self.config["mouse_sensitivity"]
                            ),
                        )
                        current_gesture = "Cursor"

                    elif (
                        gesture == "two_fingers"
                        and pyautogui
                        and current_time - self._last_action_time
                        > self.config["cooldown"]
                    ):
                        if (
                            current_time - self._last_click_time
                            < self.config["double_click_threshold"]
                        ):
                            pyautogui.doubleClick()
                            current_gesture = "Double Click"
                        else:
                            pyautogui.click()
                            current_gesture = "Click"
                        self._last_click_time = current_time
                        self._last_action_time = current_time

                    elif (
                        gesture == "three_fingers"
                        and pyautogui
                        and current_time - self._last_action_time
                        > self.config["cooldown"]
                    ):
                        pyautogui.scroll(self.config["scroll_sensitivity"])
                        current_gesture = "Scroll Up"
                        self._last_action_time = current_time

                    elif (
                        gesture == "four_fingers"
                        and pyautogui
                        and current_time - self._last_action_time
                        > self.config["cooldown"]
                    ):
                        pyautogui.scroll(-self.config["scroll_sensitivity"])
                        current_gesture = "Scroll Down"
                        self._last_action_time = current_time

                    elif (
                        gesture == "five_fingers"
                        and current_time - self._last_action_time
                        > self.config["cooldown"]
                    ):
                        _ = self._take_screenshot()
                        current_gesture = "Screenshot Saved"
                        self._last_action_time = current_time
                        overlay = cv2.addWeighted(
                            img, 0.5, np.zeros_like(img), 0.5, 0
                        )
                        img = overlay

                    elif (
                        gesture == "pinky"
                        and pyautogui
                        and current_time - self._last_action_time
                        > self.config["cooldown"]
                    ):
                        pyautogui.hotkey("alt", "tab")
                        current_gesture = "Alt+Tab"
                        self._last_action_time = current_time

                    elif (
                        gesture == "phone"
                        and pyautogui
                        and current_time - self._last_action_time
                        > self.config["cooldown"]
                    ):
                        pyautogui.hotkey("win", "d")
                        current_gesture = "Show Desktop"
                        self._last_action_time = current_time

                # HUD
                cv2.putText(
                    img,
                    f"Gesture: {current_gesture}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"Volume: {int(vol_scalar*100)}%",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                with self._lock:
                    self._current_gesture = current_gesture
                    self._current_volume = vol_scalar
                    self._last_frame = img

        finally:
            cap.release()
            with self._lock:
                self._current_gesture = "None"


class DemoController:
    """Lightweight fallback for headless servers (Render, etc.).
    Returns static data — no camera, no OS control.
    """

    def __init__(self):
        self.config = {
            "scroll_sensitivity": 40,
            "mouse_sensitivity": 1.5,
            "cooldown": 0.2,
            "screenshot_dir": "static/screenshots",
            "vol_min_distance": 0.02,
            "vol_max_distance": 0.20,
            "double_click_threshold": 0.30,
        }
        os.makedirs(self.config["screenshot_dir"], exist_ok=True)
        self._demo_mode = True

    def start(self):
        pass  # no-op

    def stop(self):
        pass  # no-op

    def is_running(self):
        return False

    def status(self):
        return {
            "running": False,
            "gesture": "Demo Mode",
            "volume": 50,
            "demo": True,
        }

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.config and v is not None:
                self.config[k] = v
        return self.config

    def get_jpeg_frame(self):
        # Return a simple placeholder frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if cv2:
            cv2.putText(
                frame,
                "Demo Mode — No Camera",
                (80, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (108, 140, 255),
                2,
            )
            cv2.putText(
                frame,
                "Run locally for gesture control",
                (90, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (170, 180, 220),
                2,
            )
            _, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()
        return b""


# Singleton — pick real or demo based on environment
if os.environ.get("RENDER"):
    print("[INFO] Running in DEMO mode (Render deployment detected)")
    controller = DemoController()
else:
    print("[INFO] Running in FULL mode (Local environment)")
    controller = GestureController()
