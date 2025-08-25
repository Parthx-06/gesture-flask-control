import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import time
import math
from datetime import datetime
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class GestureController:
    def __init__(self):
        pyautogui.FAILSAFE = False

        # Audio setup
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            vol_range = self.volume.GetVolumeRange()
            self.min_vol, self.max_vol = vol_range[0], vol_range[1]
        except Exception as e:
            print("Audio setup failed:", e)
            self.volume = None
            self.min_vol, self.max_vol = 0, 1

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Screen
        self.screen_w, self.screen_h = pyautogui.size()
        self.cam_w, self.cam_h = 1280, 720

        # Config
        self.config = {
            'scroll_sensitivity': 40,
            'mouse_sensitivity': 1.5,
            'cooldown': 0.2,
            'screenshot_dir': 'screenshots',
            'vol_min_distance': 0.02,
            'vol_max_distance': 0.2,
            'double_click_threshold': 0.3
        }
        os.makedirs(self.config['screenshot_dir'], exist_ok=True)

        # State
        self.last_action_time = 0
        self.last_click_time = 0
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.cam_w)
        self.cap.set(4, self.cam_h)

    def calculate_distance(self, p1, p2):
        return math.dist([p1.x, p1.y], [p2.x, p2.y])

    def map_value(self, value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def get_finger_state(self, hand_landmarks):
        fingers = []
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
        for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
        return fingers

    def recognize_gesture(self, fingers):
        if fingers == [0,1,0,0,0]: return 'one_finger'
        if fingers == [0,1,1,0,0]: return 'two_fingers'
        if fingers == [0,1,1,1,0]: return 'three_fingers'
        if fingers == [0,1,1,1,1]: return 'four_fingers'
        if fingers == [1,1,1,1,1]: return 'five_fingers'
        if fingers == [0,0,0,0,1]: return 'pinky'
        if fingers == [1,0,0,0,1]: return 'phone'
        return None

    def generate_frames(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_img)
            current_gesture = None
            vol = 0
            current_time = time.time()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                fingers = self.get_finger_state(hand_landmarks)
                gesture = self.recognize_gesture(fingers)

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Volume control
                vol_distance = self.calculate_distance(thumb_tip, index_tip)
                vol = np.interp(vol_distance,
                                [self.config['vol_min_distance'], self.config['vol_max_distance']],
                                [0, 1])
                vol = np.clip(vol, 0, 1)
                if self.volume:
                    self.volume.SetMasterVolumeLevelScalar(vol, None)

                # Show text
                if gesture: current_gesture = gesture

            cv2.putText(img, f"Gesture: {current_gesture or 'None'}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(img, f"Volume: {int(vol*100)}%", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

