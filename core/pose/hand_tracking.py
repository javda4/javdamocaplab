"""
hand_tracking.py

Hand tracking module using MediaPipe Hands.
"""

import cv2
import mediapipe as mp
import numpy as np
import time


class MediaPipeHandTracker:

    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        timestamp = time.time()

        hands_data = []

        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                hands_data.append({
                    "label": handedness.classification[0].label,
                    "landmarks": np.array(landmarks),
                })

        return {
            "hands": hands_data,
            "timestamp": timestamp
        }

    def draw(self, frame, hands):

        h, w, _ = frame.shape

        for hand in hands:

            for lm in hand["landmarks"]:
                x = int(lm[0] * w)
                y = int(lm[1] * h)

                cv2.circle(frame, (x,y), 3, (255,0,0), -1)

        return frame

    def close(self):
        self.hands.close()
