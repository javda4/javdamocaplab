"""
mediapipe_pose.py

Body pose tracking module using MediaPipe.
"""

import cv2
import time
import mediapipe as mp
import numpy as np


class MediaPipePoseTracker:

    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarks = None

    def process(self, frame):
        """Process frame and return pose landmarks."""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        timestamp = time.time()

        if results.pose_landmarks:

            landmarks = []

            for lm in results.pose_landmarks.landmark:
                landmarks.append([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ])

            self.landmarks = np.array(landmarks)

        else:
            self.landmarks = None

        return {
            "landmarks": self.landmarks,
            "timestamp": timestamp
        }

    def draw(self, frame):
        """Draw skeleton on frame."""

        if self.landmarks is None:
            return frame

        h, w, _ = frame.shape

        for lm in self.landmarks:
            x = int(lm[0] * w)
            y = int(lm[1] * h)

            cv2.circle(frame, (x, y), 4, (0,255,0), -1)

        return frame

    def close(self):
        self.pose.close()
