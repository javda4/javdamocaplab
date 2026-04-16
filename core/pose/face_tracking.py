"""
face_tracking.py

Face mesh tracking module using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
import time


class MediaPipeFaceTracker:

    def __init__(
        self,
        max_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        self.mp_face = mp.solutions.face_mesh

        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        timestamp = time.time()

        faces = []

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = []

                for lm in face_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                faces.append(np.array(landmarks))

        return {
            "faces": faces,
            "timestamp": timestamp
        }

    def draw(self, frame, faces):

        h, w, _ = frame.shape

        for face in faces:

            for lm in face:
                x = int(lm[0] * w)
                y = int(lm[1] * h)

                cv2.circle(frame, (x,y), 1, (0,255,255), -1)

        return frame

    def close(self):
        self.face_mesh.close()
