"""
webcam.py

Core webcam capture module for OpenMoCapLab.

Features
--------
- Threaded frame capture
- Adjustable resolution and FPS
- Timestamped frames
- Optional frame mirroring
"""

import cv2
import threading
import time


class WebcamCamera:
    def __init__(
        self,
        device_id=0,
        width=1280,
        height=720,
        fps=30,
        mirror=False,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror

        self.cap = None
        self.frame = None
        self.timestamp = None

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Start webcam capture."""
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Internal thread for grabbing frames."""
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            timestamp = time.time()

            with self.lock:
                self.frame = frame
                self.timestamp = timestamp

    def read(self):
        """Return latest frame and timestamp."""
        with self.lock:
            return self.frame, self.timestamp

    def stop(self):
        """Stop capture."""
        self.running = False

        if self.thread:
            self.thread.join()

        if self.cap:
            self.cap.release()

    def is_running(self):
        return self.running
