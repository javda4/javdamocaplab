"""
webcam.py

Core webcam capture module for MoCapLab.
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
        config=None
    ):
        """
        You can either:
        - pass explicit args (old way)
        - OR pass config dict (new project system)
        """

        if config is not None:
            device_id = config.get("index", device_id)
            width = config.get("width", width)
            height = config.get("height", height)
            fps = config.get("fps", fps)
            mirror = config.get("mirror", mirror)

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

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            with self.lock:
                self.frame = frame
                self.timestamp = time.time()

    def read(self):

        with self.lock:
            return self.frame, self.timestamp

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join()

        if self.cap:
            self.cap.release()

    def is_running(self):
        return self.running
