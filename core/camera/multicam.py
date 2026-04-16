"""
multicam.py

Multi-camera capture system for OpenMoCapLab.

Supports synchronized frame grabbing across
multiple webcams for 3D reconstruction pipelines.
"""

import time
from typing import Dict, List

from .webcam import WebcamCamera


class MultiCameraSystem:
    def __init__(
        self,
        device_ids: List[int],
        width=1280,
        height=720,
        fps=30,
        mirror=False,
    ):
        self.device_ids = device_ids
        self.cameras: Dict[int, WebcamCamera] = {}

        for dev in device_ids:
            self.cameras[dev] = WebcamCamera(
                device_id=dev,
                width=width,
                height=height,
                fps=fps,
                mirror=mirror,
            )

    def start(self):
        """Start all cameras."""
        for cam in self.cameras.values():
            cam.start()

        time.sleep(1)  # allow cameras to warm up

    def read(self):
        """
        Return frames from all cameras.

        Returns
        -------
        dict
            {
                device_id: (frame, timestamp)
            }
        """
        frames = {}

        for dev, cam in self.cameras.items():
            frame, ts = cam.read()

            if frame is not None:
                frames[dev] = (frame, ts)

        return frames

    def stop(self):
        """Stop all cameras."""
        for cam in self.cameras.values():
            cam.stop()

    def get_camera(self, device_id):
        """Return specific camera instance."""
        return self.cameras.get(device_id)

    def camera_count(self):
        return len(self.cameras)
