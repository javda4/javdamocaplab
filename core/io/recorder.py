"""
recorder.py

Mocap recording system for OpenMoCapLab.

Stores timestamped landmark data for later export.
"""

import time
import json


class MocapRecorder:

    def __init__(self):

        self.recording = False
        self.frames = []

    def start(self):

        self.recording = True
        self.frames = []

    def stop(self):

        self.recording = False

    def add_frame(
        self,
        pose=None,
        hands=None,
        face=None,
        pose3d=None,
        timestamp=None
    ):

        if not self.recording:
            return

        if timestamp is None:
            timestamp = time.time()

        frame_data = {
            "timestamp": timestamp,
            "pose": pose.tolist() if pose is not None else None,
            "hands": hands,
            "face": face,
            "pose3d": pose3d.tolist() if pose3d is not None else None
        }

        self.frames.append(frame_data)

    def save_json(self, filepath):

        with open(filepath, "w") as f:
            json.dump(self.frames, f, indent=2)

    def clear(self):

        self.frames = []
