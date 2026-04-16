"""
skeleton_drawer.py

Skeleton visualization utilities for mocap pipelines.
"""

import cv2
import numpy as np


class SkeletonDrawer:

    def __init__(
        self,
        connections,
        joint_color=(0,255,0),
        bone_color=(255,0,0),
        joint_radius=4,
        bone_thickness=2
    ):

        self.connections = connections
        self.joint_color = joint_color
        self.bone_color = bone_color
        self.joint_radius = joint_radius
        self.bone_thickness = bone_thickness

    def draw(self, frame, landmarks):

        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        pts = []

        # convert normalized coords → pixels
        for lm in landmarks:

            if lm is None:
                pts.append(None)
                continue

            x = int(lm[0] * w)
            y = int(lm[1] * h)

            pts.append((x,y))

        # draw bones
        for i, j in self.connections:

            if pts[i] is None or pts[j] is None:
                continue

            cv2.line(
                frame,
                pts[i],
                pts[j],
                self.bone_color,
                self.bone_thickness
            )

        # draw joints
        for pt in pts:

            if pt is None:
                continue

            cv2.circle(
                frame,
                pt,
                self.joint_radius,
                self.joint_color,
                -1
            )

        return frame
