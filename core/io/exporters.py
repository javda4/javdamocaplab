"""
exporters.py

Export utilities for mocap data:
- JSON
- CSV
- BVH (basic skeleton animation export)
"""

import json
import csv
import numpy as np


# ----------------------------
# BVH DEFAULT HIERARCHY
# ----------------------------

# Simple MediaPipe → BVH mapping (reduced skeleton)
BVH_JOINT_NAMES = [
    "Hips",
    "Spine",
    "Chest",
    "Neck",
    "Head",

    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",

    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",

    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",

    "RightUpLeg",
    "RightLeg",
    "RightFoot"
]


class MocapExporter:

    # ----------------------------
    # JSON EXPORT
    # ----------------------------

    def export_json(self, frames, filepath):

        with open(filepath, "w") as f:
            json.dump(frames, f, indent=2)


    # ----------------------------
    # CSV EXPORT
    # ----------------------------

    def export_csv(self, frames, filepath):

        with open(filepath, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["timestamp", "pose3d"])

            for frame in frames:
                writer.writerow([
                    frame["timestamp"],
                    frame.get("pose3d", None)
                ])


    # ----------------------------
    # BVH EXPORT (BASIC)
    # ----------------------------

    def export_bvh(self, frames, filepath, fps=30):

        """
        Basic BVH exporter.

        Assumes:
        - frames: list of dicts
        - each frame contains "pose3d" = Nx3 numpy/list
        """

        if len(frames) == 0:
            return

        skeleton = frames[0]["pose3d"]

        if skeleton is None:
            raise ValueError("No pose3d data found for BVH export")

        num_joints = min(len(BVH_JOINT_NAMES), len(skeleton))

        # ----------------------------
        # WRITE BVH HEADER
        # ----------------------------

        with open(filepath, "w") as f:

            f.write("HIERARCHY\n")
            f.write("ROOT Hips\n")
            f.write("{\n")

            f.write("  OFFSET 0 0 0\n")
            f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")

            for i in range(1, num_joints):
                f.write(f"  JOINT {BVH_JOINT_NAMES[i]}\n")
                f.write("  {\n")
                f.write("    OFFSET 0 0 0\n")
                f.write("    CHANNELS 3 Zrotation Xrotation Yrotation\n")
                f.write("  }\n")

            f.write("}\n")

            # ----------------------------
            # MOTION DATA
            # ----------------------------

            f.write("MOTION\n")
            f.write(f"Frames: {len(frames)}\n")
            f.write(f"Frame Time: {1.0 / fps:.6f}\n")

            for frame in frames:

                pose = frame.get("pose3d", None)

                if pose is None:
                    continue

                pose = np.array(pose)

                line = []

                for i in range(num_joints):

                    x, y, z = pose[i]

                    # VERY SIMPLE mapping:
                    # position only (no real joint rotation yet)

                    if i == 0:
                        # root joint (Hips)
                        line.extend([x, y, z, 0, 0, 0])
                    else:
                        line.extend([0, 0, 0])

                f.write(" ".join(map(str, line)) + "\n")
