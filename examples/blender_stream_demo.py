'''
This assumes you already have:

calibration matrices
at least 2 webcams
'''


import cv2
import numpy as np

from core.camera.multicam import MultiCameraSystem
from core.pose.mediapipe_pose import MediaPipePoseTracker
from core.math.triangulation import Triangulator
from core.visualization.debug_overlay import DebugOverlay


def extract_2d(landmarks):
    """Convert MediaPipe landmarks → Nx2"""
    if landmarks is None:
        return None
    return landmarks[:, :2]


def main():

    cams = MultiCameraSystem([0, 1])
    pose = MediaPipePoseTracker()
    overlay = DebugOverlay()

    # FAKE projection matrices (replace with calibration output later)
    P1 = np.eye(3, 4)
    P2 = np.eye(3, 4)

    triangulator = Triangulator([P1, P2])

    cams.start()

    while True:

        frames = cams.read()

        cam_ids = sorted(frames.keys())

        if len(cam_ids) < 2:
            continue

        cam1_frame, ts1 = frames[cam_ids[0]]
        cam2_frame, ts2 = frames[cam_ids[1]]

        r1 = pose.process(cam1_frame)
        r2 = pose.process(cam2_frame)

        lm1 = extract_2d(r1["landmarks"])
        lm2 = extract_2d(r2["landmarks"])

        if lm1 is not None and lm2 is not None:

            points3d = triangulator.triangulate_landmarks([lm1, lm2])

            cam1_frame = overlay.draw(cam1_frame, camera_id=0)
            cam2_frame = overlay.draw(cam2_frame, camera_id=1)

        cv2.imshow("Cam 1", cam1_frame)
        cv2.imshow("Cam 2", cam2_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cams.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
