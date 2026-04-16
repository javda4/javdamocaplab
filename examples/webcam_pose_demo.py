import cv2

from core.camera.webcam import WebcamCamera
from core.pose.mediapipe_pose import MediaPipePoseTracker
from core.visualization.skeleton_drawer import SkeletonDrawer
from core.visualization.debug_overlay import DebugOverlay


POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28),
    (27,31),(28,32)
]


def main():

    cam = WebcamCamera(mirror=True)
    pose = MediaPipePoseTracker()

    drawer = SkeletonDrawer(POSE_CONNECTIONS)
    overlay = DebugOverlay()

    cam.start()

    while True:

        frame, ts = cam.read()
        if frame is None:
            continue

        result = pose.process(frame)

        landmarks = result["landmarks"]

        frame = drawer.draw(frame, landmarks)
        frame = overlay.draw(frame, camera_id=0, timestamp=ts)

        cv2.imshow("Webcam Pose Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
