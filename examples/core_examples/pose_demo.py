import cv2

from core.camera.webcam import WebcamCamera
from core.pose.mediapipe_pose import MediaPipePoseTracker


def main():

    cam = WebcamCamera(width=1280, height=720, mirror=True)
    pose_tracker = MediaPipePoseTracker()

    cam.start()

    while True:

        frame, ts = cam.read()

        if frame is None:
            continue

        result = pose_tracker.process(frame)

        frame = pose_tracker.draw(frame)

        cv2.imshow("Pose Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()
    pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
