import cv2

from core.camera.webcam import WebcamCamera
from core.pose.hand_tracking import MediaPipeHandTracker


def main():

    cam = WebcamCamera(width=1280, height=720, mirror=True)
    hand_tracker = MediaPipeHandTracker()

    cam.start()

    while True:

        frame, ts = cam.read()

        if frame is None:
            continue

        result = hand_tracker.process(frame)

        frame = hand_tracker.draw(frame, result["hands"])

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()
    hand_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
