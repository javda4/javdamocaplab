import cv2

from core.camera.webcam import WebcamCamera
from core.pose.face_tracking import MediaPipeFaceTracker


def main():

    cam = WebcamCamera(width=1280, height=720, mirror=True)
    face_tracker = MediaPipeFaceTracker()

    cam.start()

    while True:

        frame, ts = cam.read()

        if frame is None:
            continue

        result = face_tracker.process(frame)

        frame = face_tracker.draw(frame, result["faces"])

        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()
    face_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
