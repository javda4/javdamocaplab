import cv2

from core.camera.webcam import WebcamCamera
from core.pose.mediapipe_pose import MediaPipePoseTracker
from core.pose.hand_tracking import MediaPipeHandTracker
from core.pose.face_tracking import MediaPipeFaceTracker


def main():

    cam = WebcamCamera(width=1280, height=720, mirror=True)

    pose_tracker = MediaPipePoseTracker()
    hand_tracker = MediaPipeHandTracker()
    face_tracker = MediaPipeFaceTracker()

    cam.start()

    while True:

        frame, ts = cam.read()

        if frame is None:
            continue

        pose_result = pose_tracker.process(frame)
        hand_result = hand_tracker.process(frame)
        face_result = face_tracker.process(frame)

        frame = pose_tracker.draw(frame)
        frame = hand_tracker.draw(frame, hand_result["hands"])
        frame = face_tracker.draw(frame, face_result["faces"])

        cv2.imshow("OpenMoCapLab Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()

    pose_tracker.close()
    hand_tracker.close()
    face_tracker.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
