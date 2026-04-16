import cv2

from core.camera.webcam import WebcamCamera
from core.pose.face_tracking import MediaPipeFaceTracker
from core.visualization.debug_overlay import DebugOverlay


def main():

    cam = WebcamCamera(mirror=True)
    face = MediaPipeFaceTracker()
    overlay = DebugOverlay()

    cam.start()

    while True:

        frame, ts = cam.read()
        if frame is None:
            continue

        result = face.process(frame)

        faces = result["faces"]

        frame = face.draw(frame, faces)
        frame = overlay.draw(frame, camera_id=0, timestamp=ts)

        cv2.imshow("Face Capture Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()
    face.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
