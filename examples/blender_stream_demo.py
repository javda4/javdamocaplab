import cv2
import socket
import json

from core.camera.webcam import WebcamCamera
from core.pose.mediapipe_pose import MediaPipePoseTracker


BLENDER_HOST = "127.0.0.1"
BLENDER_PORT = 5005


def send(sock, data):
    msg = json.dumps(data).encode("utf-8")
    sock.sendall(msg + b"\n")


def main():

    cam = WebcamCamera(mirror=True)
    pose = MediaPipePoseTracker()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((BLENDER_HOST, BLENDER_PORT))

    cam.start()

    while True:

        frame, ts = cam.read()
        if frame is None:
            continue

        result = pose.process(frame)

        payload = {
            "timestamp": ts,
            "pose": result["landmarks"].tolist()
            if result["landmarks"] is not None else None
        }

        send(sock, payload)

        cv2.imshow("Blender Stream Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.stop()
    pose.close()
    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
