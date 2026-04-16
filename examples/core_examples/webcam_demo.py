from core.camera.webcam import WebcamCamera
import cv2

cam = WebcamCamera()

cam.start()

while True:
    frame, ts = cam.read()

    if frame is None:
        continue

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27:
        break

cam.stop()
cv2.destroyAllWindows()
