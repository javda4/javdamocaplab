import cv2
from core.camera.multicam import MultiCameraSystem

cams = MultiCameraSystem([0,1])

cams.start()

while True:

    frames = cams.read()

    for cam_id, (frame, ts) in frames.items():
        cv2.imshow(f"Camera {cam_id}", frame)

    if cv2.waitKey(1) == 27:
        break

cams.stop()
cv2.destroyAllWindows()
