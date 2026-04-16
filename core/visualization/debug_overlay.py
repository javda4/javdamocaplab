"""
debug_overlay.py

Debug information overlay for visualization.
"""

import cv2
import time


class DebugOverlay:

    def __init__(self):

        self.prev_time = time.time()
        self.fps = 0

    def update_fps(self):

        now = time.time()

        dt = now - self.prev_time

        if dt > 0:
            self.fps = 1.0 / dt

        self.prev_time = now

        return self.fps

    def draw(
        self,
        frame,
        camera_id=None,
        timestamp=None,
        extra=None
    ):

        y = 20

        # FPS
        fps = self.update_fps()

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        y += 25

        # camera id
        if camera_id is not None:

            cv2.putText(
                frame,
                f"Camera: {camera_id}",
                (10,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2
            )

            y += 25

        # timestamp
        if timestamp is not None:

            cv2.putText(
                frame,
                f"Time: {timestamp:.3f}",
                (10,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

            y += 25

        # extra debug data
        if extra:

            for key, value in extra.items():

                cv2.putText(
                    frame,
                    f"{key}: {value}",
                    (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200,200,200),
                    2
                )

                y += 25

        return frame
