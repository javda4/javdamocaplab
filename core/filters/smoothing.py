"""
smoothing.py

Basic smoothing filters for mocap landmarks.
"""

import numpy as np
from collections import deque


class ExponentialSmoother:
    """
    Exponential smoothing filter.
    """

    def __init__(self, alpha=0.5):

        self.alpha = alpha
        self.prev = None

    def update(self, points):

        points = np.array(points)

        if self.prev is None:
            self.prev = points
            return points

        smoothed = self.alpha * points + (1 - self.alpha) * self.prev

        self.prev = smoothed

        return smoothed


class MovingAverageSmoother:
    """
    Moving average smoothing filter.
    """

    def __init__(self, window_size=5):

        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, points):

        points = np.array(points)

        self.history.append(points)

        return np.mean(self.history, axis=0)
