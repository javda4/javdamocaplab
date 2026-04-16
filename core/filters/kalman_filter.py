"""
kalman_filter.py

Kalman filtering utilities for smoothing mocap landmark data.
"""

import numpy as np


class KalmanFilter3D:
    """
    Kalman filter for a single 3D point.
    """

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):

        self.dt = 1.0

        # State vector
        self.x = np.zeros((6, 1))

        # State covariance
        self.P = np.eye(6)

        # State transition matrix
        self.F = np.array([
            [1,0,0,self.dt,0,0],
            [0,1,0,0,self.dt,0],
            [0,0,1,0,0,self.dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])

        # Measurement matrix
        self.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
        ])

        # Noise matrices
        self.Q = process_noise * np.eye(6)
        self.R = measurement_noise * np.eye(3)

    def predict(self):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:3].flatten()

    def update(self, measurement):

        z = np.array(measurement).reshape(3,1)

        y = z - (self.H @ self.x)

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)

        I = np.eye(self.P.shape[0])

        self.P = (I - K @ self.H) @ self.P

        return self.x[:3].flatten()


class SkeletonKalmanFilter:
    """
    Kalman filter for entire skeleton.
    """

    def __init__(self, num_landmarks):

        self.filters = [
            KalmanFilter3D()
            for _ in range(num_landmarks)
        ]

    def update(self, landmarks):

        filtered = []

        for i, point in enumerate(landmarks):

            if point is None:
                filtered.append(self.filters[i].predict())
            else:
                filtered.append(self.filters[i].update(point))

        return np.array(filtered)
