"""
triangulation.py

Multi-view triangulation utilities for OpenMoCapLab.
Supports 2+ cameras.
"""

import numpy as np


class Triangulator:

    def __init__(self, projection_matrices):

        if len(projection_matrices) < 2:
            raise ValueError("At least two cameras required")

        self.P = projection_matrices

    def triangulate_point(self, observations):
        """
        observations:
        [(x,y) or None, ...]
        """

        A = []

        for obs, P in zip(observations, self.P):

            if obs is None:
                continue

            x, y = obs

            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])

        if len(A) < 4:
            return None  # not enough views

        A = np.array(A)

        _, _, Vt = np.linalg.svd(A)

        X = Vt[-1]
        X = X / X[3]

        return X[:3]

    def triangulate_landmarks(self, multi_view_landmarks):

        num_points = len(multi_view_landmarks[0])

        points3d = []

        for i in range(num_points):

            observations = []

            for cam in multi_view_landmarks:

                if cam[i] is None:
                    observations.append(None)
                else:
                    observations.append(cam[i])

            pt3d = self.triangulate_point(observations)

            points3d.append(pt3d)

        return np.array(points3d)
