"""
transforms.py

3D transformation utilities for mocap.
"""

import numpy as np
import cv2


class TransformUtils:

    @staticmethod
    def normalize_landmarks(landmarks):
        """
        Center landmarks around origin.
        """

        center = np.mean(landmarks, axis=0)

        return landmarks - center

    @staticmethod
    def scale_landmarks(landmarks, scale):

        return landmarks * scale

    @staticmethod
    def rotation_matrix_from_euler(rx, ry, rz):
        """
        Create rotation matrix from Euler angles.
        """

        Rx = np.array([
        [1,0,0],
        [0,np.cos(rx),-np.sin(rx)],
        [0,np.sin(rx),np.cos(rx)]
        ])

        Ry = np.array([
        [np.cos(ry),0,np.sin(ry)],
        [0,1,0],
        [-np.sin(ry),0,np.cos(ry)]
        ])

        Rz = np.array([
        [np.cos(rz),-np.sin(rz),0],
        [np.sin(rz),np.cos(rz),0],
        [0,0,1]
        ])

        return Rz @ Ry @ Rx

    @staticmethod
    def rotate_points(points, R):
        """
        Rotate Nx3 points with rotation matrix.
        """

        return np.dot(points, R.T)

    @staticmethod
    def translate_points(points, t):
        """
        Translate points by vector t.
        """

        return points + t

    @staticmethod
    def world_to_camera(points, R, t):
        """
        Convert world coordinates → camera coordinates.
        """

        return (R @ points.T + t.reshape(3,1)).T

    @staticmethod
    def camera_to_world(points, R, t):
        """
        Convert camera coordinates → world coordinates.
        """

        return (R.T @ (points.T - t.reshape(3,1))).T
