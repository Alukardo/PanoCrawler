"""
Geometry helpers for camera and panorama transforms.
"""

import math

import numpy as np


def euler_angles_to_rotation_matrix(theta):
    """
    Convert Euler angles (rx, ry, rz) to a 3x3 rotation matrix.
    """
    r_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    r_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    r_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    return np.dot(r_z, np.dot(r_y, r_x))


def eulerAnglesToRotationMatrix(theta):
    """Backward-compatible alias for existing callers."""
    return euler_angles_to_rotation_matrix(theta)
