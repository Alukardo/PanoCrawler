"""
Unity coordinate system utilities.
Converts Euler angles to rotation matrices for 3D transformations.
"""

import math
import numpy as np

def eulerAnglesToRotationMatrix(theta):
    """
    Convert Euler angles (rx, ry, rz) to a rotation matrix.

    Args:
        theta: List or array of three angles in radians [rx, ry, rz].

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R
