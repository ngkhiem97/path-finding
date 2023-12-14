import numpy as np

def quat_to_rot(quat):
    """
    Converts a quaternion to a rotation matrix.

    Parameters:
    quat -- numpy array or list with 4 elements representing the quaternion

    Returns:
    R -- numpy 3x3 array representing the rotation matrix
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    R = np.array([[1 - 2 * qy**2 - 2 * qz**2, 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * qx**2 - 2 * qz**2, 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * qx**2 - 2 * qy**2]])
    return R

def rot_to_quat(R):
    """
    Converts a rotation matrix to a quaternion.

    Parameters:
    R -- numpy array representing the 3x3 rotation matrix

    Returns:
    q -- numpy array with 4 elements representing the quaternion
    """
    tr = np.trace(R)
    S = np.sqrt(tr + 1.0) * 2  # S=4*qw

    if tr > 0:
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[2, 1] + R[1, 2]) / S
    else:
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    return q

def rot_to_rpy_zxy(R):
    """
    Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix.

    Parameters:
    R -- A 3x3 numpy array representing the rotation matrix (world to body).

    Returns:
    roll, pitch, yaw -- Roll, pitch, and yaw angles.
    """
    roll = np.arcsin(R[1, 2])
    yaw = np.arctan2(-R[1, 0] / np.cos(roll), R[1, 1] / np.cos(roll))
    pitch = np.arctan2(-R[0, 2] / np.cos(roll), R[2, 2] / np.cos(roll))
    return roll, pitch, yaw

def rpy_to_rot_zxy(roll, pitch, yaw):
    """
    Converts roll (phi), pitch (theta), and yaw (psi) to a body-to-world Rotation matrix.
    
    :param roll: Roll angle.
    :param pitch: Pitch angle.
    :param yaw: Yaw angle.
    :return: A numpy array representing the rotation matrix.
    """
    R = np.array([[np.cos(yaw) * np.cos(pitch) - np.sin(roll) * np.sin(yaw) * np.sin(pitch),
                   np.cos(pitch) * np.sin(yaw) + np.cos(yaw) * np.sin(roll) * np.sin(pitch),
                   -np.cos(roll) * np.sin(pitch)],
                  [-np.cos(roll) * np.sin(yaw),
                   np.cos(roll) * np.cos(yaw),
                   np.sin(roll)],
                  [np.cos(yaw) * np.sin(pitch) + np.cos(pitch) * np.sin(roll) * np.sin(yaw),
                   np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.cos(pitch) * np.sin(roll),
                   np.cos(roll) * np.cos(pitch)]])
    return R

def state_to_qd(x):
    """
    Converts a state vector to a dictionary with position, velocity, Euler angles, and angular velocity.
    
    Parameters:
    x -- A 1D numpy array of state variables [pos vel quat omega] of length 13.
    
    Returns:
    qd -- A dictionary with keys 'pos', 'vel', 'rot', and 'omega'.
    """
    rot = quat_to_rot(x[6:10])
    roll, pitch, yaw = rot_to_rpy_zxy(rot)
    
    qd = {}
    qd['pos'] = x[0:3]
    qd['vel'] = x[3:6]
    qd['rot'] = np.array([roll, pitch, yaw])
    qd['omega'] = x[10:13]

    return qd