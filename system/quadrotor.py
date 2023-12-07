import numpy as np
import system.utils as utils

class Quadrotor:
    def __init__(self, controlhandle: callable, params: dict):
        self.controlhandle = controlhandle
        self.params = params

    def init_state(self, start, yaw):
        """
        Initialize 13 x 1 state vector.

        :param start: A numpy array of shape (3,) representing the starting position [x, y, z].
        :param yaw: The initial yaw angle.
        :return: A numpy array of shape (13,) representing the initialized state.
        """
        Rot0 = utils.rpy_to_rot_zxy(0.0, 0.0, yaw)
        Quat0 = utils.rot_to_quat(Rot0)
        s = np.zeros(13)
        s[0:3] = start     # x, y, z
        s[3:6] = 0         # xdot, ydot, zdot
        s[6:10] = Quat0    # qw, qx, qy, qz
        s[10:13] = 0       # p, q, r
        return s
    
    def get_prop_pos(self, pos, rot):
        """
        Get the position of the four propellers.

        :param pos: A numpy array of shape (3,) representing the position of the quadrotor.
        :param rot: A numpy array of shape (3,) representing the body-to-world rotation matrix of the quadrotor.
        :param L: The length of the quadrotor arm.
        :return: A numpy array of shape (4, 3) representing the positions of the four propellers.
        """
        L = self.params['arm_length']
        pos1 = pos + np.dot(rot, np.array([L, 0, 0]))
        pos2 = pos + np.dot(rot, np.array([0, L, 0]))
        pos3 = pos + np.dot(rot, np.array([-L, 0, 0]))
        pos4 = pos + np.dot(rot, np.array([0, -L, 0]))
        return np.array([pos1, pos2, pos3, pos4])
    
    def get_motion(self, t, state, trajhandle):
        """
        Solving quadrotor equation of motion.
        This function takes in time, state vector, controller, trajectory generator,
        and parameters and outputs the derivative of the state vector.
        The actual calculation is done in quad_EOM_readonly.

        Args:
        t (float): Time
        state (numpy.array): State vector [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
        controlhandle (function): Function handle of your controller
        trajhandle (function): Function handle of your trajectory generator
        params (dict): Parameters (output from sys_params() and any additional parameters)

        Returns:
        numpy.array: Derivative of state vector s
        """
        current_state = utils.state_to_qd(state)
        desired_state = trajhandle(t)
        thrust, M = self.controlhandle(t, current_state, desired_state, self.params)
        sdot = self._compute_sdot(state, thrust, M)
        return sdot

    def _compute_sdot(self, state, thrust, M):
        """
        Solve quadrotor equation of motion.
        This function calculates the derivative of the state vector.

        Args:
        state (numpy.array): State vector [x, y, z, xd, yd, zd, qw, qx, qy, qz, wx, wy, wz]
        thrust (float): Thrust output from controller
        M (numpy.array): Moments output from controller
        params (dict): Parameters

        Returns:
        numpy.array: Derivative of state vector s
        """
        thrust, M = self._clamp(thrust, M)

        # Assign states
        _, _, _, xdot, ydot, zdot = state[:6]
        qW, qX, qY, qZ, wX, wY, wZ = state[6:]

        quat = np.array([qW, qX, qY, qZ])
        bRw = utils.quat_to_rot(quat) # Rotation matrix from world to body
        wRb = np.transpose(bRw) # Rotation matrix from body to world

        # Acceleration
        accel = self._compute_accel(thrust, wRb)

        # Angular velocity 
        qdot = self._compute_qdot(wX, wY, wZ, quat)

        # Angular acceleration
        omegadot = self._compute_omegadot(M, wX, wY, wZ)

        # Assemble sdot
        sdot = np.zeros(13)
        sdot[0:3] = [xdot, ydot, zdot]
        sdot[3:6] = accel
        sdot[6:10] = qdot
        sdot[10:13] = omegadot

        return sdot

    def _clamp(self, thrust, M):
        """
        Clamps the thrust and moments of the quadrotor within the specified limits.

        Parameters:
        thrust (float): The total thrust of the quadrotor.
        M (numpy.ndarray): The moments of the quadrotor as a 1D numpy array.

        Returns:
        tuple: A tuple containing the clamped thrust and moments of the quadrotor.
        """
        A = np.array([[0.25, 0, -0.5 / self.params['arm_length']],
                  [0.25, 0.5 / self.params['arm_length'], 0],
                  [0.25, 0, 0.5 / self.params['arm_length']],
                  [0.25, -0.5 / self.params['arm_length'], 0]])
        
        thrust_and_moments = np.array([thrust, M[0], M[1]])
        prop_thrusts = np.dot(A, thrust_and_moments)
        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, self.params['maxF'] / 4), self.params['minF'] / 4)

        B = np.array([[1, 1, 1, 1],
                  [0, self.params['arm_length'], 0, -self.params['arm_length']],
                  [-self.params['arm_length'], 0, self.params['arm_length'], 0]])

        thrust = np.dot(B[0, :], prop_thrusts_clamped)
        M = np.concatenate([np.dot(B[1:3, :], prop_thrusts_clamped), np.array([M[2]])]).flatten()
        return thrust, M

    def _compute_accel(self, thrust, wRb):
        """
        Compute the acceleration of the quadrotor.

        Parameters:
        thrust (float): The thrust force applied to the quadrotor.
        wRb (numpy.ndarray): The rotation matrix representing the orientation of the quadrotor.

        Returns:
        numpy.ndarray: The acceleration of the quadrotor.
        """
        gravity_force = np.array([0, 0, self.params['mass'] * self.params['gravity']])
        thrust_force = np.dot(wRb, np.array([0, 0, thrust]))
        accel = 1 / self.params['mass'] * (thrust_force - gravity_force)
        return accel

    def _compute_qdot(self, wX, wY, wZ, quat):
        """
        Compute the derivative of the quaternion.

        Parameters:
        wX (float): Angular velocity around the x-axis.
        wY (float): Angular velocity around the y-axis.
        wZ (float): Angular velocity around the z-axis.
        quat (numpy.ndarray): Quaternion representing the orientation of the quadrotor.

        Returns:
        numpy.ndarray: The derivative of the quaternion.
        """
        K_quat = 2  # this enforces the magnitude 1 constraint for the quaternion
        quat_error = 1 - np.sum(quat ** 2)
        qdot_mat = -0.5 * np.array([[0, -wX, -wY, -wZ],
                                    [wX, 0, -wZ, wY],
                                    [wY, wZ, 0, -wX],
                                    [wZ, -wY, wX, 0]])
        qdot = np.dot(qdot_mat, quat) + K_quat * quat_error * quat
        return qdot

    def _compute_omegadot(self, M, wX, wY, wZ):
        """
        Compute the angular acceleration (omegadot) of the quadrotor.

        Parameters:
        - M: The total torque applied to the quadrotor.
        - wX: The roll rate of the quadrotor.
        - wY: The pitch rate of the quadrotor.
        - wZ: The yaw rate of the quadrotor.

        Returns:
        - omegadot: The angular acceleration of the quadrotor.
        """
        omega = np.array([wX, wY, wZ])
        angular_momentum = np.dot(self.params['I'], omega)
        gyroscopic_torque = np.cross(omega, angular_momentum)
        omegadot = np.dot(self.params['invI'], (M - gyroscopic_torque))
        return omegadot