from math import sin, cos

def pd_controller(t, state, des_state, params):
    """
    Calculates the control inputs (thrust and moments) for a quadcopter based on the desired state and current state.

    Args:
        t (float): Current time.
        state (dict): Dictionary containing the current state of the quadcopter, including position, velocity, orientation, and angular velocity.
        des_state (dict): Dictionary containing the desired state of the quadcopter, including position, velocity, acceleration, yaw, and yaw rate.
        params (dict): Dictionary containing the parameters of the quadcopter, including mass, gravity, and control gains.

    Returns:
        thrust (float): Thrust required to achieve the desired z acceleration.
        M (list): List of moments (torques) required for desired orientation: roll (phi), pitch (theta), and yaw (psi).
    """
    # Control gains for position (x, y, z) and orientation (phi, theta, psi)
    kpx, kdx = 16, 2.5
    kpy, kdy = 16, 2.5
    kpz, kdz = 20, 4.5
    kpphi, kdphi = 50, 2.0
    kptheta, kdtheta = 60, 2.2
    kppsi, kdpsi = 1.5, 0

    # Desired accelerations (feedforward terms) with PD control (feedback terms) for x, y, and z
    acc_x = des_state['acc'][0] + kdx * (des_state['vel'][0] - state['vel'][0]) + kpx * (des_state['pos'][0] - state['pos'][0])
    acc_y = des_state['acc'][1] + kdy * (des_state['vel'][1] - state['vel'][1]) + kpy * (des_state['pos'][1] - state['pos'][1])
    acc_z = des_state['acc'][2] + kdz * (des_state['vel'][2] - state['vel'][2]) + kpz * (des_state['pos'][2] - state['pos'][2])

    # Thrust required to achieve the desired z acceleration
    thrust = params['mass'] * (params['gravity'] + acc_z)

    # Desired roll (phi) and pitch (theta) angles based on desired x, y accelerations and yaw
    # Transforms desired accelerations into body frame (tilt angles)
    phi_c = (acc_x * sin(des_state['yaw']) - acc_y * cos(des_state['yaw'])) / params['gravity']
    theta_c = (acc_x * cos(des_state['yaw']) + acc_y * sin(des_state['yaw'])) / params['gravity']

    # Moment (torque) required for desired orientation: roll (phi), pitch (theta), and yaw (psi)
    # PD control for orientation
    M = [
        kpphi * (phi_c - state['rot'][0]) - kdphi * state['omega'][0],
        kptheta * (theta_c - state['rot'][1]) - kdtheta * state['omega'][1],
        kppsi * (des_state['yaw'] - state['rot'][2]) + kdpsi * (des_state['yawdot'] - state['omega'][2])
    ]

    return thrust, M

if __name__ == "__main__":
    # Example of usage:
    # Define the current state and desired state as dictionaries.
    # Each state should have 'pos', 'vel', 'acc', 'rot', 'omega' (and 'yaw', 'yawdot' for the desired state).
    # Define 'params' as a dictionary with keys 'mass' and 'gravity'.

    # Example state (current and desired) and params
    current_state = {'pos': [0, 0, 0], 'vel': [0, 0, 0], 'acc': [0, 0, 0], 'rot': [0, 0, 0], 'omega': [0, 0, 0]}
    desired_state = {'pos': [1, 1, 1], 'vel': [0, 0, 0], 'acc': [0, 0, 0], 'yaw': 0, 'yawdot': 0}
    vehicle_params = {'mass': 1.0, 'gravity': 9.81}

    # Compute thrust and moments
    thrust, moments = pd_controller(0, current_state, desired_state, vehicle_params)
    print("Thrust:", thrust)
    print("Moments:", moments)