import numpy as np

def get_params():
    """
    Define basic parameters for the quadrotor.

    Returns:
    params -- Dictionary of system parameters.
    """

    params = {}
    params['mass'] = 0.4  # kg
    params['gravity'] = 9.81  # m/s^2
    params['I'] = np.array([[0.00012, 0, 0],
                            [0, 0.00011, 0],
                            [0, 0, 0.00018]])
    params['invI'] = np.linalg.inv(params['I'])
    params['arm_length'] = 0.8  # m
    params['minF'] = 0.0
    params['maxF'] = 8.0 * params['mass'] * params['gravity']
    return params
