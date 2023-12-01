import numpy as np

def traj_line(t, T=30):
    # Define the maximum time of the trajectory
    t_max = 4
    # Clamp the time input to be within [0, t_max] and normalize it
    t = max(0, min(t, t_max)) / t_max

    # Calculate the position, velocity, and acceleration using polynomial time-scaling
    pos = 10 * t**3 - 15 * t**4 + 6 * t**5
    vel = (30 / t_max) * t**2 - (60 / t_max) * t**3 + (30 / t_max) * t**4
    acc = (60 / t_max**2) * t - (180 / t_max**2) * t**2 + (120 / t_max**2) * t**3

    # Scale the position, velocity, and acceleration by 4
    pos *= 3
    vel *= 3
    acc *= 3

    # Prepare the desired state output
    desired_state = {
        'pos': [pos, pos, pos],  # Same position for x, y, and z
        'vel': [vel, vel, vel],  # Same velocity for x, y, and z
        'acc': [acc, acc, acc],  # Same acceleration for x, y, and z
        'yaw': pos,              # Yaw set to the position value (uncommon choice)
        'yawdot': vel            # Yaw rate set to the velocity value
    }

    return desired_state

def traj_helix(t, T=30, r=5, z_max=2.5):
    """
    This function calculates a helical trajectory in 3D space.

    Parameters:
    t (float): Current time.
    T (float): Total duration of the trajectory.
    r (float): Radius of the helix in the xy-plane.
    z_max (float): Maximum height of the helix.

    Returns:
    dict: A dictionary containing position, velocity, acceleration, yaw, and yaw rate.
    """

    # Time settings for the start and end of the trajectory
    t0 = 0  # Start time
    tf = 18  # End time

    if t >= T:
        # Hover at the end of the trajectory
        x = np.cos(2 * np.pi) * r
        y = np.sin(2 * np.pi) * r
        z = z_max
        pos = np.array([x, y, z])
        vel = np.zeros(3)
        acc = np.zeros(3)
    else:
        # Matrix for quintic polynomial coefficients calculation
        M = np.array([
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
        ])

        # Boundary conditions for the trajectory
        b = np.array([[0, 0], [0, 0], [0, 0], [2*np.pi, z_max], [0, 0], [0, 0]])

        # Solving for the coefficients of the quintic polynomial
        a = np.linalg.solve(M, b)

        # Calculating position, velocity, and acceleration at time t
        out = np.dot(a.T, [1, t, t**2, t**3, t**4, t**5])
        outd = np.dot(a.T, [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        outdd = np.dot(a.T, [0, 0, 2, 6*t, 12*t**2, 20*t**3])

        # Angular position, velocity, acceleration around the helix
        beta, z = out
        betad, zd = outd
        betadd, zdd = outdd

        # Calculating 3D coordinates based on the helix's radius
        x = np.cos(beta) * r
        y = np.sin(beta) * r
        pos = np.array([x, y, z])

        # Calculating velocity in each dimension
        xd = -y * betad
        yd = x * betad
        vel = np.array([xd, yd, zd])

        # Calculating acceleration in each dimension
        xdd = -x * betad**2 - y * betadd
        ydd = -y * betad**2 + x * betadd
        acc = np.array([xdd, ydd, zdd])

    # Yaw and yaw rate (assumed to be zero)
    yaw = 0
    yawdot = 0

    # Returning the desired state as a dictionary
    desired_state = {
        'pos': pos, 
        'vel': vel, 
        'acc': acc, 
        'yaw': yaw, 
        'yawdot': yawdot
    }

    return desired_state

def traj_circle(t, radius=8, T=30, height=1.0, ascend_fraction=0.05, rest_fraction=0.05):
    """
    Generates a trajectory that first ascends to a specified height and then follows a circular path in the xy plane.

    Parameters:
    t (float): Current time
    radius (float): Radius of the circle
    T (float): Total time for one complete trajectory (ascending + circle)
    height (float): The height to reach before starting the circle
    ascend_fraction (float): Fraction of total time spent ascending
    rest_fraction (float): Fraction of total time spent resting at height

    Returns:
    dict: Desired state with position, velocity, acceleration, yaw, and yaw rate
    """

    # Calculate durations for each phase of the trajectory
    ascend_time = T * ascend_fraction
    rest_time = T * rest_fraction
    circle_time = T - ascend_time - rest_time

    # Phase 1: Ascending
    if t < ascend_time:
        # Ascend linearly to the specified height
        z = (height / ascend_time) * t
        vz = height / ascend_time
        az = 0
        x, y, vx, vy, ax, ay = radius, 0, 0, 0, 0, 0

    # Phase 2: Resting at Height
    elif t < ascend_time + rest_time:
        # Remain stationary at the specified height
        x, y, z = radius, 0, height
        vx, vy, vz = 0, 0, 0
        ax, ay, az = 0, 0, 0

    # Phase 3: Circular Motion
    else:
        # Normalize time for the circular motion phase
        t_circular = (t - ascend_time - rest_time) % circle_time

        # Calculate the angle for the circular motion
        angle = (2 * np.pi / circle_time) * t_circular

        # Position in the XY plane at the specified height
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        # Velocity in the circular motion
        vx = -radius * np.sin(angle) * (2 * np.pi / circle_time)
        vy = radius * np.cos(angle) * (2 * np.pi / circle_time)
        vz = 0

        # Acceleration in the circular motion
        ax = -radius * np.cos(angle) * (2 * np.pi / circle_time)**2
        ay = -radius * np.sin(angle) * (2 * np.pi / circle_time)**2
        az = 0

    # Desired state output
    desired_state = {
        'pos': np.array([x, y, z]),   # Position
        'vel': np.array([vx, vy, vz]), # Velocity
        'acc': np.array([ax, ay, az]), # Acceleration
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state

if __name__ == "__main__":
    # Example usage:
    # Get the desired state at time t = 0 to 2 seconds at intervals of 0.1 seconds
    pos = []
    for t in np.arange(0, 18, 0.05):
        # desired_state = traj_line(t)
        # desired_state = traj_helix(t)
        desired_state = traj_circle(t)
        pos.append(desired_state['pos'])
    pos = np.array(pos)

    # Plot the x, y, and z positions in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each row as a point in the 3D space
    for point in pos:
        ax.scatter(point[0], point[1], point[2])

    # Setting labels for the axes
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()
