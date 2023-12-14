import numpy as np

def traj_line(t, tstep, T=30, x_max=2, y_max=1, z_max=8):
    # Define the maximum time of the trajectory
    t0 = 0
    tf = 5
    
    if t >= tf:
        # Hover at the end of the trajectory
        x = x_max
        y = y_max
        z = z_max
        pos = np.array([x, y, z])
        vel = np.zeros(3)
        acc = np.zeros(3)
    else:
        y = [1]
        out_t, out_t_1, out_t_2 = _quintic_poly(t, t0, tstep, tf, y)
        out_t = np.squeeze(out_t)
        out_t_1 = np.squeeze(out_t_1)
        out_t_2 = np.squeeze(out_t_2)

        # Position
        x = out_t * x_max
        y = out_t * y_max
        z = out_t * z_max
        pos = np.array([x, y, z])

        x_1 = out_t_1 * x_max
        y_1 = out_t_1 * y_max
        z_1 = out_t_1 * z_max

        x_2 = out_t_2 * x_max
        y_2 = out_t_2 * y_max
        z_2 = out_t_2 * z_max

        # Calculating velocity in each dimension
        xd = (x-x_1) / tstep
        yd = (y-y_1) / tstep
        zd = (z-z_1) / tstep
        vel = np.array([xd, yd, zd])
        xd_1 = (x_1-x_2) / tstep
        yd_1 = (y_1-y_2) / tstep
        zd_1 = (z_1-z_2) / tstep

        # Calculating acceleration in each dimension
        xdd = (xd-xd_1) / tstep
        ydd = (yd-yd_1) / tstep
        zdd = (zd-zd_1) / tstep
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

def traj_helix(t, tstep, T=30, r=5, z_max=2.5):
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
        y = [2*np.pi, z_max]
        out_t, out_t_1, out_t_2 = _quintic_poly(t, t0, tstep, tf, y)

        # Angular position, velocity, acceleration around the helix
        beta, z = out_t
        beta_1, z_1 = out_t_1
        beta_2, z_2 = out_t_2

        # Calculating 3D coordinates based on the helix's radius
        x = np.cos(beta) * r
        y = np.sin(beta) * r
        pos = np.array([x, y, z])

        x_1 = np.cos(beta_1) * r
        y_1 = np.sin(beta_1) * r

        x_2 = np.cos(beta_2) * r
        y_2 = np.sin(beta_2) * r

        # Calculating velocity in each dimension
        xd = (x-x_1) / tstep
        yd = (y-y_1) / tstep
        zd = (z-z_1) / tstep
        vel = np.array([xd, yd, zd])

        xd_1 = (x_1-x_2) / tstep
        yd_1 = (y_1-y_2) / tstep
        zd_1 = (z_1-z_2) / tstep

        # Calculating acceleration in each dimension
        xdd = (xd-xd_1) / tstep
        ydd = (yd-yd_1) / tstep
        zdd = (zd-zd_1) / tstep
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

def traj_circle(t, tstep, radius=8, T=30, height=1.0, ascend_fraction=0.05, rest_fraction=0.05):
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
    ascend_rest_time = ascend_time + rest_time
    circle_time = T - ascend_rest_time

    # Phase 1: Ascending
    if t < ascend_time:
        # Ascend linearly to the specified height
        z = (height / ascend_time) * t
        vz = height / ascend_time
        az = 0
        x, y, vx, vy, ax, ay = radius, 0, 0, 0, 0, 0

        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        acc = np.array([ax, ay, az])

    # Phase 2: Resting at Height
    elif t < ascend_time + rest_time:
        # Remain stationary at the specified height
        x, y, z = radius, 0, height
        vx, vy, vz = 0, 0, 0
        ax, ay, az = 0, 0, 0

        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        acc = np.array([ax, ay, az])

    # Phase 3: Circular Motion
    else:
        # Normalize time for the circular motion phase
        t_circular = t - ascend_rest_time
        t_circular = t_circular % circle_time
        t = t_circular

        t0 = 0
        tf = circle_time

        y = [2*np.pi]
        out_t, out_t_1, out_t_2 = _quintic_poly(t, t0, tstep, tf, y)
        out_t = np.squeeze(out_t)
        out_t_1 = np.squeeze(out_t_1)
        out_t_2 = np.squeeze(out_t_2)

        # Calculating 3D coordinates based on the helix's radius
        x = np.cos(out_t) * radius
        y = np.sin(out_t) * radius
        z = z_1 = z_2 = height
        pos = np.array([x, y, z])

        x_1 = np.cos(out_t_1) * radius
        y_1 = np.sin(out_t_1) * radius

        x_2 = np.cos(out_t_2) * radius
        y_2 = np.sin(out_t_2) * radius

        # Calculating velocity in each dimension
        xd = (x-x_1) / tstep
        yd = (y-y_1) / tstep
        zd = (z-z_1) / tstep
        vel = np.array([xd, yd, zd])

        xd_1 = (x_1-x_2) / tstep
        yd_1 = (y_1-y_2) / tstep
        zd_1 = (z_1-z_2) / tstep

        # Calculating acceleration in each dimension
        xdd = (xd-xd_1) / tstep
        ydd = (yd-yd_1) / tstep
        zdd = (zd-zd_1) / tstep
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

def _quintic_poly(t, t0, dt, tf, y):
    """
    Calculates the quintic polynomial trajectory at a given time `t` between initial time `t0` and final time `tf`.

    Parameters:
        t (float): The current time.
        t0 (float): The initial time.
        tf (float): The final time.
        y (float): The desired position at the final time.

    Returns:
        tuple: A tuple containing the position at time `t`, time `t-dt`, and time `t-2*dt`.
    """

    # Matrix for quintic polynomial coefficients calculation
    M = np.array([
        [1, t0, t0**2, t0**3, t0**4, t0**5],
        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
    ])

    Y = np.zeros_like(y)

    # Boundary conditions for the trajectory
    b = np.array([Y, Y, Y, y, Y, Y])

    # Solving for the coefficients of the quintic polynomial
    a = np.linalg.solve(M, b)

    out_t = np.dot(a.T, [1, t, t**2, t**3, t**4, t**5])
    out_t_1 = np.dot(a.T, [1, (t-dt), (t-dt)**2, (t-dt)**3, (t-dt)**4, (t-dt)**5])
    out_t_2 = np.dot(a.T, [1, (t-2*dt), (t-2*dt)**2, (t-2*dt)**3, (t-2*dt)**4, (t-2*dt)**5])

    return out_t, out_t_1, out_t_2

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
