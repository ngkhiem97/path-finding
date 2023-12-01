import numpy as np
from scipy.integrate import solve_ivp
from system.params import get_params
from system.trajectories import traj_line, traj_helix, traj_circle
from system.controller import pd_controller
from system.plot import QuadPlot
import system.utils as utils
from system.quadrotor import Quadrotor

def terminate_check(x, time, stop, pos_tol, vel_tol, time_tol):
    """
    Check termination criteria, including position, velocity, and time.

    Parameters:
    x -- Current state vector [position; velocity].
    time -- Current simulation time.
    stop -- Target stop position.
    pos_tol -- Position tolerance.
    vel_tol -- Velocity tolerance.
    time_tol -- Maximum time for the simulation.

    Returns:
    terminate_cond -- Termination condition code (0, 1, or 2).
    """

    # Check position and velocity
    pos_check = np.linalg.norm(x[0:3] - stop) < pos_tol
    vel_check = np.linalg.norm(x[3:6]) < vel_tol

    # Check total simulation time
    time_check = time > time_tol

    if pos_check and vel_check:
        terminate_cond = 1  # Robot reaches goal and stops, successful
    elif time_check:
        terminate_cond = 2  # Robot doesn't reach goal within given time, not complete
    else:
        terminate_cond = 0

    return terminate_cond

# Assuming trajhandle, controlhandle, and other necessary functions are defined
def simulation_3d(trajhandle, controlhandle, max_time=30, real_time=False):
    # Parameters for simulation
    params = get_params()  # Define sys_params() or replace with actual parameters

    # Initialize quadrotor
    quad = Quadrotor(controlhandle, params)

    # Initialize plot
    plot = QuadPlot(quad, params)

    # Initial conditions
    tstep = 0.01  # Time step for solution
    cstep = 0.05  # Image capture time interval
    max_iter = int(max_time / cstep)

    des_start = trajhandle(0)
    des_stop = trajhandle(max_time)
    stop_pos = des_stop['pos']
    x0 = quad.init_state(des_start['pos'], 0)
    x = x0

    pos_tol = 0.01
    vel_tol = 0.01

    # Simulation loop
    for iter in range(max_iter):
        timeint = np.arange(iter * cstep, (iter + 1) * cstep, tstep)
        
        # Run simulation
        sol = solve_ivp(lambda t, y: quad.get_motion(t, y, trajhandle), [timeint[0], timeint[-1]], x, t_eval=timeint)
        
        # Update state
        x = sol.y[:, -1]

        # Update quad plot
        current_state = utils.state_to_qd(x)
        desired_state = trajhandle(timeint[-1])
        plot.update(timeint[-1], current_state, desired_state)

        # Check termination criteria
        if terminate_check(x, timeint[-1], stop_pos, pos_tol, vel_tol, max_time):
            break

        # Real-time delay
        if real_time:
            plot.pause(cstep)
        else:
            plot.pause(0.001)

    print('Simulation complete.')
    plot.save_animation(f'animation_{trajhandle.__name__}.gif')

# Example usage
# Prompt user for trajectory type
traj_type = input('Enter trajectory type (line, helix, circle): ')

# Run simulation
if traj_type == 'line':
    simulation_3d(traj_line, pd_controller, real_time=True)
elif traj_type == 'helix':
    simulation_3d(traj_helix, pd_controller, real_time=True)
elif traj_type == 'circle':
    simulation_3d(traj_circle, pd_controller, real_time=True)
else:
    print('Invalid trajectory type.')
