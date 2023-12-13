import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from system.quadrotor import Quadrotor
import system.utils as utils

class QuadPlot:
    """
    A class for plotting quadcopter trajectories in 3D space.

    Attributes:
        quad (Quadrotor): The quadcopter object.
        fig (Figure): The figure object for the plot.
        ax (Axes3D): The 3D axes object for the plot.
        state_history (list): A list to store the history of quadcopter states.
        des_state_history (list): A list to store the history of desired quadcopter states.
        params (dict): A dictionary of parameters for the quadcopter.
    """

    def __init__(self, quad: Quadrotor, params):
        """
        Initializes the QuadPlot object.

        Parameters:
            quad (Quadrotor): The quadcopter object.
            params (dict): A dictionary of parameters for the quadcopter.
        """
        self.quad = quad
        self.params = params
        self.state_history = []
        self.des_state_history = []
        self.imgs = []
        self._initialize_plot()

    def _initialize_plot(self):
        """Initializes the plot with default settings."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_plot_lines()
        self.set_limit((-7, 7), (-7, 7), (-0.5, 8))
        self._label_axes()

    def _setup_plot_lines(self):
        """Sets up the initial plot lines for visualization."""
        # Plot lines for quadcopter structure
        self.ax.plot([], [], [], '-', color='cyan', markersize=1)
        self.ax.plot([], [], [], '-', color='green', markersize=1)
        # Plot lines for trajectory
        self.ax.plot([], [], [], '.', color='red', markersize=0.5, label='Desired Path')
        self.ax.plot([], [], [], '.', color='blue', markersize=0.5, label='Quadcopter Path')
        self.ax.legend()

    def _label_axes(self):
        """Labels the axes of the plot."""
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')

    def set_limit(self, x, y, z):
        """
        Sets the limits of the plot.

        Parameters:
            x (tuple): The x-axis limits.
            y (tuple): The y-axis limits.
            z (tuple): The z-axis limits.
        """
        self.ax.set_xlim(x)
        self.ax.set_ylim(y)
        self.ax.set_zlim(z)

    def update(self, t, state, des_state):
        """
        Updates the plot with new quadcopter states.

        Parameters:
            t (float): The current time.
            state (dict): The current quadcopter state.
            des_state (dict): The desired quadcopter state.
        """
        self._update_state_history(state, des_state)
        self._update_trajectory_plots()
        self._update_quad_structure(state)
        self.ax.set_title('t = {:.2f}'.format(t))
        self.fig.canvas.draw_idle()
        self._save_frame(t)

    def _update_state_history(self, state, des_state):
        """Updates the history of states for the plot."""
        self.state_history.append(state['pos'])
        self.des_state_history.append(des_state['pos'])

    def _update_trajectory_plots(self):
        """Updates the trajectory plots."""
        state_history_np = np.array(self.state_history)
        des_state_history_np = np.array(self.des_state_history)
        lines = self.ax.get_lines()
        lines[3].set_data(state_history_np[:, 0], state_history_np[:, 1])
        lines[3].set_3d_properties(state_history_np[:, 2])
        lines[2].set_data(des_state_history_np[:, 0], des_state_history_np[:, 1])
        lines[2].set_3d_properties(des_state_history_np[:, 2])

    def _update_quad_structure(self, state):
        """Updates the structure of the quadcopter in the plot."""
        state['rot'] *= -1 # Invert rotation for plotting
        rot = utils.rpy_to_rot_zxy(*state['rot'])
        propeller_positions = self.quad.get_prop_pos(state['pos'], rot)
        lines = self.ax.get_lines()
        for i in range(2):
            lines[i].set_data([propeller_positions[i][0], propeller_positions[i + 2][0]],
                              [propeller_positions[i][1], propeller_positions[i + 2][1]])
            lines[i].set_3d_properties([propeller_positions[i][2], propeller_positions[i + 2][2]])

    def _save_frame(self, t):
        """Saves the current frame for animation."""
        if not os.path.exists('./img'):
            os.makedirs('./img')
        plt.savefig(f'./img/img_{t}.png', transparent = False,  facecolor = 'white')
        self.imgs.append(f'./img/img_{t}.png')

    def save_animation(self, filename):
        """
        Saves the animation as a GIF.

        Parameters:
            filename (str): The name of the file to save the animation as.
        """
        frames = []
        for img in self.imgs:
            frames.append(imageio.v2.imread(img))
        imageio.mimsave(filename, frames, 'GIF', duration=0.01)

    def pause(self, t):
        """
        Pauses the plot for a specified duration.

        Parameters:
            t (float): The duration to pause the plot.
        """
        plt.pause(t)