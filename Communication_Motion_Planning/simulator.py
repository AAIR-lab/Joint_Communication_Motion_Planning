from pathlib import Path
import pysocialforce as psf
import math
import numpy as np
import toml


def human_motion_model(human_goal, human_pos, robot_pos, robot_v, robot_theta,
                       belief, obs, plan_horizon, dt, config_filename):
    """ Simulates human motion based on the Social Forces model.

    Inputs:
        - human_goal(np.ndarray): The x,y goal coordinates of the human
        - human_pos(np.ndarray):  The current human position in x,y coord.
        - robot_pos(np.ndarray):  The current robot position in x,y coord.
        - robot_v(float):         The robot's current linear velocity
        - robot_theta(float):     The robot's current heading angle
        - belief(list):           Human belief about robot's next moves. Each element is a list [x,y,r]
                                  where x,y is the location that the human thinks the robot will be in the next cycle.
        - obs(list):              Linear static obstacles (e.g. walls)
        - plan_horizon(float):    Planning horizon in seconds
        - dt(float):              Sample time [s]
        - config_filename(string): The configuration filename
    Returns:
        - human_trajectory(np.ndarray): The (x,y) waypoints of the trajectory.
    """

    # Initial velocity factor
    vel_factor = 0.8

    # Number of steps for human simulation
    num_steps = int(plan_horizon/dt)

    # Human parameters
    human_gx, human_gy = human_goal                                 # Goal
    human_px, human_py = human_pos                                  # Initial position
    h_theta = math.atan2(human_gy - human_py, human_gx - human_px)  # Angle from init position to goal
    human_vx, human_vy = math.cos(h_theta), math.sin(h_theta)       # Initial velocity assumed

    # Robot's current location
    robot_x_curr, robot_y_curr = robot_pos

    if belief:
        # Human thinks that the robot might be in one of the 'belief' locations
        # in the next planning cycle, so create multiple virtual robots that
        # start from robot's current location and assign each robot one of these
        # locations as a goal.

        belief = np.array(belief)  # convert to np.array

        # Number of virtual robots
        num_virtual_robots = len(belief)

        # Social groups
        groups = [[x] for x in range(num_virtual_robots + 1)]

        # Robot parameters (modeled as human from the human perspective)
        robot_px = [robot_x_curr - 0.01 * i for i in range(num_virtual_robots)]            # Initial positions
        robot_vx, robot_vy = robot_v * np.cos(robot_theta), robot_v * np.sin(robot_theta)  # Initial velocities
        robot_gx, robot_gy = belief[:, 0], belief[:, 1]                                    # Goals

        # Initial states, each entry is the position, velocity and goal of a
        # pedestrian in the form of (px, py, vx, vy, gx, gy)
        human_init_state = [human_px, human_py, vel_factor * human_vx, vel_factor * human_vy, human_gx, human_gy]
        robot_init_states = []
        for i in range(num_virtual_robots):
            robot_state = [robot_px[i], robot_y_curr,
                           robot_vx, robot_vy,
                           robot_gx[i], robot_gy[i]]
            robot_init_states.append(robot_state)

        initial_state = np.array([human_init_state,
                                  *robot_init_states])

    else:
        # Human has no belief about robot's next location/goal.
        # Robot's goal is computed as a linear projection from its current position.

        # Social groups
        groups = [[0], [1]]

        # Robot parameters (modeled as human from the human perspective)
        robot_px, robot_py = robot_x_curr, robot_y_curr                                    # Initial position
        robot_vx, robot_vy = robot_v * np.cos(robot_theta), robot_v * np.sin(robot_theta)  # Initial velocities
        robot_gx, robot_gy = robot_vx * plan_horizon, robot_vy * plan_horizon              # Goal

        # Initial states, each entry is the position, velocity and goal of a
        # pedestrian in the form of (px, py, vx, vy, gx, gy)
        human_init_state = [human_px, human_py, vel_factor * human_vx, vel_factor * human_vy, human_gx, human_gy]
        robot_init_state = [robot_px, robot_py, robot_vx, robot_vy, robot_gx, robot_gy]
        initial_state = np.array([human_init_state,
                                  robot_init_state])

    # Initiate the simulator
    s = psf.Simulator(
        initial_state,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath(config_filename))

    # update x steps (e.g. update 30 steps for a 3s planning horizon if dt=0.1s)
    s.step(num_steps)

    # Pedestrian positions
    num_waypoints = s.get_states()[0].shape[0]
    x_human = s.get_states()[0][:, 0, 0].reshape((num_waypoints, 1))  # np.ndarray of size (steps+1,1)
    y_human = s.get_states()[0][:, 0, 1].reshape((num_waypoints, 1))

    # Human trajectory  (each element is array of [x, y])
    human_trajectory = np.concatenate((x_human, y_human), axis=1)

    return human_trajectory


def set_config_params(config_filename, params):
    """Sets the SF parameters in the configuration file according to each scenario.

    Inputs:
      - config_filename(string): The configuration filename
      - params(list):            The parameters for the current scenario

    Post-condition:
      - The configuration file holds the updated parameters
    """
    # Load a dictionary from the TOML configuration file
    data = toml.load(config_filename)

    # Change parameter values based on current scenario
    data["desired_force"]["factor"] = params[0]
    data["social_force"]["factor"] = params[1]
    data["obstacle_force"]["factor"] = params[2]
    data["obstacle_force"]["sigma"] = params[3]

    # Write updated dictionary to TOML configuration file
    with open(config_filename, 'w') as f:
        toml.dump(data, f)

    return
