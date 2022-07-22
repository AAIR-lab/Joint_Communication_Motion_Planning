"""

Human Predictor Model with Dynamic Window Approach

author: Keyvan Majd

"""
import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = True

def move_average_filter(data, window = None):
    X = np.convolve(data[:,0], np.array([0.1,0.8,0.1]), 'same')
    Y = np.convolve(data[:,1], np.array([0.1,0.8,0.1]), 'same')
    return np.transpose(np.array([X,Y]))

class Human:
    """""
    Human Class
    """""

    def __init__(self, dt, ob_belief,  predict_horizon, obs_trajectory, goal, world, h_radius = 0.2):
        # human location
        self.human_traj = obs_trajectory       #records human trajectory
        self.human_loc = obs_trajectory[-1]
        self.human_dir = []
        self.human_vel = []
        self.predicted_traj = []
        self.ob_belief = []
        self.goal = goal
        self.predict_horizon = predict_horizon
        self.max_yaw_rate = 90.0 * math.pi / 180.0  # [rad/s]
        self.max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss] <<<----remove?
        self.yaw_rate_resolution = 1 * math.pi / 180.0  # [rad/s]
        self.dt = dt  # [s] Time tick for motion prediction
        self.window_horizon = 3  # [s]
        self.to_goal_cost_gain = 2.0
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 3.0
        self.human_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.human_type = 'circle'    #changed <-----

        # if human_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.human_radius = h_radius  # [m] for collision check

        # Enviroment constraints static
        self.world_data = world
        self.human_update()

    

    def human_update(self, new_loc = np.array([])):
        if new_loc.size == 0:
            track_filt = np.array(self.human_traj) # write a filtering module<<<<-<<<<<<
            self.human_vel = np.linalg.norm([(track_filt[-2:,0][1]-track_filt[-2:,0][0])/self.dt,(track_filt[-2:,1][1]-track_filt[-2:,1][0])/self.dt])
            self.human_dir = np.arctan2(track_filt[-2:,1][1]-track_filt[-2:,1][0],track_filt[-2:,0][1]-track_filt[-2:,0][0])
        else: 
            self.human_traj = np.vstack((self.human_traj, new_loc))
            track_filt = np.array(self.human_traj) # write a filtering module<<<<-<<<<<<
            self.human_loc = new_loc
            self.human_vel = np.linalg.norm([(track_filt[-2:,0][1]-track_filt[-2:,0][0])/self.dt,(track_filt[-2:,1][1]-track_filt[-2:,1][0])/self.dt])
            self.human_dir = np.arctan2(track_filt[-2:,1][1]-track_filt[-2:,1][0],track_filt[-2:,0][1]-track_filt[-2:,0][0])


    def dwa_predict(self):

        """
        Dynamic Window Approach Prediction
        This mehtod simulate the human trajectory for the given prediction horizon
        """

        self.human_update()
        track_filt = np.array(self.human_traj) # write a filtering module <<<<-<<<<<<

        # initial state [x(m), y(m), yaw(rad), v(m/s), yaw_rat(rad/s)]
        x = np.array([track_filt[-1][0], track_filt[-1][1], self.human_dir, self.human_vel, 0.0]) # human last state
        predicted_trajectory = np.array([x])
        time = 0
        while time <= self.predict_horizon:
            dw = self.calc_dynamic_window(x)

            _ , trajectory = self.calc_trajectory(x, dw)

            if trajectory.shape[0] > 1:
                predicted_trajectory = np.vstack((predicted_trajectory, trajectory[1]))
                x = trajectory[1].copy()
            else: 
                predicted_trajectory = np.vstack((predicted_trajectory, trajectory[0]))
                x = trajectory[0].copy()

            time += self.dt
        
        self.predicted_traj = predicted_trajectory[:,0:2]
        output_trajectory_and_radius = np.append(predicted_trajectory[:,0:2], self.human_radius*np.ones((predicted_trajectory[:,0:2].shape[0],1)), axis = 1)
        trajectory_cost = self.calc_trajectory_cost(output_trajectory_and_radius)
        
        return trajectory_cost, output_trajectory_and_radius


    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [-self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[4] - self.max_delta_yaw_rate * self.dt,
            x[4] + self.max_delta_yaw_rate * self.dt]

        #  [yaw_rate_min, yaw_rate_max]
        #dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1])]
        dw = [min(Vs[0], Vd[0]), max(Vs[1], Vd[1])]

        return dw


    def calc_trajectory(self, x, dw):
        """
        calculation final predicted trajectory with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        # asumming human is moving with constant speed
        v = x[3]
        for w in np.arange(dw[0], dw[1], self.yaw_rate_resolution):

            trajectory = self.predict_trajectory(x_init, v, w)

            #trajectory cost
            trajectory_cost = self.calc_trajectory_cost(trajectory)

            # search minimum trajectory
            if min_cost >= trajectory_cost:
                min_cost = trajectory_cost
                best_u = [v, w]
                best_trajectory = trajectory
                if abs(best_u[0]) < self.human_stuck_flag_cons \
                        and abs(x[3]) < self.human_stuck_flag_cons:  #check this <<<----<<<
                    # to ensure the human do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory

    def calc_trajectory_cost(self, trajectory):
        to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory)
        #speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
        ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory)

        final_cost = to_goal_cost + ob_cost

        return final_cost

    def calc_obstacle_cost(self, trajectory):  #<<<<<---<<<<
        """
        calc obstacle cost inf: collision
        """

        #########
        # obstacle region for belief output and robot location    #<<<<<---<<<<
        # should be added
        # better way to call static obstacle constraints (env bounds)
        #########



        #ox = ob[:, 0]
        #oy = ob[:, 1]
        #dx = trajectory[:, 0]
        if self.ob_belief:
            ob_belief = np.array(self.ob_belief)
            ox = ob_belief[:, 0]
            oy = ob_belief[:, 1]
            dx = trajectory[:, 0] - ox[:, None]
            dy = trajectory[:, 1] - oy[:, None]
            r_belief = np.hypot(dx, dy)
            for i in range(r_belief.shape[0]):
                r_belief[i] = r_belief[i] - ob_belief[i][2]


            #dy_up = self.env_bounds.y_max - trajectory[:, 1]
            #dy_down = trajectory[:, 1] - self.env_bounds.y_min
            #r_wall = np.array([dy_up, dy_down])

            #r = np.vstack((r_belief,r_wall))
            r = r_belief

            #if np.array(r <= self.human_radius).any():
            if np.array(r <= 0).any():
                return float("Inf")

            min_r = np.min(r)
            return 1.0 / min_r  # OK

        else:
            return 0


    def calc_to_goal_cost(self, trajectory):
        """
            calc to goal cost with angle difference
        """

        dx = self.goal[0] - trajectory[trajectory.shape[0]-1, 0]
        dy = self.goal[1] - trajectory[trajectory.shape[0]-1, 1]
        error_angle = math.atan2(dy, dx)
        #if error_angle < 0:
            #error_angle += 2*np.pi   # to keep theta between 0 and 2*pi
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def predict_trajectory(self, x_init, v, w):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array([x])
        time = 0
        while time <= self.window_horizon:            
            x = self.motion(x, [v, w], self.dt)

            if self.world_data.is_occupied(x[0:2]):
                break
                
            dist_to_goal = math.hypot(x[0] - self.goal[0], x[1] - self.goal[1])  #stop the trajectory if it reaches the goal
            if dist_to_goal <= 0.5:
                break

            trajectory = np.vstack((trajectory, x))
            time += self.dt
        
        return trajectory

    @staticmethod
    def motion(x, u, dt):
        """
        motion model
        """

        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, human):  # pragma: no cover
    circle = plt.Circle((x, y), human.human_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * human.human_radius)
    plt.plot([x, out_x], [y, out_y], "-g")
