"""

CBF-BELIEF-based RRT

"""
import matplotlib.pyplot as plt
from human import Human
from rrt import RRT
from simulator import human_motion_model, set_config_params
import numpy as np
from env import ENV
from HighLevel import HighPlanner, Visual
import copy
from measure import Measurements
import time



show_animation = False
real_time_report = False

test_weights =[
    [0.25,1.5],
    [0.5, 1.5],
    [0.75,1.5],
    [1,1.5],
    [1.25, 1.5],
    [1.5, 1.5],
    [1.5, 1.25],
    [1.5, 1],
    [1.5, 0.75],
    [1.5, 0.5],
    [1.5, 0.25],
    [1.5, 0.01]
    ]
w_ratio = []
for w in test_weights:
    w_ratio.append(w[0]/w[1])
w_index = 10
#___________________________________Hyper Parameters____________________________________________________
high_level_mode = 'off'                # 'on' or 'off'
experiment_mode = 'table'
# global parameters
human_radius = 0.2
robot_radius = 0.3
grid_res = 0.3
env_map = 'map3'
if experiment_mode == 'weights': dir = 'weights' + '\\' + env_map
else: dir = env_map 
priority = 'R'
if priority == 'R': cost_weights = [1.5, 0.25, 3, 0.1, 1]
else: cost_weights = [0.25, 1.5, 1.5, 0.1, 0]
h_pers_size = 5
  # weights = [robot_cost_to goal, human_cost_to_goal, path_conflict_cost, communication cost, heading cost]
h_avg_vel = 1                       # human average velocity


if experiment_mode == 'weights':
    cost_weights = [test_weights[w_index][0], test_weights[w_index][1], 2, 0.1, 0]

## rrt parameters
rrt_goal_shift_radii = 2
rrt_state_sample_var = 1
if high_level_mode == 'on':
    rrt_max_iter = 200
    rrt_num_segments = 10
    rrt_state_sample_mode = 'explore_oriented'   # 'exploit_oriented' and 'explore_oriented'
    planning_horizon = 3
    rrt_cost_weights = [1, 1.5, 1, 1]     # [w_goal, w_human, w_heading, w_trap]
else:
    rrt_max_iter = 100
    rrt_num_segments = 5
    rrt_state_sample_mode = 'explore_oriented'
    planning_horizon = 2
    rrt_cost_weights = [1, 1.5, 1, 1]     # [w_goal, w_human, w_heading, w_trap]


# map-related paremeters
if env_map == 'map1':
    h_start, h_goal = [0.0, 2.0], [-5, 0.0]
    h_obsrv_traj = np.array([[0, 2.2], h_start])
    r_start, r_goal = [0.0, -2.0, 1.57], [[0.0, 0.0, 0.0], [-3, 0.0, 0.0]]
elif env_map == 'map2':
    h_start, h_goal  = [-0.1, -4.0], [-0.1, 3.0] 
    h_obsrv_traj = np.array([[-0.1,-4.3],[-0.1,-4.2],[-0.1, -4.1], h_start])
    r_start, r_goal = [-3.0, -0.0, 0.0], [[3.0, -1, 0.0], [3.0, -1, 0.0]]
elif env_map == 'map3':
    h_start, h_goal  = [1.0, 0.0], [-6.25, 0.0] 
    h_obsrv_traj = np.array([[1.3,0],[1.2,0],[1.1,0], h_start])
    r_start, r_goal = [-5.25, 0.0, 0.0], [[5.5, 0.0, 0.0], [5.5, 0.0, 0.0]]
elif env_map == 'map4':
    h_start, h_goal  = [1.0, 7.0], [8.0, 7.0]
    h_obsrv_traj = np.array([[1.3, 7.0], [1.2, 7.0], [1.1, 7.0], h_start])
    r_start, r_goal = [9.0, 7.0, 3.14], [[1, 7.0, 0.0], [1, 7.0, 0.0]]
#___________________________________________________________________________________________________________

sf_configuration_filename = "config_params.toml"

def main():
    for ii in np.arange(10):
        file_name = env_map + '_' + high_level_mode + '_' + priority  + '_' + str(ii+1)
        world = ENV (map_name = env_map, 
                res = grid_res, 
                p_horizon = planning_horizon, 
                human_start = h_start, 
                human_goal = h_goal, 
                human_traj = h_obsrv_traj, 
                robot_start =  r_start, 
                robot_goal = r_goal, 
                r_radius = robot_radius,
                goal_shift_radii = rrt_goal_shift_radii)
        walls = world._walls
        #test_point = [3,0]
        #ans = world.is_occupied(test_point)
        #locc = world.where(test_point)
        #world._maze[locc[1]][locc[1]]
        env_bounds = world._env_bounds
        print("start " + __file__)
        # planning parameters
        dt = world._dt                                          # sample time [s] 
        measure = Measurements(human_radius = human_radius, r_goal = r_goal [-1][0:2], h_goal = h_goal)
        # Enviroment Bounds
        env_bounds = world._env_bounds
        # robot parameters
        goal_set = world._goal_set                              # [x, y, theta]
        start = world._start                                    # [x, y, theta]
        robot = world._robot_model
        rrt = RRT(env_bounds = env_bounds,
                    robot_model = robot, 
                    controller_model = world._controller, 
                    planning_horizon = planning_horizon,
                    n_segments = rrt_num_segments,
                    max_iter = rrt_max_iter,
                    cost_weights = rrt_cost_weights,
                    state_sample_var = rrt_state_sample_var,
                    state_sample_mode = rrt_state_sample_mode,
                    occupancy_grid = world._maze, 
                    grid_res = world._res, 
                    multiple_plan = high_level_mode,
                    grid_y_offset = world._y_offset,
                    grid_x_offset = world._x_offset)              # initialize an rrt object

        # human parameters
        human_goal = world._human_goal
        human_obsv_traj = world._human_obsv_traj                # human observed trajectory
        human_obs_belief = []
        human = Human(dt = dt, 
                        ob_belief = human_obs_belief, 
                        predict_horizon = planning_horizon, 
                        obs_trajectory = human_obsv_traj, 
                        goal = human_goal, 
                        world = world, 
                        h_radius = human_radius)  #initialize a human object
    
        # high-level planner parameters
        # weights = [robot_cost_to goal, human_cost_to_goal, path_conflict_cost, communication cost]
        hp = HighPlanner(r_radius= robot_radius, 
                            h_goal = human_goal, 
                            weights = cost_weights, 
                            environment = world, 
                            pers_size = h_pers_size, 
                            avg_velocity = h_avg_vel, 
                            robot_goal = r_goal, 
                            h_radius = human_radius,
                            rrt_ph = planning_horizon)
    
        hp.update_logistics (r_goal = goal_set, e_walls = walls, r_start = np.copy(start))
        optimal_distances = hp.give_distance_norms (r_start[0:2], h_start)
        y_min, y_max = world._vis_bound_y_min - 2, world._vis_bound_y_max + 2
        x_min, x_max = world._vis_bound_x_min - 2, world._vis_bound_x_max + 2
    

        graphic = Visual(walls= walls,
                            x_lim = [x_min, x_max], 
                            y_lim = [y_min, y_max], 
                            human = human, 
                            robot = robot, 
                            rrt = rrt,
                            animation_mode = show_animation)

        # --- Human Motion Model Initializations ----------------------------------
        # Load and set parameters for the current scenario
        sf_params = world._sf_params
        set_config_params(sf_configuration_filename, sf_params)

        # Transform walls into:
        # List of linear obstacles given in the form (x_min, x_max, y_min, y_max)
        obstacles = []
        for wall in walls:
            x_min, y_min = wall[0]
            x_max, y_max = wall[1]
            obstacles.append([x_min, x_max, y_min, y_max])

        human_position = human_obsv_traj[-1]  # Initial position of human
        # -------------------------------------------------------------------------

        human_waypoints = [] 
        robot_waypoints = []
        robot_says = ''
        planning_cycles = 0
        human_time_flag = True
        # iterative motion planning
        while not robot.goal_reached_bool:
            planning_cycles = planning_cycles + 1
            if planning_cycles == 3:
                print ('pointer')
            # change the goal of robot if it reaches the first goal
            #if np.linalg.norm(robot.x[0:2]-goal_set[-2][0:2]) <=  goal_shift_radii:
            #    robot.x_goal = goal_set[-1]


            ##############
            # Trajectory Prediction 
            _ , human_pred_traj = human.dwa_predict()   # [the cost of predicted trajectory, human predicted trajectory]
            dyn_obs_list = [human_pred_traj]            # each element is array of [x, y , r]
        
            ##############
            #Tree Expansion
            best_nodes = rrt.planning(dyn_obs_list = dyn_obs_list)
            rrt._node_list_log.append(rrt._node_list)
            library = []
            for node in best_nodes:
                #node_cost_2_goal = np.linalg.norm(node.x[0:2]-robot.x_goal[0:2])
                #node_cost_2_goal = 0
                _, node_path = rrt.generate_final_course(node)
                #for i in range(node_path.shape[0]):
                #    node_cost_2_goal += np.linalg.norm(node_path[i][0:2]-robot.x_goal[0:2])
                #node_cost_2_goal = node_cost_2_goal/node_path.shape[0]
                library.append([node.x[0], node.x[1], node.cost_heading, node_path]) 
        
            ##############
            # high-level planner
            # input: 1. selected nodes, 2. tree, 
            if high_level_mode == 'on':
                human_obj_copy = copy.deepcopy(human)
                human_obj_copy.dt = 0.1 #<<<<<<<<<<<<<<<--<<<< check later
                hp_out = hp.give_plan(library, robot.x[0:2], human.human_loc, human.human_dir*180/np.pi, human_obj_copy) # [select_node_loc, x_curr_robot, list of belief based obstacles, select_node_indx]
            
                # need to show after render <<<<--<<<<
                communication_action = hp_out[1]
                human.ob_belief = []
                selected_node_indx = hp_out[3]
                print('-----------------------------')
                print('high-level report:')
                print("nominated nodes + costs:")
                for node in best_nodes:
                    print("node:" + str((node.x[0], node.x[1], node.x[2])) + ", cost:" + str(node.cost)+ ", cost_heading:" + str(node.cost_heading))
                print("selected node: " + str(hp_out[0]))
                print("communication action " + communication_action)
                print("belief " + str(human.ob_belief))
                print('-----------------------------')
                message = "communication action: " + communication_action
                robot_says = communication_action
            else:
                selected_node_indx = 0
                message = "simple CBF-TB-RRT"
        
            ##############
            # move method to move robot and human both
            selected_node = best_nodes[selected_node_indx]
            control, path = rrt.generate_final_course(selected_node)

            # human motion can be changed
            # _ , human_pred_traj = human.dwa_predict()   # predicted trajectory for t = [0, planning_horizon]
            # dyn_obs_list = [human_pred_traj] # each element is array of [x, y , r]
            graphic.render(message, planning = True)

            if high_level_mode == 'on':
                ob_belief = hp_out[2]
                #ob_belief.append([robot.x_goal[0], robot.x_goal[1], robot.r ])
            else: 
                ob_belief = []
                #ob_belief.append([robot.x_goal[0], robot.x_goal[1], robot.r ])


            if high_level_mode == 'on':
                replanning_horizon = control.shape[0]
            else:
                replanning_horizon = 1
            if high_level_mode == 'on':
                # give an extra cycle to human if robot does not move
                if np.linalg.norm(robot.x[0:2]-selected_node.x[0:2]) <0.001:
                    for _ in range(int(planning_horizon/world._dt)):
                        # --- Human Motion Model ----------------------------------
                        dyn_obs_list = human_motion_model(human_goal=human_goal,
                                                            human_pos=human_position,
                                                            robot_pos=robot.x[0:2],
                                                            robot_v=0.0,
                                                            robot_theta=robot.x[2],
                                                            belief=ob_belief,
                                                            obs=obstacles,
                                                            plan_horizon=dt,
                                                            dt=dt,
                                                            config_filename=sf_configuration_filename)
                        human_position = dyn_obs_list[1]
                        human.human_update(human_position)
                        human_waypoints.append(human.human_loc)
                        robot_waypoints.append([robot.x[0],robot.x[1],robot_says])
                        if high_level_mode == 'on':
                            graphic.render( message, motion = True, maze = hp._capture)
                        else: 
                            graphic.render( message, motion = True)
                        if real_time_report:
                                print("robot location: " + str(robot.x))
                                print("robot control action: " + str(control[i]))
                                print("human location: " + str(human_position))
                                print("----------------------------")
                        robot_says = ''
                        # ---------------------------------------------------------

            for i in range(replanning_horizon):
                # --- Human Motion Model ------------------------------------------
                dyn_obs_list = human_motion_model(human_goal=human_goal,
                                                    human_pos=human_position,
                                                    robot_pos=robot.x[0:2],
                                                    robot_v=control[i, 0],
                                                    robot_theta=robot.x[2],
                                                    belief=ob_belief,
                                                    obs=obstacles,
                                                    plan_horizon=dt,
                                                    dt=dt,
                                                    config_filename=sf_configuration_filename)
                # -----------------------------------------------------------------
                human_position = dyn_obs_list[1]
                human.human_update(human_position)
                if real_time_report:
                    print("robot location: " + str(robot.x))
                    print("robot control action: " + str(control[i]))
                    print("human location: " + str(human_position))
                    print("----------------------------")
                robot.move(control[i])  # robot moving

                if high_level_mode == 'on':
                    hp.update_geometry(human.human_loc, human.human_dir*180/np.pi)
                message ='\n'.join((
                                    r'$x=%.2f$' % (robot.x[0], ),
                                    r'$y=%.2f$' % (robot.x[1], ),
                                    r'$\theta=%.2f$' % (robot.x[2], ),
                                    r'$v=%.2f$' % (control[i][0], ),
                                    r'$\omega=%.2f$' % (control[i][1], )))
                human_waypoints.append(human.human_loc)
                robot_waypoints.append([robot.x[0],robot.x[1],robot_says])
                if high_level_mode == 'on':
                    graphic.render( message, motion = True, maze = hp._capture)
                else: 
                    graphic.render( message, motion = True)

                # if the robot reaches the goal, wait for the human to arrive
                if robot.goal_reached_bool:
                    while np.linalg.norm(human.human_loc[0:2]-h_goal[0:2]) > 0.5:
                        # --- Human Motion Model ----------------------------------
                        dyn_obs_list = human_motion_model(human_goal=human_goal,
                                                            human_pos=human_position,
                                                            robot_pos=robot.x[0:2],
                                                            robot_v=0,
                                                            robot_theta=robot.x[2],
                                                            belief=[],
                                                            obs=obstacles,
                                                            plan_horizon=dt,
                                                            dt=dt,
                                                            config_filename=sf_configuration_filename)
                        human_position = dyn_obs_list[1]
                        human.human_update(human_position)
                        robot_waypoints.append([robot.x[0],robot.x[1],robot_says])
                        human_waypoints.append(human.human_loc)
                        if high_level_mode == 'on':
                            graphic.render( message, motion = True, maze = hp._capture)
                        else: 
                            graphic.render( message, motion = True)
                        if real_time_report:
                            print("robot location: " + str(robot.x))
                            print("robot control action: " + str(control[i]))
                            print("human location: " + str(human_position))
                            print("----------------------------")
                        # ---------------------------------------------------------
                    break
                robot_says = ''
            plt.close()

        #hp.log_scenario()
        wall_info = [walls, [x_min, x_max], [y_min, y_max]]
        waypoints = [robot_waypoints,human_waypoints]
        if high_level_mode == 'on': grid_log = hp._grid_log
        else: grid_log = [[]]

        measure.log_grid_scenario(grid_log, file_name, wall_info, waypoints, planning_cycles, high_level_mode, priority, 
                                    dir, rrt._node_list_log, optimal_distances)

if __name__ == '__main__':
    main()

