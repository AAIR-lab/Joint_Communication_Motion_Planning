#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import math
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import random
import numpy as np
import time
from matplotlib import interactive
from human import Human
import pickle as pk
import copy
from datetime import datetime
from copy import copy, deepcopy
from grid import Grid, AStarPlanner
from env import ENV
import random

class HighPlanner:
  
    
    def __init__(self, loc = (25,25), rotation = 0, r_radius = 0.5, fov_c = "#ffa538", line_w = 0.5,
                face_c = '#d56c11', outline_c = "b", body_s = 60, h_goal = [-7,1], show_stats = True, weights = [0.35, 0.35, 0.2, 0.1, 0.1]
                , environment = ENV(), avg_velocity = 1, pers_size = 5, robot_goal = None, h_radius = 0.2, rrt_ph = 4):
        self._location = loc
        self._rotation = rotation
        self._robot_radius = r_radius
        self._body_color = face_c
        self._boldy_out_color = outline_c
        self._body_size = body_s
        self._signals = ['north', 'south', 'east', 'west', 'silence']
        self._belief = []
        self._tree = []
        self._scenarios = []
        self._robot_loc = None
        self._library = None
        self._robot_goal = robot_goal
        self._walls = None
        self._robot_start = None
        self._rrt_ph = rrt_ph
        self._grid = Grid(env = environment, pers = pers_size, rrt_ph = self._rrt_ph)
        self._perseption = []
        self._human_goal = h_goal
        self._human_radius = h_radius
        self._capture = None
        self._grid_log = []
        self._show_stats = show_stats
        self._weights = weights
        self._avg_velocity = avg_velocity
        self._delta_t = 0.1
        self._slow_down_thresh = 0.1
        



    def update_loc (self, newX, newY):
        self._location = (newX, newY)
    
    def update_rotation (self, newAngle):
        self._rotation = newAngle
       
    
    def update_geometry(self, new_loc, new_direction):
        self.update_loc (new_loc[0], new_loc[1])
    

    def goes_east(self, start, end):
        if (start[0] < end[0]): return True
        else: return False

    def goes_west (self, start, end):
        if (start[0] > end[0]): return True
        else: return False

    def goes_south(self, start, end):
        if (start[1] > end[1]): return True
        else: return False

    def goes_north (self, start, end):
        if (start[1] < end[1]): return True
        else: return False

    def get_dir (self,start, end):
        value = None
        if (start[0] < end[0]): value = 'f'
        elif (start[0] > end[0]): value = 'b' 
        elif (start[0] == end[0]): value = 's'
        return value


    def which_dir (self, start, end):
        direction = None
        flag = True
        dist_n, dist_s, dist_e, dist_w = 0, 0, 0, 0
        if (self.goes_east (start,end)): 
            dist_e = end[0] - start[0]
            flag = False
        if (self.goes_west(start,end)): 
            dist_w = start[0] - end[0]
            flag = False
        if (self.goes_south(start,end)): 
            dist_s =  start[1] - end[1]
            flag = False
        if (self.goes_north(start,end)): 
            dist_n = end[1] - start[1]
            flag = False
        if (flag): return ['stop','none']
        else:
            values = [dist_n, dist_s, dist_e, dist_w]
            directions = ['north','south', 'east', 'west']
            index_max1 = np.argmax(values)
            dir1 = directions [index_max1]
            values[index_max1] = - math.inf
            index_max2 = np.argmax(values)
            if (values[index_max2] > 0 ):
                dir2 = directions [index_max2]
            else: dir2 = 'none'

            return [dir1, dir2]

    def sensor_model (self, robot_loc, plan, signal, mode):
        start = robot_loc
        end = plan

        omega = mode
        
        if (signal == 'east'):
            direction = self.which_dir(start,end)
            if (direction[0] == 'east' or direction[1] == 'east'): omega = direction[0] + '/' + direction[1]
            elif (direction[0] == 'stop'): omega = signal
            #else: omega = signal [ random.randint(0, len (self._signals)-2) ]

        elif (signal == 'west'):
            direction = self.which_dir(start,end)
            if (direction[0] == 'west' or direction[1] == 'west'): omega = direction[0] + '/' + direction[1] 
            elif (direction[0] == 'stop'): omega = signal
            #else: omega = signal [ random.randint(0, len (self._signals)-2) ]

        elif (signal == 'south'):
            direction = self.which_dir(start,end)
            if (direction[0] == 'south' or direction[1] == 'south'): omega = direction[0] + '/' + direction[1] 
            elif (direction[0] == 'stop'): omega = signal
            #else: omega = signal [ random.randint(0, len (self._signals)-2) ]

        elif (signal == 'north'):
            direction = self.which_dir(start,end)
            if (direction[0] == 'north' or direction[1] == 'north'): omega = direction[0] + '/' + direction[1]
            elif (direction[0] == 'stop'): omega = signal
            #else: omega = signal [ random.randint(0, len (self._signals)-2) ]

        return omega
    
    def sep_command (self, string):
        res = []
        temp = ''
        for i in string:
            if (i != '/'): temp = temp + i
            else:
                res.append(temp)
                temp = ''
        res.append (temp)
        return res


    def test_obs (self, omega, test_omega):
        flag = False
        omega_sep = self.sep_command(omega)
        omega_test_sep = self.sep_command(test_omega)
        for om in omega_sep:
            for om_test in omega_test_sep:
                if om == om_test: flag = True
        return flag

    def belief_update (self, robot_loc, plan, signal, previous_belief):
        belief = []
        robot_loc = self._grid.where(robot_loc)
        plan = self._grid.where(plan)
        
        omega = self.sensor_model (robot_loc, plan, signal, "primary")
        for section in self._perseption:
                for b in previous_belief:
                    test_omega = self.sensor_model (b, section, signal, "secondary" )
                    if ( self.test_obs(omega,test_omega)): belief.append(section)   
        return belief
    
    def log_scenario (self):
        a = self._scenarios[0][20]
        now = datetime.now()
        filename = now.strftime('%Y_%m_%d(%H_%M_%S).pickle')
        with open(filename, 'wb') as f:
            pk.dump(self._scenarios,f)
    
    def log_grid_scenario (self, file_name,walls_info, waypoints, planning_cycles):
        metrics = self.get_metrics(waypoints, planning_cycles)
        signal_log = self.log_communication(waypoints[0])
        log = [self._grid_log, walls_info, waypoints, metrics, signal_log]
        with open(file_name, 'wb') as f:
            pk.dump(log,f)

    def log_communication (self, r_waypoints):
        log = []
        for point in r_waypoints:
            if (point[2] != ''): log.append(point)
        return log

    def update_logistics (self, r_goal = None, e_walls = None, r_start = None ):
        self._robot_goal = r_goal
        self._walls = e_walls
        self._robot_start = r_start
    
    def update_human_pers (self, human_loc):

        self._perseption = []
        pers = self._grid.get_pers (human_loc)
        aa = self._grid.where(self._location)
        for region in pers:
            if (region != self._grid.where(self._location)):  
                self._perseption.append (region)

    def modify_human_cost (self):
        waiting_cost = 0
        non_inf_costs = []
        for node in self._tree:
            if node.cost_human != math.inf: 
                non_inf_costs.append(node.cost_human)
        
        if (len(non_inf_costs) > 0 ): waiting_cost = 1.2 * max(non_inf_costs)
        else: waiting_cost = math.inf
        for node in self._tree:
            if node.cost_human == math.inf: 
                node.cost_human = waiting_cost
                node.cost_total = node.cost_robot + node.cost_human + node.cost_path + node.cost_com
        flag = False
        point1 = np.array(self._location)
        point2 = np.array(self._human_goal)
        dist = np.linalg.norm(point1 - point2)
        if dist < 0.5: flag = True
        if flag:
            for node in self._tree:
                node.cost_path = 0
                node.cost_total = node.cost_robot + node.cost_human + node.cost_path + node.cost_com

    def give_plan(self, library, robot_loc, human_loc, human_dir, human):
        self.update_geometry(human_loc, human_dir)
        self._tree = []
        #self._perseption = self._grid.get_pers (human_loc)
        bb = self.update_human_pers(human_loc)
        index = 0
        previous_belief = [self._grid.where(robot_loc)]
        start = deepcopy(robot_loc)
        self._library = library
        for plan in library:
            end = plan [0:2]
            for signal in self._signals:
                belief = self.belief_update (start, end, signal, previous_belief)
                robot_tp_path_in_grid = self._grid.path_in_grid(plan[3])
                maze, path, cost_to_goal_human, conflict_cost, human_synch_path = self._grid.give_path (human_loc, self._human_goal, 
                                                belief, False, start, plan, robot_tp_path_in_grid, self._delta_t, self._avg_velocity)
                c_human = self.inf_multiplied(cost_to_goal_human , self._weights[1])
                c_robot = self.inf_multiplied(self.robot_cost(robot_tp_path_in_grid), self._weights[0])
                p_cost = self.inf_multiplied(conflict_cost, self._weights[2])
                com_cost = self.give_com_cost (signal)
                heading_cost = plan[2]

                new_node = Node(r_loc_c = robot_loc, r_loc_g = self._grid.where(robot_loc), r_plan_c = end, r_plan_g = self._grid.where(end), 
                                signal = signal, belief_g = belief, cost_r = c_robot, cost_h = c_human, 
                                cost_traj = p_cost, signal_cost = self._weights[3]*com_cost, cost_head = heading_cost*self._weights[4],
                                human_loc_c = human_loc, human_loc_g = self._grid.where(human_loc), n_human_dir = human_dir, 
                                n_index = index, human_path = path, n_maze = maze, h_synch_path = human_synch_path, 
                                r_synch_path = robot_tp_path_in_grid)

                self._tree.append (new_node)
            index = index + 1
        self.modify_human_cost()
        min_cost = math.inf
        min_index = 0
        self._grid_log.append(self._tree)
        for i in range (len(self._tree)):
            if self._tree[i].cost_total < min_cost:
                min_cost = self._tree[i].cost_total
                min_index = i
        size_tree = len(self._tree)
        self._tree[min_index].chosen = True
        belief = self._tree[min_index].belief_g
        self._capture = self._tree[min_index].maze
        #self._grid.show_it(  self._tree[min_index].maze  )
        if self._show_stats == True:
            print ('______________________________ new tree______________________________')
            print ()
            for j in range (len(self._tree)):
                self._tree[j].print_node()
            print ('______________________________ end of tree______________________________')
            print ()
        self._robot_loc = deepcopy( robot_loc)
        if (belief != None): belief = self.belief_cont(belief) 
        return [self._tree[min_index].plan_con, self._tree[min_index].signal, belief, self._tree[min_index].index]

    def inf_multiplied (self, a, b):
        if (a == 0 and b == math.inf) or (b == 0 and a == math.inf): return 0
        else: return a * b


    def belief_cont (self, belief):
        cont = []
        for b in belief:
            cont.append( self._grid.reverse_where(b) )
        return cont

    def give_com_cost (self, signal):
        if signal == 'silence': return 0
        else: return 1
    
    def robot_cost (self, partial_path):
        g = self._grid.give_path_cost (partial_path)
        grid_size = 1.0 
        robot_radius = 0
        
        a_star = AStarPlanner(self._grid._ox_clear, self._grid._oy_clear, grid_size, robot_radius)
        sx, sy = partial_path[-1][0], partial_path[-1][1]
        robot_goal = self._robot_goal[-1]
        g_in_grid = self._grid.where(robot_goal)
        gx, gy = g_in_grid[0], g_in_grid[1]
        rx, ry, flag = a_star.planning(sx, sy, gx, gy, False)
        h = 0
        if flag == False: 
            h = math.inf
        else: _,h = self._grid.path_info (rx, ry)
        #coef = np.exp(-6/h)
        g = self._grid.give_path_cost(partial_path)
        return ( h)

    def get_metrics (self, waypoints, planning_cycles):
        r_way = waypoints[0]
        h_way = waypoints[1]
        r_actual_dist = self.dist_waypoint(r_way)
        h_actual_dist = self.dist_waypoint(h_way)
        #_______________________________


        return [r_actual_dist, h_actual_dist, planning_cycles, human_time]


    def dist_waypoint (self, waypoints):
        path = self._grid.path_in_grid_no_time(waypoints)
        distance = self._grid.give_path_cost (path)
        return distance

    def get_shortest_dist (self, start, end):
        distance = 0
        grid_size = 1.0 
        robot_radius = 0
        a_star = AStarPlanner(self._grid._ox_clear, self._grid._oy_clear, grid_size, robot_radius)
        start = self._grid.where(start)
        sx, sy = start[0], start[1]
        end = self._grid.where(end)
        gx, gy = end[0], end[1]
        rx, ry, flag = a_star.planning(sx, sy, gx, gy, False)
        if flag == False: 
            distance = math.inf
        else: _,distance = self._grid.path_info (rx, ry)
        return distance

    def give_distance_norms (self, r_start, h_start):
        h = 0
        r = 0
        grid_size = 1.0 
        robot_radius = 0
        a_star = AStarPlanner(self._grid._ox_clear, self._grid._oy_clear, grid_size, robot_radius)
        start = self._grid.where(r_start)
        sx, sy = start[0], start[1]
        end = self._grid.where(self._robot_goal[-1][0:2])
        gx, gy = end[0], end[1]
        rx, ry, flag = a_star.planning(sx, sy, gx, gy, False)
        if flag == False: 
            r = math.inf
        else: _,r = self._grid.path_info (rx, ry)

        start = self._grid.where(h_start)
        sx, sy = start[0], start[1]
        end = self._grid.where(self._human_goal)
        gx, gy = end[0], end[1]
        rx, ry, flag = a_star.planning(sx, sy, gx, gy, False)
        if flag == False: 
            h = math.inf
        else: _,h = self._grid.path_info (rx, ry)

        return [r, h]



class Node:
    def __init__(self, r_loc_c, r_loc_g, r_plan_c, r_plan_g, signal, belief_g, cost_r, cost_h, 
                 cost_traj, signal_cost, cost_head ,human_loc_c, human_loc_g, n_human_dir, n_index, 
                 human_path, n_maze, h_synch_path, r_synch_path):
        
        self.robot_loc_con = r_loc_c
        self.robot_loc_grid = r_loc_g
        self.plan_con = r_plan_c
        self.plan_grid = r_plan_g
        self.signal = signal
        self.belief_g = belief_g 
        self.cost_robot = cost_r
        self.cost_human = cost_h
        self.cost_path = cost_traj
        self.cost_com = signal_cost
        self.cost_heading = cost_head
        self.cost_total = cost_r + cost_h + cost_traj + signal_cost + cost_head
        self.human_loc_con = human_loc_c
        self.human_loc_grid = human_loc_g
        self.human_dir = n_human_dir
        self.index = n_index
        
        self.human_path = human_path
        self.human_synch_path = h_synch_path
        self.robot_synch_path = r_synch_path
        
        self.maze = n_maze
        self.chosen = False

        
        
    
    def print_node(self):
        print ()
        print ('robot g Loc: ', self.robot_loc_grid, ' robot c loc: ', self.robot_loc_con)
        print ('plan g Loc: ', self.plan_grid, ' plan c Loc: ', self.plan_con)
        print ('Signal: ', self.signal)
        print('belief: ', self.belief_g)
        print('Human g Loc: ', self.human_loc_grid, ' human c loc', self.human_loc_con,' Human dir: ', self.human_dir) 
        print ('cost human: ', self.cost_human, ' cost robot: ', self.cost_robot, ' conflict cost: ', self.cost_path, ' cost signal: ', 
               self.cost_com, ' total cost: ', self.cost_total, ' heading cost: ' ,self.cost_heading)
        print ('Human trajectory: ', self.human_path)
        print ()

class Visual:
    def __init__ (self, walls, x_lim = [-7.1,7.1], y_lim = (-2,6.7), back_c = '#5d5d5d', mon_size = (14, 14), face_c = '#d56c11', body_s = 60, rrt = None, human = None, robot = None, animation_mode = True):
        self._walls = walls
        self._xlim = x_lim
        self._ylim = y_lim
        self._back_color = back_c
        self._monitor_size = mon_size
        self._human_body_color = face_c
        self._human_body_size = body_s
        self._human = human
        self._rrt = rrt
        self._robot = robot
        self._animation_mode = animation_mode
        self._fig = plt.figure(figsize=self._monitor_size)
        self._ax = plt.gca()
                

    
    def vis_human_body (self, ax):
        human = self._human
        ax.scatter(human.human_loc[0], human.human_loc[1], s=self._human_body_size, alpha =1, color = self._human_body_color)
        human_radii = plt.Circle((human.human_loc[0], human.human_loc[1]), human.human_radius, color=self._human_body_color, fill=False)
        ax.add_patch(human_radii)
    
    def vis_human_pred_traj(self, ax):
        human = self._human
        ax.scatter(human.predicted_traj[:,0], human.predicted_traj[:,1], s=self._human_body_size, alpha =1, color = self._human_body_color)

    def vis_human_belief(self,ax):
        human = self._human
        for belief in human.ob_belief:
            ax.scatter(belief[0], belief[1], s=20, alpha =1, color = '#ff0000')
            belief_radii = plt.Circle((belief[0], belief[1]), belief[2], color='#ff0000', fill=False)
            ax.add_patch(belief_radii)

    def vis_human (self, ax):
        self.vis_human_body(ax)
        self.vis_human_pred_traj(ax)
        self.vis_human_belief(ax)

    def vis_rrt_tree(self, ax):
        rrt = self._rrt
        for node in rrt._node_list:
            if node.parent:
                x0, x1, x2, x3 = zip(*node.path_x)
                plt.plot(x0, x1, linewidth = 1, color = '#82a8ff')

    def vis_rrt_best_nodes(self, ax):
        rrt = self._rrt
        for node in rrt._best_nodes_list:
            plt.plot(node.x[0], node.x[1], color= '#031cff', marker = "X")

    def vis_rrt(self, ax):
        self.vis_rrt_tree(ax)        
        self.vis_rrt_best_nodes(ax)
    
    def vis_robot(self, ax, length=0.6, width=0.3):
        robot = self._robot
        ax.scatter(robot.x[0], robot.x[1], s=20, alpha =1, color = '#031cff')
        robot_radii = plt.Circle((robot.x[0], robot.x[1]), robot.r, color='#031cff', fill=False)
        robot_arrow = plt.arrow(robot.x[0], robot.x[1], length * math.cos(robot.x[2]), length * math.sin(robot.x[2]),
              head_length=width, head_width=width, color='#ff0000')
        ax.add_patch(robot_radii)
        ax.add_patch(robot_arrow)
        ax.scatter(robot.x_goal[0], robot.x_goal[1], s=50, marker = "*", alpha =1, color = '#000000')

    def vis_walls(self, ax):
        for wall in self._walls:
            plt.plot(wall[0:2,0],wall[0:2,1], "k")

    def vis_best_path(self, ax):
        final_path = self._rrt._best_path
        plt.plot(final_path[:,0], final_path[:,1], color='#031cff', linewidth = 1)

    
    
    def render(self, message, planning = False, motion = False, maze = []):
        if self._animation_mode:
            if planning:
                #plt.subplot(1, 2, 1)
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                ax = plt.gca()
                ax.cla()
                self.vis_human (ax)
                self.vis_rrt(ax)
                self.vis_robot(ax)
                self.vis_walls(ax)
                plt.xlim(self._xlim[0], self._xlim[1])
                plt.ylim(self._ylim[0], self._ylim[1])

                ax.text(-6.9, 6.3, message , fontsize=14, bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.5})
                # Set general font size
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(16)
                plt.xlabel("X [m]", fontsize=16)
                plt.ylabel("Y [m]", fontsize=16)
                fig = plt.gcf()
                fig.set_size_inches(4, 4)

                #fig2, ax = plt.subplot(1, 2, 2)
                #cmap = colors.ListedColormap(['white', 'black','blue', 'green', 'red', 'gray'])
                #bounds = [-0.5,0.5,1.5,2.5,3.5,4.5, 5.5]
                #norm = colors.BoundaryNorm(bounds, cmap.N)
                #plt.figure(figsize=(16, 12), dpi=80)

                #ax.imshow(maze, cmap=cmap, norm=norm)

                # draw gridlines
                #ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)
                #ax.set_xticks(np.arange(0, self._x_len+2,2))
                #ax.set_yticks(np.arange(0, self._y_len+2,2))
                #ax.invert_yaxis()
                #timer = fig.canvas.new_timer(interval = 10000) #creating a timer object and setting an interval of 10000 milliseconds
                #timer.add_callback(close_event)
                #timer.start()
                plt.show()
                #plt.pause(2)
                #plt.close()
            
            if motion:
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                ax = plt.gca()
                ax.cla()
                self.vis_rrt_tree(ax) 
                self.vis_robot(ax)
                self.vis_human_body(ax)
                self.vis_human_belief(ax)
                self.vis_walls(ax)
                self.vis_best_path(ax)


                plt.xlim(self._xlim[0], self._xlim[1])
                plt.ylim(self._ylim[0], self._ylim[1])
                ax.text(-6.9, 5, message , fontsize=14, bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.5})
                # Set general font size
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(16)
                plt.xlabel("X [m]", fontsize=16)
                plt.ylabel("Y [m]", fontsize=16)
                fig = plt.gcf()
                fig.set_size_inches(4,4)
                plt.pause(0.001)

def close_event():
    plt.close()
    
