import pickle as pk
import numpy as np
from pathlib import Path
import os
from rrt import RRT
import math

class Measurements:
    def __init__(self, human_radius = 0.3, r = 0.3, l = 0.01, threshold = 10, r_goal = None, h_goal = None):
        self.r = r
        self.l = l
        self._human_radius = human_radius
        self._threshold = threshold
        self._robot_goal = r_goal
        self._human_goal = h_goal
    
    def measure_travel_time (self, waypoints):
        r_way = waypoints [0]
        h_way = waypoints [1]
        r_timer = 0
        h_timer = 0

        for w in r_way:
            point1 = np.array(w[0:2])
            point2 = np.array(self._robot_goal)
            dist = np.linalg.norm(point1 - point2)
            if dist > 0.5: r_timer = r_timer + 1 

        for w in h_way:
            point1 = np.array(w[0:2])
            point2 = np.array(self._human_goal)
            dist = np.linalg.norm(point1 - point2)
            if dist > 0.5: h_timer = h_timer + 1
        
        r_timer = r_timer * 0.1
        h_timer = h_timer * 0.1

        return [r_timer, h_timer]

    def log_grid_scenario (self, grid_log, file_name, walls_info, waypoints, planning_cycles, 
                           high_level_mode, priority, dir, rrt, optimal_distances):
        metrics = self.get_metrics(waypoints, planning_cycles)
        signal_log_r,signal_log_h = self.log_communication(waypoints[0],waypoints[1])
        signal_log = [signal_log_r, signal_log_h]
        log = [grid_log, walls_info, waypoints, metrics, signal_log, rrt, optimal_distances]
        #if high_level_mode == 'on': file_name = map_name + '_' + high_level_mode + '_' + priority + '_'
        #else: file_name = map_name + '_' + high_level_mode + '_'
        root = os.getcwd()

        path = root + '\\' + dir + '\\' + file_name
        with open(path, 'wb') as f:
            pk.dump(log,f)

    
    def log_communication (self, r_waypoints, h_waypoints):
        log_r = []
        log_h = []
        index = []
        i = 0
        for point in r_waypoints:
           
            if (point[2] != ''): 
                log_r.append(point)
                index.append(i)
            i = i + 1
        for j in index:
            log_h.append(h_waypoints[j])
        return log_r, log_h

    def get_metrics (self, waypoints, planning_cycles):
        r_way = waypoints[0]
        h_way = waypoints[1]
        r_actual_dist = self.dist_waypoint(r_way)
        h_actual_dist = self.dist_waypoint(h_way)
        #_______________________________
        unsafe = self.unsafe_func (waypoints)
        travel_time = self.measure_travel_time (waypoints)
        return [r_actual_dist, h_actual_dist, planning_cycles, unsafe, travel_time]


    def dist_waypoint (self, waypoints):
        dist = 0
        for i in range (len(waypoints)-1):
            point1 = np.array(waypoints[i][0:2])
            point2 = np.array(waypoints[i+1][0:2])
            dist = dist + np.linalg.norm(point1 - point2)
        return dist


    def unsafe_func(self, waypoints):
        flag = False
        r = 0.5
        thresh = 1.5
        r_way = waypoints[0]
        h_way = waypoints[1]
        index = 0
        for w in r_way:
            point1 = np.array(w[0:2])
            point2 = np.array(self._robot_goal)
            dist = np.linalg.norm(point1 - point2)
            if dist > thresh: index = index + 1

        unsafe = []
        sum = 0
        for i in range (index):
            robot = r_way[i]
            human = h_way [i]
            CBF = (robot[0]-human[0])**2+(robot[1]-human[1])**2-(self.r+self._human_radius+self.l)**2
            unsafe.append(CBF)
        for each in unsafe:
            if each < self._threshold: sum = sum + each
            if each < 0: flag = True
        if flag: prox = math.inf
        else: 
            if sum == 0: prox = 0
            else: prox = 1 / (sum/index)
        return [unsafe, prox, max(unsafe)]




