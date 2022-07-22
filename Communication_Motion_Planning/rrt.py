from queue import PriorityQueue
from typing import Counter
import numpy as np
import math
import random
import copy
from scipy import signal

########
# RRT class

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, t = 0):
            self.t = t
            self.x = x
            self.path_x = []
            self.path_u = []
            self.cost = None
            self.cost_heading = 0
            self.parent = None
            

    def __init__(self,
                 env_bounds,
                 robot_model,
                 controller_model,
                 planning_horizon,  
                 cost_weights = [2, 1, 0.2, 1],           # [w_goal, w_human, w_heading, w_trap]
                 state_sample_var = 2,  
                 n_segments=10,          
                 max_iter=200,
                 selected_nodes_size = 3,
                 occupancy_grid = 0,                      # occupancy grid <<<<== for now reciev it from outside but correct it later!
                 grid_res = 0,                            # remove later
                 grid_y_offset = 0,                       # remove later
                 grid_x_offset = 0,                       # remove later
                 multiple_plan = 'on',                    # modes: 'on' and 'off'
                 state_sample_mode = 'explore_oriented'): # modes: 'exploit_oriented' and 'explore_oriented'
        """
        Setting Parameter


        """
        self._node_list = []                                         # node list 
        self._node_list_log = []
        self._best_nodes_list= []      
        self._best_node = []                                         # best selected node
        self._best_path = []
        self._selected_nodes_size = selected_nodes_size              # size of query: size of selected nodes
        self._state_sample_var = state_sample_var                    # free state sample variance p(theta) ~ exp(-(tehta-theta_goal)^2/(2*state_sample_var^2))
        self._n_segments = n_segments                                # number of segments at each edge
        self._max_iter = max_iter                                    # maximum number of nodes at each iteration
        self._env_bounds = env_bounds
        self._planning_horizon = planning_horizon                    # [s]
        self._robot_model = robot_model
        self._cost_weights = cost_weights                            # node selection cost [w_goal, w_human, w_heading, w_trap]
        self._controller= controller_model                           # controller object
        self._state_sample_mode = state_sample_mode                  # state sampling modes: 'exploit_oriented' and 'explore_oriented'
        self._multi_plan_mode = multiple_plan                        # enable or disable high level. modes: 'on' and 'off'
        self._maze = copy.deepcopy(occupancy_grid)                   # Modify this grid info <<<<< for now receive from output
        self._res = grid_res                                         # Modify this grid info <<<<< for now receive from output
        self.map_inflate()                                           # inflating map
        self._y_offset = grid_y_offset                               # Modify this grid info <<<<< for now receive from output
        self._x_offset = grid_x_offset                               # Modify this grid info <<<<< for now receive from output

    def planning(self, dyn_obs_list):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        # initialize expansion
        self._best_node = [] 
        init_node = self.Node(x = self._robot_model.x, t = 0)            # first node
        goal_node = self.Node(x = self._robot_model.x_goal)              # goal node
        final_goal_node = self.Node(x = self._robot_model.goal_set[-1])              # goal node
        #self.nodes_cum_density = np.array(init_node.x[0:2])
        #self.counter = 1
        self.calc_node_cost(node = init_node, dyn_obs_list = dyn_obs_list, goal_node = goal_node, final_goal_node = final_goal_node) # associate cost to the init_node
        init_node.path_x = [np.array([init_node.x[0], init_node.x[1], init_node.x[2], 0])]
        self._node_list = [init_node]                       # node list stores all nodes of the partial expanded tree
        sample_node_list = [init_node]                     # nodes are being sampled from this list: only store nodes that are in the time limit
        
        # start expanding tree
        for i in range(self._max_iter):

            node_sample = self.node_sampling(sample_node_list)                          # node sampling
            if self._state_sample_mode == 'exploit_oriented':
                state_sample = self.state_sampling_exploit(node_sample, goal_node)      # free state sampling (theta in bicycle model)
            else:
                state_sample = self.state_sampling_explore(node_sample)                 # free state sampling (theta in bicycle model)
            control_ref = self.input_sampling(node_sample, state_sample, goal_node)                # input sampling
            
            new_node = self.steer(node_sample, control_ref, dyn_obs_list)                 # steering function
            self.calc_node_cost(node = new_node, dyn_obs_list = dyn_obs_list, goal_node = goal_node, final_goal_node = final_goal_node) # assign cost to the new node


            # adding new_node to the node_list and sample list
            self._node_list.append(new_node)
            if new_node.t <= self._planning_horizon - self._robot_model.dt*self._n_segments:
                sample_node_list.append(new_node)

        if self._multi_plan_mode == 'on':
            selected_idx = self.node_selection_multi_plan()
        else:
            selected_idx = self.node_selection_single_plan()
            
        self._best_nodes_list = [self._node_list[i] for i in selected_idx]

        return self._best_nodes_list

    def node_selection_single_plan(self):
        selected_node_idx = [0]
        opt_cost = np.inf
        num_nodes = len(self._node_list)
        for n in range(num_nodes):
            node = self._node_list[n]
            if node.cost <= opt_cost:
                opt_cost = node.cost
                selected_node_idx[0] = n
        
        return selected_node_idx

    def node_selection_multi_plan(self):
        n = len(self._node_list)
        k = self._selected_nodes_size
        selected_indx_list = random.sample(range(n), k)          # initialize the indices of selected nodes
        optimal_cost = self.diverse_plan_cost_cal(selected_indx_list)
        prev_cost = 0
        while (np.abs(prev_cost-optimal_cost) > 0.00001):
            prev_cost = optimal_cost
            indx_list = [i for i in range(n)]                    # index list of all nodes
            for num in selected_indx_list: indx_list.remove(num) # remove the selected nodes indices from the index list of all nodes

            # test if the new index results a lower cost
            for idx in indx_list:
                cost_list = np.zeros(k)
                for j in range(k):
                    temp = selected_indx_list[j]
                    selected_indx_list[j] = idx
                    cost_list[j] = self.diverse_plan_cost_cal(selected_indx_list)
                    selected_indx_list[j] = temp
                if np.min(cost_list) < optimal_cost:
                    sub_idx = np.argmin(cost_list)
                    selected_indx_list[sub_idx] = idx

            optimal_cost = self.diverse_plan_cost_cal(selected_indx_list)

        return selected_indx_list

    def diverse_plan_cost_cal(self, node_indx_list, node_cost_w = 1.0, dist_w = 1.5):
        total_cost = 0
        for i in node_indx_list:
            node_cost = self._node_list[i].cost
            dist_list = [self.node_dist(self._node_list[i],self._node_list[j]) for j in node_indx_list]
            #total_cost += node_cost_w * node_cost/(dist_w * np.exp(-1/sum(dist_list)))
            total_cost += node_cost_w * node_cost/(dist_w * sum(dist_list))
        
        return total_cost
    
    def steer(self, from_node, control_ref, dyn_obs_list):
        time = from_node.t
        new_node = self.Node(np.array([from_node.x[0], from_node.x[1], from_node.x[2]]))
        new_node.path_x = [np.array([new_node.x[0], new_node.x[1], new_node.x[2], time])]
         
        
        for _ in range(self._n_segments):
            unsafe_list = [ human_state[int(time/self._robot_model.dt+0.00001)] for human_state in dyn_obs_list ]
            u = self._controller.control(self._env_bounds, self._robot_model, control_ref, new_node.x, unsafe_list)
            time += self._robot_model.dt
            new_node.x+=self._robot_model.motion_model(new_node.x, u)
            new_node.path_x.append(np.array([new_node.x[0], new_node.x[1], new_node.x[2], time]))
            new_node.path_u.append(np.array([u[0], u[1]]))
            

        new_node.t = time
        new_node.parent = from_node
        
        
        return new_node

    def calc_node_cost(self, node, dyn_obs_list, goal_node, final_goal_node):
        #predicted location of obstacle at this time
        ob_indx = int(node.t/self._robot_model.dt+0.00001)

        cost_to_goal = np.exp(-1/(np.linalg.norm(node.x[0:2]-self._robot_model.x_goal[0:2])))
        
        cost_to_human = 0
        for traj in dyn_obs_list:
            cost_to_human += np.exp(-(np.linalg.norm(node.x[0:2]-traj[ob_indx][0:2]))) 
        
        _, heading2goal = self.calc_distance_and_angle(node, final_goal_node)
        heading_cost = np.exp(-1/np.abs(self.angular_diff(heading2goal, node.x[2])))

        trap_cost = self.trap_cost_cal(current_x = node.x[0:2], goal_x = goal_node.x[0:2])
        trap_cost = np.exp(-1/np.abs(trap_cost))
        
        node.cost = self._cost_weights[0]*cost_to_goal + self._cost_weights[1]*cost_to_human + self._cost_weights[2]*heading_cost + self._cost_weights[3]*trap_cost
        node.cost_heading = self._cost_weights[2]*heading_cost

    def trap_cost_cal(self, current_x, goal_x, waypoints = 20):
        cost = 0.001
        for i in range(waypoints):
            l = i * (1 / (waypoints - 1))
            point = l * current_x + (1 - l) * goal_x
            if self.is_occupied(point):
                cost += 1
        return cost

    def generate_final_course(self, node):
        self._best_node = node
        control_seq = []
        path_seq = [np.array(node.path_x[-1])]
        control = []
        path = []
        while node.parent is not None:
            control = control + node.path_u
            path = path + node.path_x[0:-1]
            control.reverse()
            path.reverse()
            path_seq = path_seq + path
            control_seq = control_seq + control
            node = node.parent
            control = []
            path = []

        control_seq.reverse()
        path_seq.reverse()
        self._best_path = np.array(path_seq)
        if len(control_seq) == 0:
            control_seq = np.array([[0, 0]])
        return np.array(control_seq), np.array(path_seq)


    def node_sampling(self, sample_node_list):
        node_ind = random.randint(0,len(sample_node_list)-1)
        rnd_node = sample_node_list[node_ind]
        return rnd_node

    def state_sampling_exploit(self, sample_node, goal_node):
        _, theta_goal = self.calc_distance_and_angle(sample_node, goal_node)
        return np.random.normal(loc=theta_goal, scale=self._state_sample_var, size=None)

    def state_sampling_explore(self, sample_node):
        theta_max = sample_node.x[2] + self._robot_model.dt*self._robot_model.w_max*self._n_segments
        theta_min = sample_node.x[2] + self._robot_model.dt*self._robot_model.w_min*self._n_segments
        return self.interval_sample(theta_min-1, theta_max+1)

    def input_sampling(self, node, theta_goal, goal_node, v_gain = 1.5, w_gain = 0.3):
        control_ref = np.array([0,0],'f')
        #d = np.linalg.norm(node.x[0:2]-self._robot_model.x_goal[0:2])
        d, th = self.calc_distance_and_angle(node, goal_node)
        #control_ref[0] = np.exp(-0.8*self.angular_diff(node.x[2], th))*np.exp(-v_gain/d)*self._robot_model.v_max
        control_ref[0] = np.exp(-v_gain/d)*self._robot_model.v_max
        #control_ref[0] = self._robot_model.v_max
        control_ref[1] = w_gain*self.angular_diff(node.x[2], theta_goal)
        return control_ref

    def is_occupied (self, location):  # Modify this grid info <<<<< for now receive from output
        dim = self._maze.shape
        loc = self.where (location)
        x = loc [0]
        y = loc [1]
        if ((not (0 <= x < dim[1])) or (not (0 <= y < dim[0]))): return True
        if (self._maze[y][x] == 1): return True
        else: return False

    def where (self, location):       # Modify this grid info <<<<< for now receive from output
        x_loc = location[0]
        y_loc = location[1]
        y = int ((y_loc + self._y_offset)/self._res) +1
        x = int ((x_loc + self._x_offset)/self._res) +1
        return [x,y]

    def map_inflate(self):
        r = self._robot_model.r
        bound = np.ceil(r/self._res - 0.0001)
        mask = self.mask_create(bound)
        temp = signal.convolve2d(self._maze, mask, boundary='symm', mode='same')
        self._maze = np.where(temp >= 1, 1, 0)





    @staticmethod
    def interval_sample(min_x, max_x, res = 1000):
        return (((max_x) - (min_x))/res)*np.random.randint(0, res) + (min_x)

    @staticmethod
    def mask_create(bound):
        y,x = np.ogrid[-bound: bound+1, -bound: bound+1]
        mask = x**2+y**2 <= bound**2
        mask = 1*mask.astype(float)
        return mask
        
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        d = math.hypot(dx[0], dx[1])
        theta = math.atan2(dx[1], dx[0])
        return d, theta

    @staticmethod
    def angular_diff(x,y): #angular difference
        d = y-x
        return math.atan2(math.sin(d), math.cos(d))   # --->check out this part should be like diffang() in MATLAB
    
    @staticmethod
    def node_dist(a,b):
        return np.linalg.norm(a.x[0:2]-b.x[0:2])