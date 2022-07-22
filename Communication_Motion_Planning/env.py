import numpy as np
from controller import controller_map1, controller_map2, controller_map3, controller_map4
from robot import robot_map1, robot_map2, robot_map3, robot_map4
from matplotlib import colors
import matplotlib.pyplot as plt


class ENV:
    def __init__(self, map_name = None, res = 0.5, p_horizon = 3, human_start = [0,0], human_goal = [1,1],
                 human_traj = np.array([[7,0.0],[6.9,0.0],[6.8, 0.0]]), robot_start = [5.0, 0.0, 0.0], 
                 robot_goal = [[4.0, 0.0, 0.0]], r_radius = 0.5, goal_shift_radii = 1.5):
        self._map = map_name
        self._res = res
        self._walls = []
        if self._map == 'map1':
            self._name = 'map1'
            w = 5          # half-width of area
            h_width = 1.5    # horizontal corridor width
            v_width = 1.5    # vertical corridor width
            self._walls.append(np.array([[-w, w], [w, w]]))
            self._walls.append(np.array([[-w, -w], [w, -w]]))
            self._walls.append(np.array([[-w, -w], [-w, w]]))
            self._walls.append(np.array([[w, -w], [w, w]]))
            self._walls.append(np.array([[-w, h_width/2], [-v_width/2, h_width/2]]))
            self._walls.append(np.array([[-w, -h_width/2], [-v_width/2, -h_width/2]]))
            self._walls.append(np.array([[v_width/2, h_width/2], [w, h_width/2]]))
            self._walls.append(np.array([[v_width/2, -h_width/2], [w, -h_width/2]]))
            self._walls.append(np.array([[-v_width/2, -w], [-v_width/2, -h_width/2]]))
            self._walls.append(np.array([[v_width/2, -w], [v_width/2, -h_width/2]]))
            self._walls.append(np.array([[-v_width/2, h_width/2], [-v_width/2, w]]))
            self._walls.append(np.array([[v_width/2, h_width/2], [v_width/2, w]]))
            self._planning_horizon = p_horizon
            self._dt = 0.1
            self._env_bounds = type('', (), {})()
            self._env_bounds.y_min_map = -w              # [m]
            self._env_bounds.y_max_map = w               # [m]
            self._env_bounds.x_min_map = -w              # [m]
            self._env_bounds.x_max_map = w               # [m]
            self._env_bounds.y_max = h_width/2   # [m]
            self._env_bounds.y_min = -h_width/2  # [m]
            self._env_bounds.x_max = v_width/2   # [m]
            self._env_bounds.x_min = -v_width/2  # [m]
            self._goal_set = np.array(robot_goal)    # [x, y, theta]
            self._start = np.array(robot_start)      # [x, y, theta]
            self._human_goal = np.array(human_goal)
            self._human_obsv_traj = human_traj       # human observed trajectory
            # Human simulation parameters
            desired_force_factor = 2
            social_force_factor = 5.1
            obstacle_force_factor = 4
            obstacle_force_sigma = 0.1
            self._sf_params = [desired_force_factor,
                               social_force_factor,
                               obstacle_force_factor,
                               obstacle_force_sigma]

            # call the contoller and robot model of map1
            self._robot_model = robot_map1(self._dt, x = self._start, goal_set = self._goal_set, r = r_radius, goal_shift_radii=goal_shift_radii)
            lyp_info = self._robot_model.lyp_func()                # extract the lyapanove set info
            unsafe_info = self._robot_model.unsafe_func()                # extract the unsafe set info
            map_info = self._robot_model.map_func(self._env_bounds)            # extract the environment info
            self._controller= controller_map1(unsafe_info, map_info, lyp_info)          # controller object
            
            self._vis_bound_x_min = -w
            self._vis_bound_x_max = w
            self._vis_bound_y_min = -w
            self._vis_bound_y_max = w
            #___________________________________________________________________
            # Environment Grid Map
            self._x_bound = 2*w
            self._y_bound = 2*w
            self._x_len = int(self._x_bound/res)
            self._y_len = int(self._y_bound/res)
            self._y_offset = w  # to get non-negative coordinates
            self._x_offset = w  # to get non-negative coordinates
            self._maze = np.zeros((self._y_len+2,self._x_len+2))
            self._ox, self._oy = [], []
            self._human_loc = None
            self._robot_loc = None
            self._res = res


            a = self.where([-w,w])
            b = self.where([-v_width/2,h_width/2])
            for i in range (b[1],a[1]+1):
                for j in range (a[0],b[0]+1):
                    self._maze[i][j] = 1

            a = self.where([v_width/2,w])
            b = self.where([w,h_width/2])
            for i in range (b[1],a[1]+1):
                for j in range (a[0],b[0]+1):
                    self._maze[i][j] = 1

            a = self.where([-w,-h_width/2])
            b = self.where([-v_width/2,-w])
            for i in range (b[1],a[1]+1):
                for j in range (a[0],b[0]+1):
                    self._maze[i][j] = 1

            a = self.where([v_width/2,-h_width/2])
            b = self.where([w,-w])
            for i in range (b[1],a[1]+1):
                for j in range (a[0],b[0]+1):
                    self._maze[i][j] = 1

            # following lines add a frame to the grid map to facilitate path planning
            for i in range(0, self._x_len+2):
                self._maze[0][i] = 1

            for i in range(0, self._y_len+2):
                self._maze[i][self._x_len+2-1] = 1
        
            for i in range(0, self._x_len+2):
                self._maze[self._y_len+2-1][i] = 1
        
            for i in range(0, self._y_len+2):
                self._maze [i][0] = 1
            #_____________________________________________________________________________

        elif self._map == 'map2':
            self._name = 'map2'
            self._walls.append(np.array([[-4.1, -3.15], [1., -3.15]]))
            self._walls.append(np.array([[-4.1, -3.15], [-4.1, 2.6]]))
            self._walls.append(np.array([[1., -3.15], [1., 2.6]]))
            self._walls.append(np.array([[-4.1, 2.6], [1., 2.6]]))
            # center obstacle
            self._walls.append(np.array([[-1.35, -0.34], [-1., -0.34]]))
            self._walls.append(np.array([[-1.35, -0.34], [-1.35, 0.15]]))
            self._walls.append(np.array([[-1., -0.34], [-1., 0.15]]))
            self._walls.append(np.array([[-1.35, 0.15], [-1., 0.15]]))
            self._planning_horizon = p_horizon
            self._dt = 0.1
            self._env_bounds = type('', (), {})()
            self._env_bounds.y_min_map = -5     # [m]
            self._env_bounds.y_max_map = 5      # [m]
            self._env_bounds.x_min_map = -4     # [m]
            self._env_bounds.x_max_map = 4      # [m]
            self._env_bounds.y_min = -.25     # [m]
            self._env_bounds.y_max = .25      # [m]
            self._env_bounds.x_min = -.25    # [m]
            self._env_bounds.x_max = .25      # [m]
            self._goal_set = np.array(robot_goal)  # [x, y, theta]
            self._start = np.array(robot_start)              # [x, y, theta]
            self._human_goal = np.array(human_goal)
            self._human_obsv_traj = human_traj  # human observed trajectory
            # Human simulation parameters
            desired_force_factor = 2
            social_force_factor = 5.1
            obstacle_force_factor = 10
            obstacle_force_sigma = 0.1
            self._sf_params = [desired_force_factor,
                               social_force_factor,
                               obstacle_force_factor,
                               obstacle_force_sigma]
            
            # call the contoller and robot model of map2
            self._robot_model = robot_map2(self._dt, x = self._start, goal_set = self._goal_set, r = r_radius, goal_shift_radii=goal_shift_radii)
            unsafe_info = self._robot_model.unsafe_func()                # extract the unsafe set info
            map_info = self._robot_model.map_func(self._env_bounds)            # extract the environment info
            self._controller= controller_map2(unsafe_info, map_info)          # controller object

            self._vis_bound_x_min = -4
            self._vis_bound_x_max = 4
            self._vis_bound_y_min = -5
            self._vis_bound_y_max = 5
            #___________________________________________________________________
            # Environment Grid Map
            self._x_bound = 8
            self._y_bound = 10
            self._x_len = int (self._x_bound/res)
            self._y_len = int (self._y_bound/res)
            self._y_offset = 5  # to get non-negative coordinates
            self._x_offset = 4  # to get non-negative coordinates
            self._maze = np.zeros((self._y_len+2,self._x_len+2))
            self._ox, self._oy = [], []
            self._human_loc = None
            self._robot_loc = None
            self._res = res

            a = self.where([0.25,-0.25])
            b = self.where([-0.25,0.25])
            for i in range (a[1],b[1]):
             for j in range (b[0],a[0]):
                 self._maze[i][j] = 1

            # following lines add a frame to the grid map to facilitate path planning
            for i in range(0, self._x_len+2):
             self._maze[0][i] = 1

            for i in range(0, self._y_len+2):
             self._maze[i][self._x_len+2-1] = 1

            for i in range(0, self._x_len+2):
             self._maze[self._y_len+2-1][i] = 1

            for i in range(0, self._y_len+2):
             self._maze [i][0] = 1
            #_____________________________________________________________________________

        elif self._map == 'map3':
            self._name = 'map3'
            rw = 0.6    # room half width
            rl = 7    # room half length
            cl = 3    # small corridor half length
            cw = 1.5    # small corridor abs(y_min)
            self._walls.append(np.array([[-rl, rw], [rl, rw]]))
            self._walls.append(np.array([[-rl, rw], [-rl, -rw]]))
            self._walls.append(np.array([[rl, -rw], [rl, rw]]))
            self._walls.append(np.array([[-rl, -rw], [-cl, -rw]]))
            self._walls.append(np.array([[cl, -rw], [rl, -rw]]))
            self._walls.append(np.array([[-cl, -cw], [cl, -cw]]))
            self._walls.append(np.array([[-cl, -cw], [-cl, -rw]]))
            self._walls.append(np.array([[cl, -cw], [cl, -rw]]))
            self._planning_horizon = p_horizon
            self._dt = 0.1
            self._env_bounds = type('', (), {})()
            self._env_bounds.y_max_map = rw             # [m]
            self._env_bounds.y_min_map = -rw            # [m]
            self._env_bounds.x_min_map = -rl            # [m]
            self._env_bounds.x_max_map = rl             # [m]
            self._env_bounds.y_min_park = -cw           # [m]
            self._env_bounds.x_min_park = -cl           # [m]
            self._env_bounds.x_max_park = cl            # [m]
            self._goal_set = np.array(robot_goal)  # [x, y, theta]
            self._start = np.array(robot_start)                     # [x, y, theta]
            self._human_goal = np.array(human_goal)
            self._human_obsv_traj = human_traj  # human observed trajectory
            # Human simulation parameters
            desired_force_factor = 2.5
            social_force_factor = 4
            obstacle_force_factor = 0.5
            obstacle_force_sigma = 0.05
            self._sf_params = [desired_force_factor,
                               social_force_factor,
                               obstacle_force_factor,
                               obstacle_force_sigma]

            # call the contoller and robot model of map3
            self._robot_model = robot_map3(self._dt, x = self._start, goal_set = self._goal_set, r = r_radius, goal_shift_radii=goal_shift_radii)
            unsafe_info = self._robot_model.unsafe_func()                # extract the unsafe set info
            map_info = self._robot_model.map_func(self._env_bounds)      # extract the environment info
            self._controller= controller_map3(unsafe_info, map_info)     # controller object

            self._vis_bound_x_min = -7
            self._vis_bound_x_max = 7
            self._vis_bound_y_min = -2
            self._vis_bound_y_max = 1
            #___________________________________________________________________
            # Environment Grid Map
            self._x_bound = 2*rl
            self._y_bound = cw + rw
            self._x_len = int (self._x_bound/res)
            self._y_len = int (self._y_bound/res)
            self._y_offset = cw # to get non-negative coordinates
            self._x_offset = rl # to get non-negative coordinates
            self._maze = np.zeros((self._y_len+2,self._x_len+2))
            self._ox, self._oy = [], []
            self._human_loc = None
            self._robot_loc = None
            self._res = res

            a = self.where([-rl,-rw])
            b = self.where([-cl,-rw])
            c = self.where([-cl,-cw])
            for i in range (c[1],a[1]):
                for j in range (a[0],b[0]):
                    self._maze[i][j] = 1

            a = self.where([cl,-rw])
            b = self.where([rl,-rw])
            c = self.where([cl,-cw])
            for i in range (c[1] , a[1]):
                for j in range (a[0],b[0]):
                    self._maze[i][j] = 1

            for i in range(0, self._x_len+2):
                self._maze[0][i] = 1

            for i in range(0, self._y_len+2):
                self._maze[i][self._x_len+2-1] = 1

            for i in range(0, self._x_len+2):
                self._maze[self._y_len+2-1][i] = 1

            for i in range(0, self._y_len+2):
                self._maze [i][0] = 1
            #____________________________________________________________________________

        elif self._map == 'map4':
            self._name = 'map4'
            l = 10         # room length
            w = 8          # room width
            u_width = 2    # upper corridor width
            c_width = 2    # u-shape corridor width
            ow = 1.5       # right obstacle width
            self._walls.append(np.array([[0, 0],[l, 0]]))
            self._walls.append(np.array([[0, w],[l, w]]))
            self._walls.append(np.array([[0, 0],[0, w]]))
            self._walls.append(np.array([[l, 0],[l, w]]))
            self._walls.append(np.array([[l-ow, 0],[l-ow, w-u_width]]))
            self._walls.append(np.array([[l-ow, w-u_width],[l, w-u_width]]))
            self._walls.append(np.array([[c_width, c_width],[l-c_width-ow, c_width]]))
            self._walls.append(np.array([[c_width, c_width],[c_width, w-u_width]]))
            self._walls.append(np.array([[l-c_width-ow, c_width],[l-c_width-ow, w-u_width]]))
            self._planning_horizon = p_horizon
            self._dt = 0.1
            self._env_bounds = type('', (), {})()
            self._env_bounds.y_min_map = 0                      # [m]
            self._env_bounds.y_max_map = w                      # [m]
            self._env_bounds.x_min_map = 0                      # [m]
            self._env_bounds.x_max_map = l                      # [m]
            self._env_bounds.y_max = w-u_width          # [m]
            self._env_bounds.y_min = c_width            # [m]
            self._env_bounds.x_max = l-ow               # [m]
            self._env_bounds.x_mid = l-c_width-ow       # [m]
            self._env_bounds.x_min = c_width            # [m]
            self._goal_set = np.array(robot_goal)           # [x, y, theta]
            self._start = np.array(robot_start)             # [x, y, theta]
            self._human_goal = np.array(human_goal)
            self._human_obsv_traj = human_traj              # human observed trajectory
            # Human simulation parameters
            desired_force_factor = 2
            social_force_factor = 5.1
            obstacle_force_factor = 2
            obstacle_force_sigma = 0.1
            self._sf_params = [desired_force_factor,
                               social_force_factor,
                               obstacle_force_factor,
                               obstacle_force_sigma]

            # call the contoller and robot model of map3
            self._robot_model = robot_map4(self._dt, x = self._start, goal_set = self._goal_set, r = r_radius, goal_shift_radii=goal_shift_radii)
            unsafe_info = self._robot_model.unsafe_func()                # extract the unsafe set info
            map_info = self._robot_model.map_func(self._env_bounds)      # extract the environment info
            self._controller= controller_map4(unsafe_info, map_info)     # controller object

            self._vis_bound_x_min = 0
            self._vis_bound_x_max = l
            self._vis_bound_y_min = 0
            self._vis_bound_y_max = w
            #___________________________________________________________________
            # Environment Grid Map
            self._x_bound = l
            self._y_bound = w
            self._x_len = int(self._x_bound/res)
            self._y_len = int(self._y_bound/res)
            self._y_offset = 0  # to get non-negative coordinates
            self._x_offset = 0  # to get non-negative coordinates
            self._maze = np.zeros((self._y_len+2,self._x_len+2))
            self._ox, self._oy = [], []
            self._human_loc = None
            self._robot_loc = None
            self._res = res


            a = self.where([c_width,w-u_width])
            b = self.where([l-c_width-ow,c_width])
            for i in range (b[1],a[1]):
                for j in range (a[0],b[0]):
                    self._maze[i][j] = 1

            a = self.where([l-ow,w-u_width])
            b = self.where([l,0])
            for i in range (b[1] , a[1]):
                for j in range (a[0],b[0]):
                    self._maze[i][j] = 1

            for i in range(0, self._x_len+2):
                self._maze[0][i] = 1

            for i in range(0, self._y_len+2):
                self._maze[i][self._x_len+2-1] = 1

            for i in range(0, self._x_len+2):
                self._maze[self._y_len+2-1][i] = 1

            for i in range(0, self._y_len+2):
                self._maze [i][0] = 1
            #____________________________________________________________________________



    def is_occupied (self, location):
        dim = self._maze.shape
        loc = self.where (location)
        x = loc [0]
        y = loc [1]
        if ((not (0 <= x < dim[1])) or (not (0 <= y < dim[0]))): 
            return True
        if (self._maze[y][x] == 1): 
            return True
        else: 
            return False


    def where (self, location):
        x_loc = location[0]
        y_loc = location[1]
        y = int ((y_loc + self._y_offset)/self._res) +1
        x = int ((x_loc + self._x_offset)/self._res) +1
        return [x,y]

    def show (self):
        # free = 0, obstacle = 1, path = 2, start = 3, goal = 4, perseption =5
        cmap = colors.ListedColormap(['white', 'black','blue', 'green', 'red', 'gray'])
        bounds = [-0.5,0.5,1.5,2.5,3.5,4.5, 5.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        plt.figure(figsize=(16, 12), dpi=80)

        ax.imshow(self._maze, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)

        ax.invert_yaxis()
        plt.show()