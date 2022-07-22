import math
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from env import ENV

class Grid:
    def __init__(self, pers = 5, env = None, rrt_ph = 4):
        self._world = env
        self._x_bound = env._x_bound
        self._y_bound = env._y_bound
        self._res = env._res
        self._x_len = int (self._x_bound/self._res)
        self._y_len = int (self._y_bound/self._res)
        self._y_offset = env._y_offset
        self._x_offset = env._x_offset     
        self._maze = env._maze
        self._maze_clear = deepcopy(self._maze)
        self._ox, self._oy = [], []
        self._human_loc = None
        self._robot_loc = None
        self._pers = []
        self._pers_size = pers
        self._rrt_ph = rrt_ph
        
    
        for i in range (self._y_len+2):
            for j in range (self._x_len+2):
                if self._maze[i][j] == 1:
                    self._oy.append (i)
                    self._ox.append (j)
        self._ox_clear = deepcopy(self._ox)
        self._oy_clear = deepcopy(self._oy)

    def clear_maze (self):
        self._maze = deepcopy(self._maze_clear)
        self._ox = deepcopy(self._ox_clear)
        self._oy = deepcopy(self._oy_clear)

    def where (self, location):
        x_loc = location[0]
        y_loc = location[1]
        y = int ((y_loc + self._y_offset)/self._res) +1
        x = int ((x_loc + self._x_offset)/self._res) +1
        return [x,y]
    
    def reverse_where (self, grid_location):
        a = grid_location[0]
        b = grid_location[1]
        x = self._res*(  (a-1) - self._res/2   ) - self._x_offset
        y = self._res*(  (b-1) - self._res/2   ) - self._y_offset
        return [x, y, self._res/2]

    
    def give_path (self, start, end, belief, show_animation, robot_l, plan, plan_path, delta_t, avg_velocity):
        self.clear_maze()
        s = self.where(start)
        e = self.where(end)
        robot_loc = self.where(robot_l)
        robot_goal = self.where (plan)
        sx = s[0]
        sy = s[1]
        gx = e[0]
        gy = e[1]
        self.update_human (s)
        grid_size = 1.0 
        robot_radius = 0
        if show_animation:  # pragma: no cover
            plt.plot(self._ox, self._oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            plt.axis("equal")
        self.add_obstacle(belief, robot_loc, self.where(plan))
        a_star = AStarPlanner(self._ox, self._oy, grid_size, robot_radius)
        rx, ry, flag = a_star.planning(sx, sy, gx, gy, show_animation)
        if flag == False: 
            human_cost = math.inf
            path = [s]
        else: path,human_cost = self.path_info (rx, ry)
        self.get_pers (start)
        if (plan_path[0][0:2] == robot_loc and len(plan_path)==1): 
            plan_path = self.extend_path(plan_path)
        human_tp_path_in_grid = self.timestamp (path, avg_velocity)
        robot_tp_path_in_grid = plan_path
        human_synch_path = self.synch_paths (robot_tp_path_in_grid, human_tp_path_in_grid, avg_velocity)
        conflict_cost, potential, conflict = self.analyze_conflict (robot_tp_path_in_grid, human_synch_path)
        self.update_maze (path, s,e, robot_loc, robot_goal, plan_path, conflict, potential )
        
        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show()
        
        #self.show()

        return self._maze, path, human_cost, conflict_cost, human_synch_path
    
    def add_obstacle (self, belief, robot_loc, robot_plan):
        if (robot_loc == robot_plan): 
            self._oy.append (robot_plan[1])
            self._ox.append (robot_plan[0])
        for section in belief:
            a = section [0]
            b = section [1]
            self._oy.append (b)
            self._ox.append (a)
            self._maze[b][a] = 1

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
        ax.set_xticks(np.arange(0, self._x_len+2,2))
        ax.set_yticks(np.arange(0, self._y_len+2,2))
        ax.invert_yaxis()
        plt.show()

    def show_it (self, maze):
        # free = 0, obstacle = 1, path = 2, human = 3, human_goal = 4, perseption =5, robot = 6, robot_goal = 7, plan_path = 8, potential conflict = 9, conflict = 10
        cmap = colors.ListedColormap(['white', 'black','#baffce', '#0ee34a', '#08802a', 'gray', '#003cff', '#002291', '#92c4f0', '#f0a5a5', '#f56464' ])
        bounds = [-0.5,0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        plt.figure(figsize=(16, 12), dpi=80)

        ax.imshow(maze, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0, self._x_len+2,2))
        ax.set_yticks(np.arange(0, self._y_len+2,2))
        ax.invert_yaxis()
        plt.show()
        
    def give_path_cost (self, path):
        cost = 0
        for i in range (len(path)-1):
            x1 = path[i][0]
            y1 = path[i][1]
            x2 = path[i+1][0]
            y2 = path[i+1][1]
            if ( path[i] != path[i+1] ):
                if (x1 == x2 or y1 == y2): 
                    cost = cost + 1
                else: 
                    cost = cost + 1.4
        cost_cont = self.distance_in_continiuous (cost) 
        return cost_cont


    def path_info (self, rx, ry):
        path = []
        for i in range (len(rx)):
            path.append([rx[i],ry[i]])
        if len(rx) == 0:
            cost = math.inf
        else: cost = self.give_path_cost (path)
        return path, cost

    def get_pers (self, location):
        w = int((self._pers_size -1)/2)
        loc = self.where(location)
        a, b = loc[0], loc[1]
        self._pers = []
        for i in range (b-w,b+w+1):
            for j in range (a-w,a+w+1):
                condition1 = self.in_bound ([j,i])
                if (condition1):
                    temp = self._maze [i][j]
                    if ( temp != 1 and temp != 3): self._pers.append ([j,i])
        return self._pers

    def update_maze (self, path, start, end, robot_loc, robot_goal, plan_path, conflict, potential ):
        #print ("debug: ")
        #print ()
        for section in (self._pers):
            a = section [0]
            b = section [1]
            if (self._maze [b][a] != 1): self._maze [b][a] = 5
                    
        for i in range (len(path)):
            x = int(path [i][0])
            y = int(path [i][1])
            self._maze [y][x] = 2
        
        for i in range (len(plan_path)):
            x = int(plan_path [i][0])
            y = int(plan_path [i][1])
            self._maze [y][x] = 8

        for i in range (len(potential)):
            x = int(potential [i][0])
            y = int(potential [i][1])
            self._maze [y][x] = 9


        for i in range (len(conflict)):
            x = int(conflict [i][0])
            y = int(conflict [i][1])
            self._maze [y][x] = 10


        self._maze [start[1]] [start[0]] = 3
        self._maze [end[1]] [end[0]] = 4

        self._maze [robot_loc[1]][robot_loc[0]] = 6
        self._maze [robot_goal[1]][robot_goal[0]] = 7
    
    def update_human (self, loc):
        self._human_loc = loc
    
    def in_bound (self, point):
        flag = True
        if point[0]<0 or point[1]<0: flag = False
        if (point[0]> self._x_len +1) or (point[1]> self._y_len +1): flag = False
        return flag

    def distance_in_continiuous (self, grid_dist):
        return self._res * grid_dist


    def path_in_grid (self, cont_path):
        grid_path = []
        for node in cont_path:
            loc = self.where (node[0:2])
            grid_path.append ([loc[0], loc[1], node[3]])
        return grid_path

    def path_in_grid_no_time (self, cont_path):
        grid_path = []
        for node in cont_path:
            loc = self.where (node[0:2])
            grid_path.append ([loc[0], loc[1],])
        return grid_path

    def timestamp (self, p, avg_velocity):
        path = deepcopy (p)
        path.reverse()

        if path == []: return []
        else:
            time = 0
            delta_x = 0
            delta_t = 0
            path_t = []
            path_t.append ([path[0][0],path[0][1], time ])
            temp_dist = 0
            for i in range (len(path)-1):
                x1 = path[i][0]
                y1 = path[i][1]
                x2 = path[i+1][0]
                y2 = path[i+1][1]
                if (x1 == x2 or y1 == y2): temp_dist = self._res
                else: temp_dist = 1.4 * self._res
                delta_t = temp_dist / avg_velocity
                time = time + delta_t
                node = [x2, y2, round(time,4)]
                path_t.append (node)
            return path_t

    def analyze_conflict (self, robot_path, human_path):
        potential = []
        conflict = []
        grid_size = 1.0 
        robot_radius = 0
        min_distance = math.inf
        if human_path == []: return 0, potential, conflict
        else:
            a_star = AStarPlanner(self._ox_clear, self._oy_clear, grid_size, robot_radius)
            for i in range (len (robot_path)):
                sx, sy = robot_path[i][0], robot_path[i][1]
                gx, gy = human_path[i][0], human_path[i][1]
                rx, ry, flag = a_star.planning(sx, sy, gx, gy, False)
                if flag == False: 
                    dist = math.inf
                else: _,dist = self.path_info (rx, ry)
                if (dist < min_distance): min_distance = dist
            if min_distance == 0: conflict_cost = math.inf
            else: conflict_cost = 1 / min_distance
            return conflict_cost, potential, conflict

    def extend_path (self, path):
        loc = path[0]
        extended_path = []
        for i in range (self._rrt_ph*4):
            extended_path.append([loc[0], loc[1], i])
        return extended_path
    
    def synch_paths (self, robot_path, human_path, human_avg_vel):
        if human_path == []: return []
        else:
            human_path_synch = []
            last_index = 0
            for node in robot_path:
                human_loc = self.where_t (node[2], human_path)
                human_path_synch.append ([human_loc[0], human_loc[1], node[2]])
        return human_path_synch
           
    def where_t(self, t, human_path):
        ans = None
        for i in range (0, len(human_path)-1):
            if (t >= human_path[i][2] and t < human_path[i +1][2]):
                x = human_path[i][0]
                y = human_path[i][1]
                ans = [x,y]
                break
        if ans == None:
            if (t>= human_path[-1][2]):
                x = human_path[-1][0]
                y = human_path[-1][1]
                ans = [x,y]
        return ans

    
    

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy, show_animation):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        flag = True
        while 1:
            if len(open_set) == 0:
                #print("Open set is empty..")
                flag = False
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
#                 print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry, flag

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
#         print("min_x:", self.min_x)
#         print("min_y:", self.min_y)
#         print("max_x:", self.max_x)
#         print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
#         print("x_width:", self.x_width)
#         print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion
