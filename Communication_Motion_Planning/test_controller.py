import numpy as np
from controller import controller_map1, controller_map3
from robot import robot_map1, robot_map3


rw = 2    # room half width
rl = 5    # room half length
cl = 1    # small corridor half length
cw = 4    # small corridor abs(y_min)

# Another set of parameters for a more strict environment
rw = 1    # room half width
rl = 5    # room half length
cl = 1.5  # small corridor half length
cw = 2    # small corridor abs(y_min)

dt = 0.1
env_bounds = type('', (), {})()
env_bounds.y_max_map = rw             # [m]
env_bounds.y_min_map = -rw            # [m]
env_bounds.x_min_map = -rl            # [m]
env_bounds.x_max_map = rl             # [m]
env_bounds.y_min_park = -cw           # [m]
env_bounds.x_min_park = -cl           # [m]
env_bounds.x_max_park = cl            # [m]
goal_set = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])  # [x, y, theta]
start = np.array([-3.0, 0.0, 0.0])                     # [x, y, theta]


# call the contoller and robot model of map3
robot_model = robot_map3(dt, x = start, x_goal = goal_set[0])
unsafe_info = robot_model.unsafe_func()                # extract the unsafe set info
map_info = robot_model.map_func(env_bounds)            # extract the environment info
controller= controller_map3(unsafe_info, map_info)          # controller object
u = controller.control(env_bounds, robot_model, np.array([1, 0.2]), np.array([3,0,0]), [np.array([2, 0, 1])])
print(u)