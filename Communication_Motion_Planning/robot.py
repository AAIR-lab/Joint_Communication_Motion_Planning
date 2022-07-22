
from sympy import symbols, Matrix, sin, cos, lambdify
import numpy as np

class robot_map1:
        """
        Class for robot model (bicycle model)
        """

        def __init__(self, dt,
                     x,
                     goal_set,
                     goal_shift_radii = 1,
                     final_goal_radii = 0.5,
                     r = 0.3,
                     l = 0.01,           
                     v_max = 2.,
                     v_min = 0.,
                     w_max = 0.5,
                     w_min = -0.5):

            self.x = x         #robot position
            self.goal_shift_radii = goal_shift_radii
            self.final_goal_radii = final_goal_radii
            self.goal_set = goal_set
            self.x_goal = goal_set[0]     #robot goal
            self.goal_idx = 0  #robot goal index
            self.dt = dt       #sample time
            self.r = r         #robot radius
            self.goal_reached_bool = False
            self.l=l           #approximation parameter for bicycle model
            self.v_max = v_max #maximum linear velocity
            self.v_min = v_min #minimum linear velocity
            self.w_max = w_max #maximum angular velocity
            self.w_min = w_min #minimum angular velocity
            #Symbolic Variables

            #states and controls (symbols)
            xr1,xr2,xr3,xo1,xo2,ro,xg1,xg2,xg3 = symbols('xr1 xr2 xr3 xo1 xo2 ro xg1 xg2 xg3')  
            u1,u2 = symbols('u1,u2')
            # Vector of states and inputs + g and f:
            self.x_r_s = Matrix([xr1,xr2,xr3]) #robot state [x,y,theta]
            self.x_o_s = Matrix([xo1,xo2,ro])  #dynamic obstacle state [x,y,r]
            self.x_g_s = Matrix([xg1,xg2,xg3])  #dynamic obstacle state [x,y,theta]
            self.u_s = Matrix([u1,u2])         #control [v,w]
            # f_r = f+g*u
            self.f = Matrix([0.,0.,0.])
            self.g = Matrix([[cos(self.x_r_s[2]), -l*sin(self.x_r_s[2])], [sin(self.x_r_s[2]), l*cos(self.x_r_s[2])], [0., 1.]])
            self.f_r = self.f+self.g*self.u_s 
            self.Real_x_r = lambdify([self.x_r_s, self.u_s], Matrix([[cos(self.x_r_s[2]), 0.], [sin(self.x_r_s[2]), 0.], [0., 1.]])*self.u_s, 'numpy') #???
        
        def motion_model(self,x,u):
            return self.dt*self.Real_x_r(x,u).ravel()
        
        def move(self,u):
            self.x += self.dt*self.Real_x_r(self.x,u).ravel()
            self.goal_shift()

        def goal_shift(self):
            if np.linalg.norm(self.x[0:2]-self.goal_set[self.goal_idx][0:2]) <=  self.goal_shift_radii:
                self.goal_idx += 1
                if self.goal_idx < len(self.goal_set):
                        self.x_goal = self.goal_set[self.goal_idx]
                else:
                        self.goal_idx -= 1
                        self.x_goal = self.goal_set[-1]
                        if np.linalg.norm(self.x[0:2]-self.x_goal[0:2]) <= self.final_goal_radii:
                                self.goal_reached_bool = True
        
        def unsafe_func(self):
            unsafe_info = type('', (), {})()
            CBF = (self.x_r_s[0]-self.x_o_s[0])**2+(self.x_r_s[1]-self.x_o_s[1])**2-(self.r+self.x_o_s[2]+self.l)**2
            CBF_d = CBF.diff(Matrix([self.x_r_s]))
            unsafe_info.CBF = lambdify([self.x_r_s,self.x_o_s], CBF)
            unsafe_info.Lfh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.f)
            unsafe_info.Lgh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.g)
            return unsafe_info

        def lyp_func(self, goal_bound = 0.001):
            lyp_info = type('', (), {})()
            V = (self.x_r_s[0]-self.x_g_s[0])**2+(self.x_r_s[1]-self.x_g_s[1])**2-(goal_bound + self.l)**2
            V_d = V.diff(Matrix([self.x_r_s]))
            lyp_info.V = lambdify([self.x_r_s,self.x_g_s], V)
            lyp_info.LfV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.f)
            lyp_info.LgV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.g)
            return lyp_info

        def dist_2_goal(self):
                return np.linalg.norm(self.x[0:2]-self.x_goal[0:2])
        
        def map_func(self,env_bounds):
            map_info = type('', (), {})()
            map_info.CBF = []
            map_info.Lfh = []
            map_info.Lgh = []
            if hasattr(env_bounds,'x_max') and hasattr(env_bounds,'x_min'):
                    CBF = ((env_bounds.x_max - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_min_map') and hasattr(env_bounds,'y_max_map'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_min_map') and hasattr(env_bounds,'x_max_map'):
                    CBF = ((env_bounds.x_max_map - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max') and hasattr(env_bounds,'y_min'):
                    CBF = ((env_bounds.y_max - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_min + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max') and hasattr(env_bounds,'y_min') and hasattr(env_bounds,'x_max') and hasattr(env_bounds,'x_min'):
                    CBF = ((self.x_r_s[0]-env_bounds.x_min)**2 + (self.x_r_s[1]-env_bounds.y_min)**2 - (self.r + self.l)**2) *\
                             ((self.x_r_s[0]-env_bounds.x_max)**2 + (self.x_r_s[1]-env_bounds.y_min)**2 - (self.r + self.l)**2)*\
                                  ((self.x_r_s[0]-env_bounds.x_min)**2 + (self.x_r_s[1]-env_bounds.y_max)**2 - (self.r + self.l)**2) *\
                                          ((self.x_r_s[0]-env_bounds.x_max)**2 + (self.x_r_s[1]-env_bounds.y_max)**2 - (self.r + self.l)**2)
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'f'):
                    pass #To be filled later
            return map_info

class robot_map2:
        """
        Class for robot model (bicycle model)
        """

        def __init__(self, dt,
                     x,
                     goal_set,
                     goal_shift_radii = 1,
                     final_goal_radii = 0.5,
                     r = 0.3,
                     l = 0.01,           
                     v_max = 2.,
                     v_min = 0.,
                     w_max = 0.5,
                     w_min = -0.5):

            self.x = x         #robot position
            self.goal_shift_radii = goal_shift_radii
            self.final_goal_radii = final_goal_radii
            self.goal_set = goal_set
            self.x_goal = goal_set[0]     #robot goal
            self.goal_idx = 0  #robot goal index
            self.dt = dt       #sample time
            self.r = r         #robot radius
            self.goal_reached_bool = False
            self.l=l           #approximation parameter for bicycle model
            self.v_max = v_max #maximum linear velocity
            self.v_min = v_min #minimum linear velocity
            self.w_max = w_max #maximum angular velocity
            self.w_min = w_min #minimum angular velocity
            #Symbolic Variables

            #states and controls (symbols)
            xr1,xr2,xr3,xo1,xo2,ro,xg1,xg2,xg3 = symbols('xr1 xr2 xr3 xo1 xo2 ro xg1 xg2 xg3')  
            u1,u2 = symbols('u1,u2')
            # Vector of states and inputs + g and f:
            self.x_r_s = Matrix([xr1,xr2,xr3]) #robot state [x,y,theta]
            self.x_o_s = Matrix([xo1,xo2,ro])  #dynamic obstacle state [x,y,r]
            self.x_g_s = Matrix([xg1,xg2,xg3])  #dynamic obstacle state [x,y,theta]
            self.u_s = Matrix([u1,u2])         #control [v,w]
            # f_r = f+g*u
            self.f = Matrix([0.,0.,0.])
            self.g = Matrix([[cos(self.x_r_s[2]), -l*sin(self.x_r_s[2])], [sin(self.x_r_s[2]), l*cos(self.x_r_s[2])], [0., 1.]])
            self.f_r = self.f+self.g*self.u_s 
            self.Real_x_r = lambdify([self.x_r_s, self.u_s], Matrix([[cos(self.x_r_s[2]), 0.], [sin(self.x_r_s[2]), 0.], [0., 1.]])*self.u_s, 'numpy') #???
        
        def motion_model(self,x,u):
            return self.dt*self.Real_x_r(x,u).ravel()
        
        def move(self,u):
            self.x += self.dt*self.Real_x_r(self.x,u).ravel()
            self.goal_shift()
        
        def goal_shift(self):
            if np.linalg.norm(self.x[0:2]-self.goal_set[self.goal_idx][0:2]) <=  self.goal_shift_radii:
                self.goal_idx += 1
                if self.goal_idx < len(self.goal_set):
                        self.x_goal = self.goal_set[self.goal_idx]
                else:
                        self.goal_idx -= 1
                        self.x_goal = self.goal_set[-1]
                        if np.linalg.norm(self.x[0:2]-self.x_goal[0:2]) <= self.final_goal_radii:
                                self.goal_reached_bool = True

        def unsafe_func(self):
            unsafe_info = type('', (), {})()
            CBF = (self.x_r_s[0]-self.x_o_s[0])**2+(self.x_r_s[1]-self.x_o_s[1])**2-(self.r+self.x_o_s[2]+self.l)**2
            CBF_d = CBF.diff(Matrix([self.x_r_s]))
            unsafe_info.CBF = lambdify([self.x_r_s,self.x_o_s], CBF)
            unsafe_info.Lfh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.f)
            unsafe_info.Lgh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.g)
            return unsafe_info
        
        def lyp_func(self, goal_bound = 0.001):
            lyp_info = type('', (), {})()
            V = (self.x_r_s[0]-self.x_g_s[0])**2+(self.x_r_s[1]-self.x_g_s[1])**2-(goal_bound + self.l)**2
            V_d = V.diff(Matrix([self.x_r_s]))
            lyp_info.V = lambdify([self.x_r_s,self.x_g_s], V)
            lyp_info.LfV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.f)
            lyp_info.LgV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.g)
            return lyp_info

        def dist_2_goal(self):
                return np.linalg.norm(self.x[0:2]-self.x_goal[0:2])

        def map_func(self,env_bounds):
            map_info = type('', (), {})()
            map_info.CBF = []
            map_info.Lfh = []
            map_info.Lgh = []
            if hasattr(env_bounds,'x_min') and hasattr(env_bounds,'x_max'):
                    CBF = - (self.x_r_s[0] - (env_bounds.x_min - (self.l + self.r)))*(((self.r + self.l) + env_bounds.x_max) - self.x_r_s[0])
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_max_map') and hasattr(env_bounds,'x_min_map'):
                    CBF = ((env_bounds.x_max_map - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max_map') and hasattr(env_bounds,'y_min_map'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_min') and hasattr(env_bounds,'y_max'):
                    CBF = - (self.x_r_s[1] - (env_bounds.y_min - (self.l + self.r)))*(((self.r + self.l) + env_bounds.y_max) - self.x_r_s[1])
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'f'):
                    pass #To be filled later
            return map_info

class robot_map3:
        """
        Class for robot model (bicycle model)
        """

        def __init__(self, dt,
                     x,
                     goal_set,
                     goal_shift_radii = 1,
                     final_goal_radii = 0.5,
                     r = 0.3,
                     l = 0.01,           
                     v_max = 2.,
                     v_min = 0.,
                     w_max = 0.5,
                     w_min = -0.5):

            self.x = x         #robot position
            self.goal_shift_radii = goal_shift_radii
            self.final_goal_radii = final_goal_radii
            self.goal_set = goal_set
            self.x_goal = goal_set[0]     #robot goal
            self.goal_idx = 0  #robot goal index
            self.dt = dt       #sample time
            self.r = r         #robot radius
            self.goal_reached_bool = False
            self.l=l           #approximation parameter for bicycle model
            self.v_max = v_max #maximum linear velocity
            self.v_min = v_min #minimum linear velocity
            self.w_max = w_max #maximum angular velocity
            self.w_min = w_min #minimum angular velocity
            #Symbolic Variables

            #states and controls (symbols)
            xr1,xr2,xr3,xo1,xo2,ro,xg1,xg2,xg3 = symbols('xr1 xr2 xr3 xo1 xo2 ro xg1 xg2 xg3')  
            u1,u2 = symbols('u1,u2')
            # Vector of states and inputs + g and f:
            self.x_r_s = Matrix([xr1,xr2,xr3]) #robot state [x,y,theta]
            self.x_o_s = Matrix([xo1,xo2,ro])  #dynamic obstacle state [x,y,r]
            self.x_g_s = Matrix([xg1,xg2,xg3])  #dynamic obstacle state [x,y,theta]
            self.u_s = Matrix([u1,u2])         #control [v,w]
            # f_r = f+g*u
            self.f = Matrix([0.,0.,0.])
            self.g = Matrix([[cos(self.x_r_s[2]), -l*sin(self.x_r_s[2])], [sin(self.x_r_s[2]), l*cos(self.x_r_s[2])], [0., 1.]])
            self.f_r = self.f+self.g*self.u_s 
            self.Real_x_r = lambdify([self.x_r_s, self.u_s], Matrix([[cos(self.x_r_s[2]), 0.], [sin(self.x_r_s[2]), 0.], [0., 1.]])*self.u_s, 'numpy') #???
        
        def motion_model(self,x,u):
            return self.dt*self.Real_x_r(x,u).ravel()
        
        def move(self,u):
            self.x += self.dt*self.Real_x_r(self.x,u).ravel()
            self.goal_shift()
        
        def goal_shift(self):
            if np.linalg.norm(self.x[0:2]-self.goal_set[self.goal_idx][0:2]) <=  self.goal_shift_radii:
                self.goal_idx += 1
                if self.goal_idx < len(self.goal_set):
                        self.x_goal = self.goal_set[self.goal_idx]
                else:
                        self.goal_idx -= 1
                        self.x_goal = self.goal_set[-1]
                        if np.linalg.norm(self.x[0:2]-self.x_goal[0:2]) <= self.final_goal_radii:
                                self.goal_reached_bool = True
        def unsafe_func(self):
            unsafe_info = type('', (), {})()
            CBF = (self.x_r_s[0]-self.x_o_s[0])**2+(self.x_r_s[1]-self.x_o_s[1])**2-(self.r+self.x_o_s[2]+self.l)**2
            CBF_d = CBF.diff(Matrix([self.x_r_s]))
            unsafe_info.CBF = lambdify([self.x_r_s,self.x_o_s], CBF)
            unsafe_info.Lfh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.f)
            unsafe_info.Lgh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.g)
            return unsafe_info

        def lyp_func(self, goal_bound = 0.001):
            lyp_info = type('', (), {})()
            V = (self.x_r_s[0]-self.x_g_s[0])**2+(self.x_r_s[1]-self.x_g_s[1])**2-(goal_bound + self.l)**2
            V_d = V.diff(Matrix([self.x_r_s]))
            lyp_info.V = lambdify([self.x_r_s,self.x_g_s], V)
            lyp_info.LfV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.f)
            lyp_info.LgV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.g)
            return lyp_info
        
        def dist_2_goal(self):
                return np.linalg.norm(self.x[0:2]-self.x_goal[0:2])

        def map_func(self,env_bounds):
            map_info = type('', (), {})()
            map_info.CBF = []
            map_info.Lfh = []
            map_info.Lgh = []
            if hasattr(env_bounds,'x_max_map') and hasattr(env_bounds,'x_min_map'):
                    CBF = ((env_bounds.x_max_map - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max_map') and hasattr(env_bounds,'y_min_map'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_min_park') and hasattr(env_bounds,'y_max_map'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_min_park + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_min_park') and hasattr(env_bounds,'x_max_park'):
                    CBF = ((env_bounds.x_max_park - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_park + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max_map'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_min_map') and hasattr(env_bounds,'x_min_park') and hasattr(env_bounds,'x_max_park'):
                    CBF = ((self.x_r_s[0]-env_bounds.x_min_park)**2 + (self.x_r_s[1]-env_bounds.y_min_map)**2 - (self.r + self.l)**2) * ((self.x_r_s[0]-env_bounds.x_max_park)**2 + (self.x_r_s[1]-env_bounds.y_min_map)**2 - (self.r + self.l)**2)
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            
            if hasattr(env_bounds,'f'):
                    pass #To be filled later
            return map_info
        
class robot_map4:
        """
        Class for robot model (bicycle model)
        """

        def __init__(self, dt,
                     x,
                     goal_set,
                     goal_shift_radii = 1,
                     r = 0.3,
                     l = 0.01,           
                     v_max = 2.,
                     v_min = 0.,
                     w_max = 0.5,
                     w_min = -0.5):

            self.x = x         #robot position
            self.goal_shift_radii = goal_shift_radii
            self.goal_set = goal_set
            self.x_goal = goal_set[0]     #robot goal
            self.goal_idx = 0  #robot goal index
            self.dt = dt       #sample time
            self.r = r         #robot radius
            self.goal_reached_bool = False
            self.l=l           #approximation parameter for bicycle model
            self.v_max = v_max #maximum linear velocity
            self.v_min = v_min #minimum linear velocity
            self.w_max = w_max #maximum angular velocity
            self.w_min = w_min #minimum angular velocity
            #Symbolic Variables

            #states and controls (symbols)
            xr1,xr2,xr3,xo1,xo2,ro,xg1,xg2,xg3 = symbols('xr1 xr2 xr3 xo1 xo2 ro xg1 xg2 xg3')  
            u1,u2 = symbols('u1,u2')
            # Vector of states and inputs + g and f:
            self.x_r_s = Matrix([xr1,xr2,xr3]) #robot state [x,y,theta]
            self.x_o_s = Matrix([xo1,xo2,ro])  #dynamic obstacle state [x,y,r]
            self.x_g_s = Matrix([xg1,xg2,xg3])  #dynamic obstacle state [x,y,theta]
            self.u_s = Matrix([u1,u2])         #control [v,w]
            # f_r = f+g*u
            self.f = Matrix([0.,0.,0.])
            self.g = Matrix([[cos(self.x_r_s[2]), -l*sin(self.x_r_s[2])], [sin(self.x_r_s[2]), l*cos(self.x_r_s[2])], [0., 1.]])
            self.f_r = self.f+self.g*self.u_s 
            self.Real_x_r = lambdify([self.x_r_s, self.u_s], Matrix([[cos(self.x_r_s[2]), 0.], [sin(self.x_r_s[2]), 0.], [0., 1.]])*self.u_s, 'numpy') #???
        
        def motion_model(self,x,u):
            return self.dt*self.Real_x_r(x,u).ravel()
        
        def move(self,u):
            self.x += self.dt*self.Real_x_r(self.x,u).ravel()
        
        def unsafe_func(self):
            unsafe_info = type('', (), {})()
            CBF = (self.x_r_s[0]-self.x_o_s[0])**2+(self.x_r_s[1]-self.x_o_s[1])**2-(self.r+self.x_o_s[2]+self.l)**2
            CBF_d = CBF.diff(Matrix([self.x_r_s]))
            unsafe_info.CBF = lambdify([self.x_r_s,self.x_o_s], CBF)
            unsafe_info.Lfh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.f)
            unsafe_info.Lgh = lambdify([self.x_r_s,self.x_o_s], CBF_d.T*self.g)
            return unsafe_info
        
        def lyp_func(self, goal_bound = 0.001):
            lyp_info = type('', (), {})()
            V = (self.x_r_s[0]-self.x_g_s[0])**2+(self.x_r_s[1]-self.x_g_s[1])**2-(goal_bound + self.l)**2
            V_d = V.diff(Matrix([self.x_r_s]))
            lyp_info.V = lambdify([self.x_r_s,self.x_g_s], V)
            lyp_info.LfV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.f)
            lyp_info.LgV = lambdify([self.x_r_s,self.x_g_s], V_d.T*self.g)
            return lyp_info

        def dist_2_goal(self):
                return np.linalg.norm(self.x[0:2]-self.x_goal[0:2])

        def map_func(self,env_bounds):
            map_info = type('', (), {})()
            map_info.CBF = []
            map_info.Lfh = []
            map_info.Lgh = []
            if hasattr(env_bounds,'y_min') and hasattr(env_bounds,'y_max'):
                    CBF = - (self.x_r_s[1] - (env_bounds.y_min - (self.l + self.r)))*(((self.r + self.l) + env_bounds.y_max) - self.x_r_s[1])
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_max_map') and hasattr(env_bounds,'x_min_map'):
                    CBF = ((env_bounds.x_max_map - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max_map') and hasattr(env_bounds,'y_min_map'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_min') and hasattr(env_bounds,'x_min_map'):
                    CBF = ((env_bounds.x_min - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_max') and hasattr(env_bounds,'x_mid'):
                    CBF = ((env_bounds.x_max - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_mid + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'x_max') and hasattr(env_bounds,'x_min_map'):
                    CBF = ((env_bounds.x_max - self.l - self.r) - self.x_r_s[0])*(self.x_r_s[0] - (env_bounds.x_min_map + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'y_max_map') and hasattr(env_bounds,'y_max'):
                    CBF = ((env_bounds.y_max_map - self.l - self.r) - self.x_r_s[1])*(self.x_r_s[1] - (env_bounds.y_max + self.l + self.r))
                    map_info.CBF.append(lambdify([self.x_r_s],CBF))
                    map_info.Lfh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.f))
                    map_info.Lgh.append(lambdify([self.x_r_s] , CBF.diff(self.x_r_s).T*self.g))
            if hasattr(env_bounds,'f'):
                    pass #To be filled later
            return map_info