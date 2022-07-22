import cvxopt as cvxopt
import numpy as np

class controller_map1:
        """
        CBF-based controller
        """
        def __init__(self, unsafe_info, map_info, lyp_info, w_reg = 8.5, w_lambda = 6, w_b = 1, gamma_stat = 0.7, gamma_dyn = 2):
            self.w_reg = w_reg 
            self.w_b = w_b
            self.w_lambda = w_lambda
            self.gamma_stat = gamma_stat
            self.gamma_dyn = gamma_dyn
            self.unsafe_info = unsafe_info
            self.map_info = map_info
            self.lyp_info = lyp_info

        def simple_control(self, u_ref, x_robot, x_goal, k_v = 1.5, k_w = 1):
            dist_2_goal = np.linalg.norm(x_robot[0:2]-x_goal[0:2])
            u = np.array([0.0,0.0])
            u[0] = np.exp(-1/(k_v*dist_2_goal))*u_ref[0]
            u[1] = k_w*u_ref[1]
            return u

        def control(self, env_bounds, robot, u_ref, x_robot, unsafe_list, vacinity_r = 2):
            ###############
            # thing to add in future: 1. pass the list of unsafe areas each element [x,y,radius]
            # here we only assume one human is moving 

            map_info_temp = type('', (), {})()  #this constraint is modified based on the location of robot

            if (x_robot[0] > env_bounds.x_min and x_robot[0] < env_bounds.x_max) and (not (x_robot[1] > env_bounds.y_min and x_robot[1] < env_bounds.y_max)):
                map_info_temp.CBF = self.map_info.CBF[0:2]
                map_info_temp.Lfh = self.map_info.Lfh[0:2]
                map_info_temp.Lgh = self.map_info.Lgh[0:2]
            if (x_robot[1] > env_bounds.y_min and x_robot[1] < env_bounds.y_max) and (not (x_robot[0] > env_bounds.x_min and x_robot[0] < env_bounds.x_max)):
                map_info_temp.CBF = self.map_info.CBF[2:4]
                map_info_temp.Lfh = self.map_info.Lfh[2:4]
                map_info_temp.Lgh = self.map_info.Lgh[2:4]
            if (x_robot[1] >= env_bounds.y_min and x_robot[1] <= env_bounds.y_max) and (x_robot[0] >= env_bounds.x_min and x_robot[0] <= env_bounds.x_max):
                map_info_temp.CBF = self.map_info.CBF[1:3] + [self.map_info.CBF[4]]
                map_info_temp.Lfh = self.map_info.Lfh[1:3] + [self.map_info.Lfh[4]]
                map_info_temp.Lgh = self.map_info.Lgh[1:3] + [self.map_info.Lgh[4]]
            
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*len(unsafe_list), len(u_ref) + 1 + len(unsafe_list))) #2 and 1 are for lyapanov
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*len(unsafe_list), 1))
            else: 
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2, len(u_ref) + 1))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2, 1))

            ## wall constraints [-Lgh(x) 0 0]*[u lambda b]^T < Lfh(x) + gamma*h(x)
            for j in range(len(map_info_temp.CBF)):
                
                A[j,0:len(u_ref)] = -map_info_temp.Lgh[j](x_robot)
                b[j] = map_info_temp.Lfh[j](x_robot) + self.gamma_stat*map_info_temp.CBF[j](x_robot)



            
            ## Control Linear Constraints
            A[len(map_info_temp.CBF),0] = 1.
            b[len(map_info_temp.CBF)] = robot.v_max
            A[len(map_info_temp.CBF)+1,0] =- 1.
            b[len(map_info_temp.CBF)+1] = -robot.v_min
            A[len(map_info_temp.CBF)+2,1] = 1.
            b[len(map_info_temp.CBF)+2] = robot.w_max
            A[len(map_info_temp.CBF)+3,1] = -1.
            b[len(map_info_temp.CBF)+3] = -robot.w_min

            ## Lyapanov function constraint [LgV(x) -1 0]*[u lambda b ]^T < -LfV(x) and [0 -1 0]*[u lambda b ]^T < 0
            # Lyapanov constraint
            A[len(map_info_temp.CBF) + 2*len(u_ref), 0:2] = self.lyp_info.LgV(x_robot, robot.x_goal)
            A[len(map_info_temp.CBF) + 2*len(u_ref), len(u_ref)] = -1
            b[len(map_info_temp.CBF) + 2*len(u_ref)] = - self.lyp_info.LfV(x_robot, robot.x_goal)
                
            # constraint on -lambda < 0
            A[len(map_info_temp.CBF) + 2*len(u_ref) + 1, len(u_ref)] = -1
            b[len(map_info_temp.CBF) + 2*len(u_ref) + 1] = 0
            
            ## unsafe_list constraints [-Lgh(x) 0 -1]*[u lambda b]^T < Lfh(x) + gamma*h(x) and [0 0 1]*[u lambda b ]^T < 0
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                for j in range(len(unsafe_list)):

                    #CBF constraint
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*j, 0:2] = -self.unsafe_info.Lgh(x_robot,unsafe_list[j])
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*j, j + 1 + len(u_ref)] = -1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*j] = self.unsafe_info.Lfh(x_robot,unsafe_list[j]) + self.gamma_dyn*self.unsafe_info.CBF(x_robot,unsafe_list[j])
                
                    # Constraints on bi<0
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*j + 1, j + 1 +len(u_ref)] = 1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2 + 2*j + 1] = 0

            #if np.linalg.norm(x_robot[0:2]-robot.x_goal[0:2]) < 0.7:
                #print('here')

            #OPT weights
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                H = np.zeros((len(u_ref) + 1 + len(unsafe_list), len(u_ref) + 1 + len(unsafe_list)))
                ff = np.ones((len(u_ref) + 1 + len(unsafe_list), 1))
                ff[3] = self.w_b
            else:
                H = np.zeros((len(u_ref) + 1, len(u_ref) + 1))
                ff = np.ones((len(u_ref) + 1, 1))

            H[0,0] = self.w_reg
            H[1,1] = self.w_reg
            ff[0] = -2*self.w_reg*u_ref[0]
            ff[1] = -2*self.w_reg*u_ref[1]
            ff[2] = self.w_lambda


            try:
                uq = cvxopt_solve_qp(H, ff, A, b)
            except ValueError:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('Domain Error in cvx')

            if uq is None:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('infeasible QP')
            

            return uq[0:2]

class controller_map2:
        """
        CBF-based controller
        """

        def __init__(self, unsafe_info, map_info, w_reg = 10000, w_ref =1.5, gamma_stat = 0.7, gamma_dyn = 2):
            self.w_reg = w_reg 
            self.w_ref = w_ref
            self.gamma_stat = gamma_stat
            self.gamma_dyn = gamma_dyn
            self.unsafe_info = unsafe_info
            self.map_info = map_info

        def simple_control(self, u_ref, x_robot, x_goal, k_v = 1.5, k_w = 1):
            dist_2_goal = np.linalg.norm(x_robot[0:2]-x_goal[0:2])
            u = np.array([0.0,0.0])
            u[0] = np.exp(-1/(k_v*dist_2_goal))*u_ref[0]
            u[1] = k_w*u_ref[1]
            return u

        def control(self, env_bounds, robot, u_ref, x_robot, unsafe_list, vacinity_r = 4):
            ###############
            # thing to add in future: 1. pass the list of unsafe areas each element [x,y,radius]
            # here we only assume one human is moving 

            map_info_temp = type('', (), {})()  #this constraint is modified based on the location of robot

            if x_robot[1] < env_bounds.y_max and x_robot[1] > env_bounds.y_min:
                map_info_temp.CBF = self.map_info.CBF[0:3]
                map_info_temp.Lfh = self.map_info.Lfh[0:3]
                map_info_temp.Lgh = self.map_info.Lgh[0:3]
            elif x_robot[0] < env_bounds.x_max and x_robot[0] > env_bounds.x_min:
                map_info_temp.CBF = self.map_info.CBF[1:4]
                map_info_temp.Lfh = self.map_info.Lfh[1:4]
                map_info_temp.Lgh = self.map_info.Lgh[1:4]
            else:
                map_info_temp.CBF = self.map_info.CBF[1:3]
                map_info_temp.Lfh = self.map_info.Lfh[1:3]
                map_info_temp.Lgh = self.map_info.Lgh[1:3]

            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list), len(u_ref) + len(unsafe_list)))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list),1))
            else: 
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref), len(u_ref)))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref),1))

            
            #wall constraints [-Lgh(x) 0]*[u b]^T < Lfh(x) + gamma*h(x)
            for j in range(len(map_info_temp.CBF)):
                A[j,0:len(u_ref)] = -map_info_temp.Lgh[j](x_robot)
                b[j] = map_info_temp.Lfh[j](x_robot) + self.gamma_stat*map_info_temp.CBF[j](x_robot)
            
            #Control Linear Constraints
            A[len(map_info_temp.CBF),0] = 1.
            b[len(map_info_temp.CBF)] = robot.v_max
            A[len(map_info_temp.CBF)+1,0] =- 1.
            b[len(map_info_temp.CBF)+1] = -robot.v_min
            A[len(map_info_temp.CBF)+2,1] = 1.
            b[len(map_info_temp.CBF)+2] = robot.w_max
            A[len(map_info_temp.CBF)+3,1] = -1.
            b[len(map_info_temp.CBF)+3] = -robot.w_min

            #unsafe_list constraints [-Lgh(x) -1]*[u b]^T < Lfh(x) + gamma*h(x) and b<0
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                for j in range(len(unsafe_list)):

                    #CBF constraint
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, 0:2] = -self.unsafe_info.Lgh(x_robot,unsafe_list[j])
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, j+len(u_ref)] = -1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j] = self.unsafe_info.Lfh(x_robot,unsafe_list[j]) + self.gamma_dyn*self.unsafe_info.CBF(x_robot,unsafe_list[j])
                
                    # Constraints on bi<0
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1, j+len(u_ref)] = 1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1] = 0


            #OPT weights
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                H = np.zeros((len(u_ref) + len(unsafe_list), len(u_ref) + len(unsafe_list)))
                ff = 1.5*np.ones((len(u_ref) + len(unsafe_list), 1))
            else:
                H = np.zeros((len(u_ref), len(u_ref)))
                ff = 1.5*np.ones((len(u_ref), 1))

            H[0,0] = self.w_reg
            H[1,1] = self.w_reg
            ff[0] = -2*self.w_reg*u_ref[0]
            ff[1] = -2*self.w_reg*u_ref[1]


            try:
                uq = cvxopt_solve_qp(H, ff, A, b)
            except ValueError:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('Domain Error in cvx')

            if uq is None:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('infeasible QP')
            

            return uq[0:2]

class controller_map3:
        """
        CBF-based controller
        """

        def __init__(self, unsafe_info, map_info, w_reg = 10000, w_ref =1.5, gamma_stat = 0.7, gamma_dyn = 2):
            self.w_reg = w_reg 
            self.w_ref = w_ref
            self.gamma_stat = gamma_stat
            self.gamma_dyn = gamma_dyn
            self.unsafe_info = unsafe_info
            self.map_info = map_info

        def simple_control(self, u_ref, x_robot, x_goal, k_v = 1.5, k_w = 1):
            dist_2_goal = np.linalg.norm(x_robot[0:2]-x_goal[0:2])
            u = np.array([0.0,0.0])
            u[0] = np.exp(-1/(k_v*dist_2_goal))*u_ref[0]
            u[1] = k_w*u_ref[1]
            return u

        def control(self, env_bounds, robot, u_ref, x_robot, unsafe_list, vacinity_r = 4):
            ###############
            # thing to add in future: 1. pass the list of unsafe areas each element [x,y,radius]
            # here we only assume one human is moving 

            map_info_temp = type('', (), {})()  #this constraint is modified based on the location of robot

            if x_robot[0] < env_bounds.x_min_park and x_robot[0] > env_bounds.x_min_map:
                map_info_temp.CBF = self.map_info.CBF[0:2]
                map_info_temp.Lfh = self.map_info.Lfh[0:2]
                map_info_temp.Lgh = self.map_info.Lgh[0:2]
            elif (x_robot[0] < env_bounds.x_max_park and x_robot[0] > env_bounds.x_min_park):
                if (x_robot[1] < env_bounds.y_max_map and x_robot[1] > env_bounds.y_min_map):
                    map_info_temp.CBF = self.map_info.CBF[4:6]
                    map_info_temp.Lfh = self.map_info.Lfh[4:6]
                    map_info_temp.Lgh = self.map_info.Lgh[4:6]
                if (x_robot[1] < env_bounds.y_min_map and x_robot[1] > env_bounds.y_min_park):
                    map_info_temp.CBF = self.map_info.CBF[2:4]
                    map_info_temp.Lfh = self.map_info.Lfh[2:4]
                    map_info_temp.Lgh = self.map_info.Lgh[2:4]
            else:
                map_info_temp.CBF = self.map_info.CBF[0:2]
                map_info_temp.Lfh = self.map_info.Lfh[0:2]
                map_info_temp.Lgh = self.map_info.Lgh[0:2]

            
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list), len(u_ref) + len(unsafe_list)))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list),1))
            else: 
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref), len(u_ref)))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref),1))

            
            #wall constraints [-Lgh(x) 0]*[u b]^T < Lfh(x) + gamma*h(x)
            for j in range(len(map_info_temp.CBF)):
                A[j,0:len(u_ref)] = -map_info_temp.Lgh[j](x_robot)
                b[j] = map_info_temp.Lfh[j](x_robot) + self.gamma_stat*map_info_temp.CBF[j](x_robot)
            
            #Control Linear Constraints
            A[len(map_info_temp.CBF),0] = 1.
            b[len(map_info_temp.CBF)] = robot.v_max
            A[len(map_info_temp.CBF)+1,0] =- 1.
            b[len(map_info_temp.CBF)+1] = -robot.v_min
            A[len(map_info_temp.CBF)+2,1] = 1.
            b[len(map_info_temp.CBF)+2] = robot.w_max
            A[len(map_info_temp.CBF)+3,1] = -1.
            b[len(map_info_temp.CBF)+3] = -robot.w_min

            #unsafe_list constraints [-Lgh(x) -1]*[u b]^T < Lfh(x) + gamma*h(x) and b<0
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                for j in range(len(unsafe_list)):

                    #CBF constraint
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, 0:2] = -self.unsafe_info.Lgh(x_robot,unsafe_list[j])
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, j+len(u_ref)] = -1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j] = self.unsafe_info.Lfh(x_robot,unsafe_list[j]) + self.gamma_dyn*self.unsafe_info.CBF(x_robot,unsafe_list[j])
                
                    # Constraints on bi<0
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1, j+len(u_ref)] = 1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1] = 0


            #OPT weights
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                H = np.zeros((len(u_ref) + len(unsafe_list), len(u_ref) + len(unsafe_list)))
                ff = 1.5*np.ones((len(u_ref) + len(unsafe_list), 1))
            else:
                H = np.zeros((len(u_ref), len(u_ref)))
                ff = 1.5*np.ones((len(u_ref), 1))

            H[0,0] = self.w_reg
            H[1,1] = self.w_reg
            ff[0] = -2*self.w_reg*u_ref[0]
            ff[1] = -2*self.w_reg*u_ref[1]


            try:
                uq = cvxopt_solve_qp(H, ff, A, b)
            except ValueError:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('Domain Error in cvx')

            if uq is None:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('infeasible QP')
            

            return uq[0:2]

class controller_map4:
        """
        CBF-based controller
        """

        def __init__(self, unsafe_info, map_info, w_reg = 10000, w_ref =1.5, gamma_stat = 0.7, gamma_dyn = 2):
            self.w_reg = w_reg 
            self.w_ref = w_ref
            self.gamma_stat = gamma_stat
            self.gamma_dyn = gamma_dyn
            self.unsafe_info = unsafe_info
            self.map_info = map_info

        def simple_control(self, u_ref, x_robot, x_goal, k_v = 1.5, k_w = 1):
            dist_2_goal = np.linalg.norm(x_robot[0:2]-x_goal[0:2])
            u = np.array([0.0,0.0])
            u[0] = np.exp(-1/(k_v*dist_2_goal))*u_ref[0]
            u[1] = k_w*u_ref[1]
            return u

        def control(self, env_bounds, robot, u_ref, x_robot, unsafe_list, vacinity_r = 4):
            ###############
            # thing to add in future: 1. pass the list of unsafe areas each element [x,y,radius]
            # here we only assume one human is moving 

            map_info_temp = type('', (), {})()  #this constraint is modified based on the location of robot

            if x_robot[0] < env_bounds.x_mid and x_robot[0] > env_bounds.x_min:
                map_info_temp.CBF = self.map_info.CBF[0:3]
                map_info_temp.Lfh = self.map_info.Lfh[0:3]
                map_info_temp.Lgh = self.map_info.Lgh[0:3]
            elif x_robot[0] <= env_bounds.x_min and x_robot[0] > env_bounds.x_min_map:
                if x_robot[1] < env_bounds.y_min and x_robot[1] > env_bounds.y_min_map:
                    map_info_temp.CBF = self.map_info.CBF[1:3]
                    map_info_temp.Lfh = self.map_info.Lfh[1:3]
                    map_info_temp.Lgh = self.map_info.Lgh[1:3]
                elif x_robot[1] >= env_bounds.y_min and x_robot[1] < env_bounds.y_max:
                    map_info_temp.CBF = self.map_info.CBF[2:4]
                    map_info_temp.Lfh = self.map_info.Lfh[2:4]
                    map_info_temp.Lgh = self.map_info.Lgh[2:4]
                else:
                    map_info_temp.CBF = self.map_info.CBF[1:3]
                    map_info_temp.Lfh = self.map_info.Lfh[1:3]
                    map_info_temp.Lgh = self.map_info.Lgh[1:3]
            elif x_robot[0] <= env_bounds.x_max and x_robot[0] > env_bounds.x_mid:
                if x_robot[1] < env_bounds.y_min and x_robot[1] > env_bounds.y_min_map:
                    map_info_temp.CBF = [self.map_info.CBF[5]] + [self.map_info.CBF[2]]
                    map_info_temp.Lfh = [self.map_info.Lfh[5]] + [self.map_info.Lfh[2]]
                    map_info_temp.Lgh = [self.map_info.Lgh[5]] + [self.map_info.Lgh[2]]
                elif x_robot[1] >= env_bounds.y_min and x_robot[1] < env_bounds.y_max:
                    map_info_temp.CBF = [self.map_info.CBF[4]] + [self.map_info.CBF[2]]
                    map_info_temp.Lfh = [self.map_info.Lfh[4]] + [self.map_info.Lfh[2]]
                    map_info_temp.Lgh = [self.map_info.Lgh[4]] + [self.map_info.Lgh[2]]
                else:
                    map_info_temp.CBF = self.map_info.CBF[1:3]
                    map_info_temp.Lfh = self.map_info.Lfh[1:3]
                    map_info_temp.Lgh = self.map_info.Lgh[1:3]
            else:
                map_info_temp.CBF = [self.map_info.CBF[1]] + [self.map_info.CBF[6]]
                map_info_temp.Lfh = [self.map_info.Lfh[1]] + [self.map_info.Lfh[6]]
                map_info_temp.Lgh = [self.map_info.Lgh[1]] + [self.map_info.Lgh[6]]

            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list), len(u_ref) + len(unsafe_list)))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list),1))
            else: 
                A = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref), len(u_ref)))
                b = np.zeros((len(map_info_temp.CBF) + 2*len(u_ref),1))

            #wall constraints [-Lgh(x) 0]*[u b]^T < Lfh(x) + gamma*h(x)
            for j in range(len(map_info_temp.CBF)):
                A[j,0:len(u_ref)] = -map_info_temp.Lgh[j](x_robot)
                b[j] = map_info_temp.Lfh[j](x_robot) + self.gamma_stat*map_info_temp.CBF[j](x_robot)
            
            #Control Linear Constraints
            A[len(map_info_temp.CBF),0] = 1.
            b[len(map_info_temp.CBF)] = robot.v_max
            A[len(map_info_temp.CBF)+1,0] =- 1.
            b[len(map_info_temp.CBF)+1] = -robot.v_min
            A[len(map_info_temp.CBF)+2,1] = 1.
            b[len(map_info_temp.CBF)+2] = robot.w_max
            A[len(map_info_temp.CBF)+3,1] = -1.
            b[len(map_info_temp.CBF)+3] = -robot.w_min

            #unsafe_list constraints [-Lgh(x) -1]*[u b]^T < Lfh(x) + gamma*h(x) and b<0
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                for j in range(len(unsafe_list)):

                    #CBF constraint
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, 0:2] = -self.unsafe_info.Lgh(x_robot,unsafe_list[j])
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, j+len(u_ref)] = -1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j] = self.unsafe_info.Lfh(x_robot,unsafe_list[j]) + self.gamma_dyn*self.unsafe_info.CBF(x_robot,unsafe_list[j])
                
                    # Constraints on bi<0
                    A[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1, j+len(u_ref)] = 1
                    b[len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1] = 0


            #OPT weights
            if np.linalg.norm(x_robot[0:2]-unsafe_list[0][0:2]) < vacinity_r:
                H = np.zeros((len(u_ref) + len(unsafe_list), len(u_ref) + len(unsafe_list)))
                ff = 1.5*np.ones((len(u_ref) + len(unsafe_list), 1))
            else:
                H = np.zeros((len(u_ref), len(u_ref)))
                ff = 1.5*np.ones((len(u_ref), 1))

            H[0,0] = self.w_reg
            H[1,1] = self.w_reg
            ff[0] = -2*self.w_reg*u_ref[0]
            ff[1] = -2*self.w_reg*u_ref[1]


            try:
                uq = cvxopt_solve_qp(H, ff, A, b)
            except ValueError:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('Domain Error in cvx')

            if uq is None:
                uq = np.zeros(len(u_ref) + len(unsafe_list))
                #print('infeasible QP')
            

            return uq[0:2]

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 100
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))