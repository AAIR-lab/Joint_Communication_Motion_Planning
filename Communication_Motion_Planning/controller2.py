##########################
#
# This script relaxed the wall constraints with a slack variable
#
##########################


from scipy.integrate import odeint
from sympy import symbols, Matrix, Array, sin, cos, lambdify, exp, sqrt, log
import cvxopt as cvxopt
import numpy as np

class Controller:
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

        def control(self, env_bounds, robot, u_ref, x_robot, unsafe_list):
            ###############
            # thing to add in future: 1. pass the list of unsafe areas each element [x,y,radius]
            # here we only assume one human is moving 

            map_info_temp = type('', (), {})()  #this constraint is modified based on the location of robot

            if x_robot[0] < env_bounds.x_min or x_robot[0] > env_bounds.x_max:
                map_info_temp.CBF = self.map_info.CBF[3:5]
                map_info_temp.Lfh = self.map_info.Lfh[3:5]
                map_info_temp.Lgh = self.map_info.Lgh[3:5]
                A = np.zeros((2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list), len(u_ref) + len(unsafe_list) + len(map_info_temp.CBF)))
                b = np.zeros((2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list),1))              #zeros((len(self.map_info.CBF,1))?! 
            else:
                map_info_temp.CBF = self.map_info.CBF[0:4]
                map_info_temp.Lfh = self.map_info.Lfh[0:4]
                map_info_temp.Lgh = self.map_info.Lgh[0:4]
                A = np.zeros((2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list), len(u_ref) + len(unsafe_list) + len(map_info_temp.CBF)))
                b = np.zeros((2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*len(unsafe_list),1))

            
            #wall constraints [-Lgh(x) 0]*[u b]^T < Lfh(x) + gamma*h(x)
            for j in range(len(map_info_temp.CBF)):
                A[2*j,0:len(u_ref)] = -map_info_temp.Lgh[j](x_robot)
                A[2*j,j+len(u_ref)] = -1
                b[2*j] = map_info_temp.Lfh[j](x_robot) + self.gamma_stat*map_info_temp.CBF[j](x_robot)
                
                # Constraints on bi<0
                A[2*j+1, j + len(u_ref)] = 1
                b[2*j+1] = 0
            
            #Control Linear Constraints
            A[2*len(map_info_temp.CBF),0] = 1.
            b[2*len(map_info_temp.CBF)] = robot.v_max
            A[2*len(map_info_temp.CBF)+1,0] =- 1.
            b[2*len(map_info_temp.CBF)+1] = -robot.v_min
            A[2*len(map_info_temp.CBF)+2,1] = 1.
            b[2*len(map_info_temp.CBF)+2] = robot.w_max
            A[2*len(map_info_temp.CBF)+3,1] = -1.
            b[2*len(map_info_temp.CBF)+3] = -robot.w_min

            #unsafe_list constraints [-Lgh(x) -1]*[u b]^T < Lfh(x) + gamma*h(x) and b<0
            for j in range(len(unsafe_list)):

                #CBF constraint
                A[2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, 0:2] = -self.unsafe_info.Lgh(x_robot,unsafe_list[j])
                A[2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*j, j + len(u_ref) + len(map_info_temp.CBF)] = -1
                b[2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*j] = self.unsafe_info.Lfh(x_robot,unsafe_list[j]) + self.gamma_dyn*self.unsafe_info.CBF(x_robot,unsafe_list[j])
                
                # Constraints on bi<0
                A[2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1, j + len(u_ref) + len(map_info_temp.CBF)] = 1
                b[2*len(map_info_temp.CBF) + 2*len(u_ref) + 2*j+1] = 0





            #OPT weights
            H = np.zeros((len(u_ref) + len(unsafe_list) + len(map_info_temp.CBF), len(u_ref) + len(unsafe_list) + len(map_info_temp.CBF)))
            ff = 1.5*np.ones((len(u_ref) + len(unsafe_list) + len(map_info_temp.CBF), 1))

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