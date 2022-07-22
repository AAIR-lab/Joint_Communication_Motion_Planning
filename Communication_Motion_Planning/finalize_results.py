import pickle as pk
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import os.path
from os import path
from matplotlib.pyplot import figure
from datetime import datetime
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from datetime import datetime
import numpy as np
import math

class Results ():
    def __init__(self, number_experiments = 10, number_experiments_w =3):
        self._n_exp = number_experiments # number of experiments for each map
        self._map1 = self.load('map1') 
        self._map2 = self.load('map2')
        self._map3 = self.load('map3')
        self._map4 = None #self.load('map4')
        self._weights_map1 = None
        self._weights_map2 = None
        self._weights_map3 = None
        self._weight_exp_number = number_experiments_w
        self._weights = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]
        self._optimal_distances = {
                                   "map1": [4.26, 6.17],
                                   "map2": [6.36,7.14],
                                   "map3": [10.79, 7.19]
                                                }
        self._map_names = {
                                   "map1": "Intersection",
                                   "map2": "Basic",
                                   "map3": "Hallway"
                                                }

    def weight_study(self, map_name, n_exp):
        self._weight_exp_number = n_exp
        data = []
        size_weights = len(self._weights)
        for w in self._weights:
            temp = []
            for e in range (self._weight_exp_number):
                file_name = map_name + '_w' + str(w) + '_' + str(e)
                root_dir = os.getcwd()
                directory = root_dir + '\\' + 'weights' + '\\' + map_name  + '\\' + file_name
                d = pk.load( open( directory, "rb" ) )
                temp.append(d)
            data.append(temp)
        rd, hd, rt, ht, rv, hv, rv, hv, p = [], [], [], [], [], [], [], [], []
        we = self._weights

        for w in range (size_weights):
            rd_sum, hd_sum, p_sum, rt_sum, ht_sum, rv_sum, hv_sum = 0, 0, 0, 0, 0, 0, 0
            for e in range (self._weight_exp_number):
                rd_sum = rd_sum + data[w][e][3][0]
                hd_sum = hd_sum + data[w][e][3][1]
                rt_sum = rt_sum + data[w][e][3][4][0]
                ht_sum = ht_sum + data[w][e][3][4][1]
                rv_sum = rv_sum + self._optimal_distances[map_name][0] / data[w][e][3][4][0]
                hv_sum = hv_sum + self._optimal_distances[map_name][1] / data[w][e][3][4][1]
                p_sum = p_sum + data[w][e][3][3][1]
            rd.append (  self._optimal_distances[map_name][0] / (rd_sum/self._weight_exp_number) )
            hd.append ( self._optimal_distances[map_name][1] / (hd_sum/self._weight_exp_number))
            rt.append (  rt_sum/self._weight_exp_number )
            ht.append ( ht_sum/self._weight_exp_number)
            rv.append (  rv_sum/self._weight_exp_number )
            hv.append ( hv_sum/self._weight_exp_number)
            p.append (p_sum/self._weight_exp_number)
        
        x = range(len(we))
        ls = 'solid'
        #plt.figure(figsize=(10, 6), dpi=80)
        fig, ax1 = plt.subplots()
        
        #ax1.plot (x, p, ls='dashed', color='#a0a0a0',label = 'Proximity Cost' )
        ax2 = ax1.twinx()
        ax2.fill_between(x, p, 0, color = '#25d89c', alpha = 0.1)
        
        ax1.set_ylabel('Normalized Speed [m/s]', fontsize = 17)
        ax2.set_ylabel('Proximity Cost', fontsize = 17)
        x_tick = range(0,len(we),2)
        we_tick = []
        for i in range(0,len(we),2):
            we_tick.append(we[i])
        ax1.set_xlabel('X', fontsize = 17 )
        

        #ax1.xlabel(r'Ratio of  ($w_r/w_h$)',fontsize=14)
        #ax1.grid(color = '#e0e0e0', linestyle = (0, (5, 10)), linewidth = 1)

         

        ax1.plot (x,rv, ls='dashed', linewidth = 2, marker='o', ms=6, color='#5c63a2', mec='#2d304f', label = 'Robot Cost')  
        ax1.plot (x, hv, ls='dashed', linewidth = 2, marker='X', ms=6, color='#c068a9', mec='#5d5d5d',label = 'Human Cost' )
        #ax1.legend(prop={'size': 12})
        
        #ax1.set_yticklabels(ax1.get_yticklabels(), fontsize = 15 )
        #ax2.set_yticklabels(ax2.get_yticklabels(), fontsize = 15 )

        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        plt.title(self._map_names[map_name], fontsize = 15)
        plt.xticks(x_tick, we_tick)
        
        plt.show ()

    
    
    def load (self, map_name):
        data_on_R  = []
        data_on_H  = []
        data_off = []
        for i in range (self._n_exp):
            file_name = map_name + '/' + map_name + '_on_R_' + str(i + 1)
            res = self.sep_command (file_name)
            root_dir = os.getcwd()
            directory = root_dir + '\\' + res[0] + '\\' + res[1]
            data = pk.load( open( directory, "rb" ) )
            data_on_R.append(data)

            file_name = map_name + '/' + map_name + '_off_' + str(i + 1)
            res = self.sep_command (file_name)
            root_dir = os.getcwd()
            directory = root_dir + '\\' + res[0] + '\\' + res[1]
            data = pk.load( open( directory, "rb" ) )
            data_off.append(data)
        return [data_on_R, data_off]

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

    def get_way_data (self, waypoints):
        way_x, way_y = [], []
        for i in range (len (waypoints)):
            way_x.append(waypoints[i][0])
            way_y.append(waypoints[i][1])
        return way_x, way_y

    def show_overall(self, data):
        waypoints = data[2]
        robot_x, robot_y = self.get_way_data(waypoints [0])
        human_x, human_y = self.get_way_data(waypoints [1])
        plt.plot(robot_x, robot_y, color = '#7e12f9', linewidth = 3, label = 'Robot')
        plt.plot(human_x, human_y, color = '#e3221b', linewidth = 3, label = 'Human')


    def draw_collective (self, map_name, method, priority = None):
        figure(figsize=(6, 2), dpi=80)
        if method == 'on' and priority == 'R': method_index = 0
        elif method == 'on' and priority == 'H': method_index = 1
        elif method == 'off': method_index = 2

        if map_name == 'map1': data = self._map1[method_index]
        elif map_name == 'map2': data = self._map2[method_index]
        elif map_name == 'map3': data = self._map3[method_index]
        elif map_name == 'map4': data = self._map4[method_index]

        for d in data:
            self.show_overall(d)
       
        walls = d[1][0]
        for wall in walls:
            plt.plot(wall[0:2,0],wall[0:2,1], "k")

        plt.show()

    def export_to_excel (self):
        maps = ['map1','map2', 'map3']
        data = [self._map1, self._map2, self._map3]
        now = datetime.now()
        filename = 'results_' + now.strftime('%Y_%m_%d(%H_%M_%S).xlsx')
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')

        for i in range(len(maps)):
            rv_on_R = []
            hv_on_R = []
            pc_on_R = []
            p_on_R = []
            rd_on_R = []
            hd_on_R = []
            rv_off = []
            hv_off = []
            pc_off = []
            p_off = []
            rd_off = []
            hd_off = []
            data_current = data [i]
            data_on_R = data_current [0]
            data_off = data_current [1]
            for j in range (self._n_exp):
                metrics_on_R = data_on_R [j][3]
                metrics_off = data_off [j][3]

                r_opt_dist_on = data_on_R [j][6][0]
                h_opt_dist_on = data_on_R [j][6][1]
                
                rv_on_R.append( r_opt_dist_on / metrics_on_R [4][0] )
                hv_on_R.append( h_opt_dist_on / metrics_on_R [4][1] )
                pc_on_R.append(  metrics_on_R [2] )
                p_on_R.append(  metrics_on_R [3][1] )
                rd_on_R.append(  metrics_on_R [0] )
                hd_on_R.append(  metrics_on_R [1] )


                rv_off.append(  r_opt_dist_on /  metrics_off [4][0])
                hv_off.append(  h_opt_dist_on /  metrics_off [4][1])
                pc_off.append(  metrics_off [2] )
                p_off.append(  metrics_off [3][1] )
                rd_off.append(  metrics_off [0] )
                hd_off.append(  metrics_off [1] )


            d = {
            'RV on': rv_on_R,
            'HV on': hv_on_R, 
            'PC on': pc_on_R,
            'P on': p_on_R,
            'RD on': rd_on_R,
            'HD on': hd_on_R,
            
            'RV off': rv_off,
            'HV off': hv_off, 
            'PC off': pc_off,
            'P off': p_off,
            'RD off': rd_off,
            'HD off': hd_off
            }

            df= pd.DataFrame(d, columns = ['RV on', 'HV on', 'PC on', 'P on', 'RD on', 'HD on', 
                                           'RV off', 'HV off', 'PC off', 'P off', 'RD off', 'HD off',])

            df.to_excel (writer, sheet_name = maps[i] , index = False, header=True)
        writer.save()

def main():
    result = Results()
    #result.draw_collective ('map1', 'on','R')
    #result.draw_collective ('map1', 'on','H')
    #result.export_to_excel()
    result.weight_study('map3', 3)

if __name__ == '__main__':
    main()

