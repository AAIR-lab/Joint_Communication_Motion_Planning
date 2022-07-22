import tkinter
from tkinter import *
import tkinter.font as tkFont
import pickle as pk
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import os.path
from os import path
from matplotlib.pyplot import figure
from datetime import datetime
import os
import numpy as np


node_index = 0
scenario_index = 0
tree_size = 0
debug = None
log_size = None
flag = False


root = tkinter.Tk()
root.geometry("310x380")
root.resizable(width=0, height=0)


root.wm_title("Replay")


output = tkinter.Text(root, height = 10, width = 35, bg = "light cyan")

map_name= tkinter.StringVar()

debug =  [[]]
log_size = len (debug)

def sep_command (string):
    res = []
    temp = ''
    for i in string:
        if (i != '/'): temp = temp + i
        else:
            res.append(temp)
            temp = ''
    res.append (temp)
    return res

def load_data():
    global debug
    global log_size
    global tree_size
    global branch_label
    global node_index
    global scenario_index
    global scenario_label
    global output
    global flag
    global data

    fontTitle = tkFont.Font(family="Arial", size=9)
    output.configure(font=fontTitle)
    scenario_index = 0
    node_index = 0
    file_name = map_name.get()
    map_name.set("")
    output.delete(1.0,END)
    if path.exists(file_name):
        res = sep_command (file_name)
        root_dir = os.getcwd()
        s = len (res)
        local_path = ''
        for ii in range (0,s-1):
            local_path = local_path + res[ii] + '\\'
        directory = root_dir + '\\' + local_path + res[-1]
        data = pk.load( open( directory, "rb" ) )
        debug = data[0]
        log_size = len (debug)
        tree_size = len (debug[scenario_index])
        root.wm_title("Replay (" + file_name + ")")
        branch_label = tkinter.Label (root, text = str(node_index + 1)+ " / " + str (tree_size))
        branch_label.grid (row = 2, column = 1)
        scenario_label = tkinter.Label (root, text = str(scenario_index + 1) + ' / ' + str(log_size))
        scenario_label.grid (row = 3, column = 1)
        output.insert(END, "The file successfully loaded!" + '\n'   )
        flag = True
    else: 
        output.insert(END, "The file not found!" + '\n'   )
        debug = [[]]
        log_size = len (debug)
        tree_size = 0
        root.wm_title("Replay")
        branch_label = tkinter.Label (root, text = str(node_index + 1)+ " / " + str (tree_size))
        branch_label.grid (row = 2, column = 1)
        scenario_label = tkinter.Label (root, text = str(scenario_index + 1) + ' / ' + str(log_size))
        scenario_label.grid (row = 3, column = 1)
        flag = False



def drawBranch ():
    global node_index
    global scenario_index
    global flag
    global output
    fontTitle = tkFont.Font(family="Arial", size=9)
    output.configure(font=fontTitle)
    if flag:
        maze = debug[scenario_index][node_index].maze
        cmap = colors.ListedColormap(['white', 'black','#e75a0e', '#0ee34a', '#08802a', 'gray', '#003cff', '#002291', '#92c4f0', '#f0a5a5', '#f56464' ])
        bounds = [-0.5,0.5,1.5,2.5,3.5,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        #plt.figure(figsize=(12, 12), dpi=80)
        #x,y,z = prep_mesh(maze)

        #ax.pcolor(x, y, maze, cmap=cmap)

        ax.imshow(maze, cmap=cmap, norm=norm)
        # draw gridlines
        plt.title('Scenario: ' + str(scenario_index + 1) + '   Branch: ' + str(node_index + 1) + '   Signal: ' + 
              debug[scenario_index][node_index].signal + '   Total_cost: ' + 
              str(round(debug[scenario_index][node_index].cost_total, 3)) + '   Cost H: ' + str (round(debug[scenario_index][node_index].cost_human, 3))
              + '   Cost R: ' + str(round(debug[scenario_index][node_index].cost_robot, 3)) + '   Cost P: ' 
              + str( round(debug[scenario_index][node_index].cost_path, 3))
              + '   Cost C: ' 
              + str( round(debug[scenario_index][node_index].cost_com, 3))
              + '   Cost He: ' 
              + str( round(debug[scenario_index][node_index].cost_heading, 3)), fontsize =6)
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=2)
        #ax.set_xticks(np.arange(0, self._x_len+2,2))
        #ax.set_yticks(np.arange(0, self._y_len+2,2))
        ax.invert_yaxis()
        plt.show()
    else:
        output.delete(1.0,END)
        output.insert(END, "No file is loaded!" + '\n'   )


def infoBranch ():
    global output
    global tree_size
    global flag
    global debug
    output.delete(1.0,END)

    fontTitle = tkFont.Font(family="Arial", size=9)
    output.configure(font=fontTitle)
    if flag:
        output.insert(END, "Branch Info" + '\n'   )
        output.insert (END, '\n'  )
        output.insert (END, "Branch: " + "<"+ str(debug[scenario_index][node_index].plan_grid) + "," + debug[scenario_index][node_index].signal + ">" +'\n'  )
        if (debug[scenario_index][node_index].chosen == True):
            output.insert(END, "**CHOSEN BRANCH**" + '\n'   )
        output.insert (END, '\n'  )
        output.insert(END, "human cost-to-goal: " + str( round (debug[scenario_index][node_index].cost_human, 4 ) )  + '\n'   )
        output.insert(END, "robot cost-to-goal: " + str( round (debug[scenario_index][node_index].cost_robot, 4 ) )  + '\n'   )
        output.insert(END, "path conflict cost: " + str( round (debug[scenario_index][node_index].cost_path, 6 ) )  + '\n'   )
        output.insert(END, "communication cost: " + str( round (debug[scenario_index][node_index].cost_com, 4 ) )  + '\n'   )
        output.insert(END, "heaidng cost: " + str( round (debug[scenario_index][node_index].cost_heading, 4 ) )  + '\n'   )

        #output.insert(END, "lying cost: " + str( round (debug[scenario_index][node_index].cost_lie, 4 ) )  + '\n'   )
        output.insert(END, "branch total cost: " + str( round (debug[scenario_index][node_index].cost_total, 4 ) )  + '\n'   )
        output.insert (END, '\n'  )
        output.insert(END, "human belief: " + str( debug[scenario_index][node_index].belief_g )  + '\n'   )
        output.insert(END, "human grid location: " + str( debug[scenario_index][node_index].human_loc_grid )  + '\n'   )
        output.insert(END, "robot grid location: " + str( debug[scenario_index][node_index].robot_loc_grid )  + '\n'   )
        output.insert (END, '\n'  )
        output.insert(END, "robot synch path: " + str( debug[scenario_index][node_index].robot_synch_path )  + '\n'   )
        output.insert (END, '\n'  )
        output.insert(END, "human_synch_path: " + str( debug[scenario_index][node_index].human_synch_path )  + '\n'   )
        output.insert (END, '\n'  )
    else:
        output.delete(1.0,END)
        output.insert(END, "No file is loaded!" + '\n'   )

def infoTree ():
    global output
    global tree_size
    global flag
    global data
    output.delete(1.0,END)

    fontTitle = tkFont.Font(family="Arial", size=9)
    output.configure(font=fontTitle)
    if flag:
        output.insert(END, "Experiment Info" + '\n'   )
        output.insert (END, '\n'  )
        metrics = data[3]
        output.insert (END, "robot's dist: " + str(round(metrics[0],3)) +'\n'  )
        output.insert (END, "human's dist: " + str(round(metrics[1],3)) +'\n'  )
        output.insert (END, "planning cycles:  " + str(metrics[2]) + '\n')
        output.insert (END, "avg unsafe:  " + str(metrics[3][1]) + '\n')
        if (len(metrics[3]) == 3): output.insert (END, "min unsafe:  " + str(metrics[3][2]) + '\n') 
        output.insert (END, "unsafe:  " + str(metrics[3][0]) + '\n')
        output.insert (END, '\n'  )
    else:
        output.delete(1.0,END)
        output.insert(END, "No file is loaded!" + '\n'   )

def optBranch ():
    global branch_label
    global node_index
    global tree_size
    global scenario_index
    tree_size = len(debug[scenario_index])

    min_index = 0
    min_cost = math.inf
    for i in range (0, tree_size):
        if (debug[scenario_index][i].cost_total < min_cost ):
            min_index = i
            min_cost = debug[scenario_index][i].cost_total

    branch_label.destroy()
    node_index = min_index
    branch_label = tkinter.Label (root, text = str(node_index + 1) + " / " + str (tree_size))
    branch_label.grid (row = 2, column = 1)
    infoBranch()

def nextBranch ():
    global branch_label
    global node_index
    global tree_size
    global scenario_index
    tree_size = len(debug[scenario_index])
    if (node_index < tree_size -1):
        branch_label.destroy()
        node_index = node_index +1
        branch_label = tkinter.Label (root, text = str(node_index + 1) + " / " + str (tree_size))
        branch_label.grid (row = 2, column = 1)


def previousBranch ():
    global branch_label
    global node_index
    global tree_size
    global scenario_index
    if (node_index != 0):
        tree_size = len (debug[scenario_index])
        branch_label.destroy()
        node_index = node_index - 1
        branch_label = tkinter.Label (root, text = str(node_index + 1)+ " / " + str (tree_size))
        branch_label.grid (row = 2, column = 1)


def nextScenario ():
    global scenario_index
    global scenario_label
    global log_size
    if (scenario_index < log_size-1):
        scenario_label.destroy()
        scenario_index = scenario_index + 1
        scenario_label = tkinter.Label (root, text = str(scenario_index + 1) + ' / ' + str (log_size))
        scenario_label.grid (row = 3, column = 1)



def previousScenario ():
    global scenario_index 
    global scenario_label
    global log_size
    if (scenario_index != 0):
        scenario_label.destroy()
        scenario_index = scenario_index - 1
        scenario_label = tkinter.Label (root, text = str(scenario_index + 1) + ' / ' + str(log_size))
        scenario_label.grid (row = 3, column = 1)

def show_overall():
    global flag
    global output
    global data

    output.delete(1.0,END)
    fontTitle = tkFont.Font(family="Arial", size=9)
    output.configure(font=fontTitle)
    if flag:
        infoTree()
        figure(figsize=(6, 6), dpi=80)
        walls = data[1][0]
        waypoints = data[2]
        colors = [
                '#08F7FE',  # teal/cyan
                '#FE53BB',  # pink
                '#F5D300',  # yellow
                '#00ff41', # matrix green
                  ]
        #if (len(data) == 6):
        node_lists = data[5]
        for log in node_lists:
            for node in log:
                if node.parent:
                    x0, x1, x2, x3 = zip(*node.path_x)
                    plt.plot(x0, x1, linewidth = 0.7, color = '#cdcdcd')

        robot_x, robot_y = get_way_data(waypoints [0])
        human_x, human_y = get_way_data(waypoints [1])

        plt.scatter(human_x[-1], human_y[-1], color = '#e75a0e', s = 100, marker = '*')
        plt.scatter(robot_x[-1], robot_y[-1], color = '#135ff7', s = 100, marker = '*')

        plt.plot(robot_x, robot_y, color = '#135ff7', linewidth = 3, label = 'Robot')
        plt.plot(human_x, human_y, color = '#e75a0e', linewidth = 3, label = 'Human')


        
        

        for wall in walls:
            plt.plot(wall[0:2,0],wall[0:2,1], color = '#383941')
        signal_robot = data[4][0]
        human_signal = data[4][1]
        pc = 0
        for point in signal_robot:
            plt.text(point[0], point[1]+0.1, str(pc+1) + ': ' + str(point[2]), style = 'italic', fontsize = 9, color = "#fc38f2", label = 'Signal')   
            plt.scatter(point[0], point[1],color = "#135ff7")
            pc = pc + 1
        
        pc = 0
        for point in human_signal:
            plt.text(point[0], point[1]+0.1, str(pc+1), style = 'italic', fontsize = 9, color = "#e3221b", label = 'Signal')   
            plt.scatter(point[0], point[1],color = "#e75a0e")
            pc = pc + 1
        


        metrics = data[3]
        
        plt.legend(prop={'size': 8})
        plt.title("R dist: " + str(round(metrics[0],3)) + "   H dist: " + str(round(metrics[1],3)) + "   PC: " + str(metrics[2]) + 
                  "   avg unsafe: " + str(round(metrics[3][1],3)), fontsize =8)
        plt.show()
    else:
        output.delete(1.0,END)
        output.insert(END, "No file is loaded!" + '\n'   )

def get_way_data (waypoints):
    way_x, way_y = [], []
    for i in range (len (waypoints)):
        way_x.append(waypoints[i][0])
        way_y.append(waypoints[i][1])
    return way_x, way_y

def save_figure ():
    global flag
    global output

    fontTitle = tkFont.Font(family="Arial", size=9)
    output.configure(font=fontTitle)
    if flag:
        now = datetime.now()
        filename = now.strftime('%Y_%m_%d(%H_%M_%S).png')
        plt.savefig(filename)
    else:
        output.delete(1.0,END)
        output.insert(END, "No file is loaded!" + '\n'   )

def prep_mesh (maze):

    s = maze.shape
    xi = np.arange(0, s[1])
    yi = np.arange(0, s[0])
    X, Y = np.meshgrid(xi, yi)
    
    return X,Y,maze

show = tkinter.Button (master = root, text = "show branch", command = drawBranch, height = 5, width = 20)
info = tkinter.Button (master = root, text = "branch info", command = infoBranch, height = 5, width = 10)
optimal = tkinter.Button (master = root, text = "opt branch", command = optBranch, height = 5, width = 10)

next_branch = tkinter.Button (master = root, text = " Branch >>", command = nextBranch, height = 1, width = 10)
previous_branch = tkinter.Button (master = root, text = "<< Branch", command = previousBranch, height = 1, width = 10)
next_scenario = tkinter.Button (master = root, text = " Scenario >>", command = nextScenario, height = 1, width = 10)
previous_scenario = tkinter.Button (master = root, text = "<< Scenario", command = previousScenario, height = 1, width = 10)

tree_size = len (debug[scenario_index])
branch_label = tkinter.Label (root, text = str(node_index + 1) + ' / '+ str(tree_size))
scenario_label = tkinter.Label (root, text = str(scenario_index + 1) + ' / ' + str (log_size))

overview = tkinter.Button(root,text = 'overview', command = show_overall)
save = tkinter.Button (master = root, text = "save", command = save_figure)


name_label = tkinter.Label(root, text = 'map name')
name_entry = tkinter.Entry(root,textvariable = map_name)
load_btn = tkinter.Button(root,text = 'Load', command = load_data)

show.grid (row = 0, column = 1)
info.grid (row = 0, column = 2)
optimal.grid (row = 0, column = 0)

output.grid (row = 1, column = 0, columnspan = 3)

next_branch.grid (row = 2, column = 2)
previous_branch.grid (row = 2, column = 0)
branch_label.grid (row = 2, column = 1)

next_scenario.grid (row = 3, column = 2)
previous_scenario.grid (row = 3, column = 0)
scenario_label.grid (row = 3, column = 1)

overview.grid(row=4,column=1)
save.grid(row=4,column=2)


name_label.grid(row=5,column=0)
name_entry.grid(row=5,column=1)
load_btn.grid(row=6,column=1)


tkinter.mainloop()







     


  



