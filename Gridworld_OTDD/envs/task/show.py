#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 2023
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from os.path import dirname, abspath, join
import sys

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)

"""
Module to illustate state trajectory in the 2D gridworld
"""

class show_grid():
    def __init__(self, input=[(0, 0), (1, 0)], policy=0, setting=1):
        self.policy = policy
        self.input = input
        rew_setting = ['sps','dns']
        self.setting = rew_setting[setting]

    #display state trajectories 
    def state_traj(self,size=[3,3],show=1): #old
        n_input = np.asarray(self.input)
        u = np.unique(n_input, axis=0) #unique state value

        n_size = size[0]
        data = np.arange(1,n_size**2+1).reshape(n_size,n_size) 

        for i in range(len(u)):
            data[u[i][0],u[i][1]] = 0.5    

        #create discrete colormap
        cmap = colors.ListedColormap(['salmon','white'])
        bounds = [0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #draw gridlines
        plt.grid(axis='both',
                color='k',
                linewidth=2
                )
        
        labels=[str(x) for x in range(n_size+1)]
        plt.xticks(np.arange(-.5,n_size,1) + 1, labels=labels)
        plt.yticks(np.arange(-.5,n_size,1) + 1, labels=labels)

        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title(f'Policy_{self.policy} state-trajectory')

        plt.tight_layout()
        if show == 1:
            plt.show()
        else:
            file_name = f"{self.setting}/pi_{self.policy}_SJ.png"
            location = abspath(join(this_dir,'..', '..','evals','stt_traj',file_name))
            plt.savefig(location)
        plt.close()

        return

    #display state trajectories 
    def s_state_traj(self,size=[3,3],show=1): #new
        n_input = np.asarray(self.input)
        u = np.unique(n_input, axis=0) #unique state value

        n_size = size[0]
        data = np.arange(1,n_size**2+1).reshape(n_size,n_size) 

        for i in range(len(u)):
            data[u[i][0],u[i][1]] = 0.5    

        #create discrete colormap
        cmap = colors.ListedColormap(['salmon','white'])
        bounds = [0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #draw gridlines
        plt.grid(axis='both',
                color='k',
                linewidth=2
                )
        
        hm  = sns.heatmap(data=data,cmap=cmap)   
        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)           
        plt.title(f'Policy_{self.policy} state-trajectory')

        plt.tight_layout()
        if show == 1:
            plt.show()
        else:
            file_name = f"{self.setting}/pi_{self.policy}_SJ.png"
            location = abspath(join(this_dir,'..', '..','evals','stt_traj',file_name))
            plt.savefig(location)
        plt.close()

        return

