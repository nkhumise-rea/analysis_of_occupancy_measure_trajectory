#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 05 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import pandas as pd
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys
from collections import deque
import math

##urdf location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)


class measure():
    def __init__(self,states_size=[5,5]):
        self.algo = 'DQN'
        self.states_size = states_size

    def open_data_file(self): 
        weight_file = 'data/DQN_meta_0.npy'
        # weight_file = 'data/DQN_meta_1.npy'

        file_path = abspath(join(this_dir,weight_file))

        data = np.load(file_path, allow_pickle=True)

        # print('con_eps: ', data[2])
        # print('con_step: ', data[3])

        # print('states: \n', data[0])
        # print('actions: \n', data[1])
        # print('returns: \n', data[3])
        # print('num_states: \n', data[4])
        # print('num_actions: \n', data[5])

        return data

    # co-prime_checking_function
    def coprime_check(self,data):
        for i in range(len(data)-1):
            if math.gcd(data[i],data[i+1]) == 1:
                return True
        return False

    # episode_perturbation
    def perturbation(self,states,actions):
        eps = 0.1 #probability_of_null_state_insertion
        if np.random.rand() < eps:
            e_states = [None] + states 
            e_actions = [None] + actions 
        else:
            e_states = states 
            e_actions = actions 
    
        # print('states: ', states)
        # print('actions: ', actions)
        # print('e_states: ', e_states)
        # print('e_actions: ', e_actions)
        return e_states,e_actions

    # occupancy_measure
    def occupancy_measure(self,data):
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        states_dict,actions_dict = {},{}
        long_states_dict,long_actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout

            states,actions = self.perturbation(states,actions) #perturb_every_episode_probabilistic
            states_dict[i], actions_dict[i] = states, actions

        # create_long_episodes(length > 220)
        long_actions, long_states = [],[]
        i = 0 
        for k in states_dict.keys(): #for k in actions_dict.keys():
            # print('k: ',k)
            long_actions += actions_dict[k]
            long_states += states_dict[k]

            if len(long_actions) > 220: #min num_decisions_per_long-episode
                # print('==========================')
                long_actions_dict[i] = long_actions
                long_states_dict[i] = long_states
                long_actions, long_states = [],[]
                i += 1

        # print('long_actions: \n', long_actions_dict)
        # print('long_states: \n', long_states_dict)
        # print('long_actions: \n', len(long_actions_dict))
        # print('long_states: \n', len(long_states_dict))
        # xxx

        time_counts = defaultdict(lambda: defaultdict(int))
        total_counts_at_time =  defaultdict(int)
        probs = defaultdict(lambda: defaultdict(int)) #probabilities
        occup = defaultdict(int) #occupancy_measure

       # iterate_each_long-episode_rollout
        for k in long_actions_dict.keys():
            states = long_states_dict[k] #states_per_long-episode_rollout
            actions = long_actions_dict[k] #actions_per_long-episode_rollout
            
            # iterate_over_state-action pairs
            for t,j in enumerate(zip(states,actions)):
                time_counts[t][j] += 1
                total_counts_at_time[t] += 1
        # print('time_counts: ', time_counts)
            
        #calculate_probability_distribution
        for t in time_counts:
            for item in time_counts[t]:
                probs[t][item] = time_counts[t][item]/total_counts_at_time[t]
        # print('probs: ', probs)

        gamma = 0.99 #discount factor
        xy_space = [] #state_action_space
        y_space = ['up', 'right', 'down', 'left'] #action_space
        for i in range(self.states_size[0]):
            for j in range(self.states_size[1]):
               for k in y_space:
                   xy_space.append(((i,j),k))
                   for t in range(len(probs)):
                        occup[((i,j),k)] += (gamma**t)*probs[t][((i,j),k)]   
        
        #normalize
        normalizer = sum(occup.values())
        for k in occup.keys():
            occup[k] /= normalizer

        print('occupancy: \n', occup)
        return occup

        

#Execution
if __name__ == '__main__':
    n = 5
    ocp = measure(states_size=[n,n])
    data = ocp.open_data_file()
    ocp.occupancy_measure(data)