#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 9 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
import sys
import time
import random
from numpy import linalg as LA

"""
Deterministic 2D Gridworld of size NxN where start-state = (0,0) and goal-state = (N-1,N-1), 
Rewards can be dense or sparse
"""

class grid():
    def __init__(self, 
                 num_states=[3,3], 
                 goal_state=tuple([ [2,2] ]), 
                 start_state=tuple([0,0]),
                 rew_setting = 1, #[1:dense, 0:sparse]
                 ):
        
        #environment
        self.num_states = num_states
        self.initial_state = start_state 
        self.goal_state = goal_state[0]
        
        #actions
        self.actions = ['up', 'right', 'down', 'left'] # 0:up, 1:right, 2:down, 3:left
        self.num_actions = len(self.actions)

        #rewards
        self.reward_setting = rew_setting #[1:dense, 0:sparse]

        if self.reward_setting == 1: #dense
            self.terminal_state = self.goal_state
            vector = start_state - np.asarray(self.goal_state)
            self.rewards_dense = - LA.norm( vector, 1) #'cityblock': - np.sum(np.abs(vector))
            # self.rewards_dense = - LA.norm( start_state - np.asarray(self.goal_state) )**2
        else: #sparse
            self.rewards = np.full(self.num_states, -0.04)
            for i in goal_state:
                self.rewards[tuple(i)] = 1
    
    ## Sparse_rew_setting
    def is_terminal_state(self,current_state): #terminal_state check
        if self.rewards[tuple(current_state)] == 1: 
            return True
        return False   

    def start_state(self): #start state definition
        current_state =  tuple([0,0]) #np.random.randint(self.num_states, size=(2,))  #
        while self.is_terminal_state(current_state): #if current_state is terminal_state
            current_state = np.random.randint(self.num_states, size=(2,))
        return current_state[0], current_state[1] #current_row_index, current_col_index
    
    ## Dense_rew_setting
    def is_terminal_state_dense(self,current_state): #terminal_state check
        if np.all( np.asarray(current_state) == self.terminal_state ): 
            return True
        return False   

    def start_state_dense(self): #start state definition
        current_state =  tuple([0,0]) #np.random.randint(self.num_states, size=(2,))  #
        while self.is_terminal_state_dense(current_state): #if current_state is terminal_state
            current_state = np.random.randint(self.num_states, size=(2,))
        return current_state[0], current_state[1] #current_row_index, current_col_index
    
    def reset(self): #rest_environment
        self.step_id = 0
        if self.reward_setting == 1: 
            current_state = self.start_state_dense()
        else: 
            current_state = self.start_state()
        return current_state
    
    def reward(self, next_state): # get_rewards
        if self.reward_setting == 1: 
            vector = np.asarray(next_state) - np.asarray(self.goal_state)
            reward = - LA.norm( vector, 1) #'cityblock': - np.sum(np.abs(vector))
            # reward = - LA.norm( np.asarray(next_state) - np.asarray(self.goal_state) )**2
        else: 
            reward = self.rewards[tuple(next_state)]
        return reward
    
    def done(self, next_state): # get_done_status
        # print('step_id: ', self.step_id)
        # xxx

        if self.step_id == 15: #15,40,60,90
            done = True
            return done*1.0
        
        else:
            if self.reward_setting == 1: #dense_rew_setting
                done = self.is_terminal_state_dense(next_state)*1.0
            else: #sparse_rew_setting
                done = self.is_terminal_state(next_state)*1.0
                # print('done: ', done)
        
        return done 
    
    def step(self,current_state,action_index): #
        self.step_id += 1

        next_state = self.state_tracking(current_state, action_index) 
        done = self.done(next_state)
        reward = self.reward(next_state)

        return next_state, reward, done 

    # action_convertor    
    def action_val(self,action_index):
        #self.actions = ['up', 'right', 'down', 'left'] 
        ##action_conversion
        if self.actions[action_index] == 'up': action_val = np.array([-1,0])
        elif self.actions[action_index] == 'right': action_val = np.array([0,1])
        elif self.actions[action_index] == 'down': action_val = np.array([1,0])
        elif self.actions[action_index] == 'left': action_val = np.array([0,-1])
        return action_val

    # state_tracker
    def state_tracking(self,current_state, action_index):
        next_state = np.asarray(current_state) + self.action_val(action_index) #conversion: 1.0
        if next_state[0] < 0: 
            next_state[0] = 0 #left_edge
        elif next_state[0] > self.num_states[0] - 1: 
            next_state[0] = self.num_states[0] - 1 #right_edge
        elif next_state[1] < 0: 
            next_state[1] = 0 #up_edge
        elif next_state[1] > self.num_states[1] - 1: 
            next_state[1] = self.num_states[1] - 1 #down_edge
        return next_state[0], next_state[1]