#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys
from collections import deque
from copy import copy

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.distributions import Categorical 

##urdf location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)
# from task.grid2D import grid
# from task.grid2DHard import grid
from task.grid2Dstoc import grid
from task.show import show_grid

class gridworld():
    def __init__(self, 
                 states_size=[5,5], 
                 rew_setting = 1, #[1:dense, 0:sparse]
                 n_eps = 300,
                 ):
        
        self.rew_setting = rew_setting
        self.state_size = states_size
        self.goal_state_xx = tuple([ list(np.asarray(self.state_size) - 1) ])

        rew_set = ['sps','dns'] #dense: dns, sparse: sps
        self.setting = rew_set[rew_setting] 

        self.env = grid( #training_env
            num_states=self.state_size,
            goal_state=self.goal_state_xx,
            rew_setting=self.rew_setting, # 1:dense | 0:sparse 
                    ) 
        self.num_actions = self.env.num_actions
        self.num_states = self.env.num_states
        self.total_num_states = self.num_states[0]*self.num_states[1]
        self.num_episodes = n_eps

    def update(self, state, action, reward, next_state):
        self.N[state[0],state[1], action] += 1 #update_visits
        self.R[state[0],state[1], action] += reward
        self.r_hat[state[0],state[1], action] = self.R[state[0],state[1], action]/self.N[state[0],state[1], action]
        self.B[state[0],state[1], action] = np.sqrt(2 * np.log(self.total_num_states*self.num_actions*self.t**2/self.delta) 
                                                    / self.N[state[0],state[1], action] )
        self.q_value[state[0],state[1], action] = self.r_hat[state[0],state[1], action] + self.B[state[0],state[1], action]  

    def main(self,
             test_frq=50, #policy_eval_frequency
             runs=1, #number_of_eval_repeats
             common_policy=0, #[1: yes | 0: no]
             delta=0.1,
             ):
        
        # start_time = time.time() #timekeeping: start
                
        #visits
        self.N = np.zeros((self.num_states[0],self.num_states[1],self.num_actions))

        #sum_of_rewards
        self.R = np.zeros((self.num_states[0],self.num_states[1],self.num_actions))
        
        #average_rewards
        self.r_hat = np.zeros((self.num_states[0],self.num_states[1],self.num_actions))

        #q_value table
        self.q_value = np.zeros((self.num_states[0],self.num_states[1],self.num_actions))
        
        #exploration_bonus
        self.B = np.zeros((self.num_states[0],self.num_states[1],self.num_actions))

        #hyperameters
        self.delta = delta #[low:explore, high:exploit]
        self.t = 0

        #data collection
        steps = 0
        score_col,steps_col = [],[]
        self.ext_eps = 0
        con_eps = self.num_episodes #0
        num_eval = 0

        # OTDD misc
        trajectory_dataset = None
        stop_learning = 0
        test_frq = test_frq #test frequecy
        self.dataset_dict = {} #policy_dataset_dictionary
        self.num_eval = 0 #policy_evaluation_tracker

        #stopping_criterion
        num_stop = 5
        stop_rew = deque(maxlen=num_stop) #50, 10
       
        for episode in range(self.num_episodes):
            
            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode #steps
                num_eval = self.num_eval
                self.ext_eps = 1 #switch
            stop_learning = 0 #reset_criterion_check

            done = False
            self.store_ret = 0
            score = 0
            state = self.env.reset()
            
            while not done:
                steps += 1
                self.t += 1
                action_index = self.select_action(state)                 
                next_state, reward, done = self.env.step(state,action_index)

                # Update the UCRL estimates
                self.update(state, action_index, reward, next_state)

                score += reward
                state = next_state 
                                
                # """                
                if steps%test_frq == 0: #reward_based_stop_criteria
                    trajectory_dataset = self.pol_dataset(runs) #no_runs
                    # steps_col.append(episode) #steps, episode
                    steps_col.append(steps)
                    stop_learning, done = self.stop_criterion(steps,stop_rew,done,episode)
                # """
                                        
                #collection of return per step 
                score_col.append(score) 
                # score_col.append(self.store_ret)

            # print('eps: ', episode)  
        # print('self.q_values: ', self.q_value)

        """        
        # print('num_states: ',num_states)
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        # print('stt: ', stt)
        print('trj: ', trj)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        #"""

        # self.plot_returns(score_col,con_eps)
        # self.plot_returns(score_col)
        # xxx
           
        # end_time = time.time() #timekeeping: end
        # duration = end_time - start_time #timekeeping: span
        # print('training_duration: \n {} seconds, \t {} minutes '.format(duration,duration/60))
        # print('last_episode: ', episode)
        # print('time_col: ', steps)

        # print(' ')
        # print('num_eval: ', num_eval)
        # print('con_eps: ', con_eps)
        # xxx
        return trajectory_dataset,steps_col,score_col,con_eps, num_eval-1
        # return self.running_avg(score_col) #f/ epsilon.py 
        # return self.rolling(score_col) #f/ epsilon.py

    def illustrate_success(self):
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        # print('stt: ', stt)
        print('trj: ', trj)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        return

    #stop_training_criterion
    def stop_criterion(self,steps,stop_rew,done, episode):
        # print('here')
        start_state = self.env.initial_state
        _,_,ret,_,_ = self.learnt_agent(start_state)
        ret = round(ret,5)
        stop_rew.append(ret)
        self.store_ret = ret

        #timing_to_stop_checking_criterion
        if not self.ext_eps: 

            if self.state_size == [5,5]:
                min_dense_ret = -28.0
            elif self.state_size == [4,4]:
                min_dense_ret = -16.0
            elif self.state_size == [15,15]:
                min_dense_ret = -378.0
            else: 
                min_dense_ret = -7.0 

            n_size = self.state_size[0]
            min_sparse_ret = -0.04*(2*(n_size - 1) - 1) + 1 
            min_sparse_ret = round(min_sparse_ret,5)
            
            if self.env.reward_setting == 1: 
                if (min(stop_rew)>=min_dense_ret): 
                    # print('stop_rew: ', stop_rew)
                    # print('final_step: ', steps) 
                    # self.illustrate_success() #illustrate_convergence
                    return 1, 1.0
            else:
                if (min(stop_rew)>=min_sparse_ret): 
                    # print('stop_rew: ', stop_rew)
                    # print('final_step: ', steps) 
                    return 1, 1.0
            return 0, done
        
        else:
            return 0, done

    #action_selection_mechanism        
    def select_action(self,state,evaluation_episode=False):
        action_index = self.randargmax(self.q_value[state[0],state[1]])
        # action_index = np.argmax( self.q_value[state[0],state[1]] )
        return action_index
    
    #argmax random tie-breaker
    def randargmax(self,array):
        return random.choice(np.where(array==array.max())[0])

    # plotting returns vs episodes
    def plot_returns(self, score_col, con_eps=None):
        plt.title('Return Plot')
        # plt.plot(score_col, '-*',color='red')
        plt.plot(self.running_avg(score_col), '-',color='orchid')
        plt.xlabel('steps')
        plt.ylabel('returns')
        plt.show()
        
    #rolling window avg function
    def rolling(self,dataset, window=20):
        N = window #episode collection
        cumsum = [0] #list of cumulative sum
        moving_aves = [] #list of moving averages
        for i, x in enumerate(dataset, 1): # i: tracks index-values & x:tracks list values
            cumsum.append(cumsum[i-1] + x) #cumulative sum
            if i>=N:
                moving_ave = (cumsum[i] - cumsum[i-N])/N
                moving_aves.append(moving_ave)
        return moving_aves

    #running average function
    def running_avg(self,score_col):
        cum_score = []
        run_score = score_col[0] #-5000
        beta = 0.99
        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score

    #trained_agent
    def learnt_agent(self,t_state):
        self.env_test = grid( #test_env
            num_states=self.state_size,
            goal_state=self.goal_state_xx,
            rew_setting=self.rew_setting,#rew_setting, # 1:dense | 0:sparse 
            ) 
        
        ft_list, act_list, st_list, trj = [],[],[],[]
        t_done = False
        ep_return = 0 # episode return
        self.env_test.reset() #reset environment
        
        while not t_done:
            t_action_index = self.select_action(t_state,True)  
            t_next_state, t_reward, t_done = self.env_test.step(t_state,t_action_index)
            ep_return += t_reward

            #experience_buffer_data
            st_list.append(t_state) #stt_list
            
            # print(self.r_hat[t_state[0],t_state[1],t_action_index])

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state )

            # ft_list.append(self.pi_net.feature.numpy())
            act_list.append(self.env_test.actions[t_action_index])
            
            t_state = t_next_state

        
        # print(ft_list)
        # print(st_list)
        # print(act_list)
        # print(ep_return)

        # print('++++++++++++++++++++++++++++++++++++++++++')
        # xxx

        return ft_list, act_list, ep_return, st_list, trj
    
    #policy_dataset_collection
    def pol_dataset(self, runs=1):
        feat_list, act_list, stt_list, ret_list, trj_list = [],[],[],[],[]
        start_state = self.env.initial_state
        for _ in range(runs):
            _,acts,ret,stts,trj = self.learnt_agent(start_state) #feats
            # feat_list += feats 
            act_list += acts
            stt_list += trj #stts
            trj_list += trj
            ret_list.append(ret)

        self.dataset_dict[self.num_eval] = [stt_list,act_list,ret_list]

        # layout = show_grid(input=trj_list, policy=self.num_eval, setting=self.rew_setting)
        # layout.state_traj()

        self.num_eval += 1
    
        # print('self.dataset_dict: ', self.dataset_dict)
        # print('++++++++++++++++++++++++++++++++++++++++++')
        return self.dataset_dict



#Execution
if __name__ == '__main__':
    n = 5
    agent = gridworld(states_size=[n,n],
                 rew_setting=1, #[rew_setting, num_eps]
                 n_eps=3500) 
    agent.main()