#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys
from collections import deque
import seaborn as sns

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F
from torch.distributions import Categorical 

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)
from agent.buffer import Memory
from agent.critic import Critic_Discrete
from agent.actor import Actor_Discrete

## environments:
# from task.grid2D import grid #normal
# from task.grid2DMild import grid #single_optimal_path
from task.grid2Dstoc import grid #stochastic_actions
# from task.grid2DHard import grid #single_optimal_path + 2_sink_states + block_state

from task.show import show_grid

class gridworld():
    def __init__(self, 
                states_size=[3,3], 
                #  goal_state=tuple([ [2,2] ]), 
                #  start_state=tuple([0,0]),
                 rew_setting = 1, #[1:dense, 0:sparse]
                 n_eps = 300,
                 ):
        
        self.rew_setting = rew_setting
        self.state_size = states_size
        self.goal_state_xx = tuple([ list(np.asarray(self.state_size) - 1) ])

        self.env = grid( #training_env
            num_states=self.state_size,
            goal_state=self.goal_state_xx,
            rew_setting=self.rew_setting, # 1:dense | 0:sparse 
                    ) 
        self.num_actions = self.env.num_actions
        self.num_states = self.env.num_states
        self.num_episodes = n_eps

    # main -> trajectories   
    def main(self,
             test_frq=1, #policy_eval_frequency
             runs=1, #number_of_eval_repeats
             common_policy=1, #[1: yes | 0: no]
             epsilon=1.0, #[1.0: random | 0.0: greedy ]
             ):
                 
        if common_policy == 1:
            self.q_values = np.zeros((self.num_states[0],
                                    self.num_states[1],
                                    self.num_actions))

            
            # maxRtn = round( -0.04*(2*(self.state_size[0] - 1) - 1) + 1, 5) 
            # self.q_values = np.ones((self.num_states[0],
            #                         self.num_states[1],
            #                         self.num_actions))*1 #maxRtn
        
        else:
            self.q_values = random.randint( -100,100,
                                (self.num_states[0],
                                self.num_states[1],
                                self.num_actions))*0.01
                
        #hyperameters
        lr = 5e-2 #_(sparse/stoc) #0.5_(dense)  
        self.gamma = 0.99 #0.995
        
        #data collection
        steps = 0
        score_col,steps_col = [],[]
        state_dis = []
        self.ext_eps = 0
        con_eps = self.num_episodes #0
        num_eval = 0
        con_step = 0

        # OTDD misc
        trajectory_dataset = None
        stop_learning = 0
        test_frq = test_frq #test frequecy
        self.dataset_dict = {} #policy_dataset_dictionary
        self.num_eval = 0 #policy_evaluation_tracker

        #states_of_interest
        states_of_interest = [(0,0),(1,1),(2,2),(3,3),(4,4)]
        state_value_col = defaultdict(list) 

        #Regret
        s_k = [] #initial_states of values: V(s_k)
        V_k = [] #step_values
        opt_q_values = None

        #stopping_criterion
        num_stop = 5
        stop_rew = deque(maxlen=num_stop) 
        total_steps = 0

        for episode in range(self.num_episodes):
            
            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode #convergence_episode
                num_eval = self.num_eval #convergence_update
                self.ext_eps = 1 #switch
                con_step = steps #convergence_step
                opt_q_values = self.q_values #store optimal_Qvalue_function

                # comment_testing
                # if con_eps != 0: break #prevents_luck_optimality_landing

            stop_learning = 0 #reset_criterion_check
        
            done = False
            self.store_ret = 0
            score = 0
            state = self.env.reset()

            while not done:
                steps += 1
                total_steps +=1

                #epsilon_decay
                # if epsilon != 0:
                #     epsilon = max(epsilon*0.9999,0.0001)

                action_index = self.select_action(state,epsilon) 
                next_state, reward, done = self.env.step(state,action_index)

                max_Q_prime = self.q_values[next_state[0],next_state[0]].max()  
                Q_curr =  self.q_values[state[0],state[1],action_index] 

                #update Q-table
                self.q_values[state[0],state[1],action_index] += lr*(
                    reward + self.gamma*(1.0 - done)*max_Q_prime - Q_curr 
                    )

                #average Q-value (state_value)
                values = (1/self.num_actions)*self.q_values[state[0],state[1]].sum()
                V_k.append(values) #store state_values
                s_k.append(state)  #store states (corresponding)

                #state-value of states_of_interest
                for i in states_of_interest:
                    state_value_col[i].append((1/self.num_actions)*self.q_values[i[0],i[1]].sum())

                score += reward
                state = next_state 
                state_dis.append(next_state)

                # """                
                if steps%test_frq == 0: #reward_based_stop_criteria
                    trajectory_dataset = self.pol_dataset(runs) #no_runs
                    steps_col.append(steps)
                    stop_learning, done = self.stop_criterion(steps,stop_rew,done,episode)
                # """
                                        
            #collection of return per episode (!= step )
            score_col.append(score) 
            # score_col.append(self.store_ret)

        #Regret_array 
        if opt_q_values is None: #Did_not_converge_option
            opt_q_values = self.q_values #store final_Qvalue_function  
        regret_col = self.regret_cal(opt_q_values, V_k, s_k)   

        """     
        print('con_eps: ', con_eps)
        print('total_steps: ', total_steps)
        # self.state_visitation_dis(state_dis)
        # print('num_states: ',num_states)
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        print('trj: ', trj)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        self.plot_returns(score_col)
        xxx
        #"""
        
        return trajectory_dataset,steps_col,score_col,con_eps,con_step,num_eval-1,state_dis,regret_col,state_value_col
    
    # plotting returns vs episodes
    def plot_returns(self, score_col, con_eps=None):
        plt.title('Return Plot')
        plt.plot(self.running_avg(score_col), '-',color='orchid')
        plt.plot(score_col, '*',color='green', alpha=0.35)
        plt.plot(self.rolling(score_col), '-',color='blue', alpha=0.35)
        plt.xlabel('steps')
        plt.ylabel('returns')
        plt.grid()
        plt.show()

    # policy_path_visualization
    def illustrate_success(self):
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        return

    #rolling window avg function
    def rolling(self,dataset, window=100): #20
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
    def running_avg(self,score_col,beta = 0.99):
        cum_score = []
        run_score = score_col[0]
        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score

    #stop_training_criterion
    def stop_criterion(self,steps,stop_rew,done, episode):
        start_state = self.env.initial_state
        _,_,ret,_,_ = self.learnt_agent(start_state)
        ret = round(ret,5)
        stop_rew.append(ret)
        self.store_ret = ret

        #timing_to_stop_checking_criterion
        if not self.ext_eps: 
           
            if self.env.reward_setting == 1: #dense_setting
                if self.state_size == [5,5]:
                    min_dense_ret = -28.0
                elif self.state_size == [4,4]:
                    min_dense_ret = -16.0
                elif self.state_size == [15,15]:
                    min_dense_ret = -378.0
                else: 
                    min_dense_ret = -7.0 
            
                if (min(stop_rew)>=min_dense_ret): 
                    # print('stop_rew: ', stop_rew)
                    # print('trj: ', trj) 
                    # print('final_step: ', steps) 
                    # self.illustrate_success() #illustrate_convergence
                    return 1, 1.0
                
            else: #sparse_setting
                n_size = self.state_size[0]
                
                if self.rew_setting == 5: #strictly_single_optimal
                    min_sparse_ret = 0.85 #(0.86) 
                else: #all_sparse
                    min_sparse_ret = -0.04*(2*(n_size - 1) - 1) + 1 

                min_sparse_ret = round(min_sparse_ret,5)
                if (min(stop_rew)>=min_sparse_ret): 
                    # print('stop_rew: ', stop_rew)
                    # print('final_step: ', steps) 
                    # self.illustrate_success() #illustrate_convergence
                    return 1, 1.0
            return 0, done
        
        else:
            return 0, done

    #trained_agent
    def learnt_agent(self,t_state):
        self.env_test = grid( #test_env
            num_states=self.state_size,
            goal_state=self.goal_state_xx,
            rew_setting=self.rew_setting,#rew_setting, # 1:dense | 0:sparse 
            ) 
        
        rew_list, act_list, st_list, trj = [],[],[],[]
        t_done = False
        ep_return = 0 # episode return
        self.env_test.reset() #reset environment
        
        while not t_done:
            t_action_index = self.select_action(t_state,evaluation_episode=True)  
            t_next_state, t_reward, t_done = self.env_test.step(t_state,t_action_index)
            ep_return += t_reward

            #experience_buffer_data
            st_list.append(t_state) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state )

            # ft_list.append(self.pi_net.feature.numpy())
            act_list.append(self.env_test.actions[t_action_index])
            t_state = t_next_state
            
        return rew_list, act_list, ep_return, st_list, trj

    #action_selection_mechanism        
    def select_action(self,state,epsilon=1.0,evaluation_episode=False):
        #evaluation
        if evaluation_episode: 
            action_index = self.get_action_deterministically(state)
        else: 
        #training
            if random.random() < epsilon: 
                # print('explore') 
                action_index = self.get_action_stochastically(state)
            else:
                # print('exploit') 
                action_index = self.get_action_deterministically(state)
        return action_index

    #stochastic_action
    def get_action_stochastically(self, state):
        #sample action_index
        action_index = random.randint(self.num_actions)
        return action_index

    #greedy_action
    def get_action_deterministically(self, state):
        #select greedy action_index
        # action_index = np.argmax(
        #     self.q_values[state[0],state[1]] 
        # )

        #select greedy with random tie-breaker
        action_index = self.randargmax(
            self.q_values[state[0],state[1]] 
        )

        return action_index
    
    #argmax random tie-breaker
    def randargmax(self,array):
        # return np.argmax(random.random(array.shape)*(array == array.max()))
        return random.choice(np.where(array==array.max())[0])
        
    #policy_dataset_collection
    def pol_dataset(self, runs=3):
        # feat_list, act_list, stt_list, ret_list, trj_list = [],[],[],[],[]
        rew_list, act_list, stt_list, ret_list = [],[],[],[]
        act_du,stt_du = [],[]
        start_state = self.env.initial_state
        for i in range(runs):
            rews,acts,ret,_,trj = self.learnt_agent(start_state) #feats
            rew_list += rews #immediate_rewards
            act_list += acts #actions
            stt_list += trj #states
            ret_list.append(ret) #returns
            act_du.append(len(act_list)) #size_action_list
            stt_du.append(len(stt_list)) #size_state_list

        self.dataset_dict[self.num_eval] = [stt_list,act_list,rew_list,ret_list,stt_du,act_du]
        self.num_eval += 1  

        return self.dataset_dict
    
    #occupancy_measure
    def occupancy_measure(self, runs=3):
        _, act_list, stt_list, ret_list = [],[],[],[]
        start_state = self.env.initial_state
        for i in range(runs):
            _,acts,ret,stts,trj = self.learnt_agent(start_state) #feats
            # feat_list += feats 
            act_list += acts
            stt_list += trj #stts
            ret_list.append(ret)
            print(f'stts ({i}): \n', stts)

    # state_vistation_distribution
    def state_visitation_dis(self,state_dis):
        state_arry = state_dis
        _,freq = np.unique(
            state_arry, axis=0,return_counts=True
            )
        
        freq = 100*freq/freq.sum()
        mlt = self.num_states[0]*self.num_states[0]
        frequency = np.zeros([mlt])

        for i in range(len(freq)):
            frequency[i] = freq[i]

        data = frequency.reshape(self.num_states[0],self.num_states[0])
        sns.heatmap(data=data,
                          annot=True,
                          fmt=".2f")   
        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)   
        plt.title('State_Visitation')  
        plt.show()

    # regret_calculating_module
    def regret_cal(self,q_values, V_k, s_k):
        regrets = []
        for i in range(len(V_k)):
            #average_optimal_Q-value
            v_opt = (1/self.num_actions)*q_values[s_k[i][0],s_k[i][1]].sum()
            regret = v_opt - V_k[i]
            # print('regret: ', regret)
            regrets.append(regret)

        """        
        #Display
        plt.title('Regret Plot')
        plt.plot(regrets, '-',color='green',alpha=0.2)
        plt.plot(self.running_avg(regrets,0.9), '-',color='orchid')
        plt.xlabel('steps')
        plt.ylabel('regret')
        plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
        # plt.tickmajor_
        plt.grid()
        plt.show()    
        # """

        return regrets   


#Execution
if __name__ == '__main__':
    n = 5
    agent = gridworld(states_size=[n,n],
                      rew_setting=1, #[1:dense, 0:sparse]
                      n_eps=200) 
    agent.main(
        runs=1,
        epsilon=0 #0 #[1: random | 0: greedy ]
    )



