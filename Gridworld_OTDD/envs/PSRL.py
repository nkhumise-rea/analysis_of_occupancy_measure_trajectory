#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 06 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from scipy.stats import dirichlet
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

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)

## environments:
# from task.grid2D import grid #normal
# from task.grid2DMild import grid #single_optimal_path
from task.grid2Dstoc import grid #stochastic_actions
# from task.grid2DHard import grid #single_optimal_path + 2_sink_states + block_state


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

        self.env = grid( #training_env
            num_states=self.state_size,
            goal_state=self.goal_state_xx,
            rew_setting=self.rew_setting, # 1:dense | 0:sparse 
                    ) 
        self.num_actions = self.env.num_actions
        self.num_states = self.env.num_states
        self.total_num_states = self.num_states[0]*self.num_states[1]
        self.num_episodes = n_eps

    # sample_mean
    def sampleNormalGamma(self,Mu0,nMu0,Tau0,nTau0,Nt,rMean,rVar):
        lambda0 = nMu0
        alpha0 = nTau0/2
        beta0 = alpha0/Tau0

        nObs = Nt #estimated_num_observations
        muObs = rMean #estimated_mean_observations
        varObs = rVar #estimated_variance_observations

        mu0  = (lambda0*Mu0 + nObs*muObs) / np.clip(lambda0+nObs,1,None) #(lambda0 + nObs)
        lambda1 = lambda0 + nObs
        alpha = alpha0 + nObs/2
        beta = beta0 + .5*((nObs*varObs + lambda0*nObs*(muObs-Mu0)**2)/ np.clip(lambda0+nObs,1,None)) #(lambda0 + nObs)

        tau = np.random.gamma(alpha,1/np.clip(beta,1,None)) #beta
        sigma0 = 1/np.sqrt(np.clip(lambda1*tau,1,None)) #np.sqrt(lambda1*tau))
        mu1 = np.random.normal(mu0,sigma0)   
        
        return mu1

    # sampling from Dirichlet distribution
    def sampleDirichlet(self,alpha):
        scale = 1 
        theta = np.random.gamma(alpha,scale) #sampling_from _Gamma_distribution
        theta /= np.clip(np.linalg.norm(theta),1,None) #normalizing_probability_vector       
        return theta

    # value_iteration
    def valueIteration(self,pSample,muSample,tau,
                        num_states_row,
                        num_states_col,
                        num_actions,
                        state_pairs):

        u_0 = np.zeros((num_states_row,num_states_col)) #state_values (current) 
        u_1 = np.zeros((num_states_row,num_states_col)) #state_values (next)


        # pi_tilde = np.zeros((num_states_row,num_states_col),dtype='int') #original_policy
        pi_tilde = random.randint(0,4,(num_states_row,num_states_col)) #introduce_stochasticity

        for _ in range(tau): #loop_for_an_episode
            for state_row, state_col in state_pairs:
                for action in range(num_actions):
                    u_i = muSample[state_row,state_col,action] + (pSample[state_row,state_col,action]*u_0).sum() 

                    if u_1[state_row,state_col] < u_i or action == 0:
                        u_1[state_row,state_col] = u_i #u1[st] = u_max
                        pi_tilde[state_row,state_col] = action #action_max = action

            u_0 = u_1
            u_1 = np.zeros((self.num_states[0],self.num_states[1])) #next_state_values: resetting
        
        return pi_tilde, u_0  #policy,state_value
    
    # main -> trajectories
    def main(self,
             test_frq=1, #50 #policy_eval_frequency
             runs=1, #number_of_eval_repeats
             stop_cri=5, #num_updates w/ optimal policy
             ):
                        
        num_states_row = self.num_states[0]
        num_states_col = self.num_states[0]
        num_actions = self.num_actions

        state_pairs = []
        for state_row in range(num_states_row):
                for state_col in range(num_states_col):
                    state_pairs.append((state_row,state_col))

        Nt = np.zeros((num_states_row,num_states_col,num_actions)) #total_visits       
        Pt = np.zeros((num_states_row,num_states_col,num_actions,
                       num_states_row,num_states_col,)) #total_transitions
        
        rMean = np.zeros((num_states_row,num_states_col,num_actions)) #mean_rewards
        rVar = np.zeros((num_states_row,num_states_col,num_actions)) #variance_rewards

        alpha0 = np.zeros((num_states_row,num_states_col,num_actions,
                       num_states_row,num_states_col,)) #Prior_dirichlet_distributions

        pSample = np.zeros((num_states_row,num_states_col,num_actions,
                       num_states_row,num_states_col,)) #sample_distributions
        muSample = np.zeros((num_states_row,num_states_col,num_actions)) #mean_of_distributions

        Mu0 = np.zeros((num_states_row,num_states_col,num_actions)) #prior_mean_estimate
        nMu0 = np.zeros((num_states_row,num_states_col,num_actions)) #prior_num_observations_for_Mu0
        Tau0 = np.ones((num_states_row,num_states_col,num_actions)) #prior_tau_estimate
        nTau0 = np.zeros((num_states_row,num_states_col,num_actions)) #prior_num_observations_for_Tau0
        
        #data collection
        steps = 0
        score_col,steps_col,updates_col = [],[],[]
        state_dis = []
        self.ext_eps = 0
        con_eps = self.num_episodes #0
        num_eval = 0
        con_step = 0
        tau = 60 #num_step_per_episode

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
        opt_state_value_function = None

        #stopping_criterion
        num_stop = stop_cri #5
        stop_rew = deque(maxlen=num_stop) #50, 10
       
        for episode in range(self.num_episodes):
            
            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode #convergence_episode
                num_eval = self.num_eval #convergence_update
                self.ext_eps = 1 #switch
                con_step = steps #convergence_step
                opt_state_value_function = state_value_fuction #store optimal_value_function

                # comment_testing
                # if con_eps != 0: break #prevents_luck_optimality_landing

            stop_learning = 0 #reset_criterion_check

            done = False
            self.store_ret = 0
            score = 0
            state = self.env.reset()
            
            vi = np.zeros((num_states_row,num_states_col,num_actions)) #episodic_visits

            #Collate Bayes Information
            alpha = alpha0 + Pt #transitions
            pSample = self.sampleDirichlet(alpha) #sampling_distributions
            muSample = self.sampleNormalGamma(Mu0,nMu0,Tau0,nTau0,Nt,rMean,rVar)

            #Compute near-optima; policy for optimitic MDP
            self.pi_k,state_value_fuction = self.valueIteration(pSample,muSample,tau,
                                            num_states_row,
                                            num_states_col,
                                            num_actions,
                                            state_pairs) #policy_k
            action_index = self.pi_k[state]

            # Evaluate_policy_per_update           
            if steps%test_frq == 0: #reward_based_stop_criteria
                trajectory_dataset = self.pol_dataset(runs) #no_runs
                updates_col.append(episode) #record_of_updates

            while not done:
                steps += 1
                next_state, reward, done = self.env.step(state,action_index)
                score += reward

                vi[state[0],state[1],action_index] += 1
                Nt[state[0],state[1],action_index] += 1
                count = Nt[state[0],state[1],action_index]

                #initial_states_tracking+corresponding_values
                values = state_value_fuction[state]
                V_k.append(values) #store state_values
                s_k.append(state)  #store states (corresponding)   

                #state-value of states_of_interest
                for i in states_of_interest:
                    state_value_col[i].append( state_value_fuction[i] )

                rVar[state[0],state[1],action_index] = (
                    (count - 1)*rVar[state[0],state[1],action_index] + 
                    (reward - rMean[state[0],state[1],action_index])**2 )/count
                
                rMean[state[0],state[1],action_index] = (
                    (count - 1) * rMean[state[0],state[1],action_index] + reward )/count
                
                if not done:
                    Pt[state[0],state[1],action_index,next_state[0],next_state[1]] += 1
                    state = next_state 
                      
                action_index = self.pi_k[state]
                state_dis.append(next_state) #state_visitation 


                """                
                if steps%test_frq == 0: #reward_based_stop_criteria
                    trajectory_dataset = self.pol_dataset(runs) #no_runs
                    steps_col.append(steps)
                    stop_learning, done = self.stop_criterion(steps,stop_rew,done,episode)
                # """

                #check_converge_in_each_step (finer/more_sensitive)
                stop_learning, done = self.stop_criterion(steps,stop_rew,done,episode)
                steps_col.append(steps) #compute_time
                                        
            #collection of return per episode (!= step )
            score_col.append(score) 
            # score_col.append(self.store_ret)

        #Regret_array 
        if opt_state_value_function is None: #Did_not_converge_option
            opt_state_value_function = state_value_fuction #store final_value_function
        regret_col = self.regret_cal(opt_state_value_function,V_k,s_k) 

        """     
        print('con_step: ',con_step) 
        print('con_eps : ',con_eps ) 
         
        # print('num_states: ',num_states)
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        self.plot_returns(score_col,con_eps)  
        #"""

        return trajectory_dataset,steps_col,score_col,con_eps,con_step,num_eval-1,state_dis,regret_col,state_value_col #,updates_col

    # main -> policy_models
    def policy_models(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']
                        
        num_states_row = self.num_states[0]
        num_states_col = self.num_states[0]
        num_actions = self.num_actions

        state_pairs = []
        for state_row in range(num_states_row):
                for state_col in range(num_states_col):
                    state_pairs.append((state_row,state_col))

        Nt = np.zeros((num_states_row,num_states_col,num_actions)) #total_visits       
        Pt = np.zeros((num_states_row,num_states_col,num_actions,
                       num_states_row,num_states_col,)) #total_transitions
        
        rMean = np.zeros((num_states_row,num_states_col,num_actions)) #mean_rewards
        rVar = np.zeros((num_states_row,num_states_col,num_actions)) #variance_rewards

        alpha0 = np.zeros((num_states_row,num_states_col,num_actions,
                       num_states_row,num_states_col,)) #Prior_dirichlet_distributions

        pSample = np.zeros((num_states_row,num_states_col,num_actions,
                       num_states_row,num_states_col,)) #sample_distributions
        muSample = np.zeros((num_states_row,num_states_col,num_actions)) #mean_of_distributions

        Mu0 = np.zeros((num_states_row,num_states_col,num_actions)) #prior_mean_estimate
        nMu0 = np.zeros((num_states_row,num_states_col,num_actions)) #prior_num_observations_for_Mu0
        Tau0 = np.ones((num_states_row,num_states_col,num_actions)) #prior_tau_estimate
        nTau0 = np.zeros((num_states_row,num_states_col,num_actions)) #prior_num_observations_for_Tau0
        
        #data collection
        steps = 0
        self.ext_eps = 0
        con_eps = self.num_episodes #0
        tau = 60 #num_step_per_episode

        # OTDD misc
        stop_learning = 0
        self.num_eval = 0 #policy_evaluation_tracker

        #stopping_criterion
        num_stop = 5
        stop_rew = deque(maxlen=num_stop) #50, 10
       
        for episode in range(self.num_episodes):
            
            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode #convergence_episode
                self.ext_eps = 1 #switch

                # comment_testing
                print('converged: ', self.iteration)
                if con_eps != 0: break #prevents_luck_optimality_landing

            stop_learning = 0 #reset_criterion_check

            done = False
            self.store_ret = 0
            score = 0
            state = self.env.reset()
            
            vi = np.zeros((num_states_row,num_states_col,num_actions)) #episodic_visits

            #Collate Bayes Information
            alpha = alpha0 + Pt #transitions
            pSample = self.sampleDirichlet(alpha) #sampling_distributions
            muSample = self.sampleNormalGamma(Mu0,nMu0,Tau0,nTau0,Nt,rMean,rVar)

            #Compute near-optima; policy for optimitic MDP
            self.pi_k,_ = self.valueIteration(pSample,muSample,tau,
                                            num_states_row,
                                            num_states_col,
                                            num_actions,
                                            state_pairs) #policy_k
            action_index = self.pi_k[state]

            # policy_model_saving + stopping_criterion_check   
            self.policy_model_saver(episode)

            while not done:
                steps += 1
                next_state, reward, done = self.env.step(state,action_index)
                score += reward

                vi[state[0],state[1],action_index] += 1
                Nt[state[0],state[1],action_index] += 1
                count = Nt[state[0],state[1],action_index]

                rVar[state[0],state[1],action_index] = (
                    (count - 1)*rVar[state[0],state[1],action_index] + 
                    (reward - rMean[state[0],state[1],action_index])**2 )/count
                
                rMean[state[0],state[1],action_index] = (
                    (count - 1) * rMean[state[0],state[1],action_index] + reward )/count
                
                if not done:
                    Pt[state[0],state[1],action_index,next_state[0],next_state[1]] += 1
                    state = next_state 
                      
                action_index = self.pi_k[state]

                #check_converge_in_each_step (finer/more_sensitive)
                stop_learning, done = self.stop_criterion(steps,stop_rew,done,episode)
                                        
            #collection of return per episode (!= step )
            # score_col.append(score) 

        """     
        # print('num_states: ',num_states)
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        # self.plot_returns(score_col,con_eps)  
        #"""

        return

    #display_optimal_state-trajectory
    def illustrate_success(self):
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        return

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
 
    #action_selection_mechanism        
    def select_action(self,state,evaluation_episode=False):
        action_index = self.pi_k[state]
        return action_index
    
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
    def running_avg(self,score_col,beta = 0.99):
        cum_score = []
        run_score = score_col[0]
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
        
        rew_list, act_list, st_list, trj = [],[],[],[]
        t_done = False
        ep_return = 0 # episode return
        self.env_test.reset() #reset environment
        
        while not t_done:
            t_action_index = self.select_action(t_state,True)  
            t_next_state, t_reward, t_done = self.env_test.step(t_state,t_action_index)
            ep_return += t_reward

            #experience_buffer_data
            st_list.append(t_state) #stt_list
            rew_list.append(t_reward) #rew_list
            
            # print(self.r_hat[t_state[0],t_state[1],t_action_index])

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state )

            # ft_list.append(self.pi_net.feature.numpy())
            act_list.append(self.env_test.actions[t_action_index])
            t_state = t_next_state
        
        return rew_list, act_list, ep_return, st_list, trj
    
    #policy_dataset_collection
    def pol_dataset(self, runs=1):
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

    #~~~~~~~~~~~~~~~~~ saving_policy_model ~~~~~~~~~~~~~~~~~~~~~~
    # saving_policy_model
    def policy_model_saver(self,steps):
        file_name = f'model_{steps}'
        file_location = f'policy_models/PSRL/set_{self.problem_setting}/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,self.pi_k) 
        return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # regret_calculating_module
    def regret_cal(self,state_value_fuction, V_k, s_k):
        regrets = []
        for i in range(len(V_k)):
            #average_optimal_Q-value
            v_opt = state_value_fuction[s_k[i]] 
            regret = v_opt - V_k[i]
            # print('regret: ', regret)
            regrets.append(regret)

        """        
        # Display
        plt.title('Regret Plot')
        plt.plot(regrets, '-',color='green',alpha=0.2)
        plt.plot(self.running_avg(regrets,0.9), '-',color='orchid')
        plt.xlabel('steps')
        plt.ylabel('regret')
        # plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
        # plt.tickmajor_
        plt.grid()
        plt.show()    
        #"""

        return regrets 


#Execution
if __name__ == '__main__':
    n = 5
    agent = gridworld(states_size=[n,n],
                 rew_setting=0, #[rew_setting, num_eps]
                 n_eps=500) 
    
    # agent.main()
    for  i in range(6,10):
        agent.policy_models(
                            iteration=i,
                            problem_setting='stc'
                            )