#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 2024
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
from agent.dqn_nn import DQN

## environments:
# from task.grid2D import grid #normal
# from task.grid2DMild import grid #single_optimal_path
from task.grid2Dstoc import grid #stochastic_actions
# from task.grid2DHard import grid #single_optimal_path + 2_sink_states + block_state

from task.show import show_grid

class gridworld():
    def __init__(self, 
                states_size=[4,4], 
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
        self.num_episodes = n_eps
    
    # weight_initialization
    def initialize_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.uniform_(m.weight.data, -3e-4, 3e-4 )
            # nn.init.normal_(m.weight.data, mean=0.0, std=3e-4)
            # nn.init.constant_(m.weight.data, 1.13e-4)
            nn.init.constant_(m.bias.data,0)

    # target_model   
    def update_target_model(self, net, target_net):
        target_net.load_state_dict(net.state_dict())
        target_net.eval()

    # smooth_model_update
    def smooth_update_target_model(self, net, target_net):
        rho = 1 - (1/self.tau) #decay rate 
        for p_tgt, p in zip(target_net.parameters(),net.parameters()):
            p_tgt.data.mul_(1.0 - rho)
            p_tgt.data.add_(rho*p.data) 
    
    # compute Q losses
    def compute_loss(self,states,action_indexes,next_states,rewards,dones):
        # q-values of corresponding actions
        q = self.pi_net(states).gather(1,action_indexes.long()) 

        # state_values_computations
        states_keys_raw = states.detach().numpy().astype(int)
        states_keys_proc = [tuple(i) for i in states_keys_raw] #convert_each_pair_into_tuple
        Q_values = q.detach().numpy().flatten() #state-action_values

        state_qvalue_df = pd.DataFrame({'States':states_keys_proc,'Qvalues':Q_values})
        state_value_df = state_qvalue_df.groupby('States')['Qvalues'].mean()
        state_value_dict = defaultdict(float,state_value_df.to_dict())

        """        
        #state_value_of_interest
        for key in self.states_of_interest:
            self.state_value_col[key].append(state_value_dict[key])

        #initial_state_state_values
        self.state_value_dict[self.t_k ][self.s_k] = state_value_dict[self.s_k]

        #optimal_state_values (all_states)
        self.optimal_state_value_dict = state_value_dict
        self.optimal_state_value_dict['time'] = self.t_k
        """

        """        
        #state_values_computations
        states_keys_raw = states.detach().numpy()
        states_keys_proc = [tuple(i) for i in states_keys_raw] 
        Q_values = q1.detach().numpy().flatten() #state-action_values
        state_qvalue_df = pd.DataFrame({'States':states_keys_proc,'Qvalues':Q_values})
        state_value_df = state_qvalue_df.groupby('States')['Qvalues'].mean()
        state_value_dict = defaultdict(float,state_value_df.to_dict())

        #initial_state_state_values
        self.state_value_dict[self.t_k][self.s_k] = state_value_dict[self.s_k]

        #optimal_state_values (all_initial_states)
        self.optimal_state_value_dict = state_value_dict
        self.optimal_state_value_dict['time'] = self.t_k
        # """

        with torch.no_grad():
            tgt_q = self.tgt_net(next_states)
            y = rewards + self.gamma*(1.0 - dones)*tgt_q.max(-1)[0].unsqueeze(-1)

        loss_q = ((q-y)**2).mean() 
        return loss_q
    
    # model_training
    def train_model(self, 
                    batch, 
                    pi_optim):
    
        states =  torch.stack(batch.state) 
        next_states = torch.stack(batch.next_state) 
        action_indexes = torch.tensor(batch.action).reshape(self.batch_size,-1)
        rewards = torch.tensor(batch.reward).reshape(self.batch_size,-1)
        rewards = rewards.float()
        dones = torch.tensor(batch.done).reshape(self.batch_size,-1)
        dones = dones.float()

        #Q-functions updates
        pi_optim.zero_grad() #clear data
        loss = self.compute_loss(states,action_indexes,next_states,rewards,dones)
        loss.backward() #gradients
        pi_optim.step() #update

        #polyak updating of target networks
        with torch.no_grad():
            self.smooth_update_target_model(self.pi_net, self.tgt_net)

        return loss.item()
    
    # main -> trajectories
    def main(self,
             test_frq=1, #policy_eval_frequency
             runs=20, #number_of_eval_repeats
             common_policy=0, #[1: yes | 0: no]
             epsilon=1.0,
             ):
    
        #NN architecture
        num_states = 2
        num_actions = self.num_actions
        num_hidden_l1 = 32 #16 #32 #8

        if common_policy == 1:
            self.pi_net = self.retrieve_init_model_weights(
                                                num_states,
                                                num_actions,
                                                num_hidden_l1)
        else:
            self.pi_net = self.initialize_init_model_weights(
                                                num_states,
                                                num_actions,
                                                num_hidden_l1)
            
        #target network
        self.tgt_net = deepcopy(self.pi_net)

        #hyperameters
        self.batch_size = 64
        self.tau = 1000 
        replay_memory_capacity = int(1e4)
        memory = Memory(replay_memory_capacity)
        self.limit = 500 #1500
        initial_exploration = int(self.limit) #eps*total_eps_steps 
        lr = 5e-2 #5e-4 #1e-3 #3e-4 
        self.gamma = 0.99 #0.995
           
        #optimizer & loss function
        pi_optim = optima.Adam(self.pi_net.parameters(), lr=lr)
        
        #data collection
        steps = 0
        score_col,steps_col = [],[]
        state_dis = []
        self.ext_eps = 0
        con_eps = self.num_episodes #0
        num_eval = 0
        self.conv_num_steps = num_eval
        con_step = 0
        self.stopper = 0

        # OTDD misc
        trajectory_dataset = None
        stop_learning = 0
        test_frq = test_frq #test frequecy
        self.dataset_dict = {} #policy_dataset_dictionary
        self.num_eval = 0 #policy_evaluation_tracker
        loss_col = []

        #states_of_interest
        self.states_of_interest = [(0,0),(1,1),(2,2),(3,3),(4,4)]
        self.state_value_col = defaultdict(list)

        #Regret
        self.state_value_dict = defaultdict(lambda: defaultdict(float)) #initial_state_values
        self.optimal_state_value_dict = defaultdict(float) #optimal_state_values
        # s_k = [] #initial_states of values: V(s_k)
        opt_state_value_function = None 

        #stopping_criterion
        num_stop = 5
        stop_rew = deque(maxlen=num_stop) #50, 10
        
        for episode in range(self.num_episodes):

            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode #convergence_episode
                num_eval = self.num_eval #convergence_update
                self.conv_num_steps = num_eval #convergence_update
                self.ext_eps = 1 #switch
                con_step = steps #convergence_step               
                opt_state_value_function = self.optimal_state_value_dict #store optimal_value_function

                #comment_testing
                if con_eps != 0: break #prevents_luck_optimality_landing

            stop_learning = 0 #reset_criterion_check

            done = False
            score = 0
            self.store_ret = 0
            state = self.env.reset()

            #initial_states_tracking
            # self.t_k = steps #tracking_time
            self.s_k = state #tracking_states

            while not done:
                steps += 1

                action_index = self.select_action(state,epsilon) 
                next_state, reward, done = self.env.step(state,action_index)
                score += reward

                #tensor_conversions
                t_action = torch.tensor([action_index])   
                t_state = torch.tensor(state).float()
                t_next_state = torch.tensor(next_state).float()
                t_reward = torch.tensor([reward]).float()   
                t_done = torch.tensor([done]).float()

                #initial_states_tracking
                self.t_k = steps #tracking_time
                # self.s_k = state #tracking_states

                memory.push(t_state, t_next_state,  t_action, t_reward, t_done)  
                state = next_state 
                state_dis.append(next_state) #state_visitation      
                
                if memory.__len__() > self.batch_size:
                    if steps > initial_exploration:
                        #epsilon_decay
                        if epsilon != 0:
                            epsilon = max(epsilon*0.9995, 1e-4)
                
                    batch = memory.sample(self.batch_size)
                    loss = self.train_model(batch, pi_optim)
                    loss_col.append(loss)
                
                # """                
                if steps%test_frq == 0: #reward_based_stop_criteria
                    trajectory_dataset = self.pol_dataset(runs) #no_runs
                    steps_col.append(steps)
                    stop_learning, done = self.stop_criterion(steps,stop_rew,done,episode)
                # """
                                        
            #collection of return per episode (!= step )
            score_col.append(score) 
            # score_col.append(self.store_ret)

        # #Regret_array 
        if opt_state_value_function is None: #Did_not_converge_option
            opt_state_value_function = self.optimal_state_value_dict #store final_value_function
        regret_col = self.regret_comp(opt_state_value_function,self.state_value_dict)
                
        """        
        # print('num_states: ',num_states)
        _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        self.plot_returns(score_col,con_eps)
        #"""
        # xxx

        return trajectory_dataset,steps_col,score_col,con_eps,con_step,num_eval-1,state_dis,regret_col,self.state_value_col

    # main -> policy_models
    def policy_models(self,common_policy=0,epsilon=1.0, iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        #NN architecture
        num_states = 2
        num_actions = self.num_actions
        num_hidden_l1 = 32 

        if common_policy == 1:
            self.pi_net = self.retrieve_init_model_weights(
                                                num_states,
                                                num_actions,
                                                num_hidden_l1)
        else:
            self.pi_net = self.initialize_init_model_weights(
                                                num_states,
                                                num_actions,
                                                num_hidden_l1)
            
        #target network
        self.tgt_net = deepcopy(self.pi_net)

        #hyperameters
        self.batch_size = 64
        self.tau = 1000 
        replay_memory_capacity = int(1e4)
        memory = Memory(replay_memory_capacity)
        self.limit = 500
        initial_exploration = int(self.limit) #eps*total_eps_steps 
        lr = 5e-2 
        self.gamma = 0.99 
            
        #optimizer & loss function
        pi_optim = optima.Adam(self.pi_net.parameters(), lr=lr)
        
        #data collection
        steps = 0
        self.ext_eps = 0
        con_eps = self.num_episodes
        num_eval = 0
        self.conv_num_steps = num_eval

        # OTDD misc
        self.num_eval = 0 #policy_evaluation_tracker
        stop_learning = 0

        #stopping_criterion
        num_stop = 5
        stop_rew = deque(maxlen=num_stop) #50, 10
        
        for episode in range(self.num_episodes):

            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode #convergence_episode
                num_eval = self.num_eval #convergence_update
                self.conv_num_steps = num_eval #convergence_update
                self.ext_eps = 1 #switch

                #comment_testing
                if con_eps != 0: 
                    print('converged: ', self.iteration)
                    break #prevents_luck_optimality_landing
                    

            stop_learning = 0 #reset_criterion_check

            done = False
            score = 0
            self.store_ret = 0
            state = self.env.reset()

            while not done:
                steps += 1

                action_index = self.select_action(state,epsilon) 
                next_state, reward, done = self.env.step(state,action_index)
                score += reward

                #tensor_conversions
                t_action = torch.tensor([action_index])   
                t_state = torch.tensor(state).float()
                t_next_state = torch.tensor(next_state).float()
                t_reward = torch.tensor([reward]).float()   
                t_done = torch.tensor([done]).float()

                memory.push(t_state, t_next_state,  t_action, t_reward, t_done)  
                state = next_state    
                
                if memory.__len__() > self.batch_size:
                    if steps > initial_exploration:
                        #epsilon_decay
                        if epsilon != 0:
                            epsilon = max(epsilon*0.9995, 1e-4)
                
                    batch = memory.sample(self.batch_size)
                    loss = self.train_model(batch, pi_optim)

                # policy_model_saving + stopping_criterion_check   
                self.policy_model_saver(steps-1)
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
        self.plot_returns(score_col,con_eps)
        #"""
        # xxx

        return 

    # plotting returns vs episodes
    def plot_returns(self, score_col, con_eps):
        # plt.figure(figsize=(8,15))
        plt.title('Return Plot')
        plt.plot(score_col, '-*',color='orchid',alpha=0.2)
        plt.plot(self.running_avg(score_col,0.99), '-*',color='red')
        plt.plot(self.rolling(score_col,100), '-^',color='green')
        plt.vlines(x = con_eps,
                   ymin=min(score_col),
                   ymax=max(score_col), 
                   colors='black', 
                   ls=':',
                #    label='Convergence'
                   )
        plt.xlabel('episodes')
        plt.ylabel('returns')
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
            t_action_index = self.select_action(t_state,0.0,True)  
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
        state = torch.tensor(state).float() #convert to tensor
        #evaluation
        if evaluation_episode: 
            action_index = self.get_action_deterministically(state)
        else: 
        #training
            if random.random() < epsilon:  
                action_index = self.get_action_stochastically(state)
                # print('random')
                # print(action_index)
            else:
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

        max_action = self.pi_net(state).max(0)[1]
        action_index = max_action.item()
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

    #policy_rollouts
    def pol_rollouts(self,steps):
        rew_list, act_list, stt_list, ret_list = [],[],[],[]
        act_du,stt_du = [0],[0]
        start_state = self.env.initial_state

        while act_du[-1] <= 5000: #num_decisions
            rews,acts,ret,_,trj = self.learnt_agent(start_state) #feats
            rew_list += rews #immediate_rewards
            act_list += acts #actions
            stt_list += trj #states
            ret_list.append(ret) #returns
            act_du.append(len(act_list)) #size_action_list
            stt_du.append(len(stt_list)) #size_state_list

            # print('num_decisions: ',act_du[-1])

        meta_data = [stt_list,act_list,rew_list,ret_list,stt_du,act_du]
        self.save_traj_data(meta_data,steps)

        return 

    #saving_data
    def save_traj_data(self,meta_data,steps):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'model_{steps}'
        file_location = f'policy_models/DQN/set_{self.problem_setting}/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    # saving_policy_model
    def policy_model_saver(self,steps):
        file_name = f'model_{steps}.pth'
        file_location = f'policy_models/DQN/set_{self.problem_setting}/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        torch.save(self.pi_net.state_dict(), file_path) 
        return

    # regret_calculating_module
    def regret_comp(self,optimal_state_values_dict,state_values_dict):
        regrets = []
        for time in state_values_dict.keys():
            for key in state_values_dict[time].keys():
                regrets.append( optimal_state_values_dict[key] - state_values_dict[time][key] )

        """        
        #Display
        x_values = np.arange(len(regrets)) + self.batch_size +1
        plt.title('Regret Plot')
        plt.plot(x_values,regrets, '-',color='green',alpha=0.2)
        plt.plot(x_values,self.running_avg(regrets,0.99), '-',color='orchid')
        plt.xlabel('steps')
        plt.ylabel('regret')
        # plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)

        plt.vlines(x = self.conv_num_steps, #convergence_line
                   ymin=min(regrets),
                   ymax=max(regrets), 
                   colors='black', 
                   ls=':',)
        # plt.tickmajor_
        plt.grid()
        plt.show()  
        # """  

        return regrets  

    #running average function
    def running_avg(self,score_col,beta = 0.99):
        cum_score = []
        run_score = score_col[0]
        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score

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

    #saving model
    def save_init_model_weights(self):
        file_name = "dqn.pth"
        weight_file = 'init_weights'
        save_path = abspath(join(this_dir,weight_file,file_name))
        torch.save(self.pi_net.state_dict(),save_path)

    #retrieve_initial_model
    def retrieve_init_model_weights(self, 
                               num_states, 
                               num_actions, 
                               num_hidden_l1, 
                               ):
        
        file_name = "dqn.pth"
        weight_file = 'init_weights'
        save_path = abspath(join(this_dir,weight_file,file_name))

        #declare model
        pi_net = DQN(
                    num_states,
                    num_actions,
                    num_hidden_l1)
        
        pi_net.load_state_dict(torch.load(save_path))
        return pi_net
    
    #initialize_model
    def initialize_init_model_weights(self, 
                               num_states, 
                               num_actions, 
                               num_hidden_l1, 
                               ):
        #declare model
        pi_net = DQN(
                    num_states,
                    num_actions,
                    num_hidden_l1)

        #initialize weights
        self.initialize_weights(pi_net)

        return pi_net
        

#Execution
if __name__ == '__main__':
    n = 5
    agent = gridworld(states_size=[n,n],
                      rew_setting=1, #[rew_setting, num_eps]
                      n_eps=500) 
    # agent.main()
    for  i in range(6,10):
        agent.policy_models(
                            iteration=i,
                            problem_setting='stc'
                            )



