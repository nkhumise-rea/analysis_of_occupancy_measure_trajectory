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

#gym
import gym

#pytorch
import torch
import torch.nn as nn
import torch.optim as optima
import torch.nn.functional as F

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)
from agents.buffer import Memory
from agents.ddpg.critic import Critic
from agents.ddpg.actor import Actor
from agents.noise import Noise


class setting():
    def __init__(self, 
                 rew_setting = 1, #[1:dense, 0:sparse]
                 n_eps = 500,
                 ):
        
        # environment_declaration
        self.env = gym.make('MountainCarContinuous-v0') 

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]
        self.num_episodes = n_eps
          
    # Actor_weight_initialization
    def initialize_weights_actor(self,model):
        for i,m in enumerate( model.modules() ):
            if isinstance(m, nn.Linear):
                if i == 6:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3)
                else:
                    nn.init.uniform_(m.weight, *self.hidden_init(m) )

    # Critic_weight_initialization
    def initialize_weights_critic(self,model):
        for i,m in enumerate( model.modules() ):
            if isinstance(m, nn.Linear):
                if i == 7:
                    nn.init.uniform_(m.weight, -3e-3, 3e-3 )
                else:
                    nn.init.uniform_(m.weight, *self.hidden_init(m) )
    
    # Hidden_initialization_Module          
    def hidden_init(self,layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1.0/np.sqrt(fan_in)
        return (-lim,lim)    

    # target_model
    def update_target_model(self, net, target_net):
        target_net.load_state_dict(net.state_dict())
        target_net.eval()
    
    # smooth_model_update
    def smooth_update_target_model(self, net, target_net, tau):
        rho = 1 - (1/tau) #decay rate 
        for p_tgt, p in zip(target_net.parameters(),net.parameters()):
            p_tgt.data.mul_(1.0 - rho)
            p_tgt.data.add_(rho*p.data) 
       
    # training_of_model
    def train_model(self,
                    batch, 
                    pi_optim,
                    q1_optim,
                    ):
        
        states = torch.cat(batch.state) 
        next_states = torch.cat(batch.next_state)
        rewards = torch.tensor(batch.reward).reshape(self.batch_size,-1)
        actions = torch.tensor(batch.action).reshape(self.batch_size,-1)
        rewards = rewards.float().to(device)
        dones = torch.tensor(batch.done).reshape(self.batch_size,-1)
        dones = dones.float().to(device)

        #next_actions
        next_actions = self.tgt_pi_net(next_states)

        # q-values        
        q1 = self.q1_net(states,actions) #.squeeze(1)
        next_q1 = self.tgt_q1_net(next_states, next_actions) 
  
        with torch.no_grad():
            y = rewards + (1.0 - dones)*self.gamma*next_q1 #.detach()

        #Q-function updates
        q1_optim.zero_grad()
        loss_q1 = ((q1-y)**2).mean() 
        loss_q1.backward() 
        q1_optim.step()

        #policy updates
        pi_optim.zero_grad() #clear data
        loss_pi = -self.q1_net(states,self.pi_net(states)).mean() #-ve for gradient ascent
        loss_pi.backward()
        pi_optim.step() 
        
        #delete_data_in_GPU
        del rewards, dones
        torch.cuda.empty_cache()

        #polyak updating of target networks
        with torch.no_grad():
            self.smooth_update_target_model(self.pi_net, self.tgt_pi_net, self.tau_pi)
            self.smooth_update_target_model(self.q1_net, self.tgt_q1_net, self.tau_q1)
        
        return loss_pi.item(), loss_q1.item()

    # policy_models generation
    def policy_models(self,
             common_policy=0, #[1: yes | 0: no]
             epsilon=1.0,
             iteration=1, #iteration_num
             ):
        self.iteration = iteration #num_process_iteration
    
        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions
        num_hidden_l1 = 64 
        num_hidden_l2 = 64 

        if common_policy == 1:
            self.pi_net, self.q1_net = self.retrieve_init_model_weights(
                                                            num_states,
                                                            num_actions,
                                                            num_hidden_l1,
                                                            num_hidden_l2)
        else:
            self.pi_net, self.q1_net = self.initialize_init_model_weights(
                                                            num_states,
                                                            num_actions,
                                                            num_hidden_l1,
                                                            num_hidden_l2)
        #target networks
        self.tgt_pi_net = deepcopy(self.pi_net)
        self.tgt_q1_net = deepcopy(self.q1_net)

        #hyperameters
        self.batch_size = 128
        replay_memory_capacity = int(1e6) 
        memory = Memory(replay_memory_capacity)
        self.limit = 500
        initial_exploration = int(self.limit) #eps*total_eps_steps 
        lr_pi = 1e-4 
        lr_q1 = 1e-3 
        self.tau_pi = 1e3 
        self.tau_q1 = 1e3 
        self.gamma = 0.99 #0.999 
        self.epsilon = epsilon
           
        #optimizer & loss function
        pi_optim = optima.Adam(self.pi_net.parameters(), lr=lr_pi)
        q1_optim = optima.Adam(self.q1_net.parameters(), lr=lr_q1)

        """        
        #initialize weights
        self.initialize_weights_actor(self.pi_net)
        self.initialize_weights_critic(self.q1_net)
        self.save_init_model_weights()
        # """

        #exploration_noise
        noise_mean = np.zeros(num_actions) 
        self.noise = Noise(noise_mean) 

        #data collection
        steps = 0
        self.ext_eps = 0
        con_eps = self.num_episodes 
        num_eval = 0
        self.conv_num_steps = num_eval

        # OTDD misc
        stop_learning = 0
        self.num_eval = 0 #policy_evaluation_tracker

        #stopping_criterion
        num_stop = 10
        stop_rew = deque(maxlen=num_stop) 

        # start_time = time.time()
        for episode in range(self.num_episodes):

            #stopping_criterion_evaluation
            if stop_learning: 
                con_eps = episode - 1 #convergence_episode
                self.ext_eps = 1 #switch
                
                #comment_testing
                print('converged: ', self.iteration)
                if con_eps != 0: break #prevents_luck_optimality_landing

            stop_learning = 0 #reset_criterion_check

            done = False
            self.store_ret = 0
            score = 0
            obs,_ = self.env.reset()
            state = torch.tensor(obs).float().unsqueeze(0) 
            
            while not done:
                steps += 1

                action = self.select_action(state)
                next_obs, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated #done_function (new_approach)
                score += reward                
                next_state = torch.tensor(next_obs).float().unsqueeze(0) 

                #tensor_conversions
                t_action = torch.tensor(action).float()   
                t_state = state 
                t_next_state = next_state 
                t_reward = torch.tensor([reward]).float()   
                t_done = torch.tensor([done]).float()

                memory.push(t_state, t_next_state,  t_action, t_reward, t_done)  
                state = next_state 
                
                if steps > initial_exploration:
                    self.epsilon -= 1e-5  
                    self.epsilon = max(self.epsilon, 1e-2)
                    batch = memory.sample(self.batch_size)
                    self.train_model(batch, pi_optim, q1_optim)

                #save_models
                self.policy_model_saver(steps-1)
                   
            #collection of return per episode (!= step )
            stop_learning = self.training_stop_criterion(stop_rew,score,episode)
            # score_col.append(score) 

            # if episode % 1 == 0:
            #     print("Episode: {} | Avg_reward: {}".format(episode,round(score,4))) #round(avg_reward,4)
        
        # end_time = time.time()
        # duration = end_time - start_time
        # print('{:.2f} min'.format(duration/60))
        
        # self.plot_returns(score_col,con_eps) 
        # self.learnt_agent()         
        return
    
    # plotting returns vs episodes
    def plot_returns(self, score_col, con_eps):
        plt.title('Return_Plot')
        plt.plot(score_col, '-*',color='orchid',alpha=0.2)
        plt.plot(self.running_avg(score_col,0.99), '-*',color='red')
        plt.plot(self.rolling(score_col,50), '-^',color='green')
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

    # plotting angular-position vs episodes
    def angular_plot(self, angle_col, con_eps):
        plt.title('X-Position_Plot')
        plt.plot(angle_col, '-*',color='orchid',alpha=0.2)
        plt.plot(self.running_avg(angle_col,0.99), '-*',color='red')
        # plt.plot(self.rolling(angle_col,100), '-^',color='green')
        plt.xlabel('steps')
        plt.ylabel('X Position')
        plt.show()

    #stop_training_criterion
    def stop_criterion(self,stop_rew,done):
        ret = self.learnt_agent_testing()
        ret = round(ret,5)
        stop_rew.append(ret)
        self.store_ret = ret

        #timing_to_stop_checking_criterion
        if not self.ext_eps: 
            min_dense_ret = 90 
            
            if (min(stop_rew)>=min_dense_ret): 
                # print('stop_rew: ', stop_rew)
                # print('final_step: ', steps) 
                # self.illustrate_success() #illustrate_convergence
                return 1, 1.0
            
        return 0, done, ret

    #stop_training_criterion_simplified
    def training_stop_criterion(self,stop_rew,score,eps):
        score = round(score,5)
        stop_rew.append(score) 

        #timing_to_stop_checking_criterion
        if not self.ext_eps: 
            min_dense_ret = 90 
            
            if (min(stop_rew)>=min_dense_ret): 
                # print('stop_rew: ', stop_rew)
                # print('final_ep: ', eps) 
                # self.illustrate_success() #illustrate_convergence
                return 1
            
        return 0

    #trained_agent
    def learnt_agent(self):
        self.env_test = gym.make('MountainCarContinuous-v0')  #start_env
        
        rew_list, act_list, st_list, trj = [],[],[],[]
        Xpos_list = []
        t_done = False
        ep_return = 0 # episode return

        t_obs,_ = self.env_test.reset() #reset environment
        t_state = torch.tensor(t_obs).float().unsqueeze(0)
        
        while not t_done:
            t_action = self.select_action(t_state, True)  
            t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
            t_done = t_terminated or t_truncated #done_function (new_approach)
            ep_return += t_reward
            t_next_state = torch.tensor(t_next_obs).float().unsqueeze(0)
            t_pos = t_next_obs[0] #position_along: x-axis

            if t_pos >= 0.45:
                print('t_pos: ', t_pos)
            
            #experience_buffer_data
            st_list.append(t_state.detach().cpu().numpy()[0]) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state.detach().cpu().numpy()[0] )

            Xpos_list.append(t_pos)
            act_list.append(t_action[0])
            t_state = t_next_state
        
        self.env_test.close() #shut_env

        print('returns: ', ep_return)
        self.angular_plot(Xpos_list,0)
        return rew_list , act_list, ep_return, st_list, trj
    
    #trained_agent
    def learnt_agent_testing(self):
        self.env_test = gym.make('MountainCarContinuous-v0')  #start_env
        
        rew_list, act_list, st_list, trj = [],[],[],[]
        Xpos_list = []
        t_done = False
        ep_return = 0 # episode return

        t_obs,_ = self.env_test.reset() #reset environment
        t_state = torch.tensor(t_obs).float().unsqueeze(0)
        
        while not t_done:
            t_action = self.select_action(t_state, True)  
            t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
            t_done = t_terminated or t_truncated #done_function (new_approach)
            ep_return += t_reward
            t_next_state = torch.tensor(t_next_obs).float().unsqueeze(0)
            
            """  
            t_pos = t_next_obs[0] #position_along: x-axis

            #experience_buffer_data
            st_list.append(t_state.detach().cpu().numpy()[0]) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state.detach().cpu().numpy()[0] )

            Xpos_list.append(t_pos)
            act_list.append(t_action[0])
            """
            t_state = t_next_state
        
        self.env_test.close() #shut_env

        # print('returns: ', ep_return)
        # self.angular_plot(Xpos_list,0)
        return ep_return

    #action_selection_mechanism   
    def select_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            action = self.get_action_deterministically(state)
        else:
            action = self.get_action_stochastically(state)
        return action

    #stochastic_action
    def get_action_stochastically(self, state):
        action = self.pi_net(state).detach().cpu().numpy()[0]  

        #from: https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/README.md      
        # action = np.clip(action*.2 + self.noise(),-1,1)
        action = np.clip(action + self.noise(),-1,1)

        """        
        action += self.noise()*max(0,self.epsilon)
        action += self.noise() 
        action = np.clip(action,-1,1)
        """

        return action

    #greedy_action
    def get_action_deterministically(self, state):
        action = self.pi_net(state).detach().cpu().numpy()[0]
        return action
    
    #policy_dataset_collection
    def pol_dataset(self, runs=1):
        rew_list, act_list, stt_list, ret_list = [],[],[],[]
        act_du,stt_du = [],[]
        for i in range(runs):
            rews,acts,ret,_,trj = self.learnt_agent() 
            rew_list += rews #immediate_rewards
            act_list += acts #actions
            stt_list += trj #states
            ret_list.append(ret) #returns
            act_du.append(len(act_list)) #size_action_list
            stt_du.append(len(stt_list)) #size_state_list

        self.dataset_dict[self.num_eval] = [stt_list,act_list,rew_list,ret_list,stt_du,act_du]
        self.num_eval += 1
        return self.dataset_dict

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
        act_file_name = "ddpg_actor.pth"
        weight_file = 'init_weights'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))
        torch.save(self.pi_net.state_dict(), act_save_path)

        cri_file_name = "ddpg_critic.pth"
        cri_save_path = abspath(join(this_dir, weight_file, cri_file_name))
        torch.save(self.q1_net.state_dict(), cri_save_path)

    #retrieve_initial_model
    def retrieve_init_model_weights(self, 
                               num_states, 
                               num_actions, 
                               num_hidden_l1, 
                               num_hidden_l2,
                               ):
        
        act_file_name = "ddpg_actor.pth"
        weight_file = 'init_weights'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))

        cri_file_name = "ddpg_critic.pth"
        cri_save_path = abspath(join(this_dir, weight_file, cri_file_name))

        #declare model
        pi_net = Actor(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device)       

        q1_net = Critic(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device) 
        
        pi_net.load_state_dict(torch.load(act_save_path))
        q1_net.load_state_dict(torch.load(cri_save_path))

        return pi_net, q1_net
    
    #initialize_model
    def initialize_init_model_weights(self, 
                               num_states, 
                               num_actions, 
                               num_hidden_l1,
                               num_hidden_l2,
                               ):
        #declare model
        pi_net = Actor(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device)       

        q1_net = Critic(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device) 

        #initialize weights
        self.initialize_weights_actor(pi_net)
        self.initialize_weights_critic(q1_net)

        return pi_net, q1_net
        
    #~~~~~~~~~~~~~~~~~ saving_policy_model ~~~~~~~~~~~~~~~~~~~~~~
    #saving_model_updates
    def policy_model_saver(self,step):
        file_name = f"actor_{step}.pth"
        # file_location = f'../evals/ddpg_data/models/iter_{self.iteration}' #file_location
        
        
        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/ddpg/models/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent

        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        torch.save(self.pi_net.state_dict(), file_path) 
        return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

#Execution
if __name__ == '__main__':
    agent = setting( rew_setting=1, #[rew_setting, num_eps]
                      n_eps=100,
                      ) 
    for i in range(5,10):
        agent.policy_models(
            iteration=i
        )



