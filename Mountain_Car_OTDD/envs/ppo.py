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
from torch.distributions import MultivariateNormal

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##urdf location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)
from agents.buffer import Memory
from agents.ppo.critic import Critic
from agents.ppo.actor import Actor
# from agents.noise import Noise

## environments:
# from task.grid2D import grid #normal


class setting():
    def __init__(self, 
                 rew_setting = 1, #[1:dense, 0:sparse]
                 n_eps = 500,
                 iteration=1, #iteration_num
                 ):
        
        # print('rew_setting : ', rew_setting )
        # zzz

        self.env = gym.make('MountainCarContinuous-v0') 

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]
        self.num_episodes = n_eps
        self.itr = iteration
           
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

    # mini_batch_issuer     
    def mini_batch(self,batch_states,batch_actions,batch_log_probs,batch_rtgs):
        
        # old_v,_ = self.value_computation(batch_states,batch_actions)
        # print('old_v: ', old_v.shape)
        # print('batch_rtgs: ', batch_rtgs.shape)
        # xxx

        full_batch_size  = len(batch_states)
        for _ in range(full_batch_size // self.batch_size):
            indices = np.random.randint(0, full_batch_size, self.batch_size)
            
            # print('indices: ', indices)
            # print(batch_states[indices])
            # print(batch_actions[indices])
            # print(batch_log_probs[indices])
            # print(batch_rtgs[indices])
            # xxx
        return batch_states[indices],batch_actions[indices],batch_log_probs[indices],batch_rtgs[indices]

    # training_of_model
    def train_model(self,pi_optim,v1_optim):
        
        #policy_episodic_evaluation
        states,actions,log_probs,rtgs = self.learnt_agent_rollout()
        
        # print('states: ',len(states))
        # print('actions: ',actions)
        # print('log_probs: ',log_probs)
        # print('rtgs: ', rtgs)
        # print('rtgs: ', rtgs.shape)
        # xxx

        for epo in range(self.epochs):
            # print('epo: ', epo)
            
            # divide into mini_batch
            batch_states,batch_actions,batch_log_probs,batch_rtgs = self.mini_batch(states,
                                                                                    actions,
                                                                                    log_probs,
                                                                                    rtgs)
            
            # print('batch_states: ', len(batch_states))
            # print('batch_actions: ', batch_actions)
            # print('batch_log_probs: ', batch_log_probs)
            # print('batch_rtgs: ', batch_rtgs)
            # print('batch_rtgs: ', batch_rtgs.shape)

            #state_value
            old_v,_ = self.value_computation(batch_states,batch_actions)
            # print('old_v: ', old_v)
            
            #advantange_function
            A_k = batch_rtgs - old_v.detach()
            # print('A_k: ', A_k)

            #normalizing advantage_function
            A_k = (A_k - A_k.mean())/(A_k.std() + 1e-10)
            # print('A_k: ', A_k)
            
            # compute latest V_phi and pi_theta(a_t|s_t)
            new_v,new_log_probs = self.value_computation(batch_states,batch_actions)
            # print('new_log_probs: ', new_log_probs)
            # print('batch_log_probs: ', batch_log_probs) #.unsqueeze(0)
            # print('new_log_probs - batch_log_probs: ', new_log_probs - batch_log_probs)

            # policy_ratios
            ratios = torch.exp(new_log_probs - batch_log_probs)
            # print('ratios: ', ratios)

            # print(A_k.squeeze())
            # xxx

            # losses
            surr1 = ratios*A_k.squeeze()
            surr2 = torch.clamp(ratios,1-self.clip,1+self.clip)*A_k.squeeze()
            # print('surr1: ', surr1)
            # print('surr2: ', surr2)

            #policy updates
            pi_optim.zero_grad() #clear data
            loss_pi = -torch.min(surr1,surr2).mean() #-ve for gradient ascent
            # print('loss_pi: ', loss_pi)
            loss_pi.backward()
            pi_optim.step() 

            # print('new_v: ', new_v)
            # print('batch_rtgs: ', batch_rtgs)

            #v-function updates
            v1_optim.zero_grad()
            loss_v1 = ((new_v-batch_rtgs)**2).mean() 
            # print('loss_v1: ', loss_v1)
            loss_v1.backward() 
            v1_optim.step()
        
        #delete_data_in_GPU
        # del rewards, dones
        # torch.cuda.empty_cache()

        #updating of old policy network
        with torch.no_grad():
            self.update_target_model(self.pi_net, self.pi_net_old)
        
        return loss_pi.item(), loss_v1.item()

    def main(self,
             common_policy=0, #[1: yes | 0: no]
             iteration=1, #iteration_num
             ):
        self.iteration = iteration #num_process_iteration
        
        #NN architecture
        num_states = self.num_states
        num_actions = self.num_actions
        num_hidden_l1 = 64 
        num_hidden_l2 = 64 

        if common_policy == 1:
            self.pi_net, self.v1_net = self.retrieve_init_model_weights(
                                                            num_states,
                                                            num_actions,
                                                            num_hidden_l1,
                                                            num_hidden_l2)
        else:
            self.pi_net, self.v1_net = self.initialize_init_model_weights(
                                                            num_states,
                                                            num_actions,
                                                            num_hidden_l1,
                                                            num_hidden_l2)
        #target networks
        self.pi_net_old = deepcopy(self.pi_net)

        #hyperameters
        self.batch_size = 64 #3
        lr = 3e-4 
        self.gamma = 0.99 #0.999 
           
        #optimizer & loss function
        pi_optim = optima.Adam(self.pi_net.parameters(), lr=lr)
        v1_optim = optima.Adam(self.v1_net.parameters(), lr=lr)

        """        
        #initialize weights
        self.initialize_weights_actor(self.pi_net)
        self.initialize_weights_critic(self.q1_net)
        self.save_init_model_weights()
        # """

        #Timestep parameters
        self.clip = 0.2
        self.epochs = 5 #10
        self.horizon = 999 #5 
        self.repeats = 2

        #data collection
        steps = 0
        score_col,steps_col = [],[]
        state_dis = []
        self.ext_eps = 0
        con_eps = self.num_episodes 
        num_eval = 0
        self.conv_num_steps = num_eval
        con_step = 0

        # OTDD misc
        stop_learning = 0
        self.num_eval = 0 #policy_evaluation_tracker

        #stopping_criterion
        num_stop = 10
        stop_rew = deque(maxlen=num_stop) 

        start_time = time.time()
        t_so_far = 0 #total_timesteps_simulated
        i_so_far = 0 #total_iterations
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
            # print('state: ', state)
           
            while not done:
                steps += 1

                action,_ = self.select_action(state)
                next_obs, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated #done_function (new_approach)
                score += reward                
                next_state = torch.tensor(next_obs).float().unsqueeze(0) 

                # print('action: ', action)
                # print('next_obs: ', next_obs)
                # print('reward: ', reward)

                state = next_state
                
                #model_updates
                self.train_model(pi_optim, v1_optim)

                #save_models
                # self.save_model_weights(steps)
                # steps_col.append(steps)
                # state_dis.append(tuple(next_obs)) #state_visitation   
                   
            #collection of return per episode (!= step )
            score_col.append(score) 
            # stop_learning = self.training_stop_criterion(stop_rew,score,episode)

            if episode % 1 == 0:
                print("Episode: {} | Avg_reward: {}".format(episode,round(score,4))) #round(avg_reward,4)
        
        end_time = time.time()
        duration = end_time - start_time
        print('{:.2f} min'.format(duration/60))
        
        self.plot_returns(score_col,con_eps) 
        self.learnt_agent()         


        xxx
        meta_data = [steps_col,score_col,con_eps,con_step,state_dis]
        # print('meta_data: \n', meta_data)
        self.save_meta_data(meta_data)
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

    # policy_path_visualization
    def illustrate_success(self):
        _,act,ret,stt,trj = self.learnt_agent()
        print('act: ', act)
        print('stt: ', stt)
        print('ret: ', ret)
        layout = show_grid(input=trj, setting=self.rew_setting)
        layout.state_traj(size=self.state_size)
        return

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
    
    #current_policy_rollouts 
    def learnt_agent_rollout(self):
        self.env_test = gym.make('MountainCarContinuous-v0')  #start_env
        
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rtgs = [] #rewards_to_go

        ep_return = 0 # episode return        
        for reaps in range(self.repeats):
            t_obs,_ = self.env_test.reset() #reset environment
            t_state = torch.tensor(t_obs).float().unsqueeze(0)
            t_done = False

            ep_rewards = []
            # ep_len = 0 #episode_length_tracking
            # for _ in range(self.horizon):
            # print('reaps: ', reaps)
            while not t_done:
                t_action,t_log_prob = self.select_action(t_state)  
                t_next_obs, t_reward, t_terminated, t_truncated, _ = self.env_test.step(t_action)
                t_done = t_terminated or t_truncated #done_function (new_approach)
                ep_return += t_reward
                t_next_state = torch.tensor(t_next_obs).float().unsqueeze(0)
                
                batch_states.append(t_state.numpy().squeeze(0)) #states_collection
                ep_rewards.append(t_reward) #rewards_collection
                batch_actions.append(t_action) #actions_collection
                batch_log_probs.append(t_log_prob) #log_prob_collection
                
                """                
                if t_done: 
                    #reset environment
                    t_obs,_ = self.env_test.reset() 
                    t_state = torch.tensor(t_obs).float().unsqueeze(0)
                else: 
                    t_state = t_next_state
                """

                t_state = t_next_state
            rew2go = self.rewards_to_go(ep_rewards) #rewards_to_go
            batch_rtgs.extend(rew2go) #rewards_to_go_collection
            # batch_rtgs.append(rew2go) #rewards_to_go_collection
            # batch_lens.append(ep_len) #episode_length_collection
            # print('ep_len: ', ep_len)
            # batch_returns.append(ep_return) #returns_collection
    
        # print('batch_states: \n', batch_states)
        # print('batch_actions: \n', len(batch_actions))
        # print('batch_log_probs: \n', len(batch_log_probs))
        # print('batch_rtgs: \n', batch_rtgs)
        # print('batch_lens: \n', batch_lens)

        torch_batch_states = torch.tensor(np.array(batch_states), dtype=torch.float)
        torch_batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float)
        torch_batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).squeeze(1)
        torch_batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float).reshape(-1,1)
        # batch_lens = torch.tensor(batch_states, dtype=torch.float)

        # print('torch_batch_states: \n', len(torch_batch_states))
        # print('torch_batch_actions: \n', torch_batch_actions)
        # print('torch_batch_log_probs: \n', torch_batch_log_probs)
        # print('torch_batch_rtgs: ', torch_batch_rtgs) 
        # print('torch_batch_rtgs: ', torch_batch_rtgs.shape) 
        # print('batch_lens: \n', batch_lens)

        self.env_test.close() #shut_env
        return torch_batch_states, torch_batch_actions, torch_batch_log_probs, torch_batch_rtgs #, batch_lens
    
    # Reward_to_go computation 
    def rewards_to_go(self,rewards):
        n = len(rewards)
        rtgs = np.zeros([n]) #np.zeros_like(rew,float) #
        for i in reversed(range(n)):
            # rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0 ) #undiscounted
            rtgs[i] = (self.gamma**i)*rewards[i] + (rtgs[i+1] if i+1 < n else 0 ) #discounted

        return rtgs #.reshape(-1,1) #rtgs
    
    # value_computation of observations
    def value_computation(self, batch_states,batch_actions):
        # print('batch_states: ', batch_states)
        # print('batch_actions: ', batch_actions)
        
        # state-values 
        v = self.v1_net(batch_states) #.squeeze().unsqueeze(0)
        # print('v: ', v)
        # print('v: ', v.shape)

        # log_pis
        log_probs = self.pi_net.evaluate(batch_states,batch_actions)  
        # print('log_prob: ', log_probs)

        return v,log_probs
    
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
            return action
        else:
            action,log_pi = self.get_action_stochastically(state)
        return action, log_pi

    #stochastic_action
    def get_action_stochastically(self, state):
        action,log_pi = self.pi_net_old(state) #.detach().cpu().numpy()[0] 

        #action,log_pi = self.pi_net(state) #.detach().cpu().numpy()[0] 
        action = action.detach().cpu().numpy()[0]
        log_pi = log_pi.detach().cpu().numpy()[0]

        # print('action: ', action)
        # print('log_pi: ', log_pi)
        # xxx
        return action,log_pi
    
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
        act_file_name = "ppo_actor.pth"
        weight_file = 'init_weights'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))
        torch.save(self.pi_net.state_dict(), act_save_path)

        cri_file_name = "ppo_critic.pth"
        cri_save_path = abspath(join(this_dir, weight_file, cri_file_name))
        torch.save(self.v1_net.state_dict(), cri_save_path)

    #saving_model_updates
    def save_model_weights(self,step):
        act_file_name = f"actor_{step}.pth"
        weight_file = f'../evals/ppo_data/ppo_models_{self.itr}'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))

        os.makedirs(join(this_dir,weight_file), exist_ok=True)
        torch.save(self.pi_net.state_dict(), act_save_path) 
        return

    #saving_meta
    def save_meta_data(self,meta_data):
        meta_data = np.asarray( meta_data, dtype=object) 

        weight_file = f'../evals/ppo_data/ppo_meta_{self.itr}'
        act_save_path = abspath(join(this_dir,weight_file))
        np.save(act_save_path,meta_data)
        return

    #retrieve_initial_model
    def retrieve_init_model_weights(self, 
                               num_states, 
                               num_actions, 
                               num_hidden_l1, 
                               num_hidden_l2,
                               ):
        
        act_file_name = "ppo_actor.pth"
        weight_file = 'init_weights'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))

        cri_file_name = "ppo_critic.pth"
        cri_save_path = abspath(join(this_dir, weight_file, cri_file_name))

        #declare model
        pi_net = Actor(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device)       

        v1_net = Critic(
                        num_states,
                        num_hidden_l1,
                        num_hidden_l2).to(device) 
        
        pi_net.load_state_dict(torch.load(act_save_path))
        v1_net.load_state_dict(torch.load(cri_save_path))

        return pi_net, v1_net
    
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

        v1_net = Critic(
                        num_states,
                        num_hidden_l1,
                        num_hidden_l2).to(device) 

        #initialize weights
        self.initialize_weights(pi_net)
        self.initialize_weights(v1_net)

        return pi_net, v1_net
        

#Execution
if __name__ == '__main__':
    agent = setting( rew_setting=1, #[rew_setting, num_eps]
                      n_eps=100,
                      iteration=1
                      ) 
    agent.main()



