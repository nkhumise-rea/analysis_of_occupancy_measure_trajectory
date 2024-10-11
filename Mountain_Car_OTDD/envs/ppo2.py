#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 2024
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
from torch.distributions.normal import Normal

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir)) 
sys.path.insert(0,agent_dir)
from agents.ppo2.buffer import Memory
from agents.ppo2.actor_critic import ActorCritic
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
    def mini_batch(self,states,actions,rewards,old_log_pis,A_k):
        full_batch_size  = len(states)
    
        for _ in range(full_batch_size // self.batch_size):
            indices = np.random.randint(0, full_batch_size, self.batch_size)
            # print('indices: ', indices)
            # print(states[indices])
            # print(actions[indices])
            # print(rewards[indices])
            # print(old_log_pis[indices])
            # print(A_k[indices])

            yield states[indices],actions[indices],rewards[indices],old_log_pis[indices],A_k[indices]

    # training_of_model
    def train_model(self, 
                    pi_v_optim,
                    memory=None,
                    ):

        # """ 
        #Using_memory     
        dataset,dataset_size = memory.issue()

        states = torch.cat(dataset.state) 
        actions = torch.tensor(dataset.action).reshape(dataset_size,-1)
        old_log_pis = torch.tensor(dataset.log_pi).reshape(dataset_size,-1)
        values = torch.tensor(dataset.value).reshape(dataset_size,-1)
        
        rewards = torch.tensor(dataset.reward).reshape(dataset_size,-1)        
        rewards = rewards.float().to(device)
        dones = torch.tensor(dataset.done).reshape(dataset_size,-1)
        dones = dones.float().to(device)

        #Advantages
        A_k = self.advantage_fuction(rewards,values,dones)
        
        for epo in range(self.epochs):
         
            # divide into mini_batch
            for states,actions,rewards,old_log_pis,A_k in self.mini_batch(states,
                                                                  actions,
                                                                  rewards,
                                                                  old_log_pis,
                                                                  A_k):

                mean,std,new_values = self.pi_v_net(states)
                pi_distribution = Normal(mean, std)
                new_log_pis = pi_distribution.log_prob(actions).sum(axis=-1) #tensor([ xxx ])
                log_pi_diff = new_log_pis - old_log_pis.squeeze()


                # policy_ratios
                log_pi_diff = new_log_pis - old_log_pis.squeeze()
                ratios = torch.exp(log_pi_diff)
                surr1 = ratios*A_k #.squeeze()
                surr2 = torch.clamp(ratios,1-self.clip,1+self.clip)*A_k #.squeeze()

                #policy_critic updates
                pi_v_optim.zero_grad() #clear data
                loss_pi = -torch.min(surr1,surr2).mean() #-ve for gradient ascent
                loss_v1 = ((new_values - rewards)**2).mean() 
                entropy = pi_distribution.entropy().mean()

                mm = 0.5
                loss = loss_pi + mm*loss_v1 - self.beta*entropy

                loss.backward()
                pi_v_optim.step() 
        
        return loss.item()

    # policy_models generation
    def policy_models(self,
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
            self.pi_v_net = self.retrieve_init_model_weights(
                                                            num_states,
                                                            num_actions,
                                                            num_hidden_l1,
                                                            num_hidden_l2)
        else:
            self.pi_v_net = self.initialize_init_model_weights(
                                                            num_states,
                                                            num_actions,
                                                            num_hidden_l1,
                                                            num_hidden_l2)
        #target networks
        self.tgt_pi_v_net = deepcopy(self.pi_v_net)
    
        #hyperameters
        self.batch_size = 128 #64 
        self.epochs = 40 #5 #4
        self.clip = 0.2
        self.beta = 0.01
        self.gae_lambda = 0.95

        replay_memory_capacity = int(1e6) 
        memory = Memory(replay_memory_capacity)
        lr = 3e-4 
        self.gamma = 0.99 #0.999 
                                 
        #optimizer & loss function
        pi_v_optim = optima.Adam(self.pi_v_net.parameters(), lr=lr)

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
        score_col,steps_col = [],[]
        self.ext_eps = 0
        con_eps = self.num_episodes 
        num_eval = 0
        self.conv_num_steps = num_eval

        # OTDD misc
        stop_learning = 0
        self.dataset_dict = {} #policy_dataset_dictionary
        self.num_eval = 0 #policy_evaluation_tracker

        #stopping_criterion
        num_stop = 10
        stop_rew = deque(maxlen=num_stop) 
        
        start_time = time.time()
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

                action, log_pi = self.select_action(state)
                value = self.get_value(state)

                next_obs, reward, terminated, truncated,_ = self.env.step(action)
                done = terminated or truncated #done_function (new_approach)
                score += reward
                next_state = torch.tensor(next_obs).float().unsqueeze(0) 

                #tensor_conversions
                t_state = state 
                t_action = torch.tensor(action).float()   
                t_log_pi = torch.tensor([log_pi]).float()   
                t_reward = torch.tensor([reward]).float()   
                t_done = torch.tensor([done]).float()
                t_value = torch.tensor([value]).float()

                memory.push(t_state, t_action, t_log_pi, t_reward, t_done, t_value)

                if done:
                    self.next_value = self.get_value(next_state)
                    
                    # batch = memory.sample(self.batch_size)
                    self.train_model(pi_v_optim,memory)
                    memory.clear() #delete content
  
                state = next_state  

                #save_models
                # self.policy_model_saver(steps-1)

            #collection of return per episode (!= step )
            # stop_learning = self.training_stop_criterion(stop_rew,score,episode)
            score_col.append(score) 

            if episode % 10 == 0:
                print("Episode: {} | Avg_reward: {}".format(episode,round(score,4))) #round(avg_reward,4)

        end_time = time.time()
        duration = end_time - start_time
        print('{:.2f} min'.format(duration/60))

        self.plot_returns(score_col,con_eps) 
        self.learnt_agent()

        # meta_data = [steps_col,score_col,con_eps,con_step,state_dis]
        # print('meta_data: \n', meta_data)
        # self.save_meta_data(meta_data)
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
        plt.title('Angular_Plot')
        plt.plot(angle_col, '-*',color='orchid',alpha=0.2)
        plt.plot(self.running_avg(angle_col,0.99), '-*',color='red')
        # plt.plot(self.rolling(angle_col,100), '-^',color='green')
        plt.xlabel('steps')
        plt.ylabel('Angular Position')
        plt.show()

    #stop_training_criterion
    def stop_criterion(self,steps,stop_rew,done, episode):
        _,_,ret,_,_ = self.learnt_agent()
        ret = round(ret,5)
        stop_rew.append(ret)
        self.store_ret = ret

        #timing_to_stop_checking_criterion
        if not self.ext_eps: 
            #dense_setting
            min_dense_ret = -250 #525
            
            if (min(stop_rew)>=min_dense_ret): 
                # print('stop_rew: ', stop_rew)
                # print('final_step: ', steps) 
                # self.illustrate_success() #illustrate_convergence
                return 1, 1.0

        return 0, done

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
        return rew_list, act_list, ep_return, st_list, trj

    #action_selection_mechanism   
    def select_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            action = self.get_action_deterministically(state)
        else:
            action = self.get_action_stochastically(state)
        return action

    #stochastic_action
    def get_action_stochastically(self, state):

        mean,std,_ = self.pi_v_net(state)
        pi_distribution = Normal(mean, std)
        action_u = pi_distribution.rsample() #sample actions
        log_probs = pi_distribution.log_prob(action_u).sum(axis=-1) #tensor([ xxx ])
        action = torch.tanh(action_u) #bound action [-1,1] #tensor([[ xxx ]])

        with torch.no_grad():
            action = action.cpu().numpy()[0]

        self.epi = 1 #0.1
        action = np.clip(action + self.epi*self.noise(),-1,1)

        return action, log_probs.item() 

    #greedy_action
    def get_action_deterministically(self, state):
        with torch.no_grad():
            mean,_,_ = self.pi_v_net(state)
        action = mean.cpu().numpy()[0]
        return action
    
    #state_values
    def get_value(self,state):
        _,_,value = self.pi_v_net(state) #tensor([[ xxx ]])
        # print('value: ', value)

        return value.item()
    
    #advantage_function
    def advantage_fuction_v1(self, rewards, values, dones):
        advantage = []
        rtgs = self.next_value

        for r,v,done in zip(reversed(rewards),reversed(values),reversed(dones)):
            if done:
                rtgs = 0 #r
            else:
                rtgs = r + self.gamma*rtgs
            
            advantage.append(rtgs - v)
            advantage.reverse()
        
        #raw_advantage_function
        A_k = torch.tensor(advantage).float() 

        #normalizing advantage_function
        A_k = (A_k - A_k.mean())/(A_k.std() + 1e-10)

        return A_k
    
    # """    
    #advantage_function
    def advantage_fuction(self, rewards, values, dones):
        advantage = []
        gae = 0

        n = len(rewards)
        for step in reversed(range(n)):
            if step + 1 < n:
                error = rewards[step] + self.gamma*(1.0 - dones[step])*values[step+1] - values[step]
            else: 
                error = rewards[step] - values[step]

            gae = error + self.gamma*self.gae_lambda*(1.0 - dones[step])*gae
            advantage.insert(0,gae)        

        #raw_advantage_function
        A_k = torch.tensor(advantage).float() 

        #normalizing advantage_function
        A_k = (A_k - A_k.mean())/(A_k.std() + 1e-10)

        return A_k
    # """
   
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
        torch.save(self.pi_v_net.state_dict(), act_save_path)

    #saving_model_updates
    def save_model_weights(self,step):
        act_file_name = f"actor_{step}.pth"
        weight_file = f'../evals/ppo_data/ppo_models_{self.itr}'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))

        os.makedirs(join(this_dir,weight_file), exist_ok=True)
        torch.save(self.pi_v_net.state_dict(), act_save_path) 
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
                               num_hidden_l2
                               ):
        
        act_file_name = "ppo_actor_critic.pth"
        weight_file = 'init_weights'
        act_save_path = abspath(join(this_dir,weight_file,act_file_name))

        #declare model
        pi_v_net = ActorCritic(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device)       
        
        # self.pi_net_eval = Actor_Eval(
        #                     num_states,
        #                     num_actions,
        #                     num_hidden_l1,
        #                     num_hidden_l2,
        #                     act_limit).to(device) 

        pi_v_net.load_state_dict(torch.load(act_save_path))
        return pi_v_net
    
    #initialize_model
    def initialize_init_model_weights(self, 
                               num_states, 
                               num_actions, 
                               num_hidden_l1,
                               num_hidden_l2
                               ):
        #declare model
        pi_v_net = ActorCritic(
                        num_states,
                        num_actions,
                        num_hidden_l1,
                        num_hidden_l2).to(device)      

        # self.pi_net_eval = Actor_Eval(
        #                     num_states,
        #                     num_actions,
        #                     num_hidden_l1,
        #                     num_hidden_l2,
        #                     act_limit).to(device) 

        #initialize weights
        self.initialize_weights(pi_v_net)
        return pi_v_net
        
    #~~~~~~~~~~~~~~~~~ saving_policy_model ~~~~~~~~~~~~~~~~~~~~~~
    #saving_model_updates
    def policy_model_saver(self,step):
        file_name = f"actor_{step}.pth"
        # file_location = f'../evals/ddpg_data/models/iter_{self.iteration}' #file_location

        file_location = f'/mnt/parscratch/users/acp21rmn/Mountain_Car_OTDD/algo_data/sac/models/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent

        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        torch.save(self.pi_net.state_dict(), file_path) 
        return
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Execution
if __name__ == '__main__':
    agent = setting( rew_setting=1, #[rew_setting, num_eps]
                      n_eps=200, #500
                      ) 
    
    for i in range(5):
        agent.policy_models(
            iteration=i
        )

