#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import defaultdict
import ot
import h5py
from numpy import linalg as LA
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from copy import deepcopy
import pandas as pd
import gym
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys

#set directory/location
this_dir = dirname(__file__)
env_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,env_dir)
from envs.PSRL import gridworld

class otdd():
    def __init__(self,
                 test_freq=50,states_size=[3,3],setting=1,n_eps=1000,stop_cri=50,max_esteps=60):
        self.agent = gridworld( states_size=states_size, #grid_dimensions
                                rew_setting=setting, #rew_setting [# 1:dense | 0:sparse ]
                                n_eps=n_eps, #[num_eps] 
                               ) 
        self.test_freq = test_freq #policy_evaluation_frequency
        #dense: dns, sparse: sps, single: sgl (optimal_path), stochastic: stc, sink:snk
        #strictly-single: st_gl (optimal_path)
        rew_set = ['sps','dns','sgl','stc','snk','st_gl'] 
        self.setting = rew_set[setting] #rewards_setting
        self.states_size = states_size #grid_dimensions
        self.num_stop = stop_cri #stopping_criteria
        self.max_esteps = max_esteps #max_episode_steps
    
    # curve_lengths
    def curve_length(self,num_curve):
        trajectory_dataset,steps_col,score_col,con_eps,con_step,num_eval,state_dis,regret_col,state_value_col = self.agent.main(
                                    test_frq=self.test_freq, #policy_eval_frequency
                                    runs=1, #number_of_eval_repeats
                                    )
        
        key_init = list(trajectory_dataset)[0] #initial_key
        key_last = list(trajectory_dataset)[-1] #last_key
        initial_dataset = trajectory_dataset[key_init] #initial_policy_dataset

        if num_eval == -1: # failed_to_converge
            num_eval = key_last #key_last
            final_dataset =  trajectory_dataset[num_eval] #converged_policy_dataset
            self.optimal_traj = final_dataset[0] #trajectory_dataset[key_2ndlast][0]

            direct_dis,ds_stt_col,ds_oot_col,lip_con,ds_int_col = self.dis_calculations(trajectory_dataset,
                                                                        initial_dataset,
                                                                        final_dataset,
                                                                        )
            #plot segment_lengths_vs_steps
            outcome = 'Fail'
            self.monot(ds_stt_col,con_step,num_eval,state_dis,regret_col,num_curve,outcome) #plots segments
            return direct_dis,steps_col,score_col,con_eps,num_eval,ds_stt_col,ds_oot_col,state_dis,regret_col,outcome,lip_con,state_value_col,ds_int_col,con_step
        
        else: 
            final_dataset =  trajectory_dataset[num_eval] #converged_policy_dataset
            self.optimal_traj = final_dataset[0] #trajectory_dataset[key_2ndlast][0]
            optimal = self.check_convergence(final_dataset)

            if optimal == 1: #iff converged
                direct_dis,ds_stt_col,ds_oot_col,lip_con,ds_int_col = self.dis_calculations(trajectory_dataset,
                                                                            initial_dataset,
                                                                            final_dataset,
                                                                            )
                
                #plot segment_lengths_vs_steps
                outcome = 'Pass'
                self.monot(ds_stt_col,con_step,num_eval,state_dis,regret_col,num_curve,outcome) #plots segments
                return direct_dis,steps_col,score_col,con_eps,num_eval,ds_stt_col,ds_oot_col,state_dis,regret_col,outcome,lip_con,state_value_col,ds_int_col,con_step
            
            else: 
                num_eval = key_last #key_last
                final_dataset =  trajectory_dataset[num_eval] #converged_policy_dataset
                self.optimal_traj = final_dataset[0] #trajectory_dataset[key_2ndlast][0]

                direct_dis,ds_stt_col,ds_oot_col,lip_con,ds_int_col = self.dis_calculations(trajectory_dataset,
                                                                            initial_dataset,
                                                                            final_dataset,
                                                                            )
                #plot segment_lengths_vs_steps
                outcome = 'Fail'
                self.monot(ds_stt_col,con_step,num_eval,state_dis,regret_col,num_curve,outcome) #plots segments
                return direct_dis,steps_col,score_col,con_eps,num_eval,ds_stt_col,ds_oot_col,state_dis,regret_col,outcome,lip_con,state_value_col,ds_int_col,con_step

    # geodesics_lengths
    def dis_calculations(self,trajectory_dataset,initial_dataset,final_dataset):
        data_points = len(trajectory_dataset) - 1 #size        
        Ns = int(data_points/1)
        a,b = 0,data_points #start & end_index
        idxs = np.linspace(a,b,Ns+1) #indexes

        ds_stt_col,ds_oot_col,ds_int_col,lip_con_col = [],[],[],[]
        for i in range(idxs.shape[0] - 1):
            #between successive
            dataset1 = trajectory_dataset[round(idxs[i])]
            dataset2 = trajectory_dataset[round(idxs[i+1])]
            ds_stt,lip_con = self.OTDD(dataset1,dataset2)
            ds_stt_col.append(ds_stt) #collection_policy_distances
            lip_con_col.append(lip_con) #collection_lipschitz_constants

            #between optimal_&_others
            ds_oot,_ = self.OTDD(dataset1,final_dataset)
            ds_oot_col.append(ds_oot)

            #between initial_&_others
            ds_int,_ = self.OTDD(dataset1,initial_dataset)
            ds_int_col.append(ds_int)
            
        # curve_dis = sum(ds_stt_col[:num_eval]) #sum index 0:num_eval        
        direct_dis,_ = self.OTDD(initial_dataset,final_dataset)
        return direct_dis, ds_stt_col, ds_oot_col, max(lip_con_col),ds_int_col
    
    #describe state_size as string
    def state_label(self):     
        return str(self.states_size)

    #save large data as numpy
    def save_np(self,name,data,file_path):
        np.save(file_path,data)
        return        
    
    # iterative_computation (curve_length) for Boxplot of varying sampling rates
    def iterations(self,num_iter=20):
        for i in range(num_iter):
            data = np.asarray( self.curve_length(i) , dtype=object) #dtype=object

            #saving_data
            dataset_name = f"curve_{i}" 
            file,_ =  self.file_location()

            run_file = f"curve_{i}" #.hdf5
            file_path = abspath(join(this_dir,file,run_file)) 
            os.makedirs(join(this_dir,file), exist_ok=True)
            self.save_np(dataset_name,data,file_path)

    #plot segment_lengths_vs_steps
    def monot(self,ds_col,con_step,num_eval,state_dis,regret_col,num_curve,outcome,show=0):

        ds_col = np.asarray(ds_col)/max(np.asarray(ds_col)) #normalizing Segment_Plot

        fig = plt.figure(figsize=(12, 9))

        ## Segment_length_Plot
        plt.subplot(2,2,1)
        plt.plot(
                 ds_col,
                 '-', #'-'
                 color='gray',
                 alpha=0.2,
                #  label=f"sr: {self.test_freq} ({self.setting})"
                ) 
        
        plt.plot(
                 self.running_avg(ds_col),
                 '-',
                 color='red',
                 label=f"sr: {self.test_freq} ({self.setting})"
                 ) 
        
        plt.vlines(x = num_eval, #convergence_line
                   ymin=min(ds_col),
                   ymax=max(ds_col), 
                   colors='black', 
                   ls=':',)
        plt.xlabel('updates (episodes)') #episodes
        # plt.ylabel('segment_length/max(segment_length)')
        plt.ylabel('$\Delta / \Delta_{max}$')
        plt.title(f'segment_length vs steps [{outcome}]')
        plt.legend()
        
        ## regret_plot
        plt.subplot(2,2,2)
        plt.title('Regret Plot')
        plt.plot(regret_col,'-',color='green',alpha=0.2)
        plt.plot(self.running_avg(regret_col,0.9),'-',color='orchid')
        plt.vlines(x = con_step, #convergence_line
                   ymin=min(regret_col),
                   ymax=max(regret_col), 
                   colors='black', 
                   ls=':',)
        plt.xlabel('steps')
        plt.ylabel('regret')
        # plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
        # plt.grid()
        
        ## trajectory_visualization
        plt.subplot(2,2,3)
        data, cmap, norm = self.state_traj()
        #draw gridlines
        plt.grid(axis='both',
                color='k',
                linewidth=2
                )
        
        n_size = self.states_size[0]
        labels=[str(x) for x in range(n_size+1)]
        plt.xticks(np.arange(-.5,n_size,1) + 1, labels=labels)
        plt.yticks(np.arange(-.5,n_size,1) + 1, labels=labels)
        
        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title(f'Policy state-trajectory')
        
        ## state_visitation
        plt.subplot(2,2,4)
        state_arry = state_dis
        _,freq = np.unique(
            state_arry, axis=0,return_counts=True
            )
        
        freq = 100*freq/freq.sum()
        mlt = self.states_size[0]*self.states_size[0]
        frequency = np.zeros([mlt])

        for i in range(len(freq)):
            frequency[i] = freq[i]

        data = frequency.reshape(self.states_size[0],self.states_size[0])
        sns.heatmap(data=data,
                          annot=True,
                          fmt=".2f")   
        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)   
        plt.title('State_Visitation (%)')  


        #All_Figures
        plt.tight_layout()     
        if show == 1:
            plt.show()
        else:
            #saving_data
            _,file_loc =  self.file_location()
            os.makedirs(join(this_dir,file_loc), exist_ok=True)

            file = f"returns_{num_curve}.png"
            file_path = abspath(join(this_dir,file_loc,file)) 
            plt.savefig(file_path)
        plt.close()

    # location_of_saved_files
    def file_location(self):
        
        ## ----- Normal_operation -----
        #common_location
        common_loc = f'psrl_psrl/stop_{self.num_stop}/step_{self.max_esteps}' #PSRL
        
        #curve_datasets
        data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
        # print('data_loc: \n', data_loc)

        #curve_images
        image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
        # print('image_loc: \n', image_loc)
        """
        
        ## ----- Validation_operation ----- 

        #validation_test_location
        common_loc = f'psrl_multiruns/stop_{self.num_stop}/step_{self.max_esteps}' #psrl

        #curve_datasets
        data_loc = f"{common_loc}/runs_6/dts/{self.setting}/ss_{self.state_label()}" 
        # print('data_loc: \n', data_loc)

        #curve_images
        image_loc = f"{common_loc}/runs_6/vld/{self.setting}/ss_{self.state_label()}"                   
        # print('image_loc: \n', image_loc)
        """

        return data_loc,image_loc

    #running average function
    def running_avg(self,score_col,beta=0.9): #0.99
        cum_score = []
        run_score = score_col[0] #-5000

        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score
    
    #verify_optimality
    def check_convergence(self,final_dataset):
        opt = 0 #indicator
        traj = final_dataset[0]
        state_sizes_list = final_dataset[4]

        if len(state_sizes_list) > 1:
            opt_list = []
            state_sizes_list.insert(0,0)
            
            action_sizes_list = final_dataset[5]
            num_actions_per_run = np.diff(action_sizes_list,prepend=0)

            #through_multiple_trajectories
            for cnt,pair in enumerate(zip(state_sizes_list,state_sizes_list[1:])):
                single_opt = 0 #reset_opts
                single_traj = traj[pair[0] : pair[1] ]
                
                single_x = np.asarray(single_traj[-1]) #last_state
                single_y = num_actions_per_run[cnt] #num_actions

                x_goal = np.asarray(self.states_size) - 1 #goal_state
                y_goal = 2*(self.states_size[0] - 1) #optimal_num_actions
                
                if (single_x == x_goal).all() and single_y == y_goal: #identical_match 
                    single_opt = 1 #optimal_trajectory
                opt_list.append(single_opt) #collect_opts

            if 1 in opt_list: #check_at_least_one_trajectory_is_optimal
                opt = 1 #optimal_trajectories

        else: 
            x = np.asarray(traj[-1]) #last_state
            y = len(final_dataset[1]) #num_actions

            x_goal = np.asarray(self.states_size) - 1 #goal_state
            y_goal = 2*(self.states_size[0] - 1) #optimal_num_actions
            
            if (x == x_goal).all() and y == y_goal: #identical_match 
                opt = 1 #optimal_trajectory

        return opt
    
    #issue OPTIMAL policy path trajectory in state-space
    def state_traj(self):
        n_input = np.asarray(self.optimal_traj)
        u = np.unique(n_input, axis=0) #unique state value
        n_size = self.states_size[0]

        #grid_color values
        data = np.arange(1,n_size**2+1).reshape(n_size,n_size)                

        for i in range(len(u)):
            data[u[i][0],u[i][1]] = 0.5

        #create discrete colormap
        cmap = colors.ListedColormap(['salmon','white'])
        bounds = [0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        return data, cmap, norm

    # occupancy_measure
    def occupancy_measure(self,data):
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.insert(np.asarray(data[4]),0,0) #len(states)_of_runs
        ty1 = np.insert(np.asarray(data[5]),0,0) #len(actions)_of_runs

        time_counts = defaultdict(lambda: defaultdict(int))
        total_counts_at_time =  defaultdict(int)
        probs = defaultdict(lambda: defaultdict(int)) #probabilities
        occup = defaultdict(int) #occupancy_measure

        # iterate_each_run
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_run
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_run

            # iterate_over_state-action pairs
            for t,j in enumerate(zip(states,actions)):
                time_counts[t][j] += 1
                total_counts_at_time[t] += 1

        #calculate_probability_distribution
        for t in time_counts:
            # print(t)
            for item in time_counts[t]:
                # print(item)
                probs[t][item] = time_counts[t][item]/total_counts_at_time[t]
        
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

        return occup
    
    # expected_rewards_calculator (only for stochastic_rewards)
    def expected_rewards(self,data):
        x = data[0] #states_in_dataset
        y = data[1] #actions_in_dataset
        r = data[2] #rewards_in_dataset
        tx = np.insert(np.asarray(data[4]),0,0) #len(states)_of_runs
        ty = np.insert(np.asarray(data[5]),0,0) #len(actions)_of_runs/len(rewards)_of_runs

        D_sa = defaultdict(list) #filtered_dataset
        r_sa = defaultdict(list) #filtered_dataset

        # iterate_each_run
        for i in range(len(tx)-1):
            states = x[tx[i]:tx[i+1]] #states_per_run
            actions = y[ty[i]:ty[i+1]] #actions_per_run
            rewards = r[ty[i]:ty[i+1]] #actions_per_run

            dataset = zip(states,actions,rewards)
            state_action_rewards = [ tuple(i) for i in dataset]

            for i in state_action_rewards:
                if i[:2] in D_sa:
                    D_sa[i[:2]].append(i[2])
                else:
                    D_sa[i[:2]] = [i[2]]

            for i in D_sa.keys():
                r_sa[i] = sum(D_sa[i])/len(D_sa[i])

        return r_sa

    # OTDD 
    def OTDD(self,data1,data2):
        # Estimated/Empirical_occupancy-measures
        # occup1 = self.occupancy_measure(data1)
        # occup2 = self.occupancy_measure(data2)

        #Estimated_expected_rewards (only for stochastic rewards)
        # r1 = self.expected_rewards(data1)
        # r2 = self.expected_rewards(data2)

        # datasets_sampled_from_occupancy-measures
        # x1, x2 = np.asarray(data1[0][:-1]), np.asarray(data2[0][:-1]) #states (features)
        x1, x2 = data1[0][:-1],data2[0][:-1] #states (features)
        y1, y2 = np.asarray(data1[1]), np.asarray(data2[1]) #actions (labels)
        r1,r2 = np.asarray(data1[2]), np.asarray(data2[2]) #actions (labels)
        y_space = ['up', 'right', 'down', 'left'] #action_space

        #task_1 (feature set with label Y = y)
        col_X_y1 = {} #collection of X given Y = y
        for i in y_space:
            nd_y = [] #set of X given Y = y
            for x,y in zip(x1,y1):
                if y == i:
                    nd_y.append(x)
            col_X_y1[i] = nd_y

        #task_2 (feature set with label Y = y)
        col_X_y2 = {} #collection of X given Y = y
        for i in y_space:
            nd_y = [] #set of X given Y = y
            for x,y in zip(x2,y2):
                if y == i:
                    nd_y.append(x)
            col_X_y2[i] = nd_y

        #saving label-to-label distances
        label_dis = defaultdict(int)
        for i in y_space:
            for j in y_space:
                if len(col_X_y1[i]) != 0 and len(col_X_y2[j]) != 0: 
                    pair = (i,j)
                    label_dis[pair] = self.inner_wasserstein(
                        col_X_y1[i],
                        col_X_y2[j]) 

        otdd = self.outer_wasserstein(x1,y1,x2,y2,label_dis) #otdd
        lip_con = self.lipschitz(x1,y1,x2,y2,r1,r2,label_dis) #lipschitz_constant

        return otdd, lip_con
    
    #inner_OT (label-to-label distance)
    def inner_wasserstein(self,z1,z2): #z1:A_support, z2:B_support
        z1 = np.asarray(z1).reshape(-1,len(z1[0]))
        z2 = np.asarray(z2).reshape(-1,len(z2[0]))
        P = ot.unif(z1.shape[0])
        Q = ot.unif(z2.shape[0])
        CostMatrix = ot.dist(z1,z2, metric='cityblock') #cost matrix: 'euclidean'  

        val = ot.emd2(  P, #A_distribution 
                        Q, #B_distribution
                        M = CostMatrix, #cost_matrix pre-processing
                        numItermax=int(1e6)
                        ) #OT matrix
        return val
    
    #outer_OT (label-to-label distance)
    def outer_wasserstein(self,x1,y1,x2,y2,label_dis):
        z1 = list(zip(x1,y1))
        z2 = list(zip(x2,y2))

        p_xy1 = ot.unif(len(z1))
        p_xy2 = ot.unif(len(z2))
        Cm1 = self.outer_cost(z1,z2,label_dis) #cost matrix

        #Wasserstein_Distance (wd)
        wd = ot.emd2(  p_xy1, #A_distribution 
                       p_xy2, #B_distribution
                       M = Cm1, #cost_matrix pre-processing
                       numItermax=int(1e6)
                       ) #OT matrix
                    
        return wd**1.0
    
    #outer_cost_metric
    def outer_cost(self,z1,z2,label_dis):
        m = np.zeros([len(z1),len(z2)])
        for idx, i in enumerate(z1):
            for jdx, j in enumerate(z2):
                m[idx][jdx] = ( ( LA.norm( ( np.array(i[0]) - np.array(j[0]) ), 
                                          ord=1) )  #ord = 1: 'cityblock' norm 
                                +  ( label_dis[(i[1],j[1])] ) )
                
        return m   

    #Lipschitz_constant_calculator
    def lipschitz(self,x1,y1,x2,y2,r1,r2,label_dis):
        zr1 = list(zip(x1,y1,r1))
        zr2 = list(zip(x2,y2,r2))

        lip_con = 0
        for i in zr1:
            for j in zr2:
                if j[:2] != i[:2]:
                    d_s = ( LA.norm( ( np.array(i[0]) - np.array(j[0]) ), ord=1) )  #ord = 1: 'cityblock' norm 
                    d_a = label_dis[(i[1],j[1])] 
                    d_sa = d_s + d_a
                    if d_sa > 1e-3: #d_sa != 0 (prevent extreme values)
                        """
                        d_sa > 1e-3: ensures numerical stability, 
                        focusing on pairs with meaningful information
                        """
                        rew_diff = np.abs(i[2] - j[2])
                        ratio = rew_diff/d_sa
                        if ratio > lip_con:
                            lip_con = ratio
        return lip_con      

#Execution
if __name__ == '__main__':
    n = 5 #grid_dimension
    hardns = otdd(
        test_freq=1, #policy_evaluation_frequency 
        states_size=[n,n],#[50,50] grid_dimensions
        setting=0, # [0:sparse | 1:dense | 2:single | 3:stochastic | 4:sinks | 5: strictly-single] rewards_setting
        n_eps=500, #1000:5, 2000:15, 3000:25 [1: 100 | 0: 500] total_episodes
        stop_cri=5, #stopping_criterion
        max_esteps=15 #max_steps_per_episode
        ) 
     
    hardns.iterations(num_iter=50) #20 
    print('complete')

