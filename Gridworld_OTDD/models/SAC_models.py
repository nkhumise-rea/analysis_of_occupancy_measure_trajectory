#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
import math
from numpy import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

from copy import deepcopy
from collections import defaultdict
import pandas as pd
import ot
import seaborn as sns
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
# from torch.distributions import Categorical 

#set directory/location
this_dir = dirname(__file__)
agent_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,agent_dir)
from envs.agent.actor import Actor_Discrete

## environments:
# from task.grid2D import grid #normal
# from task.grid2DMild import grid #single_optimal_path
from envs.task.grid2Dstoc import grid #stochastic_actions
# from task.grid2DHard import grid #single_optimal_path + 2_sink_states + block_state

from envs.task.show import show_grid

class gridworld():
    def __init__(self, 
                states_size=[4,4], 
                rew_setting = 1, #[1:dense, 0:sparse]
                n_eps = 300,
                algo = 'SAC'
                 ):
        # print('rew_setting : ', rew_setting )

        self.algo = algo
        self.rew_setting = rew_setting
        self.state_size = states_size
        
        ## goal_state
        self.goal_state_xx = tuple([ list(np.asarray(self.state_size) - 1) ])
        self.goal_state = np.asarray(self.goal_state_xx[0])

        ## env call
        self.env = grid( #training_env
            num_states=self.state_size,
            goal_state=self.goal_state_xx,
            rew_setting=self.rew_setting, # 1:dense | 0:sparse 
                    ) 
        self.num_actions = self.env.num_actions
        self.num_episodes = n_eps

        #state-action space
        self.y_space = ['up', 'right', 'down', 'left'] #action_space
        self.x_space = [(i,j) for i in range(self.state_size[0]) for j in range(self.state_size[1]) ] #state_space
        self.y_pairs = [(i,j) for i in self.y_space for j in self.y_space ] #action_pairs

    # policy_data_generation
    def policy_data_generation(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        #NN architecture
        num_states = 2
        num_actions = self.num_actions
        num_hidden_l1 = 32 

        # print('self.problem_setting: ', self.problem_setting)
        # print('self.iteration: ', self.iteration)
        file_location = f'../envs/policy_models/SAC/set_{self.problem_setting}/iter_{self.iteration}'

        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))
        # xxx

        for step in range(len(os.listdir(file_location))):

            # retrieve policy_model
            self.pi_net = self.retrieve_policy_model(
                                                num_states,
                                                num_actions,
                                                num_hidden_l1,
                                                step,
                                                file_location)

            # generate_roll-outs
            self.pol_rollouts(step)

        return 
    
    # occupancy_measure_generation
    def occupancy_generation(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        # print('self.problem_setting: ', self.problem_setting)
        # print('self.iteration: ', self.iteration)
        file_location = f'policy_data/SAC/set_{self.problem_setting}/iter_{self.iteration}'
        
        final_policy_data_num = len(os.listdir(file_location)) - 1 #optimal_policy
        initial_policy_data_num = 0 #initial_policy



        # initial_policy & final_policy occupancy_measures
        final_policy_occup,_,final_states_visits= self.occupancy_measure_finite_timeless(
                        final_policy_data_num,
                        file_location
                        )
        
        initial_policy_occup,_,_= self.occupancy_measure_finite_timeless( #initl_states_visits
                        initial_policy_data_num,
                        file_location
                        )
        

        # self.plot_state_visits(final_states_visits)
        # self.plot_state_visits(initl_states_visits)

        trajctory_states_visits = np.zeros_like(final_states_visits)
        # print('trajctory_states_visits: ', trajctory_states_visits)

        # print('occu1: \n', occu1)
        # print('occu2: \n', occu2)

        direct_dis,lip_con = self.OTDD(final_policy_occup,initial_policy_occup)
        # print('direct_dis: ', direct_dis)
        # print('lip_con: ', lip_con)

        max_lip_con = lip_con # maximum lipschitz constant
        succ_dis_col,to_fin_dis_col,to_ini_dis_col = [],[],[]
        # print('max_lip_con_start: ', max_lip_con)
        for update in range(len(os.listdir(file_location))-1):

            # measure from data retrieved 
            policy_occup_1,_,policy1_states_visits = self.occupancy_measure_finite_timeless(
                            update,
                            file_location
                            )
            
            policy_occup_2,_,policy2_states_visits = self.occupancy_measure_finite_timeless(
                            update+1,
                            file_location
                            )
            
            trajctory_states_visits += (policy1_states_visits + policy2_states_visits)

            # #between successive 
            successive_dis,lip_con_1 = self.OTDD(policy_occup_1,policy_occup_2)
            succ_dis_col.append(successive_dis)
            
            # #between final(optimal)_&_others
            to_final_dis,lip_con_2 = self.OTDD(policy_occup_1,final_policy_occup)
            to_fin_dis_col.append(to_final_dis)

            # #between initial_&_others
            to_initial_dis,lip_con_3 = self.OTDD(policy_occup_2,initial_policy_occup)
            to_ini_dis_col.append(to_initial_dis)
            
            # print('lip_con_1: ', lip_con_1)
            # print('lip_con_2: ', lip_con_2)
            # print('lip_con_3: ', lip_con_3)

            max_lip_con = max(lip_con_1,lip_con_2,lip_con_3,max_lip_con)

            # print('max_lip_con: ', max_lip_con)
        
        # print(trajctory_states_visits)
        # trajctory_states_visits /= trajctory_states_visits.sum()
        # self.plot_state_visits(100*trajctory_states_visits)
        # xxx

        policy_trajectory_data = np.asarray([
            direct_dis, #geodesic_distance
            max_lip_con, #lipschitz_constant
            succ_dis_col, #y_k (stepwise distance)
            to_fin_dis_col, #x_k (distance_to_optimal)
            to_ini_dis_col, # (distance_from_initial)
            trajctory_states_visits, #state_visitaion_frequencies
            ],dtype=object) #dtype=object
        
        self.save_policy_traj_data(policy_trajectory_data)

        """  
        ## testing_visualization      
        plt.figure(figsize=(12, 9))

        ## collection_of_successive_distances 
        plt.subplot(2,2,1)
        plt.plot(succ_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(succ_dis_col,0.9),color='orchid') 
        plt.title('stepwise')
        plt.ylabel('distance')
        plt.xlabel('updates')

        ## collection_of_distances-to-optimal 
        plt.subplot(2,2,2)
        plt.plot(to_fin_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(to_fin_dis_col,0.9),color='orchid')
        plt.title('dis_to_optimal')
        plt.ylabel('distance')
        plt.xlabel('updates')

        ## collection_of_distances-to-initial
        plt.subplot(2,2,3)
        plt.plot(to_ini_dis_col,color='green',alpha=0.2)
        plt.plot(self.running_avg(to_ini_dis_col,0.9),color='orchid')
        plt.title('dis_from_initial')
        plt.ylabel('distance')
        plt.xlabel('updates') 

        plt.tight_layout()        
        plt.show()
        """

        return

    # success_rate_evaluation 
    def success_rate(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        file_location = f'policy_data/SAC/set_{self.problem_setting}/iter_{self.iteration}'
        final_policy_data_num = len(os.listdir(file_location)) - 1 #optimal_policy

        # initial_policy & final_policy occupancy_measures
        success_rate = self.success_rate_expansion(
                        final_policy_data_num,
                        file_location
                        )
        # print('SR: ', success_rate)
        if success_rate < 30: print('iter: ', iteration)

    # Occupancy Measure for stationary Policy in Finite MDP
    def success_rate_expansion(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'traj_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        returns = data[3] #returns_in_dataset
        b = [ 1 for i in returns if i == -28.0 ]
        SR_ = 100*sum(b)/len(returns) #success_rate
        # print(sum(b))
        # print(100*sum(b)/len(returns))

        return SR_

    #~~~~~~~~~~~~~~ Policy Evolution Plots ~~~~~~~~~~~~~~~~~~

    # (multiple)_policy_trajectory_evaluation
    def policy_trajectory_evaluation_stats(self,problem_setting='stc'):

        result_data = {
            "ESL" : [], #effort_of_sequential_learning
            "OMR" : [], #optimal_movement_ratio
            "UC": [], #convergence_time
            "Algo" : [],#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            "Runs" : [], #num_runs (iterations)
            "Outcome" : [], #outcome_of_run
            'geodesic': [], #initial_to_final
            }

        for iteration in range(10):
            file_name = f'traj_{iteration}.npy'
            file_location = f'policy_traj/SAC/set_{problem_setting}' #file_location        
            file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
            data = np.load(file_path, allow_pickle=True) 

            updates_file_location = f'policy_data/SAC/set_{problem_setting}/iter_{iteration}'
            num_updates = len(os.listdir(updates_file_location)) - 1 #optimal_policy


            # data_extraction
            p = 6 #float_precision (decimal_places)
            direct_dis = np.round(data[0],p) #geodesic_distance
            max_lip_con = np.round(data[1],p) #lipschitz_constant_candidate
            chords = np.round(data[2],p) #y_k (stepwise distance)
            radii = np.round(data[3],p) #x_k (distance_to_optimal)
            from_initial_dis_col = np.round(data[4],p) # (distance_from_initial)

            # print('direct_dis: ', direct_dis)
            # print('max_lip_con: ', max_lip_con)
            # print('chords: \n', chords)
            # print('radii: \n', radii)
            # print('from_initial_dis_col: \n', from_initial_dis_col)

            # data_collection
            curve_dis = np.round(sum(chords),p) # sum_{y_k}
            radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
        
            #distance-to-optimal not reduced
            non_positive_indices = np.where(radii_diff <= 0) # non-improving transition 
            non_positive_indices = non_positive_indices[0]

            #distance-to-optimal reduced
            positive_indices = np.where(radii_diff > 0) # improving transition
            positive_indices = positive_indices[0]

            non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
            positive_chords = np.array([ chords[i] for i in positive_indices])

            # wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
            usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
            # print('wasted_effort: ', wasted_effort)
            # print('usedful_effort: ', usedful_effort)

            # print('ALGO: ', self.algo)
            # print('ESL: ', np.round(curve_dis/direct_dis,3)) #effort_of_sequential_learning
            # print('OMR: ', np.round(usedful_effort/curve_dis,3)) #optimal_movement_ratio
            # print('UC: ', num_updates)
            # print('Outcome: ', 'pass')

            result_data['Algo'].append(self.algo) #algorithm_name
            result_data['ESL'].append(curve_dis/direct_dis) #effort_of_sequential_learning
            result_data['OMR'].append(usedful_effort/curve_dis) #optimal_movement_ratio
            result_data['UC'].append(num_updates) #convergence_time
            result_data['Runs'].append(500) #num_rollouts
            result_data['geodesic'].append(direct_dis) #num_rollouts

            if iteration == 0 or iteration == 6 :
                result_data['Outcome'].append('fail') #outcome_of_run
            else:
                result_data['Outcome'].append('pass') #outcome_of_run

        ####
        # print(result_data)
        # xxx

        result_display = pd.DataFrame(result_data)
        print('result_display: \n', result_display)

        averages = result_display.groupby(['Outcome']).mean() #.reset_index()
        std = result_display.groupby(['Outcome']).std() #.reset_index()
        print('averages: \n', averages)
        print('std: \n', std)

        # size = result_display.groupby(['Outcome']).size().reset_index(name='Count')
        # # print('size: \n', size)
        # size['Success'] = size['Count']*2
        # #.apply(lambda row: row['Count'] * divisors[row['Mstep']],axis=1)
        # print('size: \n', size)
        
        return

    # (single)_policy_trajectory_evaluation
    def policy_trajectory_evaluation(self,iteration=0,problem_setting='stc'):
        
        file_name = f'traj_{iteration}.npy'
        file_location = f'policy_traj/SAC/set_{problem_setting}' #file_location        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True) 

        updates_file_location = f'policy_data/SAC/set_{problem_setting}/iter_{iteration}'
        num_updates = len(os.listdir(updates_file_location)) - 1 #optimal_policy


        # data_extraction
        p = 6 #float_precision (decimal_places)

        direct_dis = np.round(data[0],p) #geodesic_distance
        max_lip_con = np.round(data[1],p) #lipschitz_constant_candidate
        chords = np.round(data[2],p) #y_k (stepwise distance)
        radii = np.round(data[3],p) #x_k (distance_to_optimal)
        from_initial_dis_col = np.round(data[4],p) # (distance_from_initial)

        # print('direct_dis: ', direct_dis)
        # print('max_lip_con: ', max_lip_con)
        # print('chords: \n', chords)
        # print('radii: \n', radii)
        # print('from_initial_dis_col: \n', from_initial_dis_col)

        # data_collection
        curve_dis = np.round(sum(chords),p) # sum_{y_k}
        radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
      
        #distance-to-optimal not reduced
        non_positive_indices = np.where(radii_diff <= 0) # non-improving transition 
        non_positive_indices = non_positive_indices[0]

        #distance-to-optimal reduced
        positive_indices = np.where(radii_diff > 0) # improving transition
        positive_indices = positive_indices[0]

        non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
        positive_chords = np.array([ chords[i] for i in positive_indices])

        # wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
        usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
        # print('wasted_effort: ', wasted_effort)
        # print('usedful_effort: ', usedful_effort)

        print('ALGO: ', self.algo)
        print('ESL: ', np.round(curve_dis/direct_dis,3)) #effort_of_sequential_learning
        print('OMR: ', np.round(usedful_effort/curve_dis,3)) #optimal_movement_ratio
        print('UC: ', num_updates)
        return

    # (single)_policy_evolution_plot
    def policy_evolution_plot(self,iteration=0,problem_setting='stc'):
        
        file_name = f'traj_{iteration}.npy'
        file_location = f'policy_traj/SAC/set_{problem_setting}' #file_location        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True) 

        # data_extraction
        p = 6 #float_precision (decimal_places)

        chords = np.round(data[2],p) #y_k (stepwise distance)
        radii = np.round(data[3],p) #x_k (distance_to_optimal)

        # data_collection
        radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}

        # plots
        plt.figure(figsize=(16, 10))

        ## Radius behaviour (1) 
        plt.subplot(3,2,1)
        plt.plot(radii/max(radii),'g')
        plt.plot(self.running_avg(radii/max(radii),0.99),'r')
        plt.ylabel('relative dis_to_opt',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.title(f'Radius_Behaviour [{self.algo}]',fontweight='bold')

        ## Chord behaviour (2) 
        plt.subplot(3,2,2)
        plt.plot(chords/max(chords),'b')
        plt.plot(self.running_avg(chords/max(chords),0.99),'r')
        plt.ylabel('relative stepwise_dis',fontweight='bold')
        plt.xlabel('updates',fontweight='bold')
        plt.title(f'Chord_Behaviour [{self.algo}]',fontweight='bold')

        ## Chords vs Radius (3)
        plt.subplot(3,2,3)
        opt_Xs = radii 
        opt_Ys = chords 
        opt_time = np.arange(len(opt_Xs))

        #coloured_segments
        points = np.array([opt_Xs,opt_Ys]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segments,
                            cmap='plasma',
                            norm=plt.Normalize(opt_time.min(),opt_time.max()),
                            alpha=0.2)
        lc.set_array(opt_time)
        plt.gca().add_collection(lc) 

        #plot
        scatter = plt.scatter(opt_Xs,opt_Ys,c=opt_time,
                                cmap='plasma',
                                edgecolor='k'
                                ) #cmap='viridis'
        cbar = plt.colorbar(scatter)
        cbar.set_label('Updates',fontweight='bold')                
        plt.xlabel('Radius',fontweight='bold')
        plt.ylabel('Chord',fontweight='bold')
        plt.title(f'Radius_vs_Chord [{self.algo}]',fontweight='bold')

        ## Chord vs Changing Radii (4)
        plt.subplot(3,2,4)
        XX = radii_diff
        YY = chords[:-1] #[:num_eval-1]
        tt = np.arange(len(XX))

        #coloured_segments
        points = np.array([XX,YY]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segments,cmap='plasma',norm=plt.Normalize(tt.min(),tt.max()),alpha=0.2)
        lc.set_array(tt)
        plt.gca().add_collection(lc)

        #plots
        scat = plt.scatter(XX,YY,c=tt,cmap='plasma',edgecolor='k') #cmap='viridis'
        cbar = plt.colorbar(scat)
        cbar.set_label('Updates',fontweight='bold')

        plt.ylabel('Chord',fontweight='bold')
        plt.xlabel('$\delta$[Radii]',fontweight='bold')
        plt.title(f'$\delta$[Radii]_vs_Chords [{self.algo}]',fontweight='bold')
        plt.xlim(-7,7)

        plt.vlines(x = 0, #convergence_line
            ymin=min(YY),
            ymax=max(YY), 
            colors='black', 
            ls=':',)

        plt.tight_layout()

        xxx
        
        ## 3D Policy Trajectory Visualization (5)
        Xs = radii 
        Ys = chords 
        self._3Dplots(Xs,Ys,name=f'{self.algo}') 

        plt.show()
        return
    
    #successive_differences [radii (distance_to_optimal)]
    def successive_diffs(self,x_col,p): #radii (dis_to_optim)

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        x_k_1 = x_col[1:] #x_{k+1}
        x_k = x_col[:-1] #x_{k}
        x_diff = np.round(x_k - x_k_1,p) # x_k - x_{k+1}#
        
        return x_diff 

    # 3Dplots (chords vs radii vs time)
    def _3Dplots(self,Xs,Ys,name='',fig=None,pos=None):
        time = np.arange(len(Xs))

        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(pos,projection='3d')

        scatter = ax.scatter(Xs,Ys,
                             time,
                             c=time,
                             cmap='plasma',
                            #  edgecolor='k'
                             ) #cmap='viridis'
        ax.plot(Xs,Ys,time,c = 'k',alpha=1)

        cbar = plt.colorbar(scatter)
        cbar.set_label('#updates',fontweight='bold')
        cbar.set_ticks([])

        ax.set_zlabel('#updates',fontweight='bold')
        ax.set_xlabel('distance-to-optimal',fontweight='bold')
        ax.set_ylabel('stepwise-distance',fontweight='bold')

        ax.view_init(elev=20.,azim=-120)
        ax.set_xlim(0,max(Xs))
        ax.set_ylim(0,max(Ys))
        ax.set_zlim(0,max(time))

        ax.set_title(f'{name}',fontweight='bold')
        ax.invert_zaxis()
        ax.set_xlim(0,14)
        ax.set_ylim(0,14)
        # ax.set_visible(False)

        plt.tight_layout()       
        return  

    # state_visits_plot
    def plot_state_visits(self,visits):
        sns.heatmap(data=visits,
                          annot=True,
                          fmt=".2f")   
        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)   
        plt.title(f'State_Visitation (%)')

        plt.show()
        return

    #~~~~~~~~~~~~~~ OTDD ~~~~~~~~~~~~~~~~~~~~

    # Wasserstein: OTDD
    def OTDD(self,occu1,occu2):
        p_sGa1 = self.prob_states_given_actions(occu1)
        p_sGa2 = self.prob_states_given_actions(occu2)

        # print('p_sGa1: ', p_sGa1)
        # print('p_sGa2: ', p_sGa2)
        # print("=======================================")

        action_dis = defaultdict(int) 
        # print(self.y_pairs)

        # action_to_action distance
        for i in self.y_pairs:
            action_dis[i] = self.inner_wasserstein(p_sGa1[i[0]],p_sGa2[i[1]])
        # print('action_dis: \n', action_dis)

        # state-action-pair_to_state-action-pair distance
        otdd, lip_con = self.outer_wasserstein(occu1,occu2,action_dis)

        # print('otdd: ', otdd)
        # print('lip_con: ', lip_con)
        return otdd, lip_con
    
    # Mapping actions -> state_distributions
    def prob_states_given_actions(self,occup):
        p_sGa = defaultdict(lambda: defaultdict(int)) #probabilities [states | actions = a]
        p_a = defaultdict(int) #probabilities [actions]

        # p(A)
        for k in self.y_space:
            for i in self.x_space:
                p_a[k] += occup[(i,k)]
        # print('p_a: \n', p_a)

        # p(S|A)
        for k in self.y_space:
            for i in self.x_space:
                if occup[(i,k)] == 0: 
                    p_sGa[k][i] = 0
                else:
                    p_sGa[k][i] = occup[(i,k)]/p_a[k]
        # print('p_sGa: \n', p_sGa)
        # print("=======================================")
        # xxxx

        return p_sGa

    # Inner_Wasserstein
    def inner_wasserstein(self,p_sGa,q_sGa):
        z = np.asarray(self.x_space)
        CostMatrix = ot.dist(z,z, metric='cityblock') #cost matrix: 'euclidean' 
        # print('CostMatrix: ',CostMatrix)
        
        P = np.asarray(list(p_sGa.values())) #prob_dis
        Q = np.asarray(list(q_sGa.values())) #prob_dis

        # print('P: \n', P)
        # print('Q: \n', Q)
        # print("=======================================")

        if P.sum() != 0 and Q.sum() != 0:
            val = ot.emd2(  P, #A_distribution 
                            Q, #B_distribution
                            M = CostMatrix, #cost_matrix pre-processing
                            numItermax=int(1e6)
                            ) #OT matrix
        else:
            val = None
        # print('val: ', val)

        return val
    
    # Outer_Wasserstein
    def outer_wasserstein(self,occu1,occu2,action_dis):
        z1 = list(occu1.keys())  #state-action space
        z2 = list(occu2.keys())  #state-action space
        CostMatrix, lip_con = self.outer_cost(z1,z2,action_dis) #cost matrix
        # print('CostMatrix: ',CostMatrix)
       
        P = np.asarray(list(occu1.values())) #prob_dis
        Q = np.asarray(list(occu2.values())) #prob_dis

        if P.sum() != 0 and Q.sum() != 0:
            val = ot.emd2(  P, #A_distribution 
                            Q, #B_distribution
                            M = CostMatrix, #cost_matrix pre-processing
                            numItermax=int(1e6)
                            ) #OT matrix
        else:
            val = None
        # print('val: ', val)

        return val, lip_con
    
    # Outer_cost_metric
    def outer_cost(self,z1,z2,action_dis):        
        len_z1 = len(z1) 
        len_z2 = len(z2)    
        m = np.zeros([len_z1,len_z2])
        lip_con = 0
        for idx, i in enumerate(z1):
            for jdx, j in enumerate(z2):
                vector = np.asarray(i[0]) - np.asarray(j[0])

                dis_states = LA.norm( vector, 1) #'cityblock'
                dis_actions = action_dis[ i[1], j[1]]

                if self.rew_setting == 1: 
                    rew_diff = np.abs(self.dense_rewards(i[0]) - self.dense_rewards(j[0]) ) #rewards_difference
                else: 
                    rew_diff = np.abs(self.sparse_rewards(i[0]) - self.sparse_rewards(j[0]) ) #rewards_difference

                if dis_actions == None:
                    m[idx][jdx] = 0
                else: 
                    dis_state_action_pairs = dis_states + dis_actions
                    m[idx][jdx] = dis_state_action_pairs

                    # Lipschitz_constant_calculator
                    if dis_state_action_pairs > 1e-4: #d_sa != 0 (prevent extreme values)
                        """
                        d_sa > 1e-4: ensures numerical stability, 
                        focusing on pairs with meaningful information
                        """
                        ratio = rew_diff/dis_state_action_pairs
                        if ratio > lip_con:
                            lip_con = ratio
        # print('m: ', m) 
        # print('lip_con: ', lip_con) 
        return m, lip_con

    #~~~~~~~~~~~~~~ Sanity Testing ~~~~~~~~~~~~~~~~~~~ 

    # regret_generation
    def regret_generation(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        file_location = f'policy_data/SAC/set_{self.problem_setting}/iter_{self.iteration}'

        optimal_policy_data_num = len(os.listdir(file_location)) - 1
        # print('optimal_policy_data_num: ', optimal_policy_data_num)


        # initial_state_sampled
        init_state = np.asarray(self.env.initial_state)
        # print('init_state: ', init_state)

        """   
        # Unrealiable_Values (Not reflecting True Policy_Performance)     
        value_optimal = self.state_value_function(
                                  optimal_policy_data_num,
                                  file_location
                                  )
        # """
             
        value_optimal = self.state_value_function_finite(
                                  optimal_policy_data_num,
                                  file_location
                                  )
        print('value_optimal: ', value_optimal)

        value_diff, values = [],[]
        for update in range(len(os.listdir(file_location)) - 1): #executes_optimal_policy_model

            # retrieve policy_model_value
            """        
            # Unrealiable_Values (Not reflecting True Policy_Performance)     
            value = self.state_value_function(
                                    update,
                                    file_location
                                    )
            # """
         
            value = self.state_value_function_finite(
                                    update,
                                    file_location
                                    )
            # print('value: ', value)

            values.append(value)
            value_diff.append(np.abs(value_optimal - value))

        
        # print('values: ', values)
        # print('max_value: ', max(values))

        # marked_policies = [i for i,j in enumerate(values) if j > value_optimal ]
        # print('marked_policies: ', marked_policies)


        # """        
        regret = np.cumsum(value_diff)

        ## collection_of_regret 
        plt.plot(regret)
        plt.title('Regret Plot')
        plt.ylabel('regret')
        plt.xlabel('updates')
        plt.show()
        # """

        return   

    # occupancy_measuring_testing && value_functions
    def occupancy_testing(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        file_location = f'policy_data/SAC/set_{self.problem_setting}/iter_{self.iteration}'

        optimal_policy_data_num = len(os.listdir(file_location)) - 1
        # print('optimal_policy_data_num: ', optimal_policy_data_num)

        value_optimal_1 = self.state_value_function_finite(
                                  optimal_policy_data_num,
                                  file_location
                                  )
        # print('value_optimal_1: ', value_optimal_1)
               
        occupancy_optimal, avg_eps_length_optimal,_ = self.occupancy_measure_finite_timeless(
                                  optimal_policy_data_num,
                                  file_location
                                  )
        # print('occupancy_optimal: ', occupancy_optimal)
        # print('occupancy_optimal[1]: ', occupancy_optimal[2])


        value_optimal_2 = self.value_function_occupancy_measure(occupancy_optimal, avg_eps_length_optimal)
        # print('value_optimal_2: ', value_optimal_2)
        print(np.abs(value_optimal_1 - value_optimal_2))
        
        value_error = []
        avg_length = []
        value_col = []
        value_relative_error = []
        for update in range(len(os.listdir(file_location)) - 1): #executes_optimal_policy_model

            # retrieve policy_model_value
            value_1 = self.state_value_function_finite(
                                    update,
                                    file_location
                                    )
            # print('value_1: ', value_1)

            occupancy, avg_eps_length,_ = self.occupancy_measure_finite_timeless(
                                    update,
                                    file_location
                                    )
            
            value_2 = self.value_function_occupancy_measure(occupancy, avg_eps_length)
            # print('value_2: ', value_2)

            value_col.append(value_1)
            # value_error.append(np.abs(value_1 - value_2))
            avg_length.append(avg_eps_length)
            value_relative_error.append(100*np.abs(value_1 - value_2)/value_1)

        
        ## collection_of_values
        plt.plot(value_col)
        plt.title('Values')
        plt.ylabel('values')
        plt.xlabel('updates')
        plt.show()        
              
        ## collection_of_value_errors      
        # plt.plot(value_error)
        # plt.title('Value Error')
        # plt.ylabel('error')
        # plt.xlabel('updates')
        # plt.show()

        ## collection_of_episode_lengths 
        plt.plot(avg_length)
        plt.title('avg episode length')
        plt.ylabel('length')
        plt.xlabel('updates')
        plt.show()

        ## collection_of_relative_value_errors 
        plt.plot(value_relative_error)
        plt.title('Relative Value Error')
        plt.ylabel('rel. error')
        plt.xlabel('updates')
        plt.show()

        return

    # testing_policy_performance
    def performance_test(self,iteration=0, problem_setting='stc'):
        self.iteration = iteration #process_iteration_num
        self.problem_setting = problem_setting #problem_setting ['stc','dns','sps']

        #NN architecture
        num_states = 2
        num_actions = self.num_actions
        num_hidden_l1 = 32 

        # print('self.problem_setting: ', self.problem_setting)
        # print('self.iteration: ', self.iteration)
        file_location = f'../envs/policy_models/SAC/set_{self.problem_setting}/iter_{self.iteration}'

        # print(os.listdir(file_location))
        # print(sorted(os.listdir(file_location)))
        # print(len(os.listdir(file_location)))

        # """     
        ## single_model_testing   
        step = len(os.listdir(file_location)) - 1 #536 #1400 #104
        print('iter: {} | last_policy: {} '.format(iteration,step))
        n = 50 #num_tests
        passed = 0 #num_times_reaching_optimality

        for _ in range(n): #test_n_times
            # retrieve policy_model
            self.pi_net = self.retrieve_policy_model(
                                                num_states,
                                                num_actions,
                                                num_hidden_l1,
                                                step,
                                                file_location)

            _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
            if ret == -28.0:
                passed += 1
                print('act: ', act)
                print('stt: ', stt)
                print('ret: ', ret)
                layout = show_grid(input=trj, setting=self.rew_setting)
                layout.state_traj(size=self.state_size)
            
            if passed >= 2: break
        # """
        
        """
        ## multiple_model_testing
        for step in [731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743]:

            n = 5 #num_tests
            for _ in range(n): #test_n_times
                # retrieve policy_model
                self.pi_net = self.retrieve_policy_model(
                                                    num_states,
                                                    num_actions,
                                                    num_hidden_l1,
                                                    step,
                                                    file_location)

                # Visualize_performance
                _,act,ret,stt,trj = self.learnt_agent(self.env.initial_state)
                print('act: ', act)
                print('stt: ', stt)
                print('ret: ', ret)
                layout = show_grid(input=trj, setting=self.rew_setting)
                layout.state_traj(size=self.state_size)
        # """
        print('complete!')
        return 

    #~~~~~~~~~~~~~~ Agent Evaluation ~~~~~~~~~~~~~~~~~~~

    #trained_agent (for: evaluation)
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
            t_action_index = self.select_action(t_state, True)  
            t_next_state, t_reward, t_done = self.env_test.step(t_state,t_action_index)
            ep_return += t_reward

            #experience_buffer_data
            st_list.append(t_state) #stt_list
            rew_list.append(t_reward) #rew_list

            if t_done:
                trj = deepcopy(st_list)
                trj.append( t_next_state )

            act_list.append(self.env_test.actions[t_action_index])
            t_state = t_next_state
            
        return rew_list, act_list, ep_return, st_list, trj
    
    #action_distribution
    def get_action_probs(self, state): #outputs: action_prob_distr
        with torch.no_grad():
            action_probs, _ = self.pi_net(state)
            # print('embedding: ',self.pi_net.embedding())
            # print('embedding: ',self.pi_net.feature)
            # print('action_probs: ', action_probs)
            return action_probs.numpy()

    #action_selection_mechanism   
    def select_action(self, state, evaluation_episode=False):
        state = torch.tensor(state).float() #convert to tensor
        if evaluation_episode:
            action_index = self.get_action_deterministically(state)
        else:
            action_index = self.get_action_stochastically(state)
        return action_index

    #stochastic_action
    def get_action_stochastically(self, state):
        action_probs = self.get_action_probs(state) #action_prob_distr

        #use action_prob_distr to sample action_index
        action_index = np.random.choice(self.num_actions, p=action_probs)
        return action_index

    #greedy_action
    def get_action_deterministically(self, state):
        action_probs = self.get_action_probs(state) #action_prob_distr

        #use action_prob_distr to select greedy action_index
        action_index = np.argmax(action_probs)
        return action_index

    # policy_rollouts
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

        meta_data = [stt_list,act_list,rew_list,ret_list,stt_du,act_du]
        self.save_traj_data(meta_data,steps) #save_trajectories (state-action samples)
        return 

    #~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~

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

    # running average function
    def running_avg(self,score_col,beta = 0.99):
        cum_score = []
        run_score = score_col[0]
        for score in score_col:
            run_score = beta*run_score + (1.0 - beta)*score
            cum_score.append(run_score)
        return cum_score

    # rolling window avg function
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

    #~~~~~~~~~~~~~~ Retrieving & Saving ~~~~~~~~~~~~~~~~~~~~

    # saving_data
    def save_policy_traj_data(self,meta_data):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'traj_{self.iteration}'
        file_location = f'policy_traj/SAC/set_{self.problem_setting}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    # saving_data
    def save_traj_data(self,meta_data,steps):
        meta_data = np.asarray( meta_data, dtype=object) #data_to_save

        file_name = f'traj_{steps}'
        file_location = f'policy_data/SAC/set_{self.problem_setting}/iter_{self.iteration}' #file_location
        os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent
        
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        np.save(file_path,meta_data) #save_command
        return

    # retrieve policy model
    def retrieve_policy_model(self,
                              num_states,
                              num_actions,
                              num_hidden_l1,
                              steps,
                              file_location):
        
        file_name = f'model_{steps}.pth'     
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name

        #declare model
        pi_net = Actor_Discrete(
                    num_states,
                    num_actions,
                    num_hidden_l1)
        
        pi_net.load_state_dict(torch.load(file_path))            
        return pi_net

    #~~~~~~~~~~~~~~ Value Functions & Occupancy Measures ~~~~~~~~~~~~~~~~~~~~
    
    # Finite value_function
    def state_value_function_finite(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'traj_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

        gamma = 0.99 #discount factor
        avg_cum_rew = 0
        cum_rew_list = []
        for k in states_dict.keys():
            cum_rew = 0
            for step,n_state in enumerate(states_dict[k]):
                if self.rew_setting == 1: 
                    reward = self.dense_rewards(n_state)
                else: 
                    reward = self.sparse_rewards(n_state)
                cum_rew += reward #(gamma**step)*reward 
            cum_rew_list.append(cum_rew)
        avg_cum_rew = np.mean(cum_rew_list)

        return avg_cum_rew
        
    # Occupancy Measure for stationary Policy in Finite MDP
    def occupancy_measure_finite_timeless(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'traj_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        ## state visitation distribution per policy
        policy_state_visits = self.visits_ditribution(x1)

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

        steps_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        probs = defaultdict(int) #probabilities [occupancy_measure]

        # iterate_each_episode_rollout
        for k in actions_dict.keys():
            states = states_dict[k] #states_per_episode_rollout
            actions = actions_dict[k] #actions_per_episode_rollout
            steps_counts[k] = len(actions) #num_decisions_per_episode

            # iterate_over_state-action pairs
            for j in zip(states,actions):
                pair_counts[j] += 1
        total_counts = sum(pair_counts.values())
        avg_eps_length = sum(steps_counts.values())/len(steps_counts)

        #calculate_probability_distribution
        for j in pair_counts:
                probs[j] = pair_counts[j]/total_counts

        return probs, avg_eps_length, policy_state_visits
    
    # state visitation distribution per policy
    def visits_ditribution(self,states):
        state_distr = defaultdict(int) #state_distribution

        uniq_stts,freq = np.unique(
            states, axis=0,return_counts=True
            )

        unique_states = list(map(tuple,uniq_stts)) 
        # freq = 100*freq/freq.sum()
        state_freque_pairs = list(zip(unique_states,freq))

        for j in state_freque_pairs:
            state_distr[j[0]] = j[1] #round(j[1],2)

        visits_distr = np.zeros([self.state_size[0],self.state_size[0]])
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                visits_distr[i][j] = state_distr[(i,j)]
        # print(visits_distr)
        return visits_distr

    # Occupancy Measure for Non-stationary Policy in Finite MDP
    def occupancy_measure_finite(self,steps,file_location): #using_finite_episodes
        
        ## load_data
        file_name = f'traj_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
        x1 = data[0] #states_in_dataset
        y1 = data[1] #actions_in_dataset
        tx1 = np.asarray(data[4]) #len(states)_of_runs
        ty1 = np.asarray(data[5]) #len(actions)_of_runs

        states_dict,actions_dict = {},{}

        # iterate_each_rollout
        for i in range(len(tx1)-1):
            states = x1[tx1[i]:tx1[i+1]] #states_per_rollout
            actions = y1[ty1[i]:ty1[i+1]] #actions_per_rollout
            states_dict[i], actions_dict[i] = states, actions

        time_counts = defaultdict(lambda: defaultdict(int))
        total_counts_at_time =  defaultdict(int)
        probs = defaultdict(lambda: defaultdict(int)) #probabilities [occupancy_measure]

       # iterate_each_episode_rollout
        for k in actions_dict.keys():
            states = states_dict[k] #states_per_episode_rollout
            actions = actions_dict[k] #actions_per_episode_rollout
            
            # iterate_over_state-action pairs
            for t,j in enumerate(zip(states,actions)):
                time_counts[t][j] += 1
                total_counts_at_time[t] += 1 #num_entries_per_step

        #calculate_probability_distribution
        for t in time_counts:
            for item in time_counts[t]:
                probs[t][item] = time_counts[t][item]/total_counts_at_time[t]

        return probs

    # Finite value_function (using Occupancy Measure)
    def value_function_occupancy_measure(self, probs, avg_eps_length):
        cum_product = 0
        y_space = ['up', 'right', 'down', 'left'] #action_space
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
               for k in y_space:
                    if self.rew_setting == 1: 
                        cum_product += self.dense_rewards((i,j))*probs[((i,j),k)]
                    else: 
                        cum_product += self.sparse_rewards((i,j))*probs[((i,j),k)]
                    
        value  = avg_eps_length*cum_product

        return value
    
    # Sparse_rewards_model
    def sparse_rewards(self,state):
        if state == self.goal_state:
            return 1
        return -0.04
    
    # Dense_rewards_model
    def dense_rewards(self,state):
        vector = np.asarray(state) - self.goal_state
        reward = -LA.norm( vector, 1) #'cityblock': - np.sum(np.abs(vector))
        return reward


    ##=========== Experimental ================##

    # Episode_perturbation
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
    
    # Infinite value_function
    def state_value_function(self,steps,file_location): #using_infinite_episodes
        
        ## load_data
        file_name = f'traj_{steps}.npy'
        file_path = abspath(join(this_dir,file_location,file_name)) #file_path: location + name
        data = np.load(file_path, allow_pickle=True)

        ## trajectories
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

            # states,actions = self.perturbation(states,actions) #perturb_every_episode_probabilistic
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

        ## cumulative rewards
        self.goal_state = np.asarray(self.goal_state_xx[0])
        # print('self.goal_state_xx : ', self.goal_state)

        gamma = 0.99 #discount factor
        avg_cum_rew = 0
        cum_rew_list = []
        for k in long_states_dict.keys():
            # print(long_states_dict[k])
            cum_rew = 0
            for step,n_state in enumerate(long_states_dict[k]):
                if self.rew_setting == 1: 
                    reward = self.dense_rewards(n_state)
                else: 
                    reward = self.sparse_rewards(n_state)
                # print(rewards)
                # print(step)
                cum_rew += (gamma**step)*reward
            # print('cum_rew: ', cum_rew)
            # xxx
            cum_rew_list.append(cum_rew)
        avg_cum_rew = np.mean(cum_rew_list)

        return avg_cum_rew


#Execution
if __name__ == '__main__':
    n = 5
    agent = gridworld(states_size=[n,n],
                      rew_setting=1, #[rew_setting, num_eps]
                      n_eps=500,
                      algo = 'SAC',
                      ) 
    
    """
    # state-actions trajectory per episode    
    for i in range(1,10): 
        agent.policy_data_generation(iteration=i,
                                    problem_setting='stc' #['stc','dns','sps']
                                        )
    #"""

    """
    # policy trajectory in occumpancy_measure_space
    for i in range(1,10):
        agent.occupancy_generation(iteration=i,
                                problem_setting='stc' #['stc','dns','sps']
                                    )
    # """

    """ 
    # check_optimality_is_reached
    for i in range(10):   
        agent.performance_test(iteration=i,
                            problem_setting='stc' #['stc','dns','sps']
                                )
    # """

    """ 
    # check_rate_of_reaching_optimality
    for i in range(10):   
        agent.success_rate(iteration=i,
                            problem_setting='stc' #['stc','dns','sps']
                                )
    # """

    """
    # trajectory_evaluation_stats
    agent.policy_trajectory_evaluation_stats(problem_setting='stc' #['stc','dns','sps']
                                )
    
    # """

    """  
    # verifying_occupancy_measure_(error_estimation) 
    # value_functions_evaluations
    agent.occupancy_testing(iteration=0,
                            problem_setting='stc' #['stc','dns','sps']
                                )
    # """

    # """    
    # (single)_policy_evolution/trajectory_plot  
    agent.policy_evolution_plot(iteration=0,
                                problem_setting='stc' #['stc','dns','sps']
                                )
    # """

    """  
    #regret_calculation  
    agent.regret_generation(iteration=0,
                            problem_setting='stc' #['stc','dns','sps']
                                )
    #"""

    """
    # (single)_policy_trajectory_evaluation_stat
    agent.policy_trajectory_evaluation(iteration=0,
                                problem_setting='stc' #['stc','dns','sps']
                                )
    # """