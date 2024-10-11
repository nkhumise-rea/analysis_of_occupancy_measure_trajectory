#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 2024
@author: rea nkhumise
"""

#import libraries
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.feature_selection import r_regression,f_regression,mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import colors
from matplotlib.collections import LineCollection
from tabulate import tabulate

from collections import defaultdict
import ot
import h5py
from numpy import linalg as LA
import seaborn as sns

from copy import deepcopy
import pandas as pd
import gym
import time
import os
import argparse
from os.path import dirname, abspath, join
import sys

##urdf location
this_dir = dirname(__file__)
env_dir = abspath(join(this_dir,'..')) 
sys.path.insert(0,env_dir)


class graph():
    def __init__(self,setting=0,states_size=[3,3],epsilon=1.0,algo='qlearn',decay=0): 
        #dense: dns, sparse: sps, single: sgl (optimal_path), stochastic: stc, sink:snk
        rew_set = ['sps','dns','sgl','stc','snk'] 
        self.setting = rew_set[setting] #rewards_setting
        self.states_size = states_size #grid_dimensions
        self.epsilon = epsilon #$\epsilon$ value
        self.num_stop = 5 #stopping_criteria
        self.max_esteps = 15 #max_episode_steps
        self.test_freq = 1 #policy_evaluation_frequency
        self.strategy_option = ['greedy','random'] #[1: random | 0: greedy]
        self.algo = algo #algorithm
        self.decay = decay #Qlearning_$\epilon$ [0: not_decay | 1: decaying]
        self.algo_label = None #label_of_algorithm

    #describe state_size as string
    def state_label(self): 
        return str(self.states_size) 

    # location_of_saved_files
    def file_location(self):
        
        strategy = self.strategy_option[self.epsilon]

        #common_location
        common_loc = f'ran_grd/stop_{self.num_stop}/step_{self.max_esteps}/{strategy}'
        
        #curve_datasets
        data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
        # print('data_loc: \n', data_loc)

        #curve_images
        image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
        # print('image_loc: \n', image_loc)

        return data_loc,image_loc
    
    # locations_of_saved_files [max_episode_steps]
    def file_location_max_episode_steps(self,max_esteps):
        if self.algo == 'qlearn':
            if self.epsilon == 0 or self.epsilon == 1:
                strategy = self.strategy_option[self.epsilon]

                # print('strategy: ', strategy)
                true_labels = [f'($\epsilon$ = 0)-greedy',f'($\epsilon$ = 1)-greedy']
                self.algo_label = true_labels[self.epsilon] #strategy

                #common_location
                common_loc = f'ran_grd/stop_{self.num_stop}/step_{max_esteps}/{strategy}' #random/greedy
                
                #curve_datasets
                data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
                # print('data_loc: \n', data_loc)

                #curve_images
                image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
                # print('image_loc: \n', image_loc)
            
            elif self.decay == 1:
                # print('strategy: ', 'e-decay')
                self.algo_label = 'e-decay'

                #common_location
                common_loc = f'eps_dcy/stop_{self.num_stop}/step_{max_esteps}/{self.epsilon}' #eps-decay
                
                #curve_datasets
                data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
                # print('data_loc: \n', data_loc)

                #curve_images
                image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
                # print('image_loc: \n', image_loc)

            else:
                # print('strategy: ', 'e-greedy')
                self.algo_label = 'e-greedy'

                #common_location
                common_loc = f'eps_grd/stop_{self.num_stop}/step_{max_esteps}/{self.epsilon}' #eps-greedy
                
                #curve_datasets
                data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
                # print('data_loc: \n', data_loc)

                #curve_images
                image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
                # print('image_loc: \n', image_loc)

        elif self.algo == 'sac': 
            # print('strategy: ', 'SAC')
            self.algo_label = 'SAC'

            #common_location
            common_loc = f'sac_sac/stop_{self.num_stop}/step_{max_esteps}' #SAC
            
            #curve_datasets
            data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
            # print('data_loc: \n', data_loc)

            #curve_images
            image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
            # print('image_loc: \n', image_loc)

        elif self.algo == 'dqn':
            # print('strategy: ', 'DQN')
            self.algo_label = 'DQN'

            #common_location
            common_loc = f'dqn_dqn/stop_{self.num_stop}/step_{max_esteps}' #SAC
            
            #curve_datasets
            data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
            # print('data_loc: \n', data_loc)

            #curve_images
            image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
            # print('image_loc: \n', image_loc)

        elif self.algo == 'ucrl':
            # print('strategy: ', 'UCRL')
            self.algo_label = 'UCRL'

            #common_location
            common_loc = f'ucrl_ucrl/stop_{self.num_stop}/step_{max_esteps}' #UCRL
            
            #curve_datasets
            data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
            # print('data_loc: \n', data_loc)

            #curve_images
            image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
            # print('image_loc: \n', image_loc)

        elif self.algo == 'psrl':
            # print('strategy: ', 'PSRL')
            self.algo_label = 'PSRL'

            #common_location
            common_loc = f'psrl_psrl/stop_{self.num_stop}/step_{max_esteps}' #UCRL
            
            #curve_datasets
            data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
            # print('data_loc: \n', data_loc)

            #curve_images
            image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
            # print('image_loc: \n', image_loc)


        # print('data_loc: \n', data_loc)
        # print('image_loc: \n', image_loc)
        # xxx

        return data_loc,image_loc 

    # location_of_saved_file [hyperparameters]
    def file_location_hyperpm(self,hyperpm):
        self.algo == 'ucrl'
        self.algo_label = 'UCRL'

        #common_location
        common_loc = f'ucrl_hyprm/stop_{self.num_stop}/step_15/delta_{hyperpm}' #UCRL
        
        #curve_datasets
        data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
        # print('data_loc: \n', data_loc)

        #curve_images
        image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
        # print('image_loc: \n', image_loc)
  
        return data_loc,image_loc 
    
    # location_of_saved_file [hyperparameters]
    def file_location_multiruns(self,num_runs):
        self.algo == 'sac'
        self.algo_label = 'SAC'

        #common_location
        # common_loc = f'sac_multiruns/stop_{self.num_stop}/step_40/runs_{num_runs}' #SAC
        # common_loc = f'ucrl_multiruns/stop_{self.num_stop}/step_40/runs_{num_runs}' #UCRL
        # common_loc = f'psrl_multiruns/stop_{self.num_stop}/step_40/runs_{num_runs}' #PSRL
        common_loc = f'dqn_multiruns/stop_{self.num_stop}/step_40/runs_{num_runs}' #DQN
        
        #curve_datasets
        data_loc = f"{common_loc}/dts/{self.setting}/ss_{self.state_label()}" 
        print('data_loc: \n', data_loc)

        #curve_images
        image_loc = f"{common_loc}/vld/{self.setting}/ss_{self.state_label()}"                   
        print('image_loc: \n', image_loc)

 
        return data_loc,image_loc 

    #testing loading_of_data
    def load(self):
        i = 0 

        file,_ = self.file_location()
        run_file = f"curve_{i}.npy" #.hdf5
        file_path = abspath(join(this_dir,file,run_file)) 
        data = np.load(file_path, allow_pickle=True)

        """
        direct_dis,
        steps_col,
        score_col,
        con_eps,
        num_eval,
        ds_stt_col,
        ds_oot_col,
        state_dis,
        regret_col,
        outcome,
        lip_con,
        state_values
        """
        direct_dis = data[0]
        steps_col =data[1]
        score_col = data[2]
        con_eps = data[3]
        num_eval = data[4] #num_updates
        ds_stt_col = data[5]
        ds_oot_col = data[6]
        state_dis = data[7]
        regret_col = data[8]
        outcome = data[9]
        lip_con = data[10]
        state_values = data[11] 
        ds_int_col = data[12]
        con_steps = data[13]

        print('direct_dis: \n', direct_dis) #geodesic_distance
        # print('steps_col: \n', steps_col) #steps_per_episode (policy_evaluation)
        # print('score_col: \n', score_col) #return_per_episode
        print('con_eps: \n', con_eps) #convergence_episode
        print('num_eval: \n', num_eval) #convergence_update
        # print('ds_stt_col: \n', ds_stt_col) #y_k
        # print('ds_oot_col: \n', ds_oot_col) #x_k
        # print('state_dis: \n', state_dis) #state_distribution
        # print('regret_col: \n', regret_col) #regret_k
        print('outcome: \n', outcome) #convergence_indicator
        print('lip_con: \n', lip_con) #lipschitz_constant_candidate
        # print('state_values: \n', state_values) #state_values
        print('ds_oot_col: \n', ds_oot_col) #xi_k (w.r.t start-state)
        print('con_steps: \n', con_steps)  #convergence_step


        self.values3Dplots(state_values)
        # self.timeplots(steps_col,ds_stt_col,ds_oot_col,regret_col,num_eval,outcome,state_values)

        return 

    #state_related_evolution
    def state_value_distribution(self):
        max_steps_col = [15,40,60,90]
        SE_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]
        for max_esteps in max_steps_col: 
            # max_esteps = 15
            outcome_list, convg_steps_list = [],[]
            visits_dict,values_dict = {},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)
            for i in range(50):

                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                num_eval = data[4] #convergence_step
                state_dis = data[7] #state_distribution
                state_values = data[11] #state_values
                outcome = data[9] #convergence_indicator
                lip_con = data[10] #lipschitz_constant_candidate

                # data_collection
                visits_dict[i] = state_dis # State_visits_collection
                values_dict[i] = state_values # State_value_collection
                convg_steps_list.append(num_eval) ##convergence_step_per_run

                outcome_list.append(outcome) #convergence_indicators
                lipsch.append(lip_con) #lipschitz_constants

            pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
            fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
            lipschitz.append(max(lipsch)) #2nd_round_lipschitz_candidate
            
            optimal_visits_dict = {key: visits_dict[key] for key in pass_indexes}
            sub_visits_dict = {key: visits_dict[key] for key in fail_indexes}

            optimal_values_dict = {key: values_dict[key] for key in pass_indexes}
            sub_values_dict = {key: values_dict[key] for key in fail_indexes}
            optimal_mean_values_dict = self.nested_value_stats(optimal_values_dict)
            sub_mean_values_dict = self.nested_value_stats(sub_values_dict)

            true_convg_steps_list = [ convg_steps_list[i] for i in pass_indexes ] #true_convergence_steps_per_run
                    
            opt_visits_nested = [*optimal_visits_dict.values()]
            sub_visits_nested = [*sub_visits_dict.values()]
            opt_visits = [ i for innerlist in opt_visits_nested for i in innerlist ]
            sub_visits = [ i for innerlist in sub_visits_nested for i in innerlist ]

            convg_steps_mean,_ = self.convg_stats(true_convg_steps_list)
            if max_esteps == 15:
                #state_plots_per_max_steps
                self.single_max_steps_state_plots(opt_visits,
                                                    sub_visits,
                                                    optimal_mean_values_dict,
                                                    sub_mean_values_dict) 

            xxxxs
            SE_max_steps[max_esteps]['visit_opt'].append(opt_visits) #optimal_segments
            SE_max_steps[max_esteps]['visit_sub'].append(sub_visits) #suboptimal_segments
            # SE_max_steps[max_esteps]['value_opt'].append(opt_values_means) #optimal_segments
            # SE_max_steps[max_esteps]['value_sub'].append(sub_values_means) #suboptimal_segments
            SE_max_steps[max_esteps]['con'].append(convg_steps_mean) #convergence_step

        # state_plots_for_all_max_steps
        self.all_max_steps_state_plots(max_steps_col,SE_max_steps)
        return 
    
    # state_plots_per_max_steps
    def single_max_steps_state_plots(self,opt_visits,
                                        sub_visits,
                                        optimal_mean_values_dict,
                                        sub_mean_values_dict):
        fig = plt.figure(figsize=(15, 10))
        font  = {'size': 12}
        font_ticks = 16

        ## state_visitation
        plt.subplot(2,2,1)
        state_arry = opt_visits
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
        plt.title(f'State_Visitation (%) [{self.strategy_option[self.epsilon]}*]')  

        plt.subplot(2,2,2)
        state_arry = sub_visits
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
        plt.title(f'State_Visitation (%) [{self.strategy_option[self.epsilon]}_sub]')  
    
        ## Value_Plot
        self.values3Dplots(optimal_mean_values_dict,
                           f'[{self.strategy_option[self.epsilon]}*]',
                           fig,223)
        self.values3Dplots(sub_mean_values_dict,
                           f'[{self.strategy_option[self.epsilon]}_sub]',
                           fig,224)
        
        plt.tight_layout()
        plt.show()
        xxxx
        return  
    
    #simple_single_run_plots
    def sanity_check_plots(self):
        max_steps_col = [15,40,60,90]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]
        for max_esteps in [15]: #max_steps_col: 
            # max_esteps = 15
            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)


            c = random.randint(0,50)
            print('runs: ', c)
            for i in [c]: #range(50):

                # i = 20 ###
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4]+1 #convergence_step
                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    # ds_stt_col[num_eval-1] = 0
                    ds_stt_col[num_eval-1] = 0


                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                ds_int_col = np.round(data[12],p) #origin_xk
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection
                og_radii = ds_int_col[:num_eval] # OX_collection
                curve_dis = np.round(sum(chords),p) #sum_{y_k}_conv
                radii_dict[i] = radii # X_collection
                chords_dict[i] = chords # Y_collection
                
                #y_k/(x_k-x_{k+1}) && x_k-x_{k+1}
                _, x_diff = self.successive_ratios(chords,radii) 
                # ratios, x_diff = self.successive_ratios(ds_stt_col,ds_oot_col) 

                non_positive_indices = np.where(x_diff <= 0)
                non_positive_indices = non_positive_indices[0]

                positive_indices = np.where(x_diff > 0)
                positive_indices = positive_indices[0]
                
                # print('non_positive_indices: ', non_positive_indices)
                # print('positive_indices: ', positive_indices)

                # non_positive_x_diff = np.array([ x_diff[i] for i in non_positive_indices])
                # positive_x_diff = np.array([ x_diff[i] for i in positive_indices])
                # print('non_positive_x_diff: \n', non_positive_x_diff)
                # print('positive_x_diff: \n', positive_x_diff)
                # print('x_diff: \n', x_diff)

                non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
                positive_chords = np.array([ chords[i] for i in positive_indices])
                # print('non_positive_chords: \n', non_positive_chords)
                # print('positive_chords: \n', positive_chords)
                # print('chords: \n', chords)

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                # print('total_effort: ', curve_dis)

                # print('effort_ratio: ', usedful_effort/wasted_effort) #proportion_of_productive_work
                # print('effort_diff: ', usedful_effort-wasted_effort) #net_productive_work

                # print('useful_effort_ratio: ', usedful_effort/curve_dis) #proportion_of_productive_work
                # print('wasted_effort_ratio: ', wasted_effort/curve_dis) #net_productive_work
                
                # print('CR: ', curve_dis/direct_dis) #competitive_ratio
                # print('num_eval: ', num_eval) #convergence_time



                # result_data = {
                #     "ER" : usedful_effort/wasted_effort, #effort_ratio
                #     "OMR" : usedful_effort/curve_dis, #optimal_movement_ratio
                #     "ESL" : curve_dis/direct_dis, #effort_of_sequential_learning
                #     "TC": num_eval, #convergence_time
                #     "Algo" : self.algo_label,#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
                # }
                # result_display = pd.DataFrame(result_data,index=['r1']) #,'r2'
                # result_display.style
                # print(tabulate(result_display,headers='keys',tablefmt='psql'))
                
                """                
                if self.epsilon == 0 or self.epsilon == 1:
                    label_opt = f'{self.strategy_option[self.epsilon]}*'
                    label_sub = f'{self.strategy_option[self.epsilon]}_sub'
                else:
                    label_opt = f'$\epsilon:${self.epsilon}*'
                    label_sub = f'$\epsilon:${self.epsilon}_sub'
                """
                label_opt = f'{self.algo_label}*'
                label_sub = f'{self.algo_label}_sub'

                fig = plt.figure(figsize=(16, 10))

                plt.subplot(3,2,1)
                plt.plot(ds_oot_col/max(ds_oot_col),'g')
                plt.plot(radii/max(ds_oot_col),'b')
                plt.plot(self.running_avg(ds_oot_col/max(ds_oot_col),0.99),'r')
                # plt.plot(self.running_avg(radii,0.99),'k')
                plt.ylabel('radii',fontweight='bold')
                plt.xlabel('steps',fontweight='bold')
                plt.title(f'Radius_Behaviour [{label_opt}]',fontweight='bold')

                plt.subplot(3,2,2)
                plt.plot(ds_stt_col/max(ds_stt_col),'g')
                plt.plot(chords/max(ds_stt_col),'b')
                plt.plot(self.running_avg(ds_stt_col/max(ds_stt_col),0.99),'r')
                plt.ylabel('chords',fontweight='bold')
                plt.xlabel('steps',fontweight='bold')
                plt.title(f'Chord_Behaviour [{label_opt}]',fontweight='bold')
                
                plt.subplot(3,2,3)
                opt_Xs = radii 
                opt_Ys = chords 
                opt_time = np.arange(len(opt_Xs))

                #coloured_segments
                colors = [(0,0,1),(1,0,0)] #blue->red
                # colors = [(0,0,1),(0,1,0),(1,0,0)] #blue->red
                n_bins = 600 #256
                cmap_name = 'blue_red'
                cm = LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)

                # colors = ['blue','lightblue','white','lightcoral','red']
                # cm = ListedColormap(colors)

                points = np.array([opt_Xs,opt_Ys]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1],points[1:]],axis=1)
                lc = LineCollection(segments,
                                    # cmap=cm,
                                    cmap='plasma',
                                    norm=plt.Normalize(opt_time.min(),opt_time.max()),
                                    alpha=0.2)
                lc.set_array(opt_time)
                plt.gca().add_collection(lc) 

                scatter = plt.scatter(opt_Xs,opt_Ys,c=opt_time,
                                    #   cmap=cm,
                                      cmap='plasma',
                                      edgecolor='k'
                                      ) #cmap='viridis'
                cbar = plt.colorbar(scatter)
                cbar.set_label('Steps',fontweight='bold')                
                plt.xlabel('radius',fontweight='bold')
                plt.ylabel('chord',fontweight='bold')
                plt.title(f'Radius_vs_Chord [{label_opt}]',fontweight='bold')

                # plt.show()
                # xxx

                """                
                plt.subplot(3,2,4)
                plt.plot(ratios,'g')
                # plt.plot(self.running_avg(ratios,0.99),'r')
                plt.ylabel('Chord/$\delta$[radii]',fontweight='bold')
                plt.xlabel('steps',fontweight='bold')
                plt.title('Cost_regret_reduction',fontweight='bold')
                """
                
                plt.subplot(3,2,4)
                XX = x_diff
                YY = chords[:-1] #[:num_eval-1]
                tt = np.arange(len(XX))

                #coloured_segments
                points = np.array([XX,YY]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1],points[1:]],axis=1)
                lc = LineCollection(segments,cmap='plasma',norm=plt.Normalize(tt.min(),tt.max()),alpha=0.2)
                lc.set_array(tt)
                plt.gca().add_collection(lc)


                scat = plt.scatter(XX,YY,c=tt,cmap='plasma',edgecolor='k') #cmap='viridis'
                cbar = plt.colorbar(scat)
                cbar.set_label('Steps',fontweight='bold')

                # plt.plot(x_diff,ds_stt_col[:num_eval-1],'^r')
                plt.ylabel('chords',fontweight='bold')
                plt.xlabel('$\delta$[radii]',fontweight='bold')
                plt.title(f'$\delta$[radii]_vs_chords [{label_opt}]',fontweight='bold')
                plt.xlim(-7,7)

                plt.vlines(x = 0, #convergence_line
                   ymin=min(YY),
                   ymax=max(YY), 
                   colors='black', 
                   ls=':',)

                plt.tight_layout()
                
                # plt.subplot(3,2,6)
                Xs = radii 
                Ys = chords 
                # self._3Dplots(Xs,Ys,name=f'{label_opt}') #,fig=fig,pos=326
                # self._3Dplots(Xs,Ys,name=f'{self.algo_label} w.r.t $\pi*$') #,fig=fig,pos=326
                self._3Dplots(Xs,Ys,name=f'{self.algo_label}') #,fig=fig,pos=326

                #polar_plot
                # self.polar_plots_3D(Xs,Ys,Xs,Ys,name=f'{self.algo_label}')

                """                
                Xs = og_radii #radii 
                Ys = chords 
                # self._3Dplots(Xs,Ys,name=f'{label_opt}') #,fig=fig,pos=326
                self._3Dplots(Xs,Ys,name=f'{self.algo_label} w.r.t $\pi_{0}$') #,fig=fig,pos=326
                """


                # plt.tight_layout()
                plt.show()

                xxx

                regrets_dict[i] = regret_col # Regret_collection
                convg_steps_list.append(num_eval) ##convergence_step_per_run
                curve_len_list.append(curve_dis) #trajectory_length
                outcome_list.append(outcome) #convergence_indicators
                lipsch.append(lip_con) #lipschitz_constants
                direct_list.append(direct_dis) #geodesics

                #cost-knowledge-transfer_per_regret-reduction 
                cost_reduction_dict[i] = self.successive_ratios(ds_stt_col,ds_oot_col) #y_k/(x_k-x_{k+1})
                cost_regret_dict[i] = self.successive_area(ds_stt_col,ds_oot_col) #Area(y_k,x_k,x_{k+1}))
                cost_change_dict[i] = self.successive_diff_chords(ds_stt_col) #y_{k+1}-y_k
                radii_change_dict[i] = self.successive_diff_radii(ds_oot_col) #y_{k+1}-y_k
                
            pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
            fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
            lipschitz.append(max(lipsch)) #2nd_round_lipschitz_candidate
            
            optimal_curve_list = [ curve_len_list[i] for i in pass_indexes]
            nu = 1e-3 #non-zero divider
            optimal_geodesic_list = [nu if direct_list[i] == 0 else direct_list[i] for i in pass_indexes ] #optimal_geodesic
            optimal_CR = np.asarray(optimal_curve_list)/np.asarray(optimal_geodesic_list)

        return

    #average_metric_results_table
    def results_check_plots(self):
        max_steps_col = [15,40,60,90]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]

        result_data = {
            "ESL" : [], #effort_of_sequential_learning
            "OMR" : [], #optimal_movement_ratio
            "TC": [], #convergence_time
            "ER" : [], #effort_ratio
            "Algo" : [],#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            "MStep" : [], #maximum_number_of_steps_per_episode
            "Outcome" : [], #outcome_of_run
            # "lip" : [], #lipschitz_constant
            }

        for max_esteps in [40]: #max_steps_col:  #[90]: #
            # max_esteps = 15
            # print('max_esteps: ', max_esteps)

            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)

            # result_data = {
            #     "ER" : usedful_effort/wasted_effort, #effort_ratio
            #     "OMR" : usedful_effort/curve_dis, #optimal_movement_ratio
            #     "ESL" : curve_dis/direct_dis, #effort_of_sequential_learning
            #     "TC": num_eval, #convergence_time
            #     "Algo" : self.algo_label,#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            # }
            # result_display = pd.DataFrame(result_data,index=['r1']) #,'r2'
            # result_display.style
            # print(tabulate(result_display,headers='keys',tablefmt='psql'))

            lenq = 50
            for i in range(lenq): #50
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4] #+1 #convergence_step
                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator
                con_eps = data[3] #convergence_episode

                # print('con_eps: ', con_eps)
                # xx

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    ds_stt_col[num_eval-1] = 0
                
                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection
                curve_dis = np.round(sum(chords),p) #sum_{y_k}_conv
                radii_dict[i] = radii # X_collection
                chords_dict[i] = chords # Y_collection

                print('1st_radii: ', radii[0])
                print('Lst_chord: ', chords[-1])
                xxxx
                
                #y_k/(x_k-x_{k+1}) && x_k-x_{k+1}
                _, x_diff = self.successive_ratios(chords,radii) 
                # ratios, x_diff = self.successive_ratios(ds_stt_col,ds_oot_col) 

                non_positive_indices = np.where(x_diff <= 0)
                non_positive_indices = non_positive_indices[0]

                positive_indices = np.where(x_diff > 0)
                positive_indices = positive_indices[0]
                
                # print('non_positive_indices: ', non_positive_indices)
                # print('positive_indices: ', positive_indices)

                # non_positive_x_diff = np.array([ x_diff[i] for i in non_positive_indices])
                # positive_x_diff = np.array([ x_diff[i] for i in positive_indices])
                # print('non_positive_x_diff: \n', non_positive_x_diff)
                # print('positive_x_diff: \n', positive_x_diff)
                # print('x_diff: \n', x_diff)

                non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
                positive_chords = np.array([ chords[i] for i in positive_indices])
                # print('non_positive_chords: \n', non_positive_chords)
                # print('positive_chords: \n', positive_chords)
                # print('chords: \n', chords)

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                # print('total_effort: ', curve_dis)

                if wasted_effort == 0: #usedful_effort == 0 and 
                    # usedful_effort
                    # print('max_esteps: ', max_esteps)
                    # print('i: ', i)
                    # print('chords: ', chords)
                    # print('curve_dis: ', curve_dis)
                    # print('wasted_effort: ', wasted_effort)
                    # print('usedful_effort: ', usedful_effort)
                    # print('direct_dis: ', direct_dis)
                    # print('non_positive_chords: \n', non_positive_chords)
                    # print('positive_chords: \n', positive_chords)
                    # print('chords: \n', chords)
                    # xxx
                    curve_dis = 1
                    wasted_effort = 1

                # print('effort_ratio: ', usedful_effort/wasted_effort) #proportion_of_productive_work
                # print('effort_diff: ', usedful_effort-wasted_effort) #net_productive_work

                # print('useful_effort_ratio: ', usedful_effort/curve_dis) #proportion_of_productive_work
                # print('wasted_effort_ratio: ', wasted_effort/curve_dis) #net_productive_work
                
                # print('CR: ', curve_dis/direct_dis) #competitive_ratio
                # print('num_eval: ', num_eval) #convergence_time

                if direct_dis == 0.0:
                    direct_dis = 1e-3

                result_data['ER'].append(usedful_effort/wasted_effort) #effort_ratio
                result_data['OMR'].append(usedful_effort/curve_dis) #optimal_movement_ratio
                result_data['ESL'].append(curve_dis/direct_dis) #effort_of_sequential_learning
                result_data['TC'].append(con_eps) #convergence_time
                result_data['Algo'].append(self.algo_label) #algorithm_name
                result_data['MStep'].append(max_esteps) #max_steps_per_episode
                # result_data['lip'].append(lip_con) #lipschitz_constant
                result_data['Outcome'].append(outcome) #outcome_of_run
                
            #     col.append(usedful_effort/wasted_effort)   
            # print(col)
            # print(np.mean(col))
            # print('==============')
            # xxx
        #         pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
        #         fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
        # print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))

            # result_display = pd.DataFrame(result_data)
            # print('result_display: \n', result_display)

            # small = result_display.groupby(['MStep','Outcome']).min()
            # # small = result_display.groupby(['TC']).min()
            # print('small: \n', small)

            # large = result_display.groupby(['MStep','Outcome']).max()
            # print('large: \n', large)
            # xxx


        result_display = pd.DataFrame(result_data)
        print('result_display: \n', result_display)

        averages = result_display.groupby(['MStep','Outcome']).mean() #.reset_index()
        std = result_display.groupby(['MStep','Outcome']).std() #.reset_index()
        print('averages: \n', averages)
        print('std: \n', std)

        size = result_display.groupby(['MStep','Outcome']).size().reset_index(name='Count')
        # print('size: \n', size)
        size['Success'] = size['Count']*2
        #.apply(lambda row: row['Count'] * divisors[row['Mstep']],axis=1)
        print('size: \n', size)
                
        return
    
    #hyperparameter_effects_metric_results_table
    def multruns_evals(self):
        num_runs_col = [1,3,6] #[1,3,6,9]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]

        result_data = {
            "ESL" : [], #effort_of_sequential_learning
            "OMR" : [], #optimal_movement_ratio
            "TC": [], #convergence_time
            "ER" : [], #effort_ratio
            "Algo" : [],#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            "Runs" : [], #num_runs (iterations)
            "Outcome" : [], #outcome_of_run
            }

        for num_runs in [6]: #num_runs_col: #[9]:#
            # num_runs = 1
            # max_esteps = 15
            # print('max_esteps: ', max_esteps)

            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_multiruns(num_runs)

            lenq = 14 #50 #20#
            for i in range(lenq): #50
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4] #+1 #convergence_step
                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    ds_stt_col[num_eval-1] = 0
                
                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection
                curve_dis = np.round(sum(chords),p) #sum_{y_k}_conv
                radii_dict[i] = radii # X_collection
                chords_dict[i] = chords # Y_collection
                
                #y_k/(x_k-x_{k+1}) && x_k-x_{k+1}
                _, x_diff = self.successive_ratios(chords,radii) 
                # ratios, x_diff = self.successive_ratios(ds_stt_col,ds_oot_col) 

                non_positive_indices = np.where(x_diff <= 0)
                non_positive_indices = non_positive_indices[0]

                positive_indices = np.where(x_diff > 0)
                positive_indices = positive_indices[0]
                
                # print('non_positive_indices: ', non_positive_indices)
                # print('positive_indices: ', positive_indices)

                # non_positive_x_diff = np.array([ x_diff[i] for i in non_positive_indices])
                # positive_x_diff = np.array([ x_diff[i] for i in positive_indices])
                # print('non_positive_x_diff: \n', non_positive_x_diff)
                # print('positive_x_diff: \n', positive_x_diff)
                # print('x_diff: \n', x_diff)

                non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
                positive_chords = np.array([ chords[i] for i in positive_indices])
                # print('non_positive_chords: \n', non_positive_chords)
                # print('positive_chords: \n', positive_chords)
                # print('chords: \n', chords)

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                # print('total_effort: ', curve_dis)

                if wasted_effort == 0: #usedful_effort == 0 and 
                    # usedful_effort
                    # print('max_esteps: ', max_esteps)
                    # print('i: ', i)
                    # print('chords: ', chords)
                    # print('curve_dis: ', curve_dis)
                    # print('wasted_effort: ', wasted_effort)
                    # print('usedful_effort: ', usedful_effort)
                    # print('direct_dis: ', direct_dis)
                    # print('non_positive_chords: \n', non_positive_chords)
                    # print('positive_chords: \n', positive_chords)
                    # print('chords: \n', chords)
                    # xxx
                    curve_dis = 1
                    wasted_effort = 1

                # print('effort_ratio: ', usedful_effort/wasted_effort) #proportion_of_productive_work
                # print('effort_diff: ', usedful_effort-wasted_effort) #net_productive_work

                # print('useful_effort_ratio: ', usedful_effort/curve_dis) #proportion_of_productive_work
                # print('wasted_effort_ratio: ', wasted_effort/curve_dis) #net_productive_work
                
                # print('CR: ', curve_dis/direct_dis) #competitive_ratio
                # print('num_eval: ', num_eval) #convergence_time

                if direct_dis == 0.0:
                    direct_dis = 1e-3

                result_data['ER'].append(usedful_effort/wasted_effort) #effort_ratio
                result_data['OMR'].append(usedful_effort/curve_dis) #optimal_movement_ratio
                result_data['ESL'].append(curve_dis/direct_dis) #effort_of_sequential_learning
                result_data['TC'].append(num_eval) #convergence_time
                result_data['Algo'].append(self.algo_label) #algorithm_name
                result_data['Runs'].append(num_runs) #num_rollouts
                # result_data['lip'].append(lip_con) #lipschitz_constant
                result_data['Outcome'].append(outcome) #outcome_of_run
                
            #     col.append(usedful_effort/wasted_effort)   
            # print(col)
            # print(np.mean(col))
            # print('==============')
            # xxx
        #         pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
        #         fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
        # print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))

        result_display = pd.DataFrame(result_data)
        print('result_display: \n', result_display)

        averages = result_display.groupby(['Runs','Outcome']).mean() #.reset_index()
        std = result_display.groupby(['Runs','Outcome']).std() #.reset_index()
        print('averages: \n', averages)
        print('std: \n', std)

        size = result_display.groupby(['Runs','Outcome']).size().reset_index(name='Count')
        # print('size: \n', size)
        size['Success'] = size['Count']*2
        #.apply(lambda row: row['Count'] * divisors[row['Mstep']],axis=1)
        print('size: \n', size)
                
        return

    # saving plots
    def save_images_for_paper(self):
        max_steps_col = [15,40,60,90]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]
        for max_esteps in [15]: #max_steps_col: 
            # max_esteps = 15
            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)


            c = random.randint(0,50)
            print('runs: ', c)
            for i in [c]: #range(50):

                # i = 20 ###
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)
                data = np.load(file_path, allow_pickle=True) 
                
                # data_extraction
                p = 10 #float_precision (decimal_places)

                num_eval = data[4]+1 #convergence_step
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':

                    # ds_stt_col[num_eval-1] = 0
                    ds_stt_col[num_eval-1] = 0
                else: xxxx

                ds_oot_col = np.round(data[6],p) #x_k

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection

                # chords = np.round(data[2],p) #y_k (stepwise distance)
                # radii = np.round(data[3],p) #x_k (distance_to_optimal)

                # data_collection
                # radii_diff = self.successive_diffs(radii,p) # x_k - x_{k+1}
                areal = self.successive_area(chords,radii,p)
                OMR,updates = self.temporal_OMR(chords,radii,p)

                # plots
                # plt.figure(figsize=(16, 10))
                fmt = 'png' #'pdf'
                dp = 'figure' #300
                # plt.rcParams['pdf.fonttype'] = 42 
                # plt.rcParams['jpg.fonttype'] = 42 
                # file_location = f'sac_data/images/iter_{self.iteration}' #file_location
                # os.makedirs(join(this_dir,file_location), exist_ok=True) #create_file_location_if_nonexistent

                ## Radius behaviour (1) 
                plt.figure(figsize=(6,3))
                plt.plot(radii,'g',alpha=1.)
                # plt.plot(radii/max(radii),'g',alpha=1.)
                # plt.plot(self.running_avg(radii/max(radii),0.99),'r')
                plt.ylabel('distance_to_optimal',fontweight='bold',fontsize=16)
                plt.xlabel('#updates',fontweight='bold',fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                # file_name1 = f'to_opt.jpg' #'to_opt.png' #
                # file_path1 = abspath(join(this_dir,file_location,file_name1)) #file_path: location + name
                # plt.savefig(file_path1,dpi=dp, bbox_inches='tight')

                ## Chord behaviour (2) 
                plt.figure(figsize=(6,3))
                plt.plot(chords,'b',alpha=1.)
                # plt.plot(chords/max(chords),'b',alpha=1.)
                # plt.plot(self.running_avg(chords/max(chords),0.99),'r')
                plt.ylabel('stepwise_distance',fontweight='bold',fontsize=16)
                plt.xlabel('#updates',fontweight='bold',fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                # file_name2 = f'stepwise.jpg' #'stepwise.png' #
                # file_path2 = abspath(join(this_dir,file_location,file_name2)) #file_path: location + name
                # plt.savefig(file_path2,dpi=dp, bbox_inches='tight')


                ## Areal Velocity behaviour (3) 
                plt.figure(figsize=(6,3))
                # plt.plot(areal,'k',alpha=0.2)
                # plt.plot(self.running_avg(areal,0.99),'r')
                plt.plot(areal/max(areal),'grey',alpha=1.)
                # plt.plot(self.running_avg(areal/max(areal),0.99),'r')
                plt.ylabel('rel. areal_velocity',fontweight='bold',fontsize=16)
                plt.xlabel('#updates',fontweight='bold',fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                # file_name3 = f'areal.jpg' #'areal.png' #
                # file_path3 = abspath(join(this_dir,file_location,file_name3)) #file_path: location + name
                # plt.savefig(file_path3,dpi=dp, bbox_inches='tight')

                ## OMR behaviour (4) 
                plt.figure(figsize=(6,3))
                plt.plot(updates,OMR,'k')
                plt.ylabel('OMR(k)',fontweight='bold',fontsize=16)
                plt.xlabel('#updates',fontweight='bold',fontsize=16)
                plt.ylim(0.4,1.05)
                # plt.ylim(0.2,0.8)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                # file_name4 = f'OMR.jpg' #'OMR.png' #
                # file_path4 = abspath(join(this_dir,file_location,file_name4)) #file_path: location + name
                # plt.savefig(file_path4,dpi=dp, bbox_inches='tight') #, transparent=True

                ## 3D Policy Trajectory Visualization (5)
                Xs = radii  #radii/max(radii) #
                Ys = chords #chords/max(chords) # 
                self._3Dplots(Xs,Ys,name=f'{self.algo_label}') 
                # file_name5 = f'3D.jpg' #'3D.png' #
                # file_path5 = abspath(join(this_dir,file_location,file_name5)) #file_path: location + name
                # plt.savefig(file_path5, dpi=dp, bbox_inches='tight')  #, format=fmt

                plt.show()
        return
    
    #hyperparameter_effects_metric_results_table
    def sac_multrun_check_plots(self):
        num_runs_col = [1,3,6] #[1,3,6,9]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]

        result_data = {
            "ESL" : [], #effort_of_sequential_learning
            "OMR" : [], #optimal_movement_ratio
            "TC": [], #convergence_time
            "ER" : [], #effort_ratio
            "Algo" : [],#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            "Runs" : [], #num_runs (iterations)
            "Outcome" : [], #outcome_of_run
            }

        for num_runs in num_runs_col: #[9]:#
            # num_runs = 1
            # max_esteps = 15
            # print('max_esteps: ', max_esteps)

            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_multiruns(num_runs)

            lenq = 50 #20#
            for i in range(lenq): #50
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4] #+1 #convergence_step
                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    ds_stt_col[num_eval-1] = 0
                
                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection
                curve_dis = np.round(sum(chords),p) #sum_{y_k}_conv
                radii_dict[i] = radii # X_collection
                chords_dict[i] = chords # Y_collection
                
                #y_k/(x_k-x_{k+1}) && x_k-x_{k+1}
                _, x_diff = self.successive_ratios(chords,radii) 
                # ratios, x_diff = self.successive_ratios(ds_stt_col,ds_oot_col) 

                non_positive_indices = np.where(x_diff <= 0)
                non_positive_indices = non_positive_indices[0]

                positive_indices = np.where(x_diff > 0)
                positive_indices = positive_indices[0]
                
                # print('non_positive_indices: ', non_positive_indices)
                # print('positive_indices: ', positive_indices)

                # non_positive_x_diff = np.array([ x_diff[i] for i in non_positive_indices])
                # positive_x_diff = np.array([ x_diff[i] for i in positive_indices])
                # print('non_positive_x_diff: \n', non_positive_x_diff)
                # print('positive_x_diff: \n', positive_x_diff)
                # print('x_diff: \n', x_diff)

                non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
                positive_chords = np.array([ chords[i] for i in positive_indices])
                # print('non_positive_chords: \n', non_positive_chords)
                # print('positive_chords: \n', positive_chords)
                # print('chords: \n', chords)

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                # print('total_effort: ', curve_dis)

                if wasted_effort == 0: #usedful_effort == 0 and 
                    # usedful_effort
                    # print('max_esteps: ', max_esteps)
                    # print('i: ', i)
                    # print('chords: ', chords)
                    # print('curve_dis: ', curve_dis)
                    # print('wasted_effort: ', wasted_effort)
                    # print('usedful_effort: ', usedful_effort)
                    # print('direct_dis: ', direct_dis)
                    # print('non_positive_chords: \n', non_positive_chords)
                    # print('positive_chords: \n', positive_chords)
                    # print('chords: \n', chords)
                    # xxx
                    curve_dis = 1
                    wasted_effort = 1

                # print('effort_ratio: ', usedful_effort/wasted_effort) #proportion_of_productive_work
                # print('effort_diff: ', usedful_effort-wasted_effort) #net_productive_work

                # print('useful_effort_ratio: ', usedful_effort/curve_dis) #proportion_of_productive_work
                # print('wasted_effort_ratio: ', wasted_effort/curve_dis) #net_productive_work
                
                # print('CR: ', curve_dis/direct_dis) #competitive_ratio
                # print('num_eval: ', num_eval) #convergence_time

                if direct_dis == 0.0:
                    direct_dis = 1e-3

                result_data['ER'].append(usedful_effort/wasted_effort) #effort_ratio
                result_data['OMR'].append(usedful_effort/curve_dis) #optimal_movement_ratio
                result_data['ESL'].append(curve_dis/direct_dis) #effort_of_sequential_learning
                result_data['TC'].append(num_eval) #convergence_time
                result_data['Algo'].append(self.algo_label) #algorithm_name
                result_data['Runs'].append(num_runs) #max_steps_per_episode
                # result_data['lip'].append(lip_con) #lipschitz_constant
                result_data['Outcome'].append(outcome) #outcome_of_run
                
            #     col.append(usedful_effort/wasted_effort)   
            # print(col)
            # print(np.mean(col))
            # print('==============')
            # xxx
        #         pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
        #         fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
        # print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))

        result_display = pd.DataFrame(result_data)
        print('result_display: \n', result_display)

        averages = result_display.groupby(['Runs','Outcome']).mean() #.reset_index()
        std = result_display.groupby(['Runs','Outcome']).std() #.reset_index()
        print('averages: \n', averages)
        print('std: \n', std)

        size = result_display.groupby(['Runs','Outcome']).size().reset_index(name='Count')
        # print('size: \n', size)
        size['Success'] = size['Count']*2
        #.apply(lambda row: row['Count'] * divisors[row['Mstep']],axis=1)
        print('size: \n', size)
                
        return

    #Effort_Seque_Learning_analysis
    def effort_seque_learning_analysis(self):
        max_steps_col = [15,40,60,90]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]

        result_data = {
            "ESL" : [], #effort_of_sequential_learning
            "OMR" : [], #optimal_movement_ratio
            "TC": [], #convergence_time
            "ER" : [], #effort_ratio
            "Algo" : [],#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            "MStep" : [], #maximum_number_of_steps_per_episode
            "Outcome" : [], #outcome_of_run
            # "lip" : [], #lipschitz_constant
            }

        for max_esteps in max_steps_col: 
            # max_esteps = 15
            # print('max_esteps: ', max_esteps)

            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)

            col = []
            lenq = 50
            # if max_esteps == 90:
            #     lenq = 46
            for i in range(lenq): #50
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4] #+1 #convergence_step
                # print('num_eval: ',num_eval )

                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator

                # print('ds_stt_col[num_eval]: ',len(ds_stt_col) )
                # print('ds_stt_col[num_eval]: ',ds_stt_col[num_eval-1] )
                # xxxx

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    ds_stt_col[num_eval-1] = 0
                
                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection
                curve_dis = np.round(sum(chords),p) #sum_{y_k}_conv
                radii_dict[i] = radii # X_collection
                chords_dict[i] = chords # Y_collection
                
                #y_k/(x_k-x_{k+1}) && x_k-x_{k+1}
                _, x_diff = self.successive_ratios(chords,radii) 
                # ratios, x_diff = self.successive_ratios(ds_stt_col,ds_oot_col) 

                non_positive_indices = np.where(x_diff <= 0)
                non_positive_indices = non_positive_indices[0]

                positive_indices = np.where(x_diff > 0)
                positive_indices = positive_indices[0]
                
                # print('non_positive_indices: ', non_positive_indices)
                # print('positive_indices: ', positive_indices)

                # non_positive_x_diff = np.array([ x_diff[i] for i in non_positive_indices])
                # positive_x_diff = np.array([ x_diff[i] for i in positive_indices])
                # print('non_positive_x_diff: \n', non_positive_x_diff)
                # print('positive_x_diff: \n', positive_x_diff)
                # print('x_diff: \n', x_diff)

                non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
                positive_chords = np.array([ chords[i] for i in positive_indices])
                # print('non_positive_chords: \n', non_positive_chords)
                # print('positive_chords: \n', positive_chords)
                # print('chords: \n', chords)

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                # print('total_effort: ', curve_dis)

                if wasted_effort == 0: #usedful_effort == 0 and 
                    # usedful_effort
                    # print('max_esteps: ', max_esteps)
                    # print('i: ', i)
                    # print('chords: ', chords)
                    # print('curve_dis: ', curve_dis)
                    # print('wasted_effort: ', wasted_effort)
                    # print('usedful_effort: ', usedful_effort)
                    # print('direct_dis: ', direct_dis)
                    # print('non_positive_chords: \n', non_positive_chords)
                    # print('positive_chords: \n', positive_chords)
                    # print('chords: \n', chords)
                    # xxx
                    curve_dis = 1
                    wasted_effort = 1

                # print('effort_ratio: ', usedful_effort/wasted_effort) #proportion_of_productive_work
                # print('effort_diff: ', usedful_effort-wasted_effort) #net_productive_work

                # print('useful_effort_ratio: ', usedful_effort/curve_dis) #proportion_of_productive_work
                # print('wasted_effort_ratio: ', wasted_effort/curve_dis) #net_productive_work
                
                # print('CR: ', curve_dis/direct_dis) #competitive_ratio
                # print('num_eval: ', num_eval) #convergence_time

                if direct_dis == 0.0:
                    direct_dis = 1e-3

                result_data['ER'].append(usedful_effort/wasted_effort) #effort_ratio
                result_data['OMR'].append(usedful_effort/curve_dis) #optimal_movement_ratio
                result_data['ESL'].append(curve_dis/direct_dis) #effort_of_sequential_learning
                result_data['TC'].append(num_eval) #convergence_time
                result_data['Algo'].append(self.algo_label) #algorithm_name
                result_data['MStep'].append(max_esteps) #max_steps_per_episode
                # result_data['lip'].append(lip_con) #lipschitz_constant
                result_data['Outcome'].append(outcome) #outcome_of_run
                
            #     col.append(usedful_effort/wasted_effort)   
            # print(col)
            # print(np.mean(col))
            # print('==============')
            # xxx
        #         pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
        #         fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
        # print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))

        result_display = pd.DataFrame(result_data)
        print('result_display: \n', result_display)

        averages = result_display.groupby(['MStep','Outcome']).mean() #.reset_index()
        std = result_display.groupby(['MStep','Outcome']).std() #.reset_index()
        print('averages: \n', averages)
        print('std: \n', std)

        size = result_display.groupby(['MStep','Outcome']).size().reset_index(name='Count')
        # print('size: \n', size)
        size['Success'] = size['Count']*2
        #.apply(lambda row: row['Count'] * divisors[row['Mstep']],axis=1)
        print('size: \n', size)
                
        return

    #hyperparameter_effects_metric_results_table
    def ucrl_hyprm_check_plots(self):
        hyperpm_col = [0.1,0.3,0.5,0.7,0.9]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]

        result_data = {
            "ESL" : [], #effort_of_sequential_learning
            "OMR" : [], #optimal_movement_ratio
            "TC": [], #convergence_time
            "ER" : [], #effort_ratio
            "Algo" : [],#algorithm ['DQN','UCRL','SAC','PSRL','Random','Greedy','E-grd','E-dcy']
            "Hyp" : [], #hyperparameter
            "Outcome" : [], #outcome_of_run
            }

        for hyperpm in hyperpm_col: 
            # max_esteps = 15
            # print('max_esteps: ', max_esteps)

            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_hyperpm(hyperpm)

            lenq = 50
            for i in range(lenq): #50
                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4]+1 #convergence_step
                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    ds_stt_col[num_eval-1] = 0
                
                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                radii = ds_oot_col[:num_eval] # X_collection
                chords = ds_stt_col[:num_eval] # Y_collection
                curve_dis = np.round(sum(chords),p) #sum_{y_k}_conv
                radii_dict[i] = radii # X_collection
                chords_dict[i] = chords # Y_collection
                
                #y_k/(x_k-x_{k+1}) && x_k-x_{k+1}
                _, x_diff = self.successive_ratios(chords,radii) 
                # ratios, x_diff = self.successive_ratios(ds_stt_col,ds_oot_col) 

                non_positive_indices = np.where(x_diff <= 0)
                non_positive_indices = non_positive_indices[0]

                positive_indices = np.where(x_diff > 0)
                positive_indices = positive_indices[0]
                
                # print('non_positive_indices: ', non_positive_indices)
                # print('positive_indices: ', positive_indices)

                # non_positive_x_diff = np.array([ x_diff[i] for i in non_positive_indices])
                # positive_x_diff = np.array([ x_diff[i] for i in positive_indices])
                # print('non_positive_x_diff: \n', non_positive_x_diff)
                # print('positive_x_diff: \n', positive_x_diff)
                # print('x_diff: \n', x_diff)

                non_positive_chords = np.array([ chords[i] for i in non_positive_indices])
                positive_chords = np.array([ chords[i] for i in positive_indices])
                # print('non_positive_chords: \n', non_positive_chords)
                # print('positive_chords: \n', positive_chords)
                # print('chords: \n', chords)

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                # print('total_effort: ', curve_dis)

                if wasted_effort == 0: #usedful_effort == 0 and 
                    # usedful_effort
                    # print('max_esteps: ', max_esteps)
                    # print('i: ', i)
                    # print('chords: ', chords)
                    # print('curve_dis: ', curve_dis)
                    # print('wasted_effort: ', wasted_effort)
                    # print('usedful_effort: ', usedful_effort)
                    # print('direct_dis: ', direct_dis)
                    # print('non_positive_chords: \n', non_positive_chords)
                    # print('positive_chords: \n', positive_chords)
                    # print('chords: \n', chords)
                    # xxx
                    curve_dis = 1
                    wasted_effort = 1

                # print('effort_ratio: ', usedful_effort/wasted_effort) #proportion_of_productive_work
                # print('effort_diff: ', usedful_effort-wasted_effort) #net_productive_work

                # print('useful_effort_ratio: ', usedful_effort/curve_dis) #proportion_of_productive_work
                # print('wasted_effort_ratio: ', wasted_effort/curve_dis) #net_productive_work
                
                # print('CR: ', curve_dis/direct_dis) #competitive_ratio
                # print('num_eval: ', num_eval) #convergence_time

                if direct_dis == 0.0:
                    direct_dis = 1e-3

                result_data['ER'].append(usedful_effort/wasted_effort) #effort_ratio
                result_data['OMR'].append(usedful_effort/curve_dis) #optimal_movement_ratio
                result_data['ESL'].append(curve_dis/direct_dis) #effort_of_sequential_learning
                result_data['TC'].append(num_eval) #convergence_time
                result_data['Algo'].append(self.algo_label) #algorithm_name
                result_data['Hyp'].append(hyperpm) #max_steps_per_episode
                # result_data['lip'].append(lip_con) #lipschitz_constant
                result_data['Outcome'].append(outcome) #outcome_of_run
                
            #     col.append(usedful_effort/wasted_effort)   
            # print(col)
            # print(np.mean(col))
            # print('==============')
            # xxx
        #         pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
        #         fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
        # print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))

        result_display = pd.DataFrame(result_data)
        print('result_display: \n', result_display)

        averages = result_display.groupby(['Hyp','Outcome']).mean() #.reset_index()
        std = result_display.groupby(['Hyp','Outcome']).std() #.reset_index()
        print('averages: \n', averages)
        print('std: \n', std)

        size = result_display.groupby(['Hyp','Outcome']).size().reset_index(name='Count')
        # print('size: \n', size)
        size['Success'] = size['Count']*2
        #.apply(lambda row: row['Count'] * divisors[row['Mstep']],axis=1)
        print('size: \n', size)
                
        return

    # regret_convolution (regret-related)
    def regret_convolution(self):
        max_steps_col = [15,40,60,90]
        RC_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]
        for max_esteps in max_steps_col: 
            # max_esteps = 15
            outcome_list,convg_steps_list,curve_len_list,direct_list = [],[],[],[]
            cost_reduction_dict,chords_dict,regrets_dict,cost_regret_dict = {},{},{},{}
            cost_change_dict,radii_change_dict,radii_dict = {},{},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)
            for i in range(50):

                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)
                
                num_eval = data[4] #convergence_step
                direct_dis = np.round(data[0],p) #geodesic_distance
                ds_stt_col = np.round(data[5],p) #y_k
                ds_oot_col = np.round(data[6],p) #x_k
                regret_col = np.round(data[8],p) #regret_k
                curve_dis = np.round(sum(ds_stt_col[:num_eval]),p) #sum_{y_k}_conv
                outcome = data[9] #convergence_indicator
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                #modification y_N = x_N (if convergence is reached)!!!
                if outcome == 'Pass':
                    ds_stt_col[num_eval-1] = 0

                # data_collection
                radii_dict[i] = ds_oot_col # X_collection
                regrets_dict[i] = regret_col # Regret_collection
                convg_steps_list.append(num_eval) ##convergence_step_per_run
                curve_len_list.append(curve_dis) #trajectory_length
                outcome_list.append(outcome) #convergence_indicators
                lipsch.append(lip_con) #lipschitz_constants
                direct_list.append(direct_dis) #geodesics

                #cost-knowledge-transfer_per_regret-reduction 
                chords_dict[i] = ds_stt_col # Y_collection
                _,cost_reduction_dict[i] = self.successive_ratios(ds_stt_col,ds_oot_col) #y_k/(x_k-x_{k+1})
                cost_regret_dict[i] = self.successive_area(ds_stt_col,ds_oot_col) #Area(y_k,x_k,x_{k+1}))
                cost_change_dict[i] = self.successive_diff_chords(ds_stt_col) #y_{k+1}-y_k
                radii_change_dict[i] = self.successive_diff_radii(ds_oot_col) #y_{k+1}-y_k
               
            pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
            fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
            lipschitz.append(max(lipsch)) #2nd_round_lipschitz_candidate
            
            optimal_curve_list = [ curve_len_list[i] for i in pass_indexes]
            nu = 1e-3 #non-zero divider
            optimal_geodesic_list = [nu if direct_list[i] == 0 else direct_list[i] for i in pass_indexes ] #optimal_geodesic
            optimal_CR = np.asarray(optimal_curve_list)/np.asarray(optimal_geodesic_list)
            
            # print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))
            # print('avg(optimal_curve_len): ', np.mean(optimal_curve_list))
            # print('avg(optimal_CR): ', np.mean(optimal_CR))
            # print('=========================================')

            optimal_cost_reduction_dict = {key: cost_reduction_dict[key] for key in pass_indexes}
            sub_cost_reduction_dict = {key: cost_reduction_dict[key] for key in fail_indexes}

            optimal_cost_regret_dict = {key: cost_regret_dict[key] for key in pass_indexes}
            sub_cost_regret_dict = {key: cost_regret_dict[key] for key in fail_indexes}

            optimal_cost_change_dict = {key: cost_change_dict[key] for key in pass_indexes}
            sub_cost_change_dict = {key: cost_change_dict[key] for key in fail_indexes}
        
            optimal_radii_change_dict = {key: radii_change_dict[key] for key in pass_indexes}
            sub_radii_change_dict = {key: radii_change_dict[key] for key in fail_indexes}

            optimal_chords_dict = {key: chords_dict[key] for key in pass_indexes}
            sub_chords_dict = {key: chords_dict[key] for key in fail_indexes}

            optimal_radii_dict = {key: radii_dict[key] for key in pass_indexes}
            sub_radii_dict = {key: radii_dict[key] for key in fail_indexes}

            optimal_regrets_dict = {key: regrets_dict[key] for key in pass_indexes}
            sub_regrets_dict = {key: regrets_dict[key] for key in fail_indexes}

            true_convg_steps_list = [ convg_steps_list[i] for i in pass_indexes ] #true_convergence_steps_per_run
                        
            opt_cost_reduction_means, _ = self.segments_stats(optimal_cost_reduction_dict)
            sub_cost_reduction_means, _ = self.segments_stats(sub_cost_reduction_dict)

            opt_cost_regret_means, _ = self.segments_stats(optimal_cost_regret_dict)
            sub_cost_regret_means, _ = self.segments_stats(sub_cost_regret_dict)

            opt_cost_change_means, _ = self.segments_stats(optimal_cost_change_dict)
            sub_cost_change_means, _ = self.segments_stats(sub_cost_change_dict)

            opt_radii_change_means, _ = self.segments_stats(optimal_radii_change_dict)
            sub_radii_change_means, _ = self.segments_stats(sub_radii_change_dict)

            opt_chords_means, _ = self.segments_stats(optimal_chords_dict)
            sub_chords_means, _ = self.segments_stats(sub_chords_dict)

            opt_radii_means, _ = self.segments_stats(optimal_radii_dict)
            sub_radii_means, _ = self.segments_stats(sub_radii_dict)

            opt_regrets_means, _ = self.segments_stats(optimal_regrets_dict)
            sub_regrets_means, _ = self.segments_stats(sub_regrets_dict)
            convg_steps_mean,_ = self.convg_stats(true_convg_steps_list)


            """            
            if opt_chords_means.size > 0:
                coefficients = np.polyfit(opt_chords_means,
                                        opt_cost_reduction_means,
                                        1)
                slope,intercept = coefficients           
                # corr = np.corrcoef(opt_chords_means,
                #                     opt_cost_reduction_means)[0,1]
                # print('corr: ',corr) 
                # print(opt_chords_means.reshape(-1,1))
                # xxx
                corr = r_regression(opt_chords_means.reshape(-1,1),
                                        opt_cost_reduction_means)[0]
                # print('f_corr: ',corr) 

                mutual_info = mutual_info_regression(opt_chords_means.reshape(-1,1),
                                        opt_cost_reduction_means)[0]

                # print('opt_chords_means: ', opt_chords_means)
                # print('opt_cost_reduction_means: ', opt_cost_reduction_means)
                data = np.asarray(list(zip(opt_chords_means,opt_cost_reduction_means)))
                # print('data: ', data)

                #standardize_data
                scaler = StandardScaler()
                data_standard = scaler.fit_transform(data)

                #apply PCA
                pca = PCA(n_components=2)
                pca.fit(data_standard)

                data_transformed = pca.transform(data_standard)
                p_com = pca.components_
                pc1 = p_com[:,0]
                pc2 = p_com[:,1]

                              
                print('pca.components_: ',pca.components_)
                print('pc1: ',pc1)
                print('pc1: ',np.arcsin(pc1[1]/pc1[0])*180/np.pi)
                print('pca.explained_variance_: ',pca.explained_variance_)
                print('pca.explained_variance_ratio_: ',pca.explained_variance_ratio_)
                # print('pca.transformed_data: ',data_transformed)

                plt.subplot(1,2,1)
                plt.plot(data_transformed[:,0],data_transformed[:,1],'*',alpha=0.2)

                plt.subplot(1,2,2)
                plt.plot(data[:,0],data[:,1],'*k',alpha=0.2)
                plt.plot(0,0,'o',color='red')
                plt.plot(*pca.components_[:,0],'o',color='k')
                plt.plot(*pca.components_[:,1],'o',color='green')

                plt.show()
                
            else: slope,intercept,corr,mutual_info = np.array(0.0),np.array(0.0),np.array(0.0),np.array(0.0)

            # print(slope,intercept,corr,mutual_info)
            
            # print('mi: ',mutual_info)
            # xx
            # """
            
            if max_esteps == 0:

                print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))
                print('avg(optimal_curve_len): ', np.mean(optimal_curve_list))
                print('avg(optimal_CR): ', np.mean(optimal_CR))


                self.single_max_steps_chords_radii(
                                                opt_chords_means,
                                                sub_chords_means,
                                                opt_radii_means,
                                                sub_radii_means,
                                                opt_radii_change_means,
                                                sub_radii_change_means,
                                                convg_steps_mean)
                # xxx
                
                # self.polar_plots(opt_chords_means,
                #                  opt_radii_means,
                #                  sub_chords_means,
                #                  sub_radii_means)
                # xxx

                #regret_convolution_plot_per_max_steps
                self.single_max_steps_regret_plots_chords(
                                                    opt_chords_means,
                                                    sub_chords_means,
                                                    opt_radii_means,
                                                    sub_radii_means,
                                                    opt_radii_change_means,
                                                    sub_radii_change_means,
                                                    convg_steps_mean) 

                self.single_max_steps_regret_plots_reduction(
                                                    opt_cost_reduction_means,
                                                    sub_cost_reduction_means,
                                                    opt_cost_change_means,
                                                    sub_cost_change_means,
                                                    opt_regrets_means,
                                                    sub_regrets_means,
                                                    convg_steps_mean) 

            # xxxxs
            RC_max_steps[max_esteps]['cost_opt'].append(opt_cost_reduction_means) #optimal_segments
            RC_max_steps[max_esteps]['cost_sub'].append(sub_cost_reduction_means) #suboptimal_segments
            RC_max_steps[max_esteps]['chrd_opt'].append(opt_chords_means) #optimal_segments
            RC_max_steps[max_esteps]['chrd_sub'].append(sub_chords_means) #suboptimal_segments
            # RC_max_steps[max_esteps]['chrd_opt'].append(opt_cost_change_means) #optimal_segments
            # RC_max_steps[max_esteps]['chrd_sub'].append(sub_cost_change_means) #suboptimal_segments
            RC_max_steps[max_esteps]['regt_opt'].append(opt_regrets_means) #optimal_segments
            RC_max_steps[max_esteps]['area_opt'].append(opt_cost_regret_means) #optimal_segments
            RC_max_steps[max_esteps]['area_sub'].append(sub_cost_regret_means) #suboptimal_segments
            RC_max_steps[max_esteps]['con'].append(convg_steps_mean) #convergence_step
            # RC_max_steps[max_esteps]['slope'].append(slope) #slope
            # RC_max_steps[max_esteps]['corr'].append(corr) #correlation_coeff
            # RC_max_steps[max_esteps]['intc'].append(intercept) #intercept
            # RC_max_steps[max_esteps]['mutual_info'].append(mutual_info) #mutual_info

        # lipschitz constant
        print('lipschitz: ', max(lipschitz))

        # #policy_evolution_plot_for_all_max_steps
        self.all_max_steps_regret_plots(max_steps_col,RC_max_steps)
        return
    
    # regret_convolution_(chords_vs_radii)
    def single_max_steps_chords_radii(self,
                                      opt_chords_means,
                                      sub_chords_means,
                                      opt_radii_means,
                                      sub_radii_means,
                                      opt_radii_change_means,
                                      sub_radii_change_means,
                                      convg_steps_mean):  
        
        # print('opt_chords_means: ', len(opt_chords_means[:-1]))
        # print('opt_chords_means: ', len(opt_radii_means))
        # print('opt_radii_change_means: ', len(opt_radii_change_means))

        # print('sub_chords_means: ', len(sub_chords_means))
        # print('sub_chords_means: ', len(sub_radii_means))
        # print('sub_radii_change_means: ', len(sub_radii_change_means))
        # xxx

        fig = plt.figure(figsize=(16, 10))
        font  = {'size': 12}
        font_ticks = 16

        if self.epsilon == 0 or self.epsilon == 1:
            label_opt = f'{self.strategy_option[self.epsilon]}*'
            label_sub = f'{self.strategy_option[self.epsilon]}_sub'
        else:
            label_opt = f'$\epsilon:${self.epsilon}*'
            label_sub = f'$\epsilon:${self.epsilon}_sub'

        ## x_{k} vs y_{k} vs steps
        plt.subplot(2,2,1)
        opt_Xs = opt_radii_means 
        opt_Ys = opt_chords_means 
        opt_time = np.arange(len(opt_Xs))

        scatter = plt.scatter(opt_Xs,opt_Ys,c=opt_time,cmap='viridis',edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Steps',fontweight='bold')
        
        plt.xlabel('Radius',fontweight='bold')
        plt.ylabel('Chord',fontweight='bold')


        plt.subplot(2,2,2)
        sub_Xs = sub_radii_means 
        sub_Ys = sub_chords_means 
        sub_time = np.arange(len(sub_Xs))

        scatter = plt.scatter(sub_Xs,sub_Ys,c=sub_time,cmap='viridis',edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Steps',fontweight='bold')
        
        plt.xlabel('Radius',fontweight='bold')
        plt.ylabel('Chord',fontweight='bold')

        ## x_{k+1}-x_{k} vs y_k vs steps
        plt.subplot(2,2,3)
        opt_dXs = opt_radii_change_means 
        opt_dYs = opt_chords_means[:-1] 
        opt_dtime = np.arange(len(opt_dXs))

        scatter = plt.scatter(opt_dXs,opt_dYs,c=opt_dtime,cmap='viridis',edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Steps',fontweight='bold')
        
        plt.xlabel('$\delta$(Radius)',fontweight='bold')
        plt.ylabel('Chord',fontweight='bold')

        ## x_{k+1}-x_{k} vs y_k vs steps
        plt.subplot(2,2,4)
        sub_dXs = sub_radii_change_means 
        sub_dYs = sub_chords_means[:-1] 
        sub_dtime = np.arange(len(sub_dXs))

        scatter = plt.scatter(sub_dXs,sub_dYs,c=sub_dtime,cmap='viridis',edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Steps',fontweight='bold')
        
        plt.xlabel('$\delta$(Radius)',fontweight='bold')
        plt.ylabel('Chord',fontweight='bold')

        plt.tight_layout()
        plt.show()
        return
    
    # regret_convolution_plot_per_max_steps (chords)
    def single_max_steps_regret_plots_chords(self,
                                      opt_chords_means,
                                      sub_chords_means,
                                      opt_radii_means,
                                      sub_radii_means,
                                      opt_cost_regret_means,
                                      sub_cost_regret_means,
                                      convg_steps_mean):  

        fig = plt.figure(figsize=(16, 10))
        font  = {'size': 12}
        font_ticks = 16

        if self.epsilon == 0 or self.epsilon == 1:
            label_opt = f'{self.strategy_option[self.epsilon]}*'
            label_sub = f'{self.strategy_option[self.epsilon]}_sub'
        else:
            label_opt = f'$\epsilon:${self.epsilon}*'
            label_sub = f'$\epsilon:${self.epsilon}_sub'

        ## Chords_Plot
        plt.subplot(3,2,1)
        if opt_chords_means.size > 0:
            opt_smooth_segments = self.running_avg(opt_chords_means,0.99) #smoothened_segments
            plt.plot(opt_smooth_segments,
                    '-r',
                    label=label_opt
                    )
            plt.plot(opt_chords_means,'-r', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
            
        plt.ylabel('$\Delta$', fontdict=font) #$\Delta / \Delta_{max}$
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(a)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.ylim(0,4) #10
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        plt.subplot(3,2,2)
        if sub_chords_means.size > 0:
            plt.plot(self.running_avg(sub_chords_means,0.99) ,
                    '-r',
                    label=label_sub
                    )
            plt.plot(sub_chords_means,'-r', alpha=0.2)
        
        plt.ylabel('$\Delta$', fontdict=font) #$\Delta / \Delta_{max}$
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(b)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.ylim(0,4) #10
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

    
        ## Radii_Plot
        plt.subplot(3,2,3)
        if opt_radii_means.size > 0:
            opt_smooth_segments = self.running_avg(opt_radii_means,0.99) #smoothened_segments
            plt.plot(opt_smooth_segments,
                    '-c',
                    label=label_opt
                    )
            plt.plot(opt_radii_means,'-c', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
        
        plt.ylabel('x_k', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(c)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.ylim(0,8)
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        plt.subplot(3,2,4)
        if sub_radii_means.size > 0:
            plt.plot(self.running_avg(sub_radii_means,0.99) ,
                    '-c',
                    label=label_sub
                    )
            plt.plot(sub_radii_means,'-c', alpha=0.2)
        
        plt.ylabel('x_k', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(d)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.ylim(0,8)
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        ## Area plots   
        plt.subplot(3,2,5)
        if opt_cost_regret_means.size > 0:
            opt_smooth_segments = self.running_avg(opt_cost_regret_means,0.99) #smoothened_segments
            plt.plot(opt_smooth_segments,
                    '-g',
                    label=label_opt
                    )
            plt.plot(opt_cost_regret_means,'-g', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Area', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(e)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.legend()
        plt.grid()

        plt.subplot(3,2,6)
        if sub_cost_regret_means.size > 0:
            sub_smooth_segments = self.running_avg(sub_cost_regret_means,0.99) #smoothened_segments
            plt.plot(sub_smooth_segments,
                    '-g',
                    label=label_sub
                    )
            plt.plot(sub_cost_regret_means,'-g', alpha=0.2) #mean_of_all_runs
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Area', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(f)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.legend()
        plt.grid()
        plt.tight_layout()

        ## x_k-vs-y_k-vs-steps
        Xs = opt_radii_means 
        Ys = opt_chords_means 
        self._3Dplots(Xs,Ys,name=label_opt,fig=None,pos=None)       
        return
    
    # regret_convolution_plot_per_max_steps (chords)
    def single_max_steps_regret_plots_reduction(self,
                                      opt_cost_reduction_means,
                                      sub_cost_reduction_means,
                                      opt_cost_change_means,
                                      sub_cost_change_means,
                                      opt_regrets_means,
                                      sub_regrets_means,
                                      convg_steps_mean):
        
        fig = plt.figure(figsize=(16, 10))
        font  = {'size': 12}
        font_ticks = 16

        if self.epsilon == 0 or self.epsilon == 1:
            label_opt = f'{self.strategy_option[self.epsilon]}*'
            label_sub = f'{self.strategy_option[self.epsilon]}_sub'
        else:
            label_opt = f'$\epsilon:${self.epsilon}*'
            label_sub = f'$\epsilon:${self.epsilon}_sub'

        ## effort_per_regret_reduction_Plot
        plt.subplot(3,2,1)
        if opt_cost_reduction_means.size > 0:
            opt_smooth_segments = self.running_avg(opt_cost_reduction_means,0.99) #smoothened_segments
            plt.plot(opt_smooth_segments,
                    '-r',
                    label=label_opt
                    )
            # plt.plot(opt_cost_reduction_means,'-r', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
        
        plt.ylabel('$\Delta$/$\delta$[X]', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(a)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.ylim(-8,18)#(-16,17) #10
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        plt.subplot(3,2,2)
        if sub_cost_reduction_means.size > 0:
            plt.plot(self.running_avg(sub_cost_reduction_means,0.99) ,
                    '-r',
                    label=label_sub
                    )
            # plt.plot(sub_cost_reduction_means,'-r', alpha=0.2)
        
        plt.ylabel('$\Delta$/$\delta$[X]', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(b)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        # plt.ylim(-1,6) #10
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        ## chord_changes_Plot
        plt.subplot(3,2,3)
        if opt_cost_change_means.size > 0:
            opt_smooth_segments = self.running_avg(opt_cost_change_means,0.99) #smoothened_segments
            plt.plot(opt_smooth_segments,
                    '-c',
                    label=label_opt
                    )
            plt.plot(opt_cost_change_means,'-c', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
        
        plt.ylabel('y_{k+1}/y_k', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(c)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.ylim(-0.5,6) #10
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        plt.subplot(3,2,4)
        if sub_cost_change_means.size > 0:
            plt.plot(self.running_avg(sub_cost_change_means,0.99) ,
                    '-c',
                    label=label_sub
                    )
            plt.plot(sub_cost_change_means,'-c', alpha=0.2)

        plt.ylabel('y_{k+1}/y_k', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(d)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        # plt.ylim(0,4) #10
        plt.xlim(0,7500) #3500
        plt.legend()
        plt.grid()

        ## Regret_plots   
        plt.subplot(3,2,5)
        if opt_regrets_means.size > 0:
            opt_smooth_segments = self.running_avg(np.abs(opt_regrets_means),0.99) #smoothened_segments
            plt.plot(opt_smooth_segments,
                    '-g',
                    label=label_opt
                    )
            plt.plot(np.abs(opt_regrets_means),'-g', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Regret', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(e)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
        plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
        plt.legend()
        plt.grid()

        plt.subplot(3,2,6)
        if sub_regrets_means.size > 0:
            sub_smooth_segments = self.running_avg(np.abs(sub_regrets_means),0.99) #smoothened_segments
            plt.plot(sub_smooth_segments,
                    '-g',
                    label=label_sub
                    )
            plt.plot(np.abs(sub_regrets_means),'-g', alpha=0.2) #mean_of_all_runs
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('pseudo-Regret', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('(f)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
        plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
        plt.close()
        
        return

    #polar_plot_computations
    def polar_plot_comp(self,y_col,x_col):

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        #y_col[:-1] -> {0:N-1} : y_k
        x_k = x_col[:-1] #a
        x_k_1 = x_col[1:] #b
        y_k = y_col[:-1] #c
        eta = 1e-4 #small value to avoid zero division

        numerator = x_k_1*x_k_1 + x_k*x_k - y_k*y_k
        denomenator = 2*x_k_1*x_k

        omit_indices = []
        for idx,val in enumerate(denomenator[:-1]):
            if val == 0: omit_indices.append(idx)

        deno = np.delete(denomenator[:-1],omit_indices)
        nume = np.delete(numerator[:-1],omit_indices)
        x_k_mod = np.delete(x_k_1,omit_indices)
        x_k_mean = self.running_avg(x_k_mod,0.99) 
        div = nume/deno


        for idx,k in enumerate(div):
            if 1 < k: div[idx] = 1
            elif k < -1: div[idx] = -1

        angle_k = np.arccos(div)
        angle_k = np.insert(angle_k,-1,0)
        cum_angle_k = np.cumsum(angle_k)
        return cum_angle_k,x_k_mod,x_k_mean

    #polar_plot
    def polar_plots_3D(self,
                       opt_radii_means, #X
                       opt_chords_means, #Y
                       sub_radii_means, #X
                       sub_chords_means, #Y
                       name
                       ): 
        
        opt_angle,opt_x,opt_mean = self.polar_plot_comp(opt_chords_means,opt_radii_means)
        sub_angle,sub_x,sub_mean = self.polar_plot_comp(sub_chords_means,sub_radii_means)
        opt_time = np.arange(len(opt_angle))
        sub_time = np.arange(len(sub_angle))

        #convert to Cartesian coordinates
        opt_xc = opt_x*np.cos(opt_angle)
        opt_yc = opt_x*np.sin(opt_angle)

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111,projection='3d') #'polar'
        font  = {'size': 12}
        font_ticks = 16
        
        sc1 = ax1.scatter(opt_xc,opt_yc,opt_time,c=opt_time,cmap='plasma') #cmap='viridis'
        ax1.plot([0,0],[0,0],[opt_time[0],opt_time[-1]],'-^r')
        ax1.plot(opt_xc,opt_yc,opt_time,'r',alpha=0.2)
        ax1.plot(opt_xc[-1],
                 opt_yc[-1],
                 opt_time[-1],
                 marker = 'o',
                 color = 'k', 
                 markersize='10',
                 )
        cbar = plt.colorbar(sc1)
        cbar.set_label('steps')
        
        # ax1.set_xlabel('radii',fontweight='bold')
        # ax1.set_ylabel('chords',fontweight='bold')
        # ax1.set_zlabel('steps',fontweight='bold')
        ax1.set_title(f'polar_plot ({name})',fontweight='bold')

        """
        for angle,radius,mean in zip(opt_angle,opt_x,opt_mean):
            sc1 = ax1.scatter(angle,radius,)
            ax1.plot(angle,radius,'*',color='tomato',alpha=0.2)
            ax1.plot(0,0,'og')
            ax1.plot(angle,mean,
                      marker = '.',
                      linestyle = '-',
                      markersize='2.5',
                      color = 'k'
                      )
            ax1.set_title(label_opt)
            # ax1.set_xlim(0,8)
        
       
        for angle,radius,mean in zip(sub_angle,sub_x,sub_mean):
            ax2.plot(angle,radius,'*',color='lightskyblue',alpha=0.2)
            ax2.plot(0,0,'og')
            ax2.plot(angle,mean,
                      marker = '.',
                      linestyle = '-',
                      markersize='2.5',
                      color = 'k'
                      )
            ax2.set_title(label_sub)
            # ax2.set_xlim(0,8)
        """
        
        plt.show()    
        return

    #polar_plot
    def polar_plots(self,opt_chords_means,
                         opt_radii_means,
                         sub_chords_means,
                         sub_radii_means,
                         ):

        opt_angle,opt_x,opt_mean = self.polar_plot_comp(opt_chords_means,opt_radii_means)
        sub_angle,sub_x,sub_mean = self.polar_plot_comp(sub_chords_means,sub_radii_means)
        opt_time = np.arange(len(opt_angle))
        sub_time = np.arange(len(sub_angle))

        # fig,ax = plt.subplot(subplot_kw={'projection':'polar'})
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(121,projection='polar')
        # ax2 = fig.add_subplot(122,projection='polar')
        font  = {'size': 12}
        font_ticks = 16

        if self.epsilon == 0 or self.epsilon == 1:
            label_opt = f'{self.strategy_option[self.epsilon]}*'
            label_sub = f'{self.strategy_option[self.epsilon]}_sub'
        else:
            label_opt = f'$\epsilon:${self.epsilon}*'
            label_sub = f'$\epsilon:${self.epsilon}_sub'
        
        sc1 = ax1.scatter(opt_angle,opt_x,c=opt_time,cmap='viridis') #,edgecolors='k'
        # ax1.plot(0,0,'or')
        ax1.plot(opt_angle,opt_x,'r',alpha=0.2)
        cbar = plt.colorbar(sc1)
        cbar.set_label('steps')

        """
        for angle,radius,mean in zip(opt_angle,opt_x,opt_mean):
            sc1 = ax1.scatter(angle,radius,)
            ax1.plot(angle,radius,'*',color='tomato',alpha=0.2)
            ax1.plot(0,0,'og')
            ax1.plot(angle,mean,
                      marker = '.',
                      linestyle = '-',
                      markersize='2.5',
                      color = 'k'
                      )
            ax1.set_title(label_opt)
            # ax1.set_xlim(0,8)
        
       
        for angle,radius,mean in zip(sub_angle,sub_x,sub_mean):
            ax2.plot(angle,radius,'*',color='lightskyblue',alpha=0.2)
            ax2.plot(0,0,'og')
            ax2.plot(angle,mean,
                      marker = '.',
                      linestyle = '-',
                      markersize='2.5',
                      color = 'k'
                      )
            ax2.set_title(label_sub)
            # ax2.set_xlim(0,8)
        """
        
        plt.show()    
        return

    # regret_convolution_plot_for_all_max_steps
    def all_max_steps_regret_plots(self,max_steps_col,RC_max_steps):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'strategy: {self.algo_label} | env: {self.setting}')
        font  = {'size': 22}
        font_ticks = 16
        plot_colors = ['r','m','g','c',]
        
        ## effort_per_regret_reduction_Plot
        plt.subplot(3,2,1)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['cost_opt'][0].size > 0:
                opt_smooth_segments = self.running_avg(RC_max_steps[i]['cost_opt'][0],0.99) #smoothened_segments
                plt.plot(opt_smooth_segments,
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}*')
                # convergence_marker_1
                plt.plot(RC_max_steps[i]['con'][0],
                         opt_smooth_segments[int(RC_max_steps[i]['con'][0])],
                         marker = 'X',
                         ms = 10,
                         color = 'k' )

                # print(f'{i}: ', RC_max_steps[i]['con'][0])

        # plt.ylabel('$\Delta$/$\delta$[X]', fontdict=font) 
        plt.ylabel('$\Delta$', fontdict=font) 
        # plt.title(f'segment_length vs steps [{self.algo_label}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        
        # plt.title('(a)', fontdict=font)
        plt.title('$\Delta$', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        # plt.xlim(-.001,3000)
        plt.legend()
        plt.grid()

        plt.subplot(3,2,2)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['cost_sub'][0].size > 0:
                plt.plot(self.running_avg(RC_max_steps[i]['cost_sub'][0],0.99),
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}_sub')
        
        # plt.ylabel('$\Delta$/$\delta$[X]', fontdict=font) 
        plt.ylabel('$\Delta$', fontdict=font) 
        # plt.title(f'segment_length vs steps [{self.algo_label}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)

        # plt.title('(b)', fontdict=font)
        plt.title('$\Delta$', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        # plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()

        """       
        ## chords_Plot
        plt.subplot(3,2,3)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['chrd_opt'][0].size > 0:
                opt_smooth_segments = self.running_avg(RC_max_steps[i]['chrd_opt'][0],0.99) #smoothened_segments
                plt.plot(opt_smooth_segments,
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}*')
                # convergence_marker_1
                plt.plot(RC_max_steps[i]['con'][0],
                         opt_smooth_segments[int(RC_max_steps[i]['con'][0])],
                         marker = 'X',
                         ms = 10,
                         color = 'k' )

        plt.ylabel('X', fontdict=font) 
        # plt.title(f'segment_length vs steps [{self.algo_label}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        
        plt.title('(c)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()

        plt.subplot(3,2,4)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['chrd_sub'][0].size > 0:
                plt.plot(self.running_avg(RC_max_steps[i]['chrd_sub'][0],0.99),
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}_sub')
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('X', fontdict=font) 
        # plt.title(f'segment_length vs steps [{self.algo_label}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)

        plt.title('(d)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()
        """
        """        
        ## Regret_Plot
        plt.subplot(3,2,5)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['regt_opt'][0].size > 0:
                opt_smooth_segments = np.abs(self.running_avg(RC_max_steps[i]['regt_opt'][0],0.99)) #smoothened_segments
                plt.plot(opt_smooth_segments,
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}*')
                # convergence_marker_1
                plt.plot(RC_max_steps[i]['con'][0],
                         opt_smooth_segments[int(RC_max_steps[i]['con'][0])],
                         marker = 'X',
                         ms = 10,
                         color = 'k' )

        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Regret', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('Regret_vs_steps', fontdict=font)
        
        plt.title('(e)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()
        """
        """
        # intercepts
        plt.subplot(3,2,3)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['corr'][0].size > 0:
                plt.plot(i,RC_max_steps[i]['corr'][0],'-*k')

        plt.xlabel('max_steps', fontdict=font)
        plt.ylabel('correlation_coeff', fontdict=font) 
        # plt.ylabel('intercepts', fontdict=font) 
        # plt.ylabel('mutual_info', fontdict=font)
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title('Corr-coef_vs_Max_steps', fontdict=font)
        # plt.title('Intercepts_vs_Max_steps', fontdict=font) 
        # plt.title('Mutual_Info_vs_Max_steps', fontdict=font)
        
        # plt.title('(e)', fontdict=font)
        plt.grid()

        # intercepts
        plt.subplot(3,2,4)
        for cnt,i in enumerate(max_steps_col): #
            if RC_max_steps[i]['intc'][0].size > 0:
                plt.plot(i,RC_max_steps[i]['intc'][0],'-*k')

            # if RC_max_steps[i]['corr'][0].size > 0:
            #     plt.plot(i,RC_max_steps[i]['corr'][0],'-*k')

        plt.xlabel('max_steps', fontdict=font)
        # plt.ylabel('correlation_coeff', fontdict=font) 
        plt.ylabel('intercepts', fontdict=font) 
        # plt.ylabel('mutual_info', fontdict=font)
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        # plt.title('Corr-coef_vs_Max_steps', fontdict=font)
        plt.title('Intercepts_vs_Max_steps {self.se}', fontdict=font) 
        # plt.title('Mutual_Info_vs_Max_steps', fontdict=font)
        
        # plt.title('(e)', fontdict=font)
        plt.grid()

        # mutual_information
        plt.subplot(3,2,5)
        for cnt,i in enumerate(max_steps_col): #
            if RC_max_steps[i]['mutual_info'][0].size > 0:
                plt.plot(i,RC_max_steps[i]['mutual_info'][0],'-*k')

            # if RC_max_steps[i]['intc'][0].size > 0:
            #     plt.plot(i,RC_max_steps[i]['intc'][0],'-*k')

            # if RC_max_steps[i]['corr'][0].size > 0:
            #     plt.plot(i,RC_max_steps[i]['corr'][0],'-*k')

        plt.xlabel('max_steps', fontdict=font)
        # plt.ylabel('correlation_coeff', fontdict=font) 
        # plt.ylabel('intercepts', fontdict=font) 
        plt.ylabel('mutual_info', fontdict=font)
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        # plt.title('Corr-coef_vs_Max_steps', fontdict=font)
        # plt.title('Intercepts_vs_Max_steps', fontdict=font) 
        plt.title('Mutual_Info_vs_Max_steps', fontdict=font)
        
        # plt.title('(e)', fontdict=font)
        plt.grid()

        #slope (best-fit-line)
        plt.subplot(3,2,6)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['slope'][0].size > 0:
                plt.plot(i,RC_max_steps[i]['slope'][0],'-*k')

        plt.xlabel('max_steps', fontdict=font)
        plt.ylabel('slopes', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        plt.title(f'Slopes_vs_Max_steps', fontdict=font)
        
        # plt.title('(e)', fontdict=font)
        plt.grid()
        # """

        """        
        ## Area_Plot
        plt.subplot(3,2,5)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['area_opt'][0].size > 0:
                opt_smooth_segments = self.running_avg(RC_max_steps[i]['area_opt'][0],0.99) #smoothened_segments
                plt.plot(opt_smooth_segments,
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}*')
                # convergence_marker_1
                plt.plot(RC_max_steps[i]['con'][0],
                         opt_smooth_segments[int(RC_max_steps[i]['con'][0])],
                         marker = 'X',
                         ms = 10,
                         color = 'k' )

        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Area', fontdict=font) 
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        
        plt.title('(e)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()

        plt.subplot(3,2,6)
        for cnt,i in enumerate(max_steps_col): 
            if RC_max_steps[i]['area_sub'][0].size > 0:
                plt.plot(self.running_avg(RC_max_steps[i]['area_sub'][0],0.99),
                        '-', color=plot_colors[cnt],
                        label=f'{self.algo_label}-{i}_sub')
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Area', fontdict=font) 
        # plt.title(f'segment_length vs steps [{self.algo_label}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)

        plt.title('(f)', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()
        """

        plt.tight_layout()
        plt.show()
        

        return

    # 3Dplots (chords vs radii vs time)
    def _3Dplots(self,Xs,Ys,name='',fig=None,pos=None):
        time = np.arange(len(Xs))

        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(pos,projection='3d')

        # font = {'size': 40, 
        #         'weight': 'bold',
        #         'family':'normal'}
        
        # plt.rc('font',**font)
        
        #option_1          
        # ax.plot(Xs,Ys,time,
        #         '*',
        #         color='salmon',
        #         alpha=0.2)
        # ax.plot(self.running_avg(Xs,0.99),
        #         self.running_avg(Ys,0.99),
        #         time,
        #         c = 'k',
        #         )

        #coloured_segments
        # points = np.array([Xs,Ys]).T.reshape(-1,1,2)
        # segments = np.concatenate([points[:-1],points[1:]],axis=1)
        # lc = Line3DCollection(segments,cmap='plasma',norm=plt.Normalize(time.min(),time.max()),alpha=0.2)
        # lc.set_array(time)
        # ax.add_collection3d(lc,zs=time,zdir='z')

        colors = [(0,0,1),(1,0,0)] #blue->red
        # colors = [(0,0,1),(0,1,0),(1,0,0)] #blue->red
        n_bins = 600 #256
        cmap_name = 'blue_red'
        cm = LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)

        scatter = ax.scatter(Xs,Ys,
                             time,
                             c=time,
                            #  cmap=cm,
                             cmap='plasma',
                            #  edgecolor='k'
                             ) #cmap='viridis'
        ax.plot(Xs,Ys,time,c = 'k',alpha=1)
        """        
        ax.quiver(Xs[:-1], Ys[:-1], time[:-1], 
                  Xs[1:]-Xs[:-1], 
                  Ys[1:]-Ys[:-1],
                  time[1:]-time[:-1], 
                #   scale_units='xyz', 
                #   angles='xyz', 
                #   scale=2,
                # normalize = True,
                # linewidth = 2,
                edgecolor='k',
                length = 1,
                arrow_length_ratio = 1,
                  )
        """


        # ax.plot(self.running_avg(Xs,0.99),self.running_avg(Ys,0.99),time,c ='r')

        # cbar = plt.colorbar(scatter)
        # cbar.set_label('#updates',fontweight='bold',fontsize=16)
        # cbar.set_ticks([])

        ax.set_zlabel('#updates',fontweight='bold',fontsize=16)
        ax.set_xlabel('distance-to-optimal',fontweight='bold',fontsize=16)
        ax.set_ylabel('stepwise-distance',fontweight='bold',fontsize=16)

        # ax.set_zlabel('steps',fontweight='bold')
        # ax.set_xlabel('radii',fontweight='bold')
        # ax.set_ylabel('chords',fontweight='bold')

        ax.view_init(elev=20.,azim=-120)
        ax.set_xlim(0,max(Xs))
        ax.set_ylim(0,max(Ys))
        ax.set_zlim(0,max(time))
        # ax.set_title(f'policy_evolution ({name})',fontweight='bold')
        ax.set_title(f'{name}',fontweight='bold',fontsize=16)
        ax.invert_zaxis()
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        ax.set_xlim(0,14)
        ax.set_ylim(0,14)
        # ax.set_xticks(fontsize=14)
        # ax.set_yaxis.ticks(fontsize=14)
        # ax.set_visible(False)
        # ax.set_rc('font',**font)

        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.zticks(fontsize=14)
        ax.tick_params(labelsize=14)
        # ax.ticklabel_format(axis='z',style='sci',scilimits=[0,2])

        plt.tight_layout()

        

        # plt.show()
        # plt.close()        
        return   
    
    # state_value landscape
    def values3Dplots(self,state_values,name='',fig=None,pos=None):
        time = np.arange(len(state_values[(0,0)]))
        states,values = [],[]
        for i in state_values.keys():
            states.append(i)
            values.append(state_values[i])
        
        values = np.asarray(values)

        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(pos,projection='3d')

        """ 
        #option_1          
        for i,state in enumerate(states):
            y = np.full(time.shape,i)
            ax.plot(time,y,values[i], label=state)
        #"""

        #"""
        #option_2
        T,Y = np.meshgrid(time,range(len(states))) #create_mesh
        surf = ax.plot_surface(T,Y,values,cmap='plasma') #cmap='viridis'
        # fig.colorbar(surf,ax=ax,shrink=0.5,aspect=5)
        #"""

        """
        #option_3
        T,Y = np.meshgrid(time,range(len(states))) #create_mesh
        surf = ax.plot_surface(T,Y,values,cmap='viridis', #edgecolor='y',
                               linewidth=0.5,rstride=8,cstride=8,alpha=0.3) #,alpha=0.8
        ax.contour(T,Y,values, zdir='z', offset=values.min(), cmap='coolwarm',alpha=0.8)
        ax.contour(T,Y,values, zdir='x',offset=time.min(), cmap='coolwarm',alpha=0.8)
        ax.contour(T,Y,values, zdir='y',offset=len(states),cmap='coolwarm',alpha=0.8)
        # fig.colorbar(surf,ax=ax,shrink=0.5,aspect=5)
        #"""


        ax.set_xlabel('steps')
        ax.set_ylabel('states')
        ax.set_zlabel('values')
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels(states)
        ax.set_title(f'state_value_landscape {name}')

        # plt.legend()
        # plt.show()
        
        return        
    
    #vs_steps_plots
    def timeplots(self,steps_col,y_col,x_col,regret_col,num_eval,outcome,state_values,show=1):

        #cost-knowledge-transfer_per_regret-reduction 
        cost_per_reduction = self.successive_diff(y_col,x_col)

        steps_col = steps_col[:-1] #all_steps
        regret_col = np.abs(regret_col) #absolute_value
        # ds_col = np.asarray(ds_col)/max(np.asarray(ds_col)) #normalizing Segment_Plot

        fig = plt.figure(figsize=(12, 9))
        ## Segment_length_Plot
        plt.subplot(2,3,1)
        plt.plot(steps_col,
                 y_col,
                 '-', #'-'
                 color='gray',
                 alpha=0.2,
                #  label=f"sr: {self.test_freq} ({self.setting})"
                ) 
        
        plt.plot(steps_col,
                 self.running_avg(y_col,0.99),
                 '-',
                 color='red',
                 label=f"sr: {self.test_freq} ({self.setting})"
                 ) 
        
        plt.vlines(x = num_eval, #convergence_line
                   ymin=min(y_col),
                   ymax=max(y_col), 
                   colors='black', 
                   ls=':',)
        plt.xlabel('steps') #episodes
        # plt.ylabel('segment_length/max(segment_length)')
        plt.ylabel('$\Delta / \Delta_{max}$')
        plt.title(f'y-Plot [{outcome}]')
        plt.legend()

        ## x_&_regret_vs_steps_plot
        plt.subplot(2,3,2)
        
        #x-values
        # plt.plot(x_col,'-',color='green',alpha=0.2)
        plt.plot(self.running_avg(x_col,0.99), #[:num_eval]
                 '-',
                 color='indigo', 
                 label="x")
        
        #regret-values
        plt.plot(self.running_avg(regret_col,0.99), #[:num_eval]
                 '-',
                 color='orchid',
                 label="regret")
        
        plt.vlines(x = num_eval, #convergence_line
                   ymin=min(x_col),
                   ymax=max(x_col), 
                   colors='black', 
                   ls=':',)
        plt.xlabel('steps')
        plt.ylabel('$\eta$, regret') #/ \eta_{max}
        plt.title('x-Plot,regret vs. step')
        # plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)
        # plt.yscale('symlog')
        plt.legend()

        ## regret_plot
        plt.subplot(2,3,3)
        plt.title('Regret Plot')
        plt.plot(regret_col,'-',color='green',alpha=0.2) #[:num_eval]
        plt.plot(self.running_avg(regret_col,0.99),'-',color='orchid') #[:num_eval]
        plt.vlines(x = num_eval, #convergence_line
                   ymin=min(regret_col),
                   ymax=max(regret_col), 
                   colors='black', 
                   ls=':',)
        plt.xlabel('steps')
        plt.ylabel('regret')
        plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)

        ## cost-knowledge-transfer_per_regret-reduction plot
        plt.subplot(2,3,4)
        plt.title(f'y/($\delta$x) vs steps')
        plt.plot(cost_per_reduction,'-',color='brown', alpha=0.2) #[:num_eval]
        plt.plot(self.running_avg(cost_per_reduction,0.99),'-',color='tomato') #[:num_eval]
        plt.vlines(x = num_eval, #convergence_line
                   ymin=min(cost_per_reduction),
                   ymax=max(cost_per_reduction), 
                   colors='black', 
                   ls=':',)
        plt.xlabel('steps')
        plt.ylabel(f'y/($\delta$x)')
        plt.yscale('symlog')
        # plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0),useMathText=True)

        #All_Figures
        plt.tight_layout()     
        if show == 1:
            plt.show()
        return
    
    #segment_length (policy_evolution)
    def policy_evolution(self):
        max_steps_col = [15,40,60,90]
        PE_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]
        for max_esteps in max_steps_col: 
            # max_esteps = 15
            outcome_list, convg_steps_list = [],[]
            segments_dict,returns_dict = {},{}
            file,_ = self.file_location_max_episode_steps(max_esteps)
            for i in range(50):

                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)

                steps_col = np.asarray(data[1])
                score_col = data[2] #returns 
                num_eval = data[4] #convergence_step
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # steps_k = steps_col[:-1]
                # steps_k_1 = steps_col[1:]
                # steps_size = steps_k_1 - steps_k
                
                # print(steps_k)
                # ccc

                # suma,tt = 0,0
                idx,tt = 0,0
                returns_per_eps = []
                for _,val in enumerate(score_col):
                    idx += 1
                    tt += val
                    if idx%max_esteps == 0:
                        returns_per_eps.append(tt/max_esteps)
                        tt = 0

                """  
                #test_ret = []
                print('score_col: ', score_col)
                print('steps_col: ', steps_col)
                # plt.plot(score_col,'-*')
                plt.plot(returns_per_eps,'-*')
                # plt.plot(self.running_avg(returns_per_eps,0.99)) #0.99
                # plt.plot(self.rolling(returns_per_eps,100))
                plt.plot(self.running_avg(score_col,0.99)) #0.99
                plt.plot(self.rolling(score_col,1000))
                plt.show()
                xxx
                # """

                # data_collection
                segments_dict[i] = ds_stt_col #unnormalized
                returns_dict[i] = returns_per_eps #score_col # # #unnormalized
                normalize = 0
                if normalize == 1:
                    segments_dict[i] = np.round(ds_stt_col/max(ds_stt_col),6) #normalizing Segment_Plot
                    returns_dict[i] = np.round(score_col/max(score_col),6)
                convg_steps_list.append(num_eval) ##convergence_step_per_run
                
                outcome_list.append(outcome) #convergence_indicators
                lipsch.append(lip_con) #lipschitz_constants

            pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
            fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
            lipschitz.append(max(lipsch)) #2nd_round_lipschitz_candidate
            # print('lipschitz: ', lipschitz)

            optimal_segments_dict = {key: segments_dict[key] for key in pass_indexes}
            sub_segments_dict = {key: segments_dict[key] for key in fail_indexes}

            optimal_returns_dict = {key: returns_dict[key] for key in pass_indexes}
            sub_returns_dict = {key: returns_dict[key] for key in fail_indexes}

            true_convg_steps_list = [ convg_steps_list[i] for i in pass_indexes ] #true_convergence_steps_per_run
            opt_segments_means, _ = self.segments_stats(optimal_segments_dict)
            sub_segments_means, _ = self.segments_stats(sub_segments_dict)
            opt_returns_means, _ = self.segments_stats(optimal_returns_dict)
            sub_returns_means, _ = self.segments_stats(sub_returns_dict)
            convg_steps_mean,_ = self.convg_stats(true_convg_steps_list)

            if max_esteps == 15:
                #policy_evolution_plot_per_max_steps
                self.single_max_steps_policy_plots(opt_segments_means,
                                                sub_segments_means,
                                                opt_returns_means,
                                                sub_returns_means,
                                                convg_steps_mean,
                                                max_esteps) 

            PE_max_steps[max_esteps]['opt'].append(opt_segments_means) #optimal_segments
            PE_max_steps[max_esteps]['sub'].append(sub_segments_means) #suboptimal_segments
            PE_max_steps[max_esteps]['con'].append(convg_steps_mean) #convergence_step

        #policy_evolution_plot_for_all_max_steps
        self.all_max_steps_policy_plots(max_steps_col,PE_max_steps,normalize)
        return
    
    # policy_evolution_plot_for_all_max_steps
    def all_max_steps_policy_plots(self,max_steps_col,PE_max_steps,normalize):
        fig = plt.figure(figsize=(15, 7))
        font  = {'size': 22}
        font_ticks = 16
        plot_colors = ['r','m','g','c',]

        if self.epsilon == 0 or self.epsilon == 1:
            label_opt = f'{self.strategy_option[self.epsilon]}*'
            label_sub = f'{self.strategy_option[self.epsilon]}_sub'
        else:
            label_opt = f'$\epsilon:${self.epsilon}*'
            label_sub = f'$\epsilon:${self.epsilon}_sub'
        
        ## Segment_length_Plot
        plt.subplot(1,2,1)
        for cnt,i in enumerate(max_steps_col): 
            if PE_max_steps[i]['opt'][0].size > 0:
                opt_smooth_segments = self.running_avg(PE_max_steps[i]['opt'][0],0.99) #smoothened_segments
                plt.plot(opt_smooth_segments,
                        '-', color=plot_colors[cnt],
                        label=label_opt+f'-{i}*')
                # convergence_marker_1
                plt.plot(PE_max_steps[i]['con'][0],
                         opt_smooth_segments[int(PE_max_steps[i]['con'][0])],
                         marker = 'X',
                         ms = 10,
                         color = 'k' )

                print(f'{i}: ', PE_max_steps[i]['con'][0])

        plt.xlabel('steps', fontdict=font)
        plt.ylabel('$\Delta$', fontdict=font) #$\Delta / \Delta_{max}$
        # plt.ylim(0,4.5)
        if normalize == 1:
            plt.ylabel('$\Delta / \Delta_{max}$', fontdict=font) 
            plt.ylim(0,1)
        # plt.title(f'segment_length vs steps [{self.strategy_option[self.epsilon]}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        
        plt.title(f'env: {self.setting}-opt', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        # plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()


        plt.subplot(1,2,2)
        for cnt,i in enumerate(max_steps_col): 
            if PE_max_steps[i]['sub'][0].size > 0:
                plt.plot(self.running_avg(PE_max_steps[i]['sub'][0],0.99),
                        '-', color=plot_colors[cnt],
                        label=label_sub+f'-{i}_sub')
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('$\Delta$', fontdict=font) #$\Delta / \Delta_{max}$
        # plt.ylim(0,4.5)
        if normalize == 1:
            plt.ylabel('$\Delta / \Delta_{max}$', fontdict=font) 
            plt.ylim(0,1)
        # plt.title(f'segment_length vs steps [{self.strategy_option[self.epsilon]}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)

        plt.title(f'env: {self.setting}-sub', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(3,4))
        # plt.xlim(-.001,5000)
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

        return
    
    # policy_evolution_plot_per_max_steps
    def single_max_steps_policy_plots(self,opt_segments_means,
                                      sub_segments_means, 
                                      opt_returns_means,
                                      sub_returns_means,
                                      convg_steps_mean,max_esteps):
        fig = plt.figure(figsize=(15, 7))
        font  = {'size': 22}
        font_ticks = 16

        if self.epsilon == 0 or self.epsilon == 1:
            label_opt = f'{self.strategy_option[self.epsilon]}*'
            label_sub = f'{self.strategy_option[self.epsilon]}_sub'
        else:
            label_opt = f'$\epsilon:${self.epsilon}*'
            label_sub = f'$\epsilon:${self.epsilon}_sub'

        ## Segment_length_Plots
        plt.subplot(2,2,1)
        if opt_segments_means.size > 0:
            opt_smooth_segments = self.running_avg(opt_segments_means,0.99) #smoothened_segments
            # opt_smooth_segments = self.rolling(opt_segments_means)
            plt.plot(opt_smooth_segments,
                    '-g',
                    label=label_opt
                    )
            plt.plot(opt_segments_means,'-g', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            plt.plot(convg_steps_mean,
                    opt_smooth_segments[int(convg_steps_mean)],
                    marker = 'X',
                    ms = 10,
                    color = 'k' )
        
        # convergence_marker_2
        # plt.vlines(x = convg_steps_mean,
        #         ymin=0, 
        #         ymax=1, 
        #         colors='black', 
        #         ls=':')
        
        plt.xlabel('steps', fontdict=font)
        # plt.ylabel('$\Delta / \Delta_{max}$', fontdict=font) 
        plt.ylabel('$\Delta$', fontdict=font) #$\Delta / \Delta_{max}$
        # plt.ylabel('segment length', fontdict=font)
        # plt.title(f'segment_length vs steps [{outcome}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        # plt.ylim(0,1)
        # plt.xlim(0,800)
        plt.title(f'(a) env: {self.setting} | horizon: {max_esteps}', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
        plt.legend()
        plt.grid()

        plt.subplot(2,2,2)
        if sub_segments_means.size > 0:
            plt.plot(self.running_avg(sub_segments_means,0.99) ,
                    #self.rolling(sub_segments_means),
                    '-g',
                    label=label_sub
                    )
            plt.plot(sub_segments_means,'-g', alpha=0.2)
        
        plt.xlabel('steps', fontdict=font)
        # plt.ylabel('$\Delta / \Delta_{max}$', fontdict=font) 
        plt.ylabel('$\Delta$', fontdict=font) #$\Delta / \Delta_{max}$
        # plt.ylabel('segment length', fontdict=font)
        # plt.title(f'segment_length vs steps [{outcome}]')
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        # plt.ylim(0,1)
        # plt.xlim(0,800)
        plt.title(f'(b) env: {self.setting} | horizon: {max_esteps}', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
        plt.legend()
        plt.grid()

        ##Returns
        plt.subplot(2,2,3)
        if opt_returns_means.size > 0:
            opt_smooth_returns = self.running_avg(opt_returns_means,0.9) #smoothened_returns
            # opt_smooth_returns = self.rolling(opt_returns_means,50) #smoothened_returns
            plt.plot(opt_smooth_returns,
                    '-r',
                    label=label_opt
                    )
            plt.plot(opt_returns_means,'-r', alpha=0.2) #mean_of_all_runs

            # convergence_marker_1
            # plt.plot(convg_steps_mean,
            #         opt_smooth_returns[int(convg_steps_mean)],
            #         marker = 'X',
            #         ms = 10,
            #         color = 'k' )
                
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Returns', fontdict=font)
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        # plt.ylim(0,1)
        # plt.xlim(0,800)
        plt.title(f'(a) env: {self.setting} | horizon: {max_esteps}', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
        plt.legend()
        plt.grid()

        plt.subplot(2,2,4)
        if sub_returns_means.size > 0:
            plt.plot(self.running_avg(sub_returns_means,0.9),
                    #  self.rolling(sub_returns_means,20),
                    '-r',
                    label=label_sub
                    )
            plt.plot(sub_returns_means,'-r', alpha=0.2)
        
        plt.xlabel('steps', fontdict=font)
        plt.ylabel('Returns', fontdict=font)
        plt.xticks(fontsize=font_ticks)
        plt.yticks(fontsize=font_ticks)
        # plt.ylim(0,1)
        # plt.xlim(0,800)
        plt.title(f'(b) env: {self.setting} | horizon: {max_esteps}', fontdict=font)
        plt.ticklabel_format(axis='x',style='sci', scilimits=(0,0))
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
        return

    # convergence_steps_stats
    def convg_stats(self,array):
        s = np.asarray(array)
        mean = np.mean(s)
        median = np.median(s)
        std = np.std(s)
        return median, std

    # segments_mean_standard_deviation
    def segments_stats(self,dict):
        dict_df = pd.DataFrame.from_dict(dict, orient='index').T
        dict_data = dict_df.values
        dict_means = np.nanmean(dict_data, axis=1)
        dict_stds = np.nanstd(dict_data, axis=1)
        return dict_means, dict_stds
    
    # nested_value_stats
    def nested_value_stats(self,values_dict):
        #swap_nested_dictionary
        new_values_dict = defaultdict(dict)
        for k1,v1 in values_dict.items():
            for k2,v2 in v1.items():
                new_values_dict[k2][k1] = v2

        values_means = {}
        for k1 in new_values_dict.keys():
            means, _ = self.segments_stats(new_values_dict[k1])
            values_means[k1] = means
        return values_means

    #competitve_ratios_per_episode_length
    def competitive_ratios_eps_len(self):
        CR_max_steps = defaultdict(lambda: defaultdict(list))
        lipschitz,lipsch = [],[]
        for max_esteps in [15,40,60,90]: 
            curve_list,direct_list,outcome_list= [],[],[]
            file,_ = self.file_location_max_episode_steps(max_esteps)
            for i in range(50):

                run_file = f"curve_{i}.npy" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                data = np.load(file_path, allow_pickle=True)

                # data_extraction
                p = 6 #float_precision (decimal_places)

                direct_dis = np.round(data[0],p) #geodesic_distance
                num_eval = data[4] #convergence_step
                ds_stt_col = np.round(data[5],p) #y_k
                outcome = data[9] #convergence_indicator
                lip_con = np.round(data[10],p) #lipschitz_constant_candidate

                # data_collection
                curve_list.append(sum(ds_stt_col[:num_eval])) #curve_dis
                direct_list.append(direct_dis) #geodesics
                outcome_list.append(outcome) #convergence_indicators
                lipsch.append(lip_con) #lipschitz_constants

            pass_indexes = [i for i,j in enumerate(outcome_list) if j=='Pass'] #optimal_runs
            fail_indexes = [i for i,j in enumerate(outcome_list) if j=='Fail'] #suboptimal_runs
            lipschitz.append(max(lipsch)) #2nd_round_lipschitz_candidate
            # print('fail_indexes: \n', fail_indexes)
            # print('pass_indexes: \n', pass_indexes)
            print('success_rate: ', 100*(len(pass_indexes)/(len(pass_indexes)+len(fail_indexes))))
            
            nu = 1e-3 #non-zero divider
            optimal_geodesic_list = [nu if direct_list[i] == 0 else direct_list[i] for i in pass_indexes ] #optimal_geodesic
            optimal_curve_list = [ curve_list[i] for i in pass_indexes ] #optimal_curve

            sub_geodesic_list = [ nu if direct_list[i] == 0 else direct_list[i] for i in fail_indexes ] #suboptimal_geodesic
            sub_curve_list = [ curve_list[i] for i in fail_indexes ] #suboptimal_curve

            optimal_CR = np.asarray(optimal_curve_list)/np.asarray(optimal_geodesic_list)
            sub_CR = np.asarray(sub_curve_list)/np.asarray(sub_geodesic_list)

            CR_max_steps[max_esteps]['opt'].append(optimal_CR)
            CR_max_steps[max_esteps]['sub'].append(sub_CR)
        
        self.CR_Ran_Grd_boxplots(CR_max_steps) #boxplots_CR_(random/greedy)
        return

    #boxplot of CR for (random-greedy) strategies
    def CR_Ran_Grd_boxplots(self,CR_max_steps):
        fig = plt.figure(figsize=(12, 9))
        #plotting/visualizing 
        bp = plt.boxplot(
                [
                CR_max_steps[15]['opt'][0],
                CR_max_steps[15]['sub'][0],
                CR_max_steps[40]['opt'][0],
                CR_max_steps[40]['sub'][0],
                CR_max_steps[60]['opt'][0],
                CR_max_steps[60]['sub'][0],
                CR_max_steps[90]['opt'][0],
                CR_max_steps[90]['sub'][0],
                ], 
                # patch_artist=True, 
                medianprops = dict(color = "black"),
                labels=[
                    "CR*-15",
                    "$CR_{sb}$-15",
                    "CR*-40",
                    "$CR_{sb}$-40",
                    "CR*-60",
                    "$CR_{sb}$-60",
                    "CR*-90",
                    "$CR_{sb}$-90",
                    ],
                )
        
        X,Y = [],[]
        for m in bp['medians']: #'medians'
            [[x0,x1],[y0,y1]] = m.get_data()
            X.append(np.mean((x0,x1)))
            Y.append(np.mean((y0,y1)))

        X_opt = [X[i] for i in [0,2,4,6] ]
        Y_opt = [Y[i] for i in [0,2,4,6] ]
        X_sub = [X[i] for i in [1,3,5,7] ]
        Y_sub = [Y[i] for i in [1,3,5,7] ]
        plt.plot(X_opt,Y_opt,"^-",c="red",label="CR*") #optimal_CR
        plt.plot(X_sub,Y_sub,"^-",c="C1",label="$CR_{sb}$") #sub_CR

        plt.grid()
        plt.legend()
        # plt.ylim(1,1000)
        plt.yscale("log")
        plt.ylabel("Competitive Ratio")
        plt.xlabel("Group")
        plt.title(f"env: {self.setting} | strategy: {self.strategy_option[self.epsilon]}")
        plt.show()

        return

    #successive_ratios
    def successive_ratios(self,y_col,x_col):
        p = 6 #float_precision (decimal_places)

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        x_k_1 = x_col[1:] #x_{k+1}
        x_k = x_col[:-1] #x_{k}
        y_k = y_col[:-1] #y_{k}
        x_diff = np.round(x_k - x_k_1,p) # x_k - x_{k+1}#
        x_diff_true = np.round(x_k - x_k_1,p) # x_k - x_{k+1}#
        eta = 1e-3 #small value to avoid zero division
        
        """
        #verify triangular_inequality
        print('len(x_k_1): ',len(x_k_1))
        test1,test2 = 0,0
        for i in range(len(x_k_1)):
            diff = np.round(np.abs(x_k_1[i] - x_k[i]),4)
            sum = np.round(x_k_1[i] + x_k[i],4)
            y =  np.round(y_k[i],4)
            if diff <= y:
                test1 += 1
            if sum >= y:
                test2 += 1

        print('forward_triangle: ', 100*(test2/len(x_k_1)))
        print('reverse_triangle: ', 100*(test1/len(x_k_1)))
        xxx
        # """

        #absolute values
        # x_diff_abs = np.abs(np.asarray(x_col[:-1]) - np.asarray(x_col[1:])) # x_k - x_{k+1}#
        # x_diff = np.where(x_diff_abs<eta,eta,x_diff_abs)

        # for idx,i in enumerate(x_diff):
        #     if 0 <= i < eta: 
        #         x_diff[idx] = eta
        #     elif -eta < i < 0:
        #         x_diff[idx] = -eta

        # x_diff[x_diff< eta] = eta
        y_k[y_k< eta] = eta

        #compute ratios
        ratios = np.round(x_diff/y_k,p) #np.round(y_k/x_diff,p)
        return ratios, x_diff_true #x_diff
    
    #successive_difference
    def successive_diff_chords(self,y_col):
        p = 6 #float_precision (decimal_places)

        #y_col[:-1] -> {0:N-1} : y_k
        #y_col[1:] -> {1:N} : y_{k+1}
        y_k_1 = y_col[1:] #y_{k+1}
        y_k = y_col[:-1] #y_{k}
        eta = 1e-3 #small value to avoid zero division

        y_diff = []
        for i in range(len(y_k)):
            if y_k_1[i] == 0: y_diff.append(0)
            elif y_k[i] == 0: continue #y_diff.append(y_k_1[i]/eta)
            else: 
                y_op = np.round(y_k_1[i]/y_k[i],p)
                y_diff.append(y_op)

        return y_diff
    
    #successive_difference
    def successive_diff_radii(self,x_col):
        p = 6 #float_precision (decimal_places)

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        x_k_1 = x_col[1:] #x_{k+1}
        x_k = x_col[:-1] #x_{k}
        eta = 1e-3 #small value to avoid zero division

        """        
        x_diff = []
        for i in range(len(x_k)):
            # if x_k_1[i] == 0: x_diff.append(0)
            # elif x_k[i] == 0: continue #y_diff.append(y_k_1[i]/eta)
            # else: 
            x_op = np.round(x_k_1[i] - x_k[i],p)
            x_diff.append(x_op)
        """

        x_diff = np.round(x_k - x_k_1,p) # x_k - x_{k+1}#

        return x_diff

    #successive_differences [radii (distance_to_optimal)]
    def successive_diffs(self,x_col,p): #radii (dis_to_optim)

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        x_k_1 = x_col[1:] #x_{k+1}
        x_k = x_col[:-1] #x_{k}
        x_diff = np.round(x_k - x_k_1,p) # x_k - x_{k+1}#
        
        return x_diff 

    #successive_area
    def successive_area(self,y_col,x_col,p):
        # p = 6 #float_precision (decimal_places)

        #x_col[:-1] -> {0:N-1} : x_k
        #x_col[1:] -> {1:N} : x_{k+1}
        #y_col[:-1] -> {0:N-1} : y_k
        x_k = x_col[:-1] #a
        x_k_1 = x_col[1:] #b
        y_k = y_col[:-1] #c
        eta = 1e-4 #small value to avoid zero division
        
        #compute areas
        s_k = (x_k+x_k_1+y_k)*0.5 # s = (a+b+c)/2

        d = s_k - y_k
        e = s_k - x_k_1
        f = s_k - y_k

        d = np.where(np.abs(d)>eta,d,0)        
        e = np.where(np.abs(e)>eta,e,0)
        f = np.where(np.abs(f)>eta,f,0)

        area_k = np.round((s_k*d*e*f)**0.5,p) #A = (s(s-a)(s-b)(s-c))^.5
        return area_k
    
    #temporal OMR
    def temporal_OMR(self,chords,radii,p):
        
        segments = 12
        div = 1 #5 #len(chords)//segments
        omr,ups  = [],[]
        for k in range(len(chords)):
            if k%div == 0 and k <= (len(chords)-10): #100
                # print(i)

                # data_collection
                n_chords = chords[k:]
                n_radii = radii[k:]

                # curve_dis = np.round(sum(n_chords),p) # sum_{y_k} #[i:]
                radii_diff = self.successive_diffs(n_radii,p) # x_k - x_{k+1}
            
                #distance-to-optimal not reduced
                non_positive_indices = np.where(radii_diff <= 0) # non-improving transition 
                non_positive_indices = non_positive_indices[0]

                #distance-to-optimal reduced
                positive_indices = np.where(radii_diff > 0) # improving transition
                # print(positive_indices)
                # print(positive_indices[0])
                
                positive_indices = positive_indices[0]
                
                # test = len(positive_indices)+len(non_positive_indices)
                # print(len(positive_indices), len(non_positive_indices), len(positive_indices)/test )
                # # xxx

                non_positive_chords = np.array([ n_chords[i] for i in non_positive_indices])
                positive_chords = np.array([ n_chords[i] for i in positive_indices])

                wasted_effort = non_positive_chords.sum() #paid_to_further_from_goal
                usedful_effort = positive_chords.sum() #paid_to_closer_to_goal
                # print('wasted_effort: ', wasted_effort)
                # print('usedful_effort: ', usedful_effort)
                total_travel = wasted_effort + usedful_effort

                # print(curve_dis - total_travel)
                # print(usedful_effort/total_travel)


                # print('OMR: ', np.round(usedful_effort/curve_dis,3))
                # omr.append(usedful_effort/curve_dis)
                omr.append(usedful_effort/total_travel)
                ups.append(k)

 
        return omr, ups 

    #running average function
    def running_avg(self,score_col,beta=0.9): #0.99
        cum_score = []
        run_score = score_col[0] #-5000

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
        
#Execution
if __name__ == '__main__':
    n = 5
    gr = graph( setting=1, #[0:sparse | 1:dense | 2:single | 3:stochastic | 4:sinks] rewards_setting
                states_size=[n,n],
                algo='qlearn', #['qlearn','sac','ucrl','dqn', 'psrl']
                epsilon=1, #[1: random | 0: greedy ]
                decay=0, # [0: not_decay | 1: decaying] Qlearning_$\epilon$ 
                )
    
    # gr.load() #testing_data_loading
    # gr.competitive_ratios_eps_len() #competitve_ratios_per_episode_length
    # gr.policy_evolution() #tracking_policy_changes
    # gr.regret_convolution() #regret_related_changes
    # gr.state_value_distribution() #state_related_changes
    # gr.sanity_check_plots() #policy_evolution_plots
    # gr.results_check_plots() #average_metrics_evaluation
    # gr.ucrl_hyprm_check_plots() #hyperparameter_effects [UCRL]
    # gr.sac_multrun_check_plots() #num_runs_effects [SAC - stochastic-env]
    # gr.effort_seque_learning_analysis() #Effort_sequential_learning_analysis
    # gr.multruns_evals() #[stochastic-env]
    gr.save_images_for_paper()
    
    # gr.boxplots_B()
    # gr.plots()
    # gr.cum_len_plots()

    # gr.curve_len_eps()
    # gr.curve_len_epsB()
    # gr.curve_len_eps_ds()

    