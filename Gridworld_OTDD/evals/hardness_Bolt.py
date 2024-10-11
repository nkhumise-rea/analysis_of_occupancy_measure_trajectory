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
from matplotlib import colors
from collections import defaultdict
import ot
import h5py
from numpy import linalg as LA

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
from envs.Boltzman import gridworld

class otdd():
    def __init__(self,test_freq=50,states_size=[3,3],setting=1,n_eps=1000,tau=1.0,stop_cri=50):
        self.agent = gridworld( states_size=states_size,
                                rew_setting=setting, #rew_setting [# 1:dense | 0:sparse ]
                                n_eps=n_eps, #[num_eps] #4000 /6000
                               ) 
        self.test_freq = test_freq
        rew_set = ['sps','dns'] #dense: dns, sparse: sps
        self.setting = rew_set[setting] 
        self.states_size = states_size
        # self.it = 1
        self.tau = tau
        self.num_stop = stop_cri
    
    # curve_lengths: only converged policies
    def curve_length_converged(self,num_curve):

        trajectory_dataset, steps_col, score_col,con_eps,num_eval = self.agent.main(
                                    test_frq=self.test_freq, #policy_eval_frequency
                                    runs=1, #number_of_eval_repeats
                                    common_policy=0, #[1: yes | 0: no]
                                    tau=self.tau,
                                    )
        
        if num_eval == -1: 
            # print('fail-to-converge')
            return [0]
            
        else: 
            key_init = list(trajectory_dataset)[0]
            # key_last = list(trajectory_dataset)[-1]
            # key_2ndlast = list(trajectory_dataset)[-2]

            initial_dataset = trajectory_dataset[key_init] #initial_policy_dataset
            final_dataset =  trajectory_dataset[num_eval] #converged_policy_dataset
            #final_dataset = trajectory_dataset[key_last] #final_policy_dataset

            # print('final_dataset: ', final_dataset)
            # print('final_dataset: ', final_dataset[0])
            self.optimal_traj = final_dataset[0] #trajectory_dataset[key_2ndlast][0]
            # print('keys: \n',trajectory_dataset.keys())
            # print('key_last: ', key_last)
            # print('conv_dataset: \n', trajectory_dataset[num_eval])
            # print('final_dataset: \n',trajectory_dataset[key_last])

            ##trajectory_display
            # self.state_traj()
            # self.illustration()
            # self.all_trajectories(trajectory_dataset,key_init,num_eval)
            # xxxx

            """        
            direct_dis = self.OTDD(initial_dataset,final_dataset)
            print('initial_dataset: \n', initial_dataset)
            print('final_dataset: \n', final_dataset)
            print('direct_dis: \n', direct_dis)
            """

            optimal = self.check_convergence(trajectory_dataset,key_init,num_eval)

            if optimal == 1: #iff converged
                data_points = len(trajectory_dataset) - 1 #size        
                Ns = int(data_points/1)
                a,b = 0,data_points #start & end_index
                idxs = np.linspace(a,b,Ns+1) #indexes

                ds_stt_col, ds_stt_cum = [],[]
                for i in range(idxs.shape[0] - 1):
                    dataset1 = trajectory_dataset[round(idxs[i])]
                    dataset2 = trajectory_dataset[round(idxs[i+1])]
                    ds_stt = self.OTDD(dataset1,dataset2)
                    ds_stt_col.append(ds_stt)
                    ds_stt_cum.append( sum(ds_stt_col))
                    
                # print('ds_stt_col', ds_stt_col )
                # print('ds_stt_cum', ds_stt_cum)
                
                # curve_dis = sum(ds_stt_col)  #ds_stt_cum[-1]
                # print('curve_dis_all: ', curve_dis)
                curve_dis = sum(ds_stt_col[:num_eval]) #sum index 0:num_eval
                # print('curve_dis_con: ', curve_dis)
                
                direct_dis = self.OTDD(initial_dataset,final_dataset)
                # print('direct_dis', direct_dis)
                
                #plot segment_lengths_vs_steps
                ##episodes
                # self.monot(ds_stt_col,ds_stt_cum,steps_col,score_col,con_eps,num_curve) #plots segments
                
                ##steps
                self.monot(ds_stt_col,ds_stt_cum,steps_col,score_col,num_eval,con_eps,num_curve) #plots segments

                # self.save_converge_data(initial_dataset,final_dataset,num_curve)
                # return curve_dis, direct_dis, ds_stt_cum, steps_col, score_col, con_eps
                return curve_dis, direct_dis, ds_stt_cum, steps_col, score_col, con_eps, num_eval, ds_stt_col
            
            else: return [0]

    # curve_lengths
    def curve_length(self,num_curve):
        trajectory_dataset, steps_col, score_col,con_eps,num_eval = self.agent.main(
                                    test_frq=self.test_freq, #policy_eval_frequency
                                    runs=1, #number_of_eval_repeats
                                    common_policy=0, #[1: yes | 0: no]
                                    )
        key_init = list(trajectory_dataset)[0]
        key_last = list(trajectory_dataset)[-1]
        # key_2ndlast = list(trajectory_dataset)[-2]

        if num_eval == -1: 
            num_eval = key_last

        initial_dataset = trajectory_dataset[key_init] #initial_policy_dataset
        final_dataset =  trajectory_dataset[num_eval] #converged_policy_dataset
        #final_dataset = trajectory_dataset[key_last] #final_policy_dataset

        # print('final_dataset: ', final_dataset)
        # print('final_dataset: ', final_dataset[0])
        self.optimal_traj = final_dataset[0] #trajectory_dataset[key_2ndlast][0]
        # print('keys: \n',trajectory_dataset.keys())
        # print('key_last: ', key_last)
        # print('conv_dataset: \n', trajectory_dataset[num_eval])
        # print('final_dataset: \n',trajectory_dataset[key_last])

        ##trajectory_display
        # self.state_traj()
        # self.illustration()
        # self.all_trajectories(trajectory_dataset,key_init,num_eval)
        # xxxx

        """        
        direct_dis = self.OTDD(initial_dataset,final_dataset)
        print('initial_dataset: \n', initial_dataset)
        print('final_dataset: \n', final_dataset)
        print('direct_dis: \n', direct_dis)
        """

        data_points = len(trajectory_dataset) - 1 #size        
        Ns = int(data_points/1)
        a,b = 0,data_points #start & end_index
        idxs = np.linspace(a,b,Ns+1) #indexes

        ds_stt_col, ds_stt_cum = [],[]
        for i in range(idxs.shape[0] - 1):
            dataset1 = trajectory_dataset[round(idxs[i])]
            dataset2 = trajectory_dataset[round(idxs[i+1])]
            ds_stt = self.OTDD(dataset1,dataset2)
            ds_stt_col.append(ds_stt)
            ds_stt_cum.append( sum(ds_stt_col))
            
        # print('ds_stt_col', ds_stt_col )
        # print('ds_stt_cum', ds_stt_cum)
        
        # curve_dis = sum(ds_stt_col)  #ds_stt_cum[-1]
        # print('curve_dis_all: ', curve_dis)
        curve_dis = sum(ds_stt_col[:num_eval]) #sum index 0:num_eval
        # print('curve_dis_con: ', curve_dis)
        
        direct_dis = self.OTDD(initial_dataset,final_dataset)
        # print('direct_dis', direct_dis)
        
        #plot segment_lengths_vs_steps
        ##episodes
        # self.monot(ds_stt_col,ds_stt_cum,steps_col,score_col,con_eps,num_curve) #plots segments
        
        ##steps
        self.monot(ds_stt_col,ds_stt_cum,steps_col,score_col,num_eval,con_eps,num_curve) #plots segments

        # self.save_converge_data(initial_dataset,final_dataset,num_curve)
        # return curve_dis, direct_dis, ds_stt_cum, steps_col, score_col, con_eps
        return curve_dis, direct_dis, ds_stt_cum, steps_col, score_col, con_eps, num_eval, ds_stt_col


    def save_converge_data(self,initial_dataset,final_dataset,num_curve):
        # file_loc = f"curve/fun_{self.it}/{self.setting}/ss_{self.state_label()}"
        # os.makedirs(join(this_dir,file_loc), exist_ok=True)
        # dataset_name = f"curve_{num_curve}" #.hdf5
        # file_path = abspath(join(this_dir,file_loc,dataset_name)) 
        # self.save_np(dataset_name,dataset,file_path)

        file_loc_fun = f"curve_{self.it}/fun/{self.setting}/ss_{self.state_label()}"
        file_loc_ini = f"curve_{self.it}/ini/{self.setting}/ss_{self.state_label()}"
        os.makedirs(join(this_dir,file_loc_fun), exist_ok=True)
        os.makedirs(join(this_dir,file_loc_ini), exist_ok=True)
        
        dataset_name = f"curve_{num_curve}" #.hdf5
        file_path_fun = abspath(join(this_dir,file_loc_fun,dataset_name)) 
        file_path_ini = abspath(join(this_dir,file_loc_ini,dataset_name)) 
        # print('initial_dataset: ', initial_dataset)
        # print('final_dataset: ', final_dataset)
        # xxx
        self.save_np(dataset_name,final_dataset,file_path_fun)
        self.save_np(dataset_name,initial_dataset,file_path_ini)
        return

    # iterative_computation (curve_length) for Boxplot of varying sampling rates
    def iterations(self,num_iter=20):
        for i in range(num_iter):
            # print('iteration: ', i)
            # data = np.asarray( self.curve_length_converged(i) ) 
            data = np.asarray( self.curve_length(i) ) 

            if len(data) == 1 : pass
            else: 
                dataset_name = f"curve_{i}" 
                file = f"stop_criteria/stoc/{self.num_stop}/tau_n/eps_{self.tau}/dts/{self.setting}/ss_{self.state_label()}" #curve_datasets
                run_file = f"curve_{i}" #.hdf5
                file_path = abspath(join(this_dir,file,run_file)) 
                os.makedirs(join(this_dir,file), exist_ok=True)
                self.save_np(dataset_name,data,file_path)

    #describe state_size as string
    def state_label(self):     
        return str(self.states_size)

    #save large data as numpy
    def save_np(self,name,data,file_path):
        np.save(file_path,data)
        return        

    # iterative_computation (curve_length)
    def plot_iterations(self,num_iter=20, block='[3x3]'): #average_cum/segment_lengths
        curve_dis_cum, steps_col_cum = {},{}
        steps_col = []
        for i in range(num_iter):
            print('iteration: ', i)
            _,direct_dis, ds_stt_cum, steps_col = self.curve_length()
            # _,direct_dis, ds_stt_cum = self.simcurve_length()
            print('direct_dis: ', direct_dis)
            # print('len(cum): ', len(ds_stt_cum))
            curve_dis_cum[i] = ds_stt_cum
            # steps_col.append(max_steps)
            steps_col_cum[i] = steps_col[1:]

        print('curve_dis_cum: ', curve_dis_cum)
        print('steps_col_cum: ', steps_col_cum)
        xxx
        
        print('steps_col: ',steps_col)
        curve_cum_df = pd.DataFrame.from_dict(curve_dis_cum, orient='index').T
        data = curve_cum_df.values
        means = np.nanmean(data, axis=1)
        stds = np.nanstd(data, axis=1)
        # print(curve_cum_df)
        # print(data)
        # print(np.nanmean(data, axis=1))
        # print(np.nanstd(data, axis=1))

        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(means,'-m*',label=f"tf: {self.test_freq} ({self.setting})" )
        plt.fill_between(range(len(means)), 
                         means-stds, 
                         means+stds, 
                         alpha=0.2,
                         color='m')
        plt.xlabel('steps')
        plt.ylabel('cumulative segment length')
        plt.title('Avg. Cumulative segment vs steps')
        plt.legend()

        plt.subplot(1,2,2)
        for i in range(num_iter):
            plt.plot(curve_dis_cum[i],
                     steps_col_cum[i],
                     color=np.random.rand(3,), 
                     alpha=0.70, 
                    #  label=f"{i}"
                     )
        plt.xlabel('steps')
        plt.ylabel('cumulative segment length')
        plt.title('Ind. Cumulative segment vs steps')
        
        """        
        ext = ['png','pkl']
        name1 = f"nev_{block}.{ext[0]}"
        name2 = f"nev_{block}.{ext[1]}"

        subfile = f"comp/n_{self.setting}/"
        file_path1 = abspath(join(this_dir,subfile,name1))
        file_path2 = abspath(join(this_dir,subfile,name2))

        # print('file_path1: ',file_path1)
        # print('file_path2: ',file_path2)

        curve_cum_df.to_pickle(file_path2)
        plt.savefig(file_path1)
        """

        # file_name = f"test_freq/{self.setting}/Sampling_Boxplot.png"
        
        # plt.close()
        plt.show()
    
    # boxplots of curve_length & geodesic_dis for varyinf sampling rates
    def plot_stats(self,curve,direct,show=1):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        labels = [f"tf: {self.test_freq}"] #test_frequency
        bp1 = ax1.boxplot(  curve, 
                            patch_artist=True, 
                            medianprops = dict(color = "black"),
                            labels=labels
                            )
        ax1.set_title(f"Curve_Len Boxplot [tf:{self.test_freq}] ({self.setting})") #tf:test_frequency

        bp2 = ax2.boxplot(  direct, 
                            patch_artist=True, 
                            medianprops = dict(color = "black"),
                            labels=labels
                            )
        ax2.set_title(f"Geodesic_Dis Boxplot [tf:{self.test_freq}] ({self.setting})") #tf:test_frequency

        plt.tight_layout()
        if show == 1:
            plt.show()
        else:
            plt.show()
            file_name = f"test_freq/{self.setting}/Sampling_Boxplot.png"
            plt.savefig(file_name)
        plt.close()
    
    #plot segment_lengths_vs_steps
    def monot(self,ds_col,ds_cum,steps_col,score_col,num_eval,con_eps,num_curve,show=0):
        # steps_col = steps_col[1:]
        steps_col = steps_col[:-1]

        fig = plt.figure(figsize=(12, 9))
        plt.subplot(2,2,1)
        plt.plot(steps_col,ds_col,'-',color='gray',label=f"sr: {self.test_freq} ({self.setting})") 
        plt.vlines(x = num_eval, #con_eps,
                   ymin=min(ds_col),
                   ymax=max(ds_col), 
                   colors='black', 
                   ls=':',
                #    label='Convergence'
                   )
        plt.xlabel('episodes')
        plt.ylabel('segment length')
        plt.title('segment length vs steps')
        plt.legend()

        plt.subplot(2,2,2)
        plt.plot(steps_col, ds_cum,'-',color='teal',label=f"sr: {self.test_freq} ({self.setting})") 
        plt.vlines(x = num_eval, #con_eps,
                   ymin=min(ds_cum),
                   ymax=max(ds_cum), 
                   colors='black', 
                   ls=':',
                #    label='Convergence'
                   )
        plt.xlabel('episodes')
        plt.ylabel('cumulative segment length')
        plt.title('cumulative segment length vs steps')
        plt.legend()

        plt.subplot(2,2,3)
        plt.title('Return Plot')
        plt.plot(score_col, '-',color='orchid')
        plt.vlines(x = num_eval, #con_eps,
                   ymin=min(score_col),
                   ymax=max(score_col), 
                   colors='black', 
                   ls=':',
                #    label='Convergence'
                   )
        plt.xlabel('episodes')
        plt.ylabel('returns')
        #plt.legend()
        
        plt.subplot(2,2,4)
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

        plt.tight_layout()
        
        if show == 1:
            plt.show()
        else:
            # print('saving')
            #curve_valid
            # file_loc = f"tau_n/eps_{self.tau}/vld/{self.setting}/ss_{self.state_label()}B"
            # file_loc = f"stop_criteria/{self.num_stop}/tau_n/eps_{self.tau}/vld/{self.setting}/ss_{self.state_label()}"
            file_loc = f"stop_criteria/stoc/{self.num_stop}/tau_n/eps_{self.tau}/vld/{self.setting}/ss_{self.state_label()}"
            
            os.makedirs(join(this_dir,file_loc), exist_ok=True)

            file = f"returns_{num_curve}.png"
            file_path = abspath(join(this_dir,file_loc,file)) 
            # print('file_name: ', file_path)
            plt.savefig(file_path)
        plt.close()


    def all_trajectories(self,trajectory_dataset,key_init,num_eval):
        
        for i in range(key_init,num_eval):  #num_eval
            print(i)
            print(trajectory_dataset[i])
            print('')
            traj = trajectory_dataset[i][0]
            self.illustration(traj,i)
            
            
        #     x = np.asarray(traj[:-1])
        #     y = np.asarray(trajectory_dataset[i][1])
        #     n_y = np.array(list( map(self.action_val,y) ))
        #     z = np.hstack([x,n_y])

        #     # print('z: ', z)
        #     u_z = np.unique(z, axis=0) 
        #     # print('u_z: ', u_z)

        #     n_z = StandardScaler().fit_transform(z)
        #     # print('n_z: ', n_z)

        #     pca = PCA(n_components=2)
        #     pc_z = pca.fit_transform(n_z)
        #     # print('pc_z: ',pc_z)
        #     u_pc_z = np.unique(pc_z, axis=0) 
        #     # print('u_pc_z: ',u_pc_z)

        #     self.data_plot(u_pc_z,i)
        # plt.show()
        # xxx
            
        return
    
    def check_convergence(self,trajectory_dataset,num_eval):
        opt = 0

        traj = trajectory_dataset[num_eval][0]
        x = np.asarray(traj[-1])
        y = len(trajectory_dataset[num_eval][1])

        x_goal = np.asarray(self.states_size) - 1
        y_goal = 2*(self.states_size[0] - 1)

        if (x == x_goal).all() and y == y_goal: 
            opt = 1

        # self.illustration(traj)
        # plt.show()
        # print('optimal: ', opt)
                    
        return opt
    
    def generate_colors(self):
        a = [str(k) for k in '0123456789ABCDEF' ]
        color = "#"+''.join(random.choice(a,6))        
        return color

    def data_plot(self,u_pc_z,i):
        plt.plot(u_pc_z[:,0],u_pc_z[:,1],'o',marker=f'${i}$',c=self.generate_colors())
        plt.xlabel('principal_comp_1')
        plt.ylabel('principal_comp_2')
        # plt.show()
        return
    
    def illustration(self,traj,pol=0,show=1):
        data, cmap, norm = self.state_traj_any(traj)
        n_size = self.states_size[0]

        #draw gridlines
        plt.grid(axis='both',
                color='k',
                linewidth=2
                )
        
        labels=[str(x) for x in range(n_size+1)]
        plt.xticks(np.arange(-.5,n_size,1) + 1, labels=labels)
        plt.yticks(np.arange(-.5,n_size,1) + 1, labels=labels)

        plt.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)
        plt.imshow(data, cmap=cmap, norm=norm)
        plt.title(f'Policy state-trajectory')

        plt.tight_layout()

        if show == 0:
            plt.show()
        else: 
            # file_loc = f"stop_criteria/{self.num_stop}/epsilon_decay/eps_{self.epsilon}/vld/{self.setting}/ss_{self.state_label()}/paths"
            file_loc = f"stop_criteria/stoc/{self.num_stop}/epsilon_decay/eps_{self.epsilon}/vld/{self.setting}/ss_{self.state_label()}/paths"
            
            file = f"p_{pol}.png"
            file_path = abspath(join(this_dir,file_loc,file)) 
            plt.savefig(file_path)

    #issue ANY policy path trajectory in state-space
    def state_traj_any(self,traj):
        n_input = np.asarray(traj)
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

    # curve_lengths
    def simcurve_length(self):
        trajectory_dataset = self.agent.main(
                                    test_frq=self.test_freq, #policy_eval_frequency
                                    runs=6, #number_of_eval_repeats
                                    common_policy=1, #[1: yes | 0: no]
                                    )
        
        key_init = list(trajectory_dataset)[0]
        key_last = list(trajectory_dataset)[-1]
        initial_dataset = trajectory_dataset[key_init] #initial_policy_dataset
        final_dataset = trajectory_dataset[key_last] #final_policy_dataset

        data_points = len(trajectory_dataset) - 1 #size        
        Ns = int(data_points/1)
        a,b = 0,data_points #start & end_index
        idxs = np.linspace(a,b,Ns+1) #indexes

        ds_stt_col, ds_stt_cum = [],[]
        simds_stt_col, simds_stt_cum = [],[]
        for i in range(idxs.shape[0] - 1):
            dataset1 = trajectory_dataset[round(idxs[i])]
            dataset2 = trajectory_dataset[round(idxs[i+1])]
            ds_stt = self.OTDD(dataset1,dataset2)
            simds_stt = self.simOTDD(dataset1,dataset2)

            ds_stt_col.append(ds_stt)
            ds_stt_cum.append( sum(ds_stt_col))
    
            simds_stt_col.append(simds_stt)
            simds_stt_cum.append( sum(simds_stt_col))

        # print('ds_stt_col', ds_stt_col)
        # print('ds_stt_cum', ds_stt_cum)
        # print('simds_stt_col', simds_stt_col)
        # print('simds_stt_cum', simds_stt_cum)

        curve_dis = sum(ds_stt_col)  #ds_stt_cum[-1]
        simcurve_dis = sum(simds_stt_col)  #ds_stt_cum[-1]
        print('curve_dis: ', curve_dis)
        print('simcurve_dis: ', simcurve_dis)
        print('curve/simcurve: ', curve_dis/simcurve_dis )

        direct_dis = self.OTDD(initial_dataset,final_dataset)
        simdirect_dis = self.simOTDD(initial_dataset,final_dataset)
        print('direct_dis', direct_dis)
        print('simdirect_dis', simdirect_dis)
        print('direct/simdirect: ', direct_dis/simdirect_dis )

        #plot segment_lengths_vs_steps
        self.simmonot(ds_stt_col,simds_stt_col,ds_stt_cum,simds_stt_cum) #plots segments

        return curve_dis, direct_dis, ds_stt_cum
    
    #plot segment_lengths_vs_steps
    def simmonot(self,ds_col,simds_col,ds_cum,simds_cum,show=1):
        fig = plt.figure(figsize=(12, 5))

        plt.subplot(1,2,1)
        plt.plot(ds_col,'-gd',label="OTDD") 
        plt.plot(simds_col,'-rd',label="simOTDD") 
        plt.xlabel('steps')
        plt.ylabel('segment length')
        plt.title('segment length vs steps')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(ds_cum,'-m*',label="OTDD") 
        plt.plot(simds_cum,'-b*',label="simOTDD") 
        plt.xlabel('steps')
        plt.ylabel('cumulative segment length')
        plt.title('cumulative segment length vs steps')
        
        plt.tight_layout()
        plt.legend()
        if show == 1:
            plt.show()
        else:
            name = f"otdd_v_sim[3x3].png"
            subfile = f"test_freq/{self.setting}/"
            file_path = abspath(join(this_dir,subfile,name))
            # file_name = f"test_freq/{self.setting}/Sampling_Boxplot.png"
            plt.savefig(file_path)
        plt.close()

    # save_simulation_data
    def test_save(self,curve,direct):
        data = [curve,direct]
        name =  self.test_freq #'50'
        data_file = f"test_freq/{self.setting}/{name}.npy"
        np.save(data_file, data)
        return

    def OTDD(self,data1,data2):
        x1, x2 = np.asarray(data1[0][:-1]), np.asarray(data2[0][:-1]) #states (features)
        y1, y2 = np.asarray(data1[1]), np.asarray(data2[1]) #actions (labels)
        y_space = ['up', 'right', 'down', 'left'] #action_space
    
        # print('x1,y1: ', x1,y1)
        # print('x2,y2: ', x2,y2)

        #task_1 (feature set with label Y = y)
        col_X_y1 = {} #collection of X given Y = y
        for i in y_space:
            nd_y = [] #set of X given Y = y
            for x,y in zip(x1,y1):
                if y == i:
                    nd_y.append(x)
            col_X_y1[i] = nd_y
        # print('col_X_y1: ', col_X_y1)

        #task_2 (feature set with label Y = y)
        col_X_y2 = {} #collection of X given Y = y
        for i in y_space:
            nd_y = [] #set of X given Y = y
            for x,y in zip(x2,y2):
                if y == i:
                    nd_y.append(x)
            col_X_y2[i] = nd_y
        # print('col_X_y2: ', col_X_y2)

        #saving label-to-label distances
        label_dis = defaultdict(int)
        for i in y_space:
            for j in y_space:
                if len(col_X_y1[i]) != 0 and len(col_X_y2[j]) != 0: 
                    pair = (i,j)
                    label_dis[pair] = self.inner_wasserstein(
                        col_X_y1[i],
                        col_X_y2[j]
                        ) 
        # print('label_dis: ', label_dis)

        otdd = self.outer_wasserstein(x1,y1,x2,y2,label_dis)
        # print('otdd: ', otdd)        

        return otdd

    def simOTDD(self,data1,data2):
        x1, x2 = np.asarray(data1[0][:-1]), np.asarray(data2[0][:-1]) #states (features)
        y1, y2 = np.asarray(data1[1]), np.asarray(data2[1]) #actions (labels)
    
        # print('x1,y1: ', x1,y1)
        # print('x2,y2: ', x2,y2)

        n_y1 = np.array(list( map(self.action_val,y1) ))
        n_y2 = np.array(list( map(self.action_val,y2) ))
        z1 = np.hstack([x1,n_y1])
        z2 = np.hstack([x2,n_y2])

        sim_otdd = self.sim_wasserstein(x1,x2,n_y1,n_y2)
        # print('sim_otdd: ', sim_otdd)

        return sim_otdd
    
    #simple_OT
    def sim_wasserstein(self,x1,x2,y1,y2):
        P = ot.unif(x1.shape[0])
        Q = ot.unif(x2.shape[0])
        dx = ot.dist(x1,x2, metric='cityblock') #cost matrix: 'euclidean'
        dy = ot.dist(y1,y2, metric='euclidean') #cost matrix: 'cityblock'
        dz = dx + dy

        """        
        print('dz: ', dz)
        m = self.test_cost(x1,x2,y1,y2)
        print('m: ', m)
        print( np.all(m == dz) )
        """

        val = ot.emd2(  P, #A_distribution 
                        Q, #B_distribution
                        M = dz, #cost_matrix pre-processing
                        numItermax=int(1e6)
                        ) #OT matrix
        return val

    #test_cost_metric
    def test_cost(self,x1,x2,y1,y2):
        m = np.zeros([len(x1),len(x2)])
        for idx, i in enumerate(x1):
            for jdx, j in enumerate(x2):
                m[idx][jdx] =  LA.norm( (i - j ), ord=1) + LA.norm( (y1[idx] - y2[jdx] ), ord=1)   #ord = 1: 'cityblock' norm 
        return m 

    # action_convertor    
    def action_val(self,action):
        if action == 'up': return np.array([-1,0])
        elif action == 'down': return np.array([1,0])
        elif action == 'right': return np.array([0,1])
        elif action == 'left': return np.array([0,-1])

    def state_val(self,state):
        s0 = np.interp(state[0], (0,2), (0,1) ) 
        s1 = np.interp(state[1], (0,2), (0,1) ) 
        # print(np.array([s0,s1]))

        if all(state == 0): n_state = state
        else: n_state = state/LA.norm(state)
        # print(n_state)

        return n_state, np.array([s0,s1])
    
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
        val = ot.emd2(  p_xy1, #A_distribution 
                        p_xy2, #B_distribution
                        M = Cm1, #cost_matrix pre-processing
                        numItermax=int(1e6)
                        ) #OT matrix
        return val**1.0 #L1_norm
    
    #outer_cost_metric
    def outer_cost(self,z1,z2,label_dis):
        m = np.zeros([len(z1),len(z2)])
        for idx, i in enumerate(z1):
            for jdx, j in enumerate(z2):
                m[idx][jdx] = ( ( LA.norm( (i[0] - j[0]), ord=1) )  #ord = 1: 'cityblock' norm 
                                +  ( label_dis[(i[1],j[1])] ) )
        return m 


#Execution
if __name__ == '__main__':
    n = 5 #33
    hardns = otdd(
        test_freq=1, #1 
        states_size=[n,n],#[50,50],
        setting=1, # [1:dense | 0:sparse ]
        n_eps=2500, #1000:5, 2000:15, 3000:25 [1: 100 | 0: 500]
        tau=2, #[0.5, 2, 4, 10]
        stop_cri=5,
        ) 
     
    hardns.iterations(num_iter=100) #20 
    # hardns.curve_length()

    # hardns.plot_iterations(num_iter=2, block='[3x3]')
    # hardns.simcurve_length()


    # test_freqs = [1]
    # for tf in test_freqs:
    #     iter_hardns = otdd(tf,0)
    #     iter_hardns.iterations()

    print('complete')
    """    
    xx
    tf = 1
    setting = ['sps', 'dns']
    stg = 0 # [0: sps, 1:dns]
    i_hardns = otdd(tf,stg) #1 
    print('')
    print(f"test_freq: {tf} | setting: {setting[stg]}")
    print('')
    i_hardns.plot_iterations(num_iter=50, block='[3x3]') #100, 50
    """


