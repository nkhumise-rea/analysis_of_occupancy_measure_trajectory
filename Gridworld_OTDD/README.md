# Gridworld (discrete state-action space)
This depository contains scripts used for analysing various reinforcement learning (RL) algorithms in a discrete state-action space. The paper, published in *Transactions on Machine Learning Research* (TMLR, 2025), can be found on [OpenReview](https://openreview.net/forum?id=pdC092Nn8N).

## Executing an existing algorithm
We provide details about running an exemplary algorithm (e.g. DQN in the Gridworld)

### Creating & saving policy models 
1. Navigate to directory <code>/Gridworld_OTDD/envs</code>. 
2. Open <code>DQN.py</code> file.  
3. Select the desired task environment in </code>lines 37-38</code>
4. Under the <code>if \__name__ == "\__main__"</code> code block specify:
	- Grid size using <code>states_sizes</code> variable.
	- Reward setting using <code>rew_setting</code> variable. (Note 1=dense rewards, 0=sparse rewards)
	- Number of training episodes using <code>n_eps</code> variable. 
	- Problem setting by labelling it as desired. (This will aid in tracking down files.)   
5. Save file and run <code>python DQN.py</code> in command line to execute.

### Using policy models
1. Navigate to directory <code>/Gridworld_OTDD/models</code>. 
2. Open <code>DQN_models.py</code> file. 
3. Ensure _task environment_, _grid size_, _reward setting_, _number of training episodes_, and the _problem setting_ match with those in the <code>DQN.py</code> file. 
4. Under the <code>if \__name__ == "\__main__"</code> code block, ensure only the relevant function is uncommented while the rest are commented. 
	- It mandatory to run <code>agent.policy_data_generation(...)</code> and then <code>agent.occupancy_generation(...)</code> in this order. 
	> These generates state-action pair rollouts and policy trajectories in the occupancy measure space.
	- Use <code>agent.policy_evolution_plot(...)</code> to visualize the policy evoluation.
	- Use <code>agent.policy_trajecotry_evaluation(...)</code> to assess metrics of a single trajectory.
	- Otherwise use <code>agent.policy_trajecotry_evaluation_stats(...)</code> to statistically assess the metrics where means and standard deviations are outputted. 
	
## Executing a custom algorithm
Suppose you like to run your own algorithm, e.g. OWN in the Gridworld.

### Creating, saving and using policy models 
1. Navigate to directory <code>/Gridworld_OTDD/envs</code>. 
2. Create <code>OWN.py</code> file.  
3. Import task environment following exisiting script.
4. Specify:
	- Grid size using <code>states_sizes</code> variable.
	- Reward setting using <code>rew_setting</code> variable. (Note 1=dense rewards, 0=sparse rewards)
	- Number of training episodes using <code>n_eps</code> variable. 
	- Problem setting by labelling it as desired. (This will aid in tracking down files.)   
5. Develop a <code>policy_models</code> function that saves updating policy models during training. Follow along with how existing scripts have been written. 
6. Save file and run <code>python OWN.py</code> in command line to execute.
7. To use policy models follow the same steps above.

