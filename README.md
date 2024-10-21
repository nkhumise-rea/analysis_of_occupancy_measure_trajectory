# analysis_of_occupancy_measure_trajectory
Scripts for analyzing various RL algorithms using optimal transport. This is supplementary material for the paper: "How does your RL agent explore? An optimal transport analysis of occupancy measure trajectories"

## Table of contents
- [Installation] (##installation)

## Installation 
We recommend to set up a virtual environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

1. Clone the repo: <code>git clone https://github.com/nkhumise-rea/analysis_of_occupancy_measure_trajectory.git</code>   
2. Navigate to the directory where the clone exists.
3. Open command line and run the following. 
4. Run <code>conda env create -f environment.yml</code>  to create the environment. 
5. Run <code>conda activate analysis</code> to activate the environment. 
6. Verify installations by running <code>conda list</code> and <code>pip list</code>. 

## Execution
We provide details about running an exemplary algorithm (e.g. DQN in Gridworld)

### Creating & saving policy models 
1. Navigate to directory <code>/Gridworld_OTDD/envs</code>. 
2. Open <code>DQN.py</code> file.  
3. Select the task environment in 'lines 37-38</code>
4. Under the <code>if \__name__ == "\__main__"</code> code block specify
	- Grid size using <code>states_sizes</code> variable.
	- reward setting using <code>rew_setting</code> variable. Note that 1 = dense rewards, 0 = sparse rewards
	- Number of training episodes using <code>n_eps</code> variable. 
	- Problem setting by labelling it as desired. This will aid in tracking down files.   
5. Save file and run <code>python DQN.py</code> in command line to execute.

### Using policy models
1. Navigate to directory <code>/Gridworld_OTDD/models</code>. 
2. Open <code>DQN_models.py</code> file. 
3. Ensure task environment, grid size, reward setting, number of training episodes, and the problem setting match with those in the <code>DQN.py</code> file. 
4. Under the <code>if \__name__ == "\__main__"</code> code block, ensure only the relevant function is uncommented while the rest are commented. 
	- It mandatory to run <code>agent.policy_data_generation(...)</code> and then <code>agent.occupancy_generation(...)</code> in this order. 
	> These generates state-action pair rollouts and policy trajectories in the occupancy measure space.
	- Use <code>agent.policy_evolution_plot(...)</code> to visualize the policy evoluation.
	- Use <code>agent.policy_trajecotry_evaluation(...)</code> to assess metrics of a single trajectory.
	- Otherwise use <code>agent.policy_trajecotry_evaluation_stats(...)</code> to statistically assess the metrics where means and standard deviations are outputted. 
