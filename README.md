# analysis_of_occupancy_measure_trajectory
Scripts for analyzing various RL algorithms using optimal transport. This is supplementary material for the paper: "How does your RL agent explore? An optimal transport analysis of occupancy measure trajectories"

## Table of contents
- [Installation] (#installation)

## Installation 
We recommend to set up a virtual environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

1. Clone the repo: <code> git clone https://github.com/nkhumise-rea/  analysis_of_occupancy_measure_trajectory.git </code>   
2. Navigate to the directory where the clone exists.
3. Open command line and run the following. 
4. Run 'conda env create -f environment.yml' to create the environment. 
5. Run 'conda activate analysis' to activate the environment. 
6. Verify installations by running 'conda list' and 'pip list'

## Execution
We provide details about running an exemplary algorithm (e.g. DQN in Gridworld)

### Creating & saving policy models 
1. Navigate to directory '/Gridworld_OTDD/envs'. 
2. Open 'DQN.py' file.  
3. Select the task environment in 'lines 37-38'
4. Under the 'if **name** == "**main**"' code block specify
	- Grid size using 'states_sizes' variable.
	- reward setting using 'rew_setting' variable. Note that 1 = dense rewards, 0 = sparse rewards
	- Number of training episodes using 'n_eps' variable. 
	- Problem setting by labelling it as desired. This will aid in tracking down files.   
5. Save file and run 'python DQN.py' in command line to execute.

### Using policy models
1. Navigate to directory '/Gridworld_OTDD/models'. 
2. Open 'DQN_models.py' file. 
3. Ensure task environment, grid size, reward setting, number of training episodes, and the problem setting match with those in the 'DQN.py' file. 
4. Under the 'if __name__ == "__main__"' code block, ensure only the relevant function is uncommented while the rest are commented. 
	- It mandatory to run 'agent.policy_data_generation(...)' and then 'agent.occupancy_generation(...)' in this order. 
	> These generates state-action pair rollouts and policy trajectories in the occupancy measure space.
	- Use 'agent.policy_evolution_plot(...)' to visualize the policy evoluation.
	- Use 'agent.policy_trajecotry_evaluation(...)' to assess metrics of a single trajectory.
	- Otherwise use 'agent.policy_trajecotry_evaluation_stats(...)' to statistically assess the metrics where means and standard deviations are outputted. 
