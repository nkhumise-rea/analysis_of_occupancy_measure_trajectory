# analysis_of_occupancy_measure_trajectory
This depository contains scripts used for analysing various reinforcement learning (RL) algorithms by studying policy updates in the space of occupancy measures using optimal transport. This is supplementary material for the [paper](https://arxiv.org/abs/2402.09113): "How does your RL agent explore? An optimal transport analysis of occupancy measure trajectories."

## Installation 
We recommend to set up a virtual environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

1. Clone the repo: <code>git clone https://github.com/nkhumise-rea/analysis_of_occupancy_measure_trajectory.git</code>   
2. Navigate to the directory where the clone exists.
3. Open command line and run the following: 
	- <code>conda env create -f environment.yml</code>  to create the environment. 
 	- <code>conda activate analysis</code> to activate the environment. 
6. Verify installations by running <code>conda list</code> and <code>pip list</code>. 

## Execution
You can find relevant instructions to evaluate algorithms in the <code>README.md files</code> located at:

1. <code>Gridworld_OTDD directory</code> for an environment with discrete states and actions.
2. <code>Mountain_Car_OTDD directory</code> for an environment with continuous states and actions.  
