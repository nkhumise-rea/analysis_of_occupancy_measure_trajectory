# Mountain Car (continuous state-action space)
This depository contains scripts used for analysing various reinforcement learning (RL) algorithms in a continuous state-action space. The paper, published in *Transactions on Machine Learning Research* (TMLR, 2025), can be found on [OpenReview](https://openreview.net/forum?id=pdC092Nn8N).

## Executing an existing algorithm
We provide details about running an exemplary algorithm (e.g. DDPG in the Mountain-Car)

### Creating & saving policy models 
1. Navigate to directory <code>/Mountain_Car_OTDD/envs</code>. 
2. Open <code>ddpg.py</code> file.  
4. Under the <code>if \__name__ == "\__main__"</code> code block specify:
	- Number of training episodes using <code>n_eps</code> variable. 
	- Number of training iterations.    
5. Save file and run <code>python DDPG.py</code> in command line to execute.

### Using policy models
1. Navigate to directory <code>/Mountain_Car_OTDD/models</code>. 
2. Open <code>ddpg_models.py</code> file. 
3. Specify the number of discretization bins for states and actions or use default.
4. Under the <code>if \__name__ == "\__main__"</code> code block, ensure only the relevant function is uncommented while the rest are commented. 
	- Match number of training episodes <code>n_eps</code> with those in <code>ddpg.py</code> file.
	- First, run <code>agent.policy_data_generation(...)</code> for the _number of training iterations_.
	- Second, run <code>agent.occupancy_generation(...)</code>. 
	> These generates state-action pair rollouts and policy trajectories in the occupancy measure space.
	- Use <code>agent.policy_evolution_plot(...)</code> to visualize the policy evoluation.
	- Use <code>agent.policy_trajecotry_evaluation(...)</code> to assess metrics of a single trajectory.
	
## Executing a custom algorithm
Suppose you like to run your own algorithm, e.g. OWN in the Mountain-Car.

### Creating, saving and using policy models 
1. Navigate to directory <code>/Mountain_Car_OTDD/envs</code>. 
2. Create <code>OWN.py</code> file.  
3. Import task environment following exisiting script.
4. Specify _number of training episodes_ using <code>n_eps</code> variable.   
5. Develop a <code>policy_models</code> function that saves updating policy models during training. Follow along with how existing scripts have been written. 
6. Save file and run <code>python OWN.py</code> in command line to execute.
7. To use policy models follow the same steps above.
