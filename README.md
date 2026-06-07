# Analysis of Occupancy Measure Trajectories

Code accompanying the paper **"How does your RL agent explore? An optimal transport analysis of occupancy measure trajectories"** ([arXiv:2402.09113](https://arxiv.org/abs/2402.09113)).

This repository provides the scripts used to analyse reinforcement-learning (RL) algorithms by tracking how their policies evolve in the space of **occupancy measures**, using optimal transport to measure the distance between successive policies along a training trajectory.

## Repository structure

| Directory | Environment | State / action space |
|---|---|---|
| [`Gridworld_OTDD/`](Gridworld_OTDD) | Gridworld | Discrete states and actions |
| [`Mountain_Car_OTDD/`](Mountain_Car_OTDD) | Mountain Car | Continuous states and actions |

Each directory has its own `README.md` with step-by-step instructions for training policies, generating occupancy-measure trajectories, and evaluating the optimal-transport metrics.

## Installation

We recommend setting up a virtual environment with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Clone the repository:
   ```bash
   git clone https://github.com/nkhumise-rea/analysis_of_occupancy_measure_trajectory.git
   cd analysis_of_occupancy_measure_trajectory
   ```
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate analysis
   ```
3. Verify the installation with `conda list` and `pip list`.

## Usage

See the per-environment instructions:

- **[`Gridworld_OTDD/README.md`](Gridworld_OTDD/README.md)** — discrete state–action experiments (e.g. DQN).
- **[`Mountain_Car_OTDD/README.md`](Mountain_Car_OTDD/README.md)** — continuous state–action experiments (e.g. DDPG).

Both guides cover running an existing algorithm and adding your own.

## Citation

If you use this code, please cite:

```bibtex
@article{nkhumise2024explore,
  title   = {How does Your RL Agent Explore? An Optimal Transport Analysis of Occupancy Measure Trajectories},
  author  = {Nkhumise, Reabetswe M. and Basu, Debabrota and Prescott, Tony J. and Gilra, Aditya},
  journal = {arXiv preprint arXiv:2402.09113},
  year    = {2024}
}
```
