# Studying Exploration in RL: An Optimal Transport Analysis of Occupancy Measure Trajectories

[![Published in TMLR](https://img.shields.io/badge/TMLR-2025-b31b1b.svg)](https://openreview.net/forum?id=pdC092Nn8N)
[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-blue.svg)](https://openreview.net/pdf?id=pdC092Nn8N)

Reference implementation for the paper **"Studying Exploration in RL: An Optimal Transport Analysis of Occupancy Measure Trajectories"** (Nkhumise, Basu, Prescott & Gilra), published in *Transactions on Machine Learning Research* (TMLR), 2025 — [OpenReview](https://openreview.net/forum?id=pdC092Nn8N).

## Motivation

Modern reinforcement-learning (RL) algorithms differ widely in *how* they explore and optimise, yet they are routinely compared on a single axis — cumulative reward. A reward curve says little about the *process* by which an agent reaches its policy: how directly it travels, how much of its movement is productive, and how task difficulty shapes that journey. This work provides a quantitative, algorithm-agnostic framework for comparing the **learning processes** of RL algorithms.

## Approach

We represent the learning process as the sequence of policies $\{\pi_0, \pi_1, \dots, \pi_T\}$ produced during training, and study the **trajectory this sequence induces on the manifold of state–action occupancy measures**. Distances between successive policies are measured with an **optimal-transport metric** (a nested / hierarchical Wasserstein distance over the occupancy measures, in the spirit of the Optimal Transport Dataset Distance). The geometry of this policy trajectory — its length, curvature, and direction relative to the optimum — exposes the exploration behaviour that reward curves hide.

## Metrics

The framework introduces two complementary, theoretically grounded metrics:

- **Effort of Sequential Learning (ESL)** — the length of the policy path travelled in occupancy-measure space, relative to the shortest (geodesic) path from the initial to the optimal policy. ESL quantifies how *circuitous* an algorithm's learning is.
- **Optimal Movement Ratio (OMR)** — the fraction of policy movement that effectively reduces an analogue of regret. OMR connects occupancy-measure dynamics to suboptimality, quantifying how *productive* each update is.

The paper derives **finite-sample approximation guarantees** that allow ESL and OMR to be estimated from samples *without access to an optimal policy*.

## Experimental scope

The metrics are evaluated across both discrete and continuous MDPs, spanning value-based, policy-gradient, and model-based exploration algorithms:

| Environment | State–action space | Algorithms studied |
|---|---|---|
| [`Gridworld_OTDD/`](Gridworld_OTDD) | Discrete | DQN, Q-learning, SARSA, discrete SAC, PSRL, UCRL2, Boltzmann exploration |
| [`Mountain_Car_OTDD/`](Mountain_Car_OTDD) | Continuous | DDPG, SAC, PPO |

Each environment directory additionally provides **task-hardness analyses** (`evals/hardness_*.py`) examining how MDP difficulty interacts with exploration effort.

## Repository structure

```
.
├── Gridworld_OTDD/          # Discrete state–action experiments
│   ├── envs/                #   training scripts per algorithm (DQN.py, PSRL.py, UCRL2.py, ...)
│   ├── models/              #   policy/occupancy-measure generation & evaluation
│   └── evals/               #   task-hardness analyses
├── Mountain_Car_OTDD/       # Continuous state–action experiments
│   ├── envs/                #   training scripts (ddpg.py, sac.py, ppo.py)
│   └── evals/               #   occupancy-measure model evaluation
└── environment.yml          # Conda environment specification
```

## Installation

We recommend an isolated environment via [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
git clone https://github.com/nkhumise-rea/analysis_of_occupancy_measure_trajectory.git
cd analysis_of_occupancy_measure_trajectory
conda env create -f environment.yml
conda activate analysis
```

**Core dependencies:** Python 3.10, PyTorch 1.13, [POT (Python Optimal Transport)](https://pythonot.github.io/) 0.9, [Pymanopt](https://pymanopt.org/) 2.1 (manifold optimisation), and Gym 0.26.

## Reproducing the experiments

The full ESL/OMR pipeline — (1) train and checkpoint a sequence of policies, (2) generate occupancy-measure trajectories, (3) evaluate the metrics and visualise the policy evolution — is documented per environment:

- **[`Gridworld_OTDD/README.md`](Gridworld_OTDD/README.md)** — discrete state–action experiments.
- **[`Mountain_Car_OTDD/README.md`](Mountain_Car_OTDD/README.md)** — continuous state–action experiments.

Both guides also explain how to drop in your own algorithm and have it analysed under the same framework.

## Citation

```bibtex
@article{nkhumise2025studying,
  title   = {Studying Exploration in {RL}: An Optimal Transport Analysis of Occupancy Measure Trajectories},
  author  = {Nkhumise, Reabetswe M. and Basu, Debabrota and Prescott, Tony J. and Gilra, Aditya},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2025},
  url     = {https://openreview.net/forum?id=pdC092Nn8N}
}
```
