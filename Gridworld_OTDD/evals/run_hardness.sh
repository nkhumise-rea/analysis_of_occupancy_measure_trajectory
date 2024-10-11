#!/bin/bash
#SBATCH --mem=128G #65G
#SBATCH --job-name=d_st[5] ##d_, s_
#SBATCH --time=30:00:00
##SBATCH --mail-user=rmnkhumise1@sheffield.ac.uk
##SBATCH --mail-type=ALL
#SBATCH --output=output.%j.dns_st[5].out ##dns_, sps_

# module load Anaconda3/2022.10
module load Anaconda3/2022.05
# module load cuDNN/8.8.0.121-CUDA-12.0.0


source activate gym3pot

# python hardness_Q2.py
python hardness_SAC2.py
# python hardness_UCRL2.py
# python hardness_DQN2.py 
# python hardness_PSRL2.py

# python hardness_DQN.py 
# python hardness_SAC.py
# python hardness_Q.py
# python hardness_Bolt.py
# python hardness_UC.py
# python hardness_UCRL.py
# python hardness_PSRL.py
