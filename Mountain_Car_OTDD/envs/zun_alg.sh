#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=MCC #MCC: MountainCar_Continuous 
#SBATCH --time=30:00:00
##SBATCH --mail-user=rmnkhumise1@sheffield.ac.uk
##SBATCH --mail-type=ALL
#SBATCH --output=output.%j.MCC.out 

module load Anaconda3/2022.05
source activate gym3pot

# python ddpg.py #ddpg
python sac.py #sac