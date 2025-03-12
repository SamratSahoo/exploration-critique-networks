#!/bin/bash
#SBATCH -JGPUDDPGTraining
#SBATCH -N1 --gres=gpu:1 -C A100-40GB
#SBATCH --time=16:00:00
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out

module load anaconda3/2023.03
source activate ecn
cd /home/hice1/ssahoo61/Documents/exploration-critique-networks
srun python ddpg.py