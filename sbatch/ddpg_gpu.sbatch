#!/bin/bash
#SBATCH -JGPUDDPGTraining
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --time=16:00:00
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ssahoo61@gatech.edu

module load anaconda3/2023.03
source activate ecn
cd /home/hice1/ssahoo61/Documents/exploration-critique-networks
srun python ddpg.py