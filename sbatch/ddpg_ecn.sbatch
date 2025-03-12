#!/bin/bash
#SBATCH -JECNTraining
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out

module load anaconda3/2023.03
source activate ecn
cd /home/hice1/ssahoo61/Documents/exploration-critique-networks
srun python ddpg_ecn.py