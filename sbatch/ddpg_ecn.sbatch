#!/bin/bash
#SBATCH -JECNTraining
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G --cpus-per-task=4
#SBATCH --time=17:59:59
#SBATCH -q coc-ice
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ssahoo61@gatech.edu

cd /home/hice1/ssahoo61/Documents/exploration-critique-networks
source venv/bin/activate
srun python ddpg_ecn.py