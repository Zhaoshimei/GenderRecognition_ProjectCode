#!/bin/bash
# 1 node, 8 tasks per node = 8 cores
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-24:24:00
#SBATCH --partition gpgpu

# Load required modules
module load Python/3.5.2-goolf-2015a

# Launch multiple process python code
time mpiexec -n 8 python3 train.py
