#!/bin/bash

#SBATCH --job-name=PI_um
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=48:00:00
#SBATCH --no-requeue

#SBATCH --mem=100G

# Load required modules first
module purge  # Clear any existing modules
#module load python/3.12
module load netcdf-c
module load openmpi

source /sw/spack-levante/jupyterhub/jupyterhub/etc/profile.d/conda.sh

# Activate virtual environment
conda activate $HOME/.conda/envs/hk25


#python3 /home/b/b383007/hk25-hamburg/hk25-Teams/hk25-TropCyc/notebooks/PI_compute.py
mpirun -np 16 python3 /home/b/b383007/hk25-hamburg/hk25-teams/hk25-TropCyc/notebooks/PI_compute.py
