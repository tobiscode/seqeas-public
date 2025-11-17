#!/bin/bash

# General settings 
#SBATCH --job-name=
#SBATCH --nodes=16
#SBATCH --tasks-per-node=2
#SBATCH --mem=100000M
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
 
# check the configuration
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NTASKS"=$SLURM_NTASKS
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# load the altar environment, e.g.
# module load cuda/12.2
# mamba activate altar

# launch the altar application
date
time mpirun SEAS3D --config=seas.pfg --shell.hosts=$SLURM_NNODES --shell.tasks=$SLURM_NTASKS --shell.auto=no
pyexit=$?
mkdir -p profiling steps && mv prof-*.csv profiling/ || true && mv step*.h5 steps/ || true
date
exit $pyexit
