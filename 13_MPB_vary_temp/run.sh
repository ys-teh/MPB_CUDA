#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=01:30:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
### #SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "MC113"   # job name
#SBATCH --mail-user=yteh@caltech.edu   # email address
### #SBATCH --mail-type=BEGIN
### #SBATCH --mail-type=END
### #SBATCH --qos=debug
#SBATCH --output=slurm.out

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load cuda/9.1 
module load intel/19.1

make clean
make

TEST=1

PERCENT=50
time ./MonteCarlo ${PERCENT} ${TEST} > results/log_percent${PERCENT}_test${TEST}


