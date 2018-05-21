#!/bin/bash
#set a job name  
#SBATCH --job-name=abs
#################  
#a file for job output, you can check job progress
#SBATCH --output=abs.out
#################
# a file for errors from the job
#SBATCH --error=abs.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting, N for all cores
#SBATCH -N 1
#SBATCH --exclusive
#################
#partition to use
#SBATCH --partition=par-gpu
#################
#number of nodes to distribute n tasks across
#################

python abs_net_train_vs_datasize.py $1 