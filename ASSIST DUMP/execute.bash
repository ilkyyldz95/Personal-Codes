#!/bin/bash
#set a job name  
#SBATCH --job-name=run1
#################  
#a file for job output, you can check job progress
#SBATCH --output=run1.out
#################
# a file for errors from the job
#SBATCH --error=run1.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting
#SBATCH -n 1
#################
#partition to use
#SBATCH --partition=par-gpu-2
#################
#number of nodes to distribute n tasks across
#################

python abs_net_train_g.py $1
