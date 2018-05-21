#!/bin/bash
#set a job name  
#SBATCH --job-name=only_comp
#################  
#a file for job output, you can check job progress
#SBATCH --output=only_comp.out
#################
# a file for errors from the job
#SBATCH --error=only_comp.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting
#SBATCH -N 1
#SBATCH --exclusive
#################
#partition to use
#SBATCH --partition=par-gpu
#################
#number of nodes to distribute n tasks across
#################

python main_onlyComparisonNN.py train 0 $1 $2 80
