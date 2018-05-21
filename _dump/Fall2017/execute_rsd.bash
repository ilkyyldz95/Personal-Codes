#!/bin/bash
#set a job name  
#SBATCH --job-name=rsd
#################  
#a file for job output, you can check job progress
#SBATCH --output=rsd.out
#################
# a file for errors from the job
#SBATCH --error=rsd.err
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

python main_deepROP.py train $1 $2
