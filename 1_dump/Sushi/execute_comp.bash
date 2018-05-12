#!/bin/bash
#set a job name  
#SBATCH --job-name=comp
#################  
#a file for job output, you can check job progress
#SBATCH --output=comp.out
#################
# a file for errors from the job
#SBATCH --error=comp.err
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
#SBATCH --partition=ioannidis
#################
#number of nodes to distribute n tasks across
#################

python main_comparison_network.py train $1 $2
