#!/bin/bash
for kthFold in "0" "1" "2" "3" "4"
do
    work=/gss_gpfs_scratch/tian.pe/ASSIST-Ilkay/
    cd $work
    sbatch execute.bash $kthFold
done