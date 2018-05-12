#!/bin/bash
for samp_per_user in "1" "5" "10"
do
    for no_user in "10" "50" "100" "500" "1000" "3000" "5000"
    do
        work=/gss_gpfs_scratch/yildiz.i/
        cd $work
        sbatch execute_abs.bash $no_user $samp_per_user
    done
done
