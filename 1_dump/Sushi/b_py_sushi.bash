#!/bin/bash
for samp_per_user in "1" "20" "40" "100"
do
    for sim_thr in "0.2" "0.4" "0.6" "0.8"
    do
        for train_set in "abs" "comp" "both"
        do
            work=/gss_gpfs_scratch/yildiz.i/
            cd $work
            sbatch execute_sushi.bash $train_set $sim_thr $samp_per_user
        done
    done
done
