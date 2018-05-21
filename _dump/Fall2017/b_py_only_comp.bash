#!/bin/bash
for loss in "Thurstone" "diff" "BT"
do
    for layers in "1" "2" "3" "4" "5" "6"
    do
        work=/gss_gpfs_scratch/yildiz.i/
        cd $work
        sbatch execute_only_comp.bash $layers $loss
    done
done
