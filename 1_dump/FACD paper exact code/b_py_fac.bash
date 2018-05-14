#!/bin/bash
for lambda in "0.02" "0.002" "0.0002" "0.0"
do
    for lr in "1e-06" "1e-05" "1e-04"
    do
        work=/gss_gpfs_scratch/yildiz.i/
        sbatch execute_fac.bash $lambda $lr
    done
done



