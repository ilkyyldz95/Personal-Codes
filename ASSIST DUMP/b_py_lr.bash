#!/bin/bash
for kthFold in "5e-08" "1e-08" "5e-09" "1e-09"
do
    work=/gss_gpfs_scratch/yildiz.i/
    cd $work
    sbatch execute_lr.bash $kthFold
done
