#!/bin/bash
for kthFold in "0" "1" "2" "3" "4"
do
    work=/gss_gpfs_scratch/yildiz.i/
    cd $work
    sbatch execute_abs_fold.bash $kthFold
done
