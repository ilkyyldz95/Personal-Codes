#!/bin/bash
for size in "35000" "29000" "20000" "11000" "6000"
do
    work=/gss_gpfs_scratch/yildiz.i/
    cd $work
    sbatch execute.bash $size
done