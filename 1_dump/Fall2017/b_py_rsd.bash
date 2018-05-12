#!/bin/bash
for kthFold in "0" "1" "2" "3" "4"
#for kthFold in "5"
do
    for no_im in "80" "75" "70" "65" "60" "55" "50" "45" "40"
    #for no_im in "100" "90" "80" "70" "60" "50" "40"
    do
        work=/gss_gpfs_scratch/yildiz.i/
        cd $work
        sbatch execute_rsd.bash $kthFold $no_im
    done
done
