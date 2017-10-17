#!/bin/bash
for alpha in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
do
    for lamda in "1e-6" "1e-5" "1e-4" "1e-3" "1e-2" "1e-1" "2e-1" "3e-1" "4e-1" "5e-1" "6e-1" "7e-1" "8e-1" "9e-1" "1" "3" "5" "7" "9" "10" "100" "1000" "1e4"
    do
        work=/home/tian.pe/Python/GridSearch/S1/
        cd $work
        sbatch execute.bash $alpha $lamda
    done
done
