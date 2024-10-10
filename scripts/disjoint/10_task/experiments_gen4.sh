# /bin/bash

# GPU=$1
# DATA=$2
# SPLIT=$3
# MEM_SIZE=$4
# MODE=$5
# TRAIN_SETUP=$6
# TASK=$7
# SEED=$8
# TAS_LR=$9
# TCA_LR=${10}
# CASE=${11}

g=3
split=4
seed=7

for ((t=0 ; t <= 8 ; t++));
do
    bash ./task_run.sh $g breakfast $split 60 gen T10_disjoint $t $seed 5e-4 1e-3 test
    bash ./task_run.sh $g breakfast $split 60 sample T10_disjoint $t $seed 5e-4 1e-3 test
done