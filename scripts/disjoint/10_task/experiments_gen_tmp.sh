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
# SPLIT_SIL=${11}
# CASE=${12}

g=0
sd=default

bash ./task_run.sh $g breakfast 1 60 sample T10_disjoint 3 $sd 5e-4 1e-3 1 fix_order &
bash ./task_run.sh $g breakfast 2 60 sample T10_disjoint 3 $sd 5e-4 1e-3 1 fix_order &
bash ./task_run.sh $g breakfast 3 60 sample T10_disjoint 3 $sd 5e-4 1e-3 1 fix_order &
bash ./task_run.sh $g breakfast 4 60 sample T10_disjoint 3 $sd 5e-4 1e-3 1 fix_order
