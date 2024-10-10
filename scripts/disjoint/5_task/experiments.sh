# /bin/bash

# GPU=$1
# DATA=$2
# splitLIT=$3
# MEM_SIZE=$4
# MODE=$5
# TRAIN_SETUP=$6
# TASK=$7
# SEED=$8
# TAS_LR=$9
# TCA_LR=${10}
# splitLIT_SIL=${11}
# CASE=${12}

g=0
split=1
case=none

# bash ./task_run.sh $g breakfast $split 60 segm T5_disjoint 0 5e-4 1e-3 $case
bash ./task_run.sh $g breakfast $split 60 segm T5_disjoint 1 5e-4 1e-3 $case
bash ./task_run.sh $g breakfast $split 60 segm T5_disjoint 2 5e-4 1e-3 $case
bash ./task_run.sh $g breakfast $split 60 segm T5_disjoint 3 5e-4 1e-3 $case
bash ./task_run.sh $g breakfast $split 60 segm T5_disjoint 4 5e-4 1e-3 $case
