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
seed=7
case=none
tas_lr=5e-4
tca_lr=1e-3

bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 0 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 1 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 2 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 3 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 4 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 5 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 6 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 7 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 8 $seed $tas_lr $tca_lr $case
# bash ./task_run.sh $g breakfast $split 60 segm T10_disjoint 9 $seed $tas_lr $tca_lr $case
