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
split=1

# for ((t=0 ; t <= 4 ; t++));
# do
#     bash ./task_run.sh $g breakfast $split 60 gen T5_disjoint $t 5e-4 1e-3 none
#     bash ./task_run.sh $g breakfast $split 60 sample T5_disjoint $t 5e-4 1e-3 none
# done

bash ./task_run.sh 0 breakfast 1 60 gen T5_disjoint 1 5e-4 1e-3 none
bash ./task_run.sh 0 breakfast 1 60 sample T5_disjoint 1 5e-4 1e-3 none

# bash ./task_run.sh 0 breakfast 1 60 gen T5_disjoint 2 5e-4 1e-3 none --> *
# bash ./task_run.sh 1 breakfast 2 60 gen T5_disjoint 2 5e-4 1e-3 none --> *
# bash ./task_run.sh 2 breakfast 3 60 gen T5_disjoint 2 5e-4 1e-3 none --> *
# bash ./task_run.sh 3 breakfast 4 60 gen T5_disjoint 2 5e-4 1e-3 none --> *
# bash ./task_run.sh 0 breakfast 1 60 gen T5_disjoint 3 5e-4 1e-3 none --> *
# bash ./task_run.sh 1 breakfast 2 60 gen T5_disjoint 3 5e-4 1e-3 none --> *
# bash ./task_run.sh 2 breakfast 3 60 gen T5_disjoint 3 5e-4 1e-3 none --> *
# bash ./task_run.sh 3 breakfast 4 60 gen T5_disjoint 3 5e-4 1e-3 none --> *
# bash ./task_run.sh 0 breakfast 1 60 gen T5_disjoint 4 5e-4 1e-3 none --> *
# bash ./task_run.sh 1 breakfast 2 60 gen T5_disjoint 4 5e-4 1e-3 none --> *
# bash ./task_run.sh 2 breakfast 3 60 gen T5_disjoint 4 5e-4 1e-3 none --> *
# bash ./task_run.sh 3 breakfast 4 60 gen T5_disjoint 4 5e-4 1e-3 none --> *



# bash ./task_run.sh 0 breakfast 1 60 sample T5_disjoint 2 5e-4 1e-3 none
# bash ./task_run.sh 1 breakfast 2 60 sample T5_disjoint 2 5e-4 1e-3 none
# bash ./task_run.sh 2 breakfast 3 60 sample T5_disjoint 2 5e-4 1e-3 none
# bash ./task_run.sh 3 breakfast 4 60 sample T5_disjoint 2 5e-4 1e-3 none

# bash ./task_run.sh 0 breakfast 1 60 sample T5_disjoint 3 5e-4 1e-3 none
# bash ./task_run.sh 1 breakfast 2 60 sample T5_disjoint 3 5e-4 1e-3 none
# bash ./task_run.sh 2 breakfast 3 60 sample T5_disjoint 3 5e-4 1e-3 none
# bash ./task_run.sh 3 breakfast 4 60 sample T5_disjoint 3 5e-4 1e-3 none

# bash ./task_run.sh 0 breakfast 1 60 sample T5_disjoint 4 5e-4 1e-3 none
# bash ./task_run.sh 1 breakfast 2 60 sample T5_disjoint 4 5e-4 1e-3 none
# bash ./task_run.sh 2 breakfast 3 60 sample T5_disjoint 4 5e-4 1e-3 none
# bash ./task_run.sh 3 breakfast 4 60 sample T5_disjoint 4 5e-4 1e-3 none
