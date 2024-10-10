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

GPU=0
DATA=breakfast
SPLIT=1
MEM_SIZE=60
MODE=segm
TRAIN_SETUP=T10_disjoint
TASK=9
TAS_LR=5e-4
TCA_LR=1e-3
CASE=none
TEST_TYPE=best
TEST_ACTION=test_all

CUDA_VISIBLE_DEVICES=$GPU python3 main.py --action $TEST_ACTION \
 --dataset $DATA --split $SPLIT --memory_size $MEM_SIZE \
 --mode $MODE --train_setup $TRAIN_SETUP \
 --task $TASK --tas_lr $TAS_LR --tca_lr $TCA_LR \
 --case $CASE --test_type $TEST_TYPE

