# /bin/bash

GPU=$1
DATA=$2
SPLIT=$3
MEM_SIZE=$4
MODE=$5
TRAIN_SETUP=$6
TASK=$7
SEED=$8
TAS_LR=$9
TCA_LR=${10}
CASE=${11}

echo 
echo "Train (Dataset: "$DATA" / Task: "$TASK" / Split: "$SPLIT" / Scenario: "$TRAIN_SETUP" / Mode: $MODE / Using GPU: "$GPU" / TAS_LR: $TAS_LR / TCA_LR: $TCA_LR)"
CUDA_VISIBLE_DEVICES=$GPU python3 main.py --action train \
 --dataset $DATA --split $SPLIT --memory_size $MEM_SIZE \
 --mode $MODE --train_setup $TRAIN_SETUP \
 --task $TASK --tas_lr $TAS_LR --tca_lr $TCA_LR \
 --seed $SEED --case $CASE
 
# echo 
# echo "Predict (Dataset: "$DATA" / Task: "$TASK" / TCN-Stage: 4 / SEED: "$SEED" / Using GPU: "$GPU" / TAS LR: $TAS_LR)"
# CUDA_VISIBLE_DEVICES=$GPU python3 main.py --action predict \
#  --dataset $DATA --split $SPLIT \
#  --mode $MODE --train_setup $TRAIN_SETUP \
#  --task $TASK --seed $SEED --tas_lr $TAS_LR --tca_lr $TCA_LR \
#  --split_SIL $SPLIT_SIL --case $CASE

# echo 
# echo "Run eval mode (Dataset: "$DATA" / Task: $TASK / TCN-Stage: 4 / SEED: "$SEED" / Using GPU: "$GPU")"
# CUDA_VISIBLE_DEVICES=$GPU python3 eval.py --dataset $DATA \
#     --task $TASK \
#     --seed $SEED \
#     --tas_lr $TAS_LR \
#     --case $CASE

# python3 integrate.py --dataset $DATA --stages $STAGE --seed $SEED --case $CASE


