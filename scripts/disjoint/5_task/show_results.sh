# /bin/bash

train_setup=T5_disjoint
dataset=breakfast
memory_size=60
tas_lr=5e-4
tca_lr=1e-3
split_SIL=1
case=none

python3 integrate.py --train_setup $train_setup \
    --dataset $dataset \
    --memory_size $memory_size \
    --tas_lr $tas_lr \
    --tca_lr $tca_lr \
    --split_SIL $split_SIL \
    --case $case
