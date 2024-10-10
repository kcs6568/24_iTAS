# /bin/bash

train_setup=T10_disjoint
dataset=breakfast
memory_size=60
tas_lr=5e-4
tca_lr=1e-3
case=optAdamW

python3 integrate.py --train_setup $train_setup \
    --dataset $dataset \
    --memory_size $memory_size \
    --seed default \
    --tas_lr $tas_lr \
    --tca_lr $tca_lr \
    --case $case
