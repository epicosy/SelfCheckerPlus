#!/bin/bash
dataset="PD"
model="PD1"
num_classes=2

python -u textual_kde.py --dataset $dataset --model $model --num_classes $num_classes --flag "test" | tee "log_kde.txt"

python -u sc.py $dataset $model $num_classes | tee "log_sc.txt"