#!/bin/bash
dataset="PD"
model="PD1"
num_classes=2
num_layers=2

python -u textual_kde.py --dataset $dataset --model $model --num_classes $num_classes -f train | tee "log_kde_train.txt"

for((i=0;i<$num_classes;i++));
do
	python layer_selection_agree.py $dataset $model $num_classes $num_layers $((i)) &
done

wait


for((i=0;i<$num_classes;i++));
do
	python layer_selection_condition.py $dataset $model $num_classes $num_layers $((i)) &
done

wait

for((i=0;i<$num_classes;i++));
do
	python layer_selection_condition_neg.py $dataset $model $num_classes $num_layers $((i)) &
done

wait
