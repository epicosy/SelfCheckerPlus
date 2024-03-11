#!/bin/bash

datasets=("PD" "HP")
pd_models=("PD1" "PD2" "PD3" "PD4")
hp_models=("HP1" "HP2" "HP3" "HP4")
num_classes=2

for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "PD" ]]; then
        models=("${pd_models[@]}")
    elif [[ "$dataset" == "HP" ]]; then
        models=("${hp_models[@]}")
    fi

    for model in "${models[@]}"; do
        echo "Running for dataset $dataset and model $model"
        python -u textual_kde.py --dataset $dataset --model $model --num_classes $num_classes --flag "test" | tee "results/log_kde_${dataset}_${model}.txt"
        python -u sc.py $dataset $model $num_classes | tee "results/log_sc_${dataset}_${model}.txt"
    done
done