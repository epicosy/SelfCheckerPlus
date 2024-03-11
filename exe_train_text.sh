#!/bin/bash

datasets=("PD" "HP")
pd_models=("PD1" "PD2" "PD3" "PD4")
hp_models=("HP1" "HP2" "HP3" "HP4")
num_classes=2
num_layers=2

for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "PD" ]]; then
        models=("${pd_models[@]}")
    elif [[ "$dataset" == "HP" ]]; then
        models=("${hp_models[@]}")
    fi

    for model in "${models[@]}"; do
        echo "Running for dataset $dataset and model $model"
        python -u textual_kde.py --dataset $dataset --model $model --num_classes $num_classes -f train | tee "results/log_kde_train_${dataset}_${model}.txt"

        for ((i = 0; i < $num_classes; i++)); do
            python layer_selection_agree.py $dataset $model $num_classes $num_layers $((i)) &
        done

        wait

        for ((i = 0; i < $num_classes; i++)); do
            python layer_selection_condition.py $dataset $model $num_classes $num_layers $((i)) &
        done

        wait

        for ((i = 0; i < $num_classes; i++)); do
            python layer_selection_condition_neg.py $dataset $model $num_classes $num_layers $((i)) &
        done

        wait
    done
done