#!/bin/bash

types=("stand" "chair" "bed" "all")
frames=(1 5 10 20)
mkdir -p results

generate_commands() {
    for type_fall in "${types[@]}"; do
        for frame in "${frames[@]}"; do
            
            list="lists/${type_fall}_${frame}.csv"
            all="frames.csv"

            # geom
            echo "python3 geom.py $list $all > results/${type_fall}_${frame}_geom.txt 2>&1"

            # models
            for model in knn svm rf mlp xgb; do
                echo "python3 models.py $model $list $all > results/${type_fall}_${frame}_${model}.txt 2>&1"
            done
        done
    done
}

# Gera os comandos, envia para o xargs rodar 10 em paralelo
generate_commands | xargs -I CMD -P 10 bash -c CMD

