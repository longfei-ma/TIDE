#!/bin/bash

dataset=$1
device=$2
cpunum=$3
run=$4
llm_name=${5:-"Llama-3.2-1B"}
batch_size=${6:-200}


mkdir -p logs/train_node_tide


taskset -c $cpunum  nohup python train_tide_node.py --dataset_name $dataset  --llm_predictor $llm_name --gpu $device --run $run $num_neighbors --batch_size $batch_size > logs/train_node_tide/$dataset-TIDE-$llm_name.$run 2>&1 &
