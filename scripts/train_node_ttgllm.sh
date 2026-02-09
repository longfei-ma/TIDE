#!/bin/bash

dataset=$1
model_name=$2
llm_name=$3
device=$4
cpunum=$5
num_neighbors=$6
run=$7
patch_size=${8:-1}


mkdir -p logs/train_node_ttgllm

taskset -c $cpunum  nohup python train_node_ttgllm.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --run $run --num_neighbors $num_neighbors --max_input_sequence_length $num_neighbors --patch_size $patch_size > logs/train_node_ttgllm/$dataset-$model_name.$run 2>&1 &
