#!/bin/bash

dataset=$1
model_name=$2
llm_name=$3
device=$4
cpunum=$5
num_neighbors=$6
run=$7
lr=${8:-0.0001}
patch_size=${9:-1}


mkdir -p logs/train_node_ttgllm

taskset -c $cpunum  nohup python train_node_ttgllm_u.py --dataset_name $dataset  --model_name $model_name --llm_name $llm_name --gpu $device --run $run --num_neighbors $num_neighbors --patience 10 --learning_rate $lr > logs/train_node_ttgllm/$dataset-$model_name.u$run-$lr 2>&1 &
