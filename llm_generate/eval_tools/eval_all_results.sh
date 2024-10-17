#!/bin/bash

# 使用示例路径，替换为你的实际路径
foler_path="/home/u2308283050/paper_code/data/eval_output"
data_name="gsm8k_v0_test"
input_path="${foler_path}/${data_name}.jsonl"
output_path="${foler_path}/${data_name}_benchresult.jsonl"
# input_path="/mnt/wx_feature/home/trenousqiu/codes/paper_version/paper_code/data/eval_data/pot_same_stand_gpt_rewrite_output.jsonl"
# output_path="/mnt/wx_feature/home/trenousqiu/codes/paper_version/paper_code/data/eval_output/pot_same_stand_gpt_rewrite_output_same.jsonl"
dataset="math"  # 替换为实际的 dataset 名称
# dataset="college"  # 替换为实际的 dataset 名称

# 运行Python脚本
python3 multithreading_compare_source.py --input_path "$input_path" --output_path "$output_path" --dataset "$dataset"
