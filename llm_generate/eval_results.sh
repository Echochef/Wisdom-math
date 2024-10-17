#!/bin/bash

# 使用示例路径，替换为你的实际路径
foler_path="root_path"
data_name="math_SFT_seek_answer_stage1_stage2_stage3_deepseek_test"
input_path="${foler_path}/${data_name}.jsonl"
output_path="${foler_path}/${data_name}_benchresult.jsonl"

dataset="math"  # 替换为实际的 dataset 名称
# dataset="college"  # 替换为实际的 dataset 名称

# 运行Python脚本
python3 eval_tools/multithreading_compare_source.py --input_path "$input_path" --output_path "$output_path" --dataset "$dataset"
