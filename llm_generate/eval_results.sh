#!/bin/bash

# 使用示例路径，替换为你的实际路径
foler_path="../data/eval_output"
data_name="deepseek_math"
input_path="${foler_path}/${data_name}.jsonl"
output_path="${foler_path}/${data_name}_benchresult.jsonl"
echo $input_path

## dataset options ['college','Olympiad',else dataset(same eval methods)]
#dataset="math"  # 替换为实际的 dataset 名称
dataset="college"  # 替换为实际的 dataset 名称

# 运行Python脚本
python3 eval_tools/multithreading_compare.py --input_path "$input_path" --output_path "$output_path" --dataset "$dataset"
