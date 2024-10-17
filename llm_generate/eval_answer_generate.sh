#!/bin/bash
#source ~/.zshrc
# conda activate vllm
python3 --version
#export NCCL_SOCKET_IFNAME=bond1
# Function to split JSONL file
split_jsonl() {
    input_file=$1
    num_splits=$2
    base_output_file=$3

    total_lines=$(wc -l < "$input_file")
    lines_per_file=$(( (total_lines + num_splits - 1) / num_splits ))

    awk -v lpf=$lines_per_file -v prefix=$base_output_file \
        'NR%lpf==1 {close(f); f=sprintf("%s.split.%02d.jsonl", prefix, int(NR/lpf))} {print > f}' "$input_file"
}
# test_name="tabmwp"
# test_name="aime24"
test_name="gsm8k"

# version="DPMath_instruct_7b"
# version="dartmath_llama3_8b"
version="v0"

# First inference step

model_path="/home/u2308283050/model/models--Wisdom-math--wisdom-dsmath-7b/snapshots/5075fe0e6382672d84876db4301a9e0ade9e4d9f"

input_json_dataset_path="../data/benchmarks/${test_name}_test.json"
# input_json_dataset_path="../data/vllm_hint/math_eval_gpt.jsonl"
#rm -r ./tmp/*
intermediate_save_path="./tmp/"
# final_output_path="../data/eval_output/deepseek_seed_answer_stage1_stage2_stage3_tabmwp_test.jsonl"
final_output_path="../data/eval_output/${test_name}_${version}_test.jsonl"
dataset="alpaca"
gpu_tp=1
load_dtype="bf16"
model_max_length=4096
use_vllm="--use_vllm"  # Uncomment this line if you want to use vLLM

num_splits=1
gpus=(0 1 2 3 4 5 6 7)
# gpus=0
mkdir -p "$intermediate_save_path"
mkdir -p "../data/eval_output/"
chmod -R 777  "$intermediate_save_path"

# Split the input JSONL file for the first inference step
split_jsonl "$input_json_dataset_path" $num_splits "$input_json_dataset_path"

# Log the split files
echo "Split files created for first inference step:"
ls "${input_json_dataset_path}.split."*.jsonl

# Run the first inference step on each split using a different GPU
for i in $(seq 0 $(($num_splits - 1))); do
    split_file="${input_json_dataset_path}.split.$(printf "%02d" $i).jsonl"
    # conda activate vllm
    output_file="${intermediate_save_path}tmp_part${i}.jsonl"
    CUDA_VISIBLE_DEVICES=${gpus[$i]} python3 generate/answer_generate.py \
        --model_path "$model_path" \
        --input_json_dataset_path "$split_file" \
        --save_path "$output_file" \
        --dataset "$dataset" \
        --gpu_tp $gpu_tp \
        --load_dtype $load_dtype \
        --model_max_length $model_max_length \
        $use_vllm \
        --print &
done

# Wait for all background processes to finish
wait

echo "First inference step completed on all GPUs."

# Merge the split files into one final output file for the first inference step
cat ${intermediate_save_path}tmp_part*.jsonl > "$final_output_path"

echo "Merged all split files into $final_output_path."

