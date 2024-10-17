import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from dataset import BatchDatasetLoader
from tqdm import tqdm
from collections import Counter
from transformers import GenerationConfig
import os
from util import read_jsonl, write_jsonl,prepare_tokenizer,vllm_hint_answer_generate



import pandas as pd
def main():
    # import ray  # Import Ray

    parser = argparse.ArgumentParser(description='Run model evaluation on JSONL dataset.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--input_json_dataset_path', type=str, required=True, help='Path to the input JSONL dataset.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output JSONL file.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--use_vllm', action='store_true', help='Use vLLM for inference.')
    parser.add_argument('--gpu_tp', type=int, default=1, help='Tensor parallelism size for vLLM.')
    parser.add_argument('--load_dtype', type=str, choices=['bf16', 'fp16'], default='fp16', help='Data type for loading the model.')
    parser.add_argument('--model_max_length', type=int, default=300, help='Max length for model generation.')
    parser.add_argument('--print', action='store_true', help='Print the intermediate results.')
    # parser.add_argument('--ray_address', type=str, default=None, help='Ray cluster address (e.g., "auto" or "ray://<head_node_ip>:<port>").')

    args = parser.parse_args()
    # if args.use_vllm:
    #     ray_port = os.getenv("RAY_PORT", "6379")  # 默认端口6379
    #     ray_address = ray_port = os.getenv("RAY_ADDRESS", "6379")

    #     # 初始化 Ray
    #     ray.init(address=ray_address, _temp_dir=f"/tmp/ray/ray_{ray_port}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left', trust_remote_code=True)
    tokenizer = prepare_tokenizer(tokenizer)

    if args.use_vllm:
        from vllm import LLM, SamplingParams
        model = LLM(
            model=args.model_path,
            tensor_parallel_size=args.gpu_tp,
            dtype=torch.bfloat16 if args.load_dtype == 'bf16' else torch.float16,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        # model.generation_config = GenerationConfig.from_pretrained(args.model_path)
        # model.generation_config.pad_token_id = model.generation_config.eos_token_id
        # print(1)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.load_dtype == 'bf16' else torch.float16,
            trust_remote_code=True
        )
        model.eval()
    # model=None
    print('Writing the output to new', args.save_path)
    file_handle = open(args.save_path, 'w')
    data_list = []
    loaded = read_jsonl(args.input_json_dataset_path)

    for d in loaded:
        tmp = {
            'question': d['question'],
            # 'category': d['category'],
            # 'hint': d['hint'],
            'gt': d['gt'],
            # 'instruction': d['instruction'],
        }
        data_list.append(tmp)
    # print(data_list[:2])
    result = []
    for data in tqdm(data_list):
        questions = data['question']
        answers = vllm_hint_answer_generate(questions,tokenizer, args, model)
        for question, answer in zip([questions], answers):
            example = {
                'question': question,
                # 'hint': data['hint'],
                # 'category': data['category'],
                'gt': data['gt'],
                'answer': answer,
                # 'instruction':data['instruction']
            }
            if args.print:
                print(example['answer'])
            # pd.DataFrame(example).to_json(args.save_path,force_ascii=False,lines=True,orient='records')
            # file_handle.write(json.dumps(example) + '\n')
            result.append(example)
    if result:
        pd.DataFrame(result).to_json(args.save_path,force_ascii=False,lines=True,orient='records')

    # file_handle.close()

if __name__ == "__main__":
    main()
