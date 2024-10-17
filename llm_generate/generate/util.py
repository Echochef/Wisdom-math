from transformers import HfArgumentParser, TrainingArguments
import torch.nn as nn
import re
import torch

from transformers import GenerationConfig
from collections import Counter

def get_alpaca_step_answer_prompt(qas: list):

    tmp = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    prefix = tmp + '### Instruction:\n' + '{question}' + '\n\n' + '### Response:'

    return prefix
def prepare_tokenizer(tokenizer):
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"

    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return tokenizer

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            print("Conversion to floating number fails: {}".format(n))
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n

def vllm_hint_answer_generate(question,tokenizer,args,model):
    # if type(question) == 'str':
    question = [question]
    prompt_prefixs = get_alpaca_step_answer_prompt([])
    prompt_prefixs = [prompt_prefixs]
    # input_strs = [p.format(question=q) for p, q in zip(prompt_prefixs, question)]
    input_strs = [p.replace("{question}",q) for p, q in zip(prompt_prefixs, question)]

    # input_strs = [
    #     {"role":"user","content":input_strs[0]}
    # ]
    # input_strs = [
    # # {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "user", "content": input_strs[0]}
    # ]
    # input_strs = tokenizer.apply_chat_template(input_strs,tokenize=False, add_generation_prompt=True)
    # input_strs = [
    #     {"role":"user","content":question}
    # ]
    from vllm import SamplingParams
    # stop_tokens = ["USER:", "ASSISTANT:", "### Instruction:", "Response:", "<start_of_turn>", "[INST]", "<|eot_id|>"]
    # stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    # sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=args.model_max_length, stop=stop_tokens)
    outputs = model.generate(input_strs, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    return outputs
def read_jsonl(path):
    import json
    json_list = []
    with open(path,'r',encoding='UTF-8') as file:
        for line in file:
            data = json.loads(line)
            json_list.append(data)
    return json_list
def write_jsonl(path,data):
    import json
    # 将列表写入 JSON Lines 文件
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
