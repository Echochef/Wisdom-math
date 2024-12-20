#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import re
import os
import multiprocessing as mp
from tqdm import tqdm
import threading
from queue import Queue, Empty
import argparse
from eval import eval
from util import extract_math_answer,extract_ans
from functools import partial


class ProgressCounter:
    def __init__(self, total):
        self.counter = mp.Value('i', 0)
        self.total = total

    def increment(self):
        with self.counter.get_lock():
            self.counter.value += 1

    def get_value(self):
        return self.counter.value


def init_worker(counter):
    global progress_counter
    progress_counter = counter


def extract_special_text(text):
    match = re.search(r"\$(.*?)\$", text)
    return match.group(1) if match else text


def extract_pair_numbers(text):
    pattern = r'\(([^)]+)\)|(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    results = [f"({match[0].replace(' ', '')})" if match[0] else match[1] for match in matches]
    return ",".join(results)


def extract_boxed(resp: str) -> str:
    ans = resp.split("oxed")[-1]
    if ans and ans[0] == "{":
        stack, a = 1, ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
            a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def get_paire_output(pred):
    pred = extract_boxed(pred)
    return extract_pair_numbers(pred)


def extract_math_expression_mathrm(text):
    pattern = r'(\d+(?:,\d+)*\s*\\pi|\d+\s*\\sqrt{\d+}|\d+\s*/\s*\(\d+\\s*\\pi\)|\d+\s*/\s*\d+|\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\\?mathrm'
    matches = re.findall(pattern, text)
    return matches[0] if len(matches) == 1 else tuple(matches) if matches else text


def clean_ref(ref):
    ref = extract_special_text(ref) if "and $" not in ref else ref
    if '=' in ref:
        ref = ref.split('=')[-1].split('=0')[0]
    return extract_math_expression_mathrm(ref)


def evaluate_with_timeout(data, timeout=0.5, dataset='None'):
    result_queue = Queue()

    def target(data, queue):
        try:
            labels = [data['gt']] if isinstance(data['gt'], str) else data['gt']

            pred = extract_math_answer(data['answer'], None)

            if dataset == 'college' or dataset == 'Olympiad':
                easy_pred = get_paire_output(data['answer'])
                flags = [
                    eval(pred, *labels),
                    eval(pred, clean_ref(*labels)),
                    eval(easy_pred, clean_ref(*labels))
                ]
            else:
                if type(labels) == int:labels = str(labels)
                if type(labels) == list:labels = labels[0]
                flags = [
                    eval(pred, labels),
                ]
            data['compare_flag'] = any(flags)
        except Exception as e:
            print(e)
            data['compare_flag'] = False
        queue.put(data)

    thread = threading.Thread(target=target, args=(data, result_queue))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        thread.join(0)
        data['compare_flag'] = False
    else:
        try:
            data = result_queue.get_nowait()
        except Empty:
            data['compare_flag'] = False

    progress_counter.increment()
    return data


def parallel_process(dataframe, num_processes, dataset):
    progress_counter = ProgressCounter(len(dataframe))

    # Use partial to pass the dataset argument to the evaluate_with_timeout function
    evaluate_func = partial(evaluate_with_timeout, dataset=dataset)

    with mp.Pool(num_processes, initializer=init_worker, initargs=(progress_counter,)) as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_func, [row for _, row in dataframe.iterrows()]), total=len(dataframe)))

    return pd.DataFrame(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate JSONL files.')
    parser.add_argument('--input_path',
                        default='/Users/echoch/Downloads/data/revolve/eval/llama3_8b_math_eval_answer.jsonl', type=str,
                        help='Path to the input JSONL file.')
    parser.add_argument('--output_path', default='../../data/answer/math_gsm8k_train_input_wrong_answer1.jsonl',
                        type=str, help='Path to the output JSONL file.')
    parser.add_argument('--dataset', type=str, default='math', help='Dataset name.')
    parser.add_argument('--print', default=True, help='Print the intermediate results.')

    args = parser.parse_args()

    df = pd.read_json(args.input_path, lines=True)
    print(df.shape, args.input_path)
    dataset = args.dataset
    if args.print:
        print(f"Dataset Name: {dataset}")
        print(df.shape)
        print(df.sample(5))

    result_df = parallel_process(df, num_processes=20, dataset=dataset)
    if args.print:
        # print(result_df.sample(20))
        print(result_df['compare_flag'].value_counts())
        print('final accuracy:',len(result_df[result_df['compare_flag']==True])/len(result_df))
        print(result_df.shape)
    print(args.output_path)
    result_df.to_json(args.output_path, lines=True, orient='records', force_ascii=False)