import pandas as pd
from sympy import sympify
import json
import argparse
import re
from tqdm import tqdm
# from util import compare_answer_with_groundtruth, answer_clean, delete_extra_zero
from util import extract_math_answer
from eval import eval
def get_seperation_trigger(dataset: str):
    triggers = ['The answer is:', 'The answer is', 'the answer is']
    if dataset == 'gsm8k':
        triggers.append('####')
    return triggers

def extract_and_evaluate_frac(frac_str):
    """
    Extracts numerator and denominator from LaTeX \\frac expression and evaluates it using sympy.
    """
    pattern = r'\\frac{([^{}]*)}{([^{}]*)}'
    match = re.search(pattern, frac_str)
    if match:
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        expr = sympify(f'({numerator})/({denominator})')
        return float(expr.evalf())
    else:
        raise ValueError("Input string is not a valid LaTeX \\frac expression")

def get_latex_pred_eval(pred, label):
    try:
        if 'frac' in pred:
            pred = extract_and_evaluate_frac(pred)
        else:
            pred = float(pred)
        if 'frac' in label:
            label = extract_and_evaluate_frac(label)
        else:
            label = float(label)
    except Exception as e:
        print('exception ', pred, label)
        return False
    return pred == label

def main():
    parser = argparse.ArgumentParser(description='Evaluate JSONL files.')
    parser.add_argument('--eval_path', default='/Users/echoch/Downloads/chrome/colledge_math_stage2_stage3_test.jsonl',type=str, help='Path to the input JSONL file.')
    parser.add_argument('--eval_path_output', default='../../../data/eval_output/deepseek_stage2_stage3_Bench_compare.jsonl',type=str, help='Path to the output JSONL file.')
    parser.add_argument('--dataset', type=str, default='college', help='Dataset name.')
    parser.add_argument('--print', default=False, help='Print the intermediate results.')

    args = parser.parse_args()

    df = pd.read_json(args.eval_path, lines=True)
    correct = 0
    wrong = 0
    file_handle = open(args.eval_path_output, 'w')

    for index, data in tqdm(df.iterrows()):
        if index < 6:
            continue
        flag = 0
        question = data['question']
        pred = data['answer']
        category = args.dataset
        if category == 'gsm8k':
            labels_unresolved = data['gt']
            labels = delete_extra_zero(labels_unresolved.split("#### ")[-1].replace(",", ""))
            if labels is None:
                continue
            pred_clean = answer_clean(category, get_seperation_trigger(args.dataset), pred)
            labels = answer_clean(category, get_seperation_trigger(args.dataset), labels)
            if isinstance(labels, str):
                labels = [labels]
            if eval(pred_clean, *labels):
                correct += 1
                flag = 1
            elif get_latex_pred_eval(pred_clean, *labels):
                correct += 1
                flag = 1
            else:
                wrong += 1
        elif category == 'math':
            labels = data['gt']
            # labels[0] = labels[0].strip('$')
            # labels = delete_extra_zero(labels_unresolved.split("#### ")[-1].replace(",", ""))
            if labels is None:
                continue
            pred_clean = extract_math_answer(pred,None)
            # pred_clean = pred
            # labels = answer_clean(category, get_seperation_trigger(args.dataset), labels)
            if isinstance(labels, str):
                labels = [labels]
            print(pred_clean,labels)
            if eval(pred_clean, labels):
                correct += 1
                flag = 1
            # elif get_latex_pred_eval(pred_clean, *labels):
            #     correct += 1
            #     flag = 1
            else:
                wrong += 1
        elif category == 'college':
            if index !=1614:
                continue
            labels = data['gt']
            # if inde
            # labels[0] = labels[0].strip('$')
            # labels = delete_extra_zero(labels_unresolved.split("#### ")[-1].replace(",", ""))
            if labels is None:
                continue
            pred_clean = extract_math_answer(pred,None)
            # pred_clean = pred
            # labels = answer_clean(category, get_seperation_trigger(args.dataset), labels)
            if isinstance(labels, str):
                labels = [labels]
            # print(labels)
            # if pred_clean !=r'\frac{\sin^4x}{4}+C':
            #     continue
            if eval(pred_clean, *labels):
                correct += 1
                flag = 1
            # elif get_latex_pred_eval(pred_clean, *labels):
            #     correct += 1
            #     flag = 1
            else:
                wrong += 1
        if args.print:
            print(pred_clean, '#', labels, '#', correct / (correct + wrong))

        example = {
            'question': question,
            'gt': labels,
            'pred_clean': pred_clean,
            'pred': pred,
            'category': data['category'],
            'correct': flag,
            # 'hint':data['hint']
        }

        file_handle.write(json.dumps(example) + '\n')

    print('final accuracy: ', correct / (correct + wrong))
    file_handle.close()

if __name__ == "__main__":
    main()
