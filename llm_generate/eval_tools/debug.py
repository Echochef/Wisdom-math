#!/usr/bin/env python
# coding: utf-8

import re
from eval import eval


# print(math_evaluator.eq('5.333','\\frac{16}{3}'))
c = '\frac{16}{3}'
def extract_numbers(text):
    # 正则表达式匹配数字，确保数字后面跟着单位
    pattern = r'(\d+\.?\d*)\s*\\?mathrm'
    pattern1 = r'(\d{1,3}(?:,\d{3})*(?:\.\d*)?)\s*\\?mathrm'
    # 匹配普通小数或整数，如"1.42"或"30"
    pattern2 = r'(\d+\.?\d*)\s*\\?mathrm'
    pattern = f"{pattern}｜{pattern1}|{pattern2}"
    # 使用 findall 方法提取所有匹配的数字
    matches = re.findall(pattern, text)
    matches = [m[0] for m in matches if m[0]]

    # 如果没有匹配到任何数字，返回原始字符串
    if not matches:
        return text
    print(matches)
    # 将提取到的数字转换为浮点数
    numbers = list(map(lambda x: float(x.replace(',', '')), matches))

    # 根据提取到的数字个数返回不同格式的结果
    if len(numbers) == 1:
        return numbers[0]  # 如果只有一个数字，返回浮点数
    else:
        return tuple(numbers)  # 如果有多个数字，返回元组
def extract_special_text(text):
    match = re.search(r"\$(.*?)\$", text)
    if match:
        return match.group(1)
    else:
        return text
def clean_ref(ref):
    ref = extract_special_text(ref)
    # print(ref)
    if len(ref.split('='))==2:
        print(2)
        ref = ref.split('=')[-1]

    if '=0' in ref:
        ref = ref.split('=0')[0]
    ref = extract_numbers(ref)
    return ref
print(eval(clean_ref("0.0319753"),('3.46621207')))
# print(eval('$\log _{36}(36)=1$',"1"))
