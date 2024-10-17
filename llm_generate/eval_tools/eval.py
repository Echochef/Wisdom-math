from util import (
    norm_str2date_time,
    latex2sympy_fix,
    latex2sympy_interval,
    norm_ans_str,
    latex2matrix,
    is_num_eq,
    is_sym_eq,
    parse,
)
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from mammoth import compare_answer_with_groundtruth
from constant import *
import re
import math
def parse_latex(expr):
    # 处理分数表达式 \frac{a}{b}
    expr = re.sub(r'\\frac\{([^\}]*)\}\{([^\}]*)\}', r'(\1)/(\2)', expr)
    # 将 e 转换为 sympy 自然常数
    expr = expr.replace('e', 'E')
    expr = expr.replace(r'\pi', '3.1415926')

    return expr
def pre_norm_str_compare(pred,ref):
    # datetime
    pred_datetime = norm_str2date_time(str(pred))
    ref_datetime = norm_str2date_time(str(ref))
    if (
            pred_datetime is not None
            and ref_datetime is not None
            and pred_datetime == ref_datetime
    ):
        return True  # Stricter than ratio
    return False
def lower_compare_equal(pred,ref):
    # 1. literally equal
    lower_pred = pred.lower()
    lower_ref = ref.lower()
    if lower_pred == lower_ref:
        return True
def evaluate_string(s):
    try:
        # 尝试直接转换为浮点数
        # print(s)
        s = float(s)
        return s
    except ValueError:
        # 如果转换失败，尝试解析 LaTeX 表达式
        try:
            # 解析 LaTeX 表达式
            parsed_expr = parse_latex(s)
            # 使用 sympy 解析字符串表达式
            expr = sp.sympify(parsed_expr, locals={"E": sp.E})
            return float(expr.evalf())
        except Exception as e:
            # print(e)
            # 捕获所有解析错误并返回 None
            return None

def latex_expr_equal(expr1, expr2, tolerance=1e-2):
    try:
        # 将 expr2 列表中的每个元素处理为符号表达式并转换为小数
        # for e in expr2:print(e)
        # print(expr2,expr2.replace(r'\sqrt{', 'sqrt(').replace('}', ')'))
        expr2_processed = [sp.sympify(expr2.replace(r'\sqrt{', 'sqrt(').replace('}', ')')).evalf()]

        # 将 expr1 列表中的每个元素转换为 SymPy 表达式并确保转换为小数
        expr1_processed = [sp.sympify(expr1)]
        # print(expr1_processed,expr2_processed)
        # 检查两个表达式是否在允许的误差范围内相等
        return all(abs(a - b) < tolerance for a, b in zip(expr1_processed, expr2_processed))
    except Exception as e:
        # print(f"Error parsing expressions: {e}")
        return False
def evaluate_tuple(expr1,expr2):
    # 定义两个 LaTeX 表达式

    # 拆分元组表达式，并逐一比较
    expr1_elements = expr1.strip('()').split(', ')
    expr2_elements = expr2.strip('()').split(', ')

    # 检查每个元素是否相等
    are_all_equal = all(latex_expr_equal(e1, e2) for e1, e2 in zip(expr1_elements, expr2_elements))

    return are_all_equal
def are_strings_equal(s1, s2):
    # 计算两个字符串的值
    value1 = evaluate_string(s1)
    value2 = evaluate_string(s2)

    # 如果任一值为 None，则返回 False
    if value1 is None or value2 is None:
        return False

    # 使用 math.isclose 进行比较
    if 'e' in s1 or 'e' in s2 or 'pi' in s1 or 'pi' in s2:
        # print(value1, value2,s1,s2)
        result = math.isclose(value1, value2, rel_tol=1e-8)
        # if result:
        #     print(value1, value2,s1,s2)
        return result
    else:
        result = math.isclose(value1, value2, rel_tol=1e-8)
        # if result == True:
        #     print(value2,value1)
        return result
def dart_equal(pred_str,ref_str,ref_num):
    pred_parse_errs = []
    ref_parse_errs = []

    # 2. Numerically equal
    # no `norm_float_str` for possible mistakes like "123,456"(123 and 456) -> "123456"
    # print(pred_str,ref_str)
    pred_num = parse(float, pred_str, pred_parse_errs)
    if ref_num is None:
        ref_num = parse(float, ref_str, ref_parse_errs)
    num_eq = is_num_eq(ref_num, pred_num)
    if num_eq is not None:
        return num_eq
    # 3. Symbolically equal (w/ SymPy and antlr4)
    # Return `True` if the two expressions can be interpreted as equal in **any** unified form.
    # NOTE: possible ambiguity 1,234 -> (1,234) / 1234 ?

    # 3.1 Python object
    # NOTE: parse_expr("1,234") == (1, 234)
    pred_obj = parse(parse_expr, pred_str, pred_parse_errs)
    ref_obj = parse(parse_expr, ref_str, ref_parse_errs)
    # print(pred_obj, ref_obj, symbol_equal(pred_obj, ref_obj))  # debug
    if pred_obj is not None and ref_obj is not None and pred_obj == ref_obj:
        return True
    # 3.2 SymPy object
    # ImportError: LaTeX parsing requires the antlr4 Python package, provided by pip (antlr4-python3-runtime) or conda (antlr-python-runtime), version 4.11
    pred_spobj = parse(latex2sympy_interval, pred_str, pred_parse_errs)
    ref_spobj = parse(latex2sympy_interval, ref_str, ref_parse_errs)
    # print(pred_spobj, ref_spobj, symbol_equal(pred_spobj, ref_spobj))  # debug
    if (
            pred_spobj is not None
            and ref_spobj is not None
            and is_sym_eq(pred_spobj, ref_spobj)
    ):
        return True
    pred_spobj = parse(latex2matrix, pred_str, pred_parse_errs)
    ref_spobj = parse(latex2matrix, ref_str, ref_parse_errs)
    # print(pred_spobj, ref_spobj, symbol_equal(pred_spobj, ref_spobj))  # debug
    if (
            pred_spobj is not None
            and ref_spobj is not None
            and is_sym_eq(pred_spobj, ref_spobj)
    ):
        return True

    # WARNING: parse_latex("a,b") -> a but parse_latex("1,234") -> 1234, `latex2sympy_fix` fixed the former by raising a `LaTeXParsingError``
    pred_spobj = parse(latex2sympy_fix, pred_str, pred_parse_errs)
    ref_spobj = parse(latex2sympy_fix, ref_str, ref_parse_errs)
    # print(pred_spobj, ref_spobj, symbol_equal(pred_spobj, ref_spobj))  # debug
    if (
            pred_spobj is not None
            and ref_spobj is not None
            and is_sym_eq(pred_spobj, ref_spobj)
    ):
        return True

    if (
            pred_spobj is not None
            and ref_obj is not None
            and is_sym_eq(pred_spobj, ref_obj)
    ):
        return True

    if (
            pred_obj is not None
            and ref_spobj is not None
            and is_sym_eq(pred_obj, ref_spobj)
    ):
        return True

    n_checks = 5
    expr_parse_errs = {}
    if len(pred_parse_errs) == n_checks:
        expr_parse_errs["pred"] = pred_parse_errs
    if len(ref_parse_errs) == n_checks:
        expr_parse_errs["ref"] = ref_parse_errs

    # print(expr_parse_errs)
    if len(expr_parse_errs) > 0:
        # print(expr_parse_errs)
        return False
    else:
        return False

def fix_bug_eqal(pred,ref,ref_num):
    if are_strings_equal(ref, pred):
        # print(pred,ref,'math')
        return True
    if evaluate_tuple(ref, pred):
        # print(pred,ref,'tuple')
        return True
    return False
def sort_and_compare(str1, str2):
    try:
        # 将字符串拆分为数字列表，并转换为整数类型
        list1 = sorted(map(int, str1.split(',')))
        list2 = sorted(map(int, str2.split(',')))

        # 比较排序后的列表
        return list1 == list2
    except (ValueError, TypeError):
        # 如果发生错误（例如字符串中没有逗号，无法转换为整数等），返回 False
        return False
def eval(pred,ref):
    if isinstance(ref, list) and len(ref) == 2:
        ref, ref_num = ref
    else:
        ref_num = None

    if ref is None:
        return None

    if pred is None:
        return False
    # 0. Normalize
    pred_str = norm_ans_str(pred)
    ref_str = norm_ans_str(ref)
    if len(pred_str) == 0:
        return False
    if lower_compare_equal(pred_str,ref_str):return True


    if dart_equal(pred_str,ref_str,ref_num):return True
    if fix_bug_eqal(pred,ref,ref_num):return True
    if compare_answer_with_groundtruth(pred,ref,ref_num):return True
    if sort_and_compare(pred,ref):return True
    # if are_strings_equal(ref, pred):
    #     # print(pred_str,ref_str)
    #     return True
    # if evaluate_tuple(ref,pred):
    #     return True
    return False