
import re
import warnings
from datetime import datetime
from math import isclose
from typing import Any, Callable
import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.exceptions import SymPyDeprecationWarning
from constant import *
from sympy import (
    E,
    FiniteSet,
    I,
    Intersection,
    Interval,
    Matrix,
    N,
    Union,
    pi,
    simplify,
    sqrt,
)

include_percentage = True
rel_tol: float = 1e-6
abs_tol: float = 1e-6
percent_rel_tol: float = 1e-6
def extract_boxed(resp: str) -> str:
    ans = resp.split("oxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def extract_explicit_ans( resp_str: str) -> str:
    resp_str = clean_trailing(resp_str)
    # might be answer only
    if "herefore" in resp_str:
        resp_str = resp_str.split("herefore")[-1].strip()
    if GSM8K_ANS_PREFIX in resp_str:
        resp_str = resp_str.split(GSM8K_ANS_PREFIX)[-1].strip()
    if PRM800K_ANS_PRRFIX in resp_str:
        resp_str = resp_str.split(PRM800K_ANS_PRRFIX)[-1].strip()

    if "oxed{" in resp_str:
        resp = extract_boxed(resp_str)
    else:
        resp = resp_str

        # should be answer only
        if "is the ans" in resp:
            resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[
                -1
            ].strip()
        elif "is our ans" in resp:
            resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[
                -1
            ].strip()
        elif "answer is" in resp:
            resp = resp.split("answer is")[-1].strip()
        elif "answer:" in resp:
            resp = resp.split("answer:")[-1].strip()
        elif "answer :" in resp:
            resp = resp.split("answer :")[-1].strip()
        elif "statement" in resp:
            bool_resp = norm_str2bool(resp.split("is ")[-1].strip())
            if bool_resp is not None:
                return str(bool_resp)
        else:
            return None

        if resp.startswith("$") and resp.endswith("$"):
            resp = resp[1:-1]

    return resp
def extract_ans(resp_str: str) -> str:
    """Extract answer segment from complete `resp`."""

    resp = extract_explicit_ans(resp_str)
    if resp is None:  # use the last number
        pattern = r"-?\d*\.?\d+"
        resp = re.findall(pattern, resp_str.replace(",", ""))
        if len(resp) >= 1:
            resp = resp[-1]
        else:
            resp = ""

    return resp
def norm_str2date_time( string: str):
    """Normalize date or time string to a standard and precise format."""

    for fmt in DATETIME_FMTS:
        try:
            dt = datetime.strptime(string, fmt)
            has_time, has_date = ":" in string, "/" in string or "-" in string
            if has_date and has_time:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            elif has_date:
                return dt.strftime("%Y-%m-%d")
            elif has_time:
                return dt.strftime("%H:%M:%S")
            else:
                pass
        except ValueError:
            continue
    return None
def norm_str2weekday(s: str) -> str | None:
    """Converts a string representation of a weekday to its normalized form. Returns `None` if the input is not a valid weekday"""
    s = str(s).lower().strip()
    if " " in s:  # not a word
        return None

    for i_day in range(NDAYS_PER_WEEK):
        if s.startswith(WEEKDAY_ABBRS[i_day]):
            return WEEKDAY_FULLS[i_day].capitalize()
    return None
def norm_str2bool(s: str) -> bool | None:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None
def is_set(s: str):
    return (
            re.search(r"[^a-z]or(x|[^a-z])", s) is not None
            or (s.startswith("{") and s.endswith("}"))
            or (s.startswith("\\{") and s.endswith("\\}"))
    )
def latex2sympy_fix(s: str):
    sp_symbol = parse_latex(s)

    if "," in s:
        first_term = None
        try:
            first_term = parse_latex(s.split(",")[0])
        except Exception:
            pass
        if sp_symbol == first_term:
            raise LaTeXParsingError(f"{s} != {first_term}")

    return sp_symbol
def latex2sympy_interval(s: str):
    """Parse LaTeX expression like (-\\infty,0] as SymPy Interval object."""
    s = s.replace(" ", "")

    if "\\cup" in s:
        exps = s.split("\\cup")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Union(*intervals)

    if "\\cap" in s:
        exps = s.split("\\cap")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Intersection(*intervals)

    if s.startswith("\\{") and s.endswith("\\}"):
        return FiniteSet(simplify(latex2sympy_fix(s[2:-2])))
    elif s.startswith("{") and s.endswith("}"):
        return FiniteSet(simplify(latex2sympy_fix(s[1:-1])))

    if s.startswith("("):
        left_open = True
        s = s[1:]
    elif s.startswith("\\("):
        left_open = True
        s = s[2:]
    elif s.startswith("["):
        left_open = False
        s = s[1:]
    elif s.startswith("\\["):
        left_open = False
        s = s[2:]
    else:
        raise ValueError(f"Invalid interval: {s}")

    if s.endswith(")"):
        right_open = True
        s = s[:-1]
    elif s.endswith("\\)"):
        right_open = True
        s = s[:-2]
    elif s.endswith("]"):
        right_open = False
        s = s[:-1]
    elif s.endswith("\\]"):
        right_open = False
        s = s[:-2]
    else:
        raise ValueError(f"Invalid interval: {s}")

    left, right = s.split(",")
    left = simplify(latex2sympy_fix(left))
    right = simplify(latex2sympy_fix(right))
    if left.is_comparable and right.is_comparable and left >= right:
        raise ValueError(f"Invalid interval: {left}, {right}")
    interval = Interval(left, right, left_open, right_open)

    return interval
def norm_ans_str(ans: str) -> str:
    """Normalize answer string for **all kinds** of answers."""
    ans = str(ans)
    ans = ans.replace("\n", "")  # no answer must need \n
    ans = ans.strip()

    # remove impropriate trailing punctuations
    ans = clean(ans)

    # cornor cases

    # bool
    ans_bool = norm_str2bool(ans)
    if ans_bool is not None:
        return str(ans_bool)

    # weekdays
    ans_weekday = norm_str2weekday(ans)
    if ans_weekday is not None:
        return ans_weekday

    # math normalize
    ans = norm_math_str(ans)

    return ans
def clean(ans: str) -> str:
    """Clean the extracted answer."""

    ans = ans.strip()
    ans = clean_preceding(ans)
    ans = clean_trailing(ans)

    return ans
def clean_preceding(
        s: str,  # The input string.
) -> str:  # The cleaned string with preceding punctuation marks removed.
    """Removes preceding punctuation marks from a string."""
    s = str(s).strip()
    while s != "" and s[0] in NO_PRECEDING_PUNCS:
        s = s[1:].strip()

    return s
def clean_trailing(
        s: str,  # The input string.
) -> str:  # The cleaned string with trailing punctuation marks removed.
    """Removes trailing punctuation marks from a string."""
    s = str(s).strip()
    while s != "" and s[-1] in NO_TRAILING_STRS:
        s = s[:-1].strip()
    return s
def remove_latex_cmd(s: str, cmd: str) -> str:
    try:
        cmd_idx = s.index(cmd)
    except ValueError:
        return s

    pfx = s[:cmd_idx].strip()
    sfx = s[cmd_idx + len(cmd) :].strip()

    if len(sfx) > 0 and sfx[0] == "{":  # Common command
        sfx = remove_first_paren_pair(sfx, "{")
    elif len(pfx) > 0 and pfx[-1] == "{":  # Declaration command
        left_idx_in_sfx = sfx.find("}")
        if left_idx_in_sfx != -1:
            pfx = pfx[:-1]
            sfx = sfx[:left_idx_in_sfx] + sfx[left_idx_in_sfx + 1 :]
    else:  # Indepedent command
        pass

    return pfx + sfx
def latex2matrix(latex_mat_str: str):
        """This function convert latex matrix into sympy matrix (always 2)"""
        if not isinstance(latex_mat_str, str):
            raise ValueError(f"{latex_mat_str} is not a `str`!")
        latex_mat_str = latex_mat_str.replace(" ", "")

        pattern = r"(?:\[|\()?\\begin{[a-zA-Z]?(?:matrix|array)}(?:\[lcr\])*?(.*)\\end{[a-zA-Z]?(?:matrix|array)}(?:\]|\))?"
        data = re.search(pattern, latex_mat_str)
        python_matrix = []
        if data is not None:
            data = data[1]
            # \+ not followed by frac or sqrt
            rows = re.split(r"\\+(?!frac|sqrt)", data)
            for row in rows:
                elements_list = row.split("&")
                python_matrix.append(elements_list)
        else:
            if "," in latex_mat_str:
                # if is_set(latex_mat_str):
                #     # print("set")
                #     python_matrix = [extract_set(latex_mat_str)]
                # else:
                    python_matrix = [remove_out_paren(latex_mat_str).split(",")]
            else:
                raise LaTeXParsingError(
                    f"{latex_mat_str} can not be parsed in a `Matrix`!"
                )

        # print(data)
        # print(python_matrix)
        sympy_matrix = []
        for row in python_matrix:
            # print(row)
            sympy_row = [latex2sympy_fix(element) for element in row]
            sympy_matrix.append(sympy_row)

        matrix = Matrix(sympy_matrix)

        # print(s)
        # unify one row/col into vector
        if len(matrix.shape) == 2 and matrix.shape[1] == 1:
            matrix = matrix.T
        return matrix
def could_be_percent(v) -> bool:
    """Check if a value could be a percentage."""
    return 0 < v < 1 or 1 < v < 100
def is_num_eq(ref_num, pred_num) -> bool | None:
    """Compare two numbers with specified feautures:
    - relative tolerance
    - flexible percentage surface forms
    """
    if ref_num is None or pred_num is None:
        return None
    # print(ref_num,pred_num,123)

    if isclose(ref_num, pred_num, rel_tol=rel_tol, abs_tol=abs_tol):
        return True

    if include_percentage and could_be_percent(pred_num):
        percent_ref_nums = [
            num
            for num in [ref_num / 100, ref_num * 100]
            if could_be_percent(num)
        ]
        for item in percent_ref_nums:
            # "For the values to be considered close, the difference between them must be smaller than at least one of the tolerances."
            if isclose(
                    item, pred_num, rel_tol=percent_rel_tol, abs_tol=abs_tol
            ):
                return True
    return None
def remove_first_paren_pair(
        s: str,
        l: str,  # Left parenthesis
) -> str:
    i_l, i_r = index_first_paren_pair(s, l)
    if i_l != -1 and i_r != -1:
        len_paren = len(l)
        s = s[:i_l] + s[i_l + len_paren : i_r] + s[i_r + len_paren :]

    return s
def index_first_paren_pair(s: str, l: str) -> tuple[int, int]:
    r = PAREN_MAP[l]
    try:
        i_l = s.index(l)
    except ValueError:
        return -1, -1
    len_paren = len(l)

    depth = 0
    i_r = -1
    for i_c in range(i_l, len(s)):
        if s[i_c : i_c + len_paren] == l:
            depth -= 1
        elif s[i_c : i_c + len_paren] == r:
            depth += 1
        if depth == 0:
            i_r = i_c
            break

    return i_l, i_r
def rm_latex_env(s: str, env: str) -> str:
    """Remove LaTeX environment from a string.

    Parameters
    ----------
    s : str
        The input string.
    env : str
        The LaTeX environment name to remove.

    Returns
    -------
    str
        The string with the specified LaTeX environment removed.
    """
    s = s.replace(f"\\begin{{{env}}}", "")
    s = s.replace(f"\\end{{{env}}}", "")
    return s
def norm_deg(s: str) -> str:
    """Normalize expressions including degrees, except independent <num>\\circ"""
    s = s.replace("rad", "")
    s = re.sub(r"^(\d+) ?\^?\\?circ$", r"\1", s)
    s = re.sub(r"(\d+) ?\^?\\?circ", r"{\1*\\frac{\\pi}{180}}", s)

    return s
def norm_basic_fn( s: str) -> str:
    """Avoid potential LaTex errors caused by removing spaces:
    - \\{fn}[a-z] : followed by some letter without middle spaces
    - \\{fn}^{pow}{expr}

    Returns
    -------
    str
        Normalized format of basic function expression: \\{fn}^{{pow}}{{expr}}
    """
    # \2 matches \d+ without {} around, if there has been {}, there is no need to normalize
    # Existing nude power, i.e. ^<pow_d+>
    s = re.sub(rf"\\?({'|'.join(BASIC_FN_NAMES)})\^(\d+)", r"\\\1^{\2}", s)
    # No power
    s = re.sub(rf"\\?({'|'.join(BASIC_FN_NAMES)})(?!\^)", r"\\\1^{1}", s)
    return s
def fix_fracs(s: str) -> str:
    """Fixes the formatting of fractions in a given string."""
    substrs = s.split("\\frac")
    _s = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            _s += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                _s += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return s
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        _s += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        _s += "{" + a + "}" + b + post_substr
                    else:
                        _s += "{" + a + "}" + b
    return _s
def remove_out_paren(s: str) -> str:
    """Remove until there are no parentheses outside."""
    done = False
    while not done:
        done = True
        for left, _ in PAREN_MAP.items():
            len_paren = len(left)
            i_l, i_r = index_first_paren_pair(s, left)
            if i_l == 0 and i_r == len(s) - len_paren:
                s = s[len_paren:-len_paren]
                done = False
    return s
def is_sym_eq(a: Any, b: Any) -> bool | None:
    """Compare two objects symbolically."""
    if a is None or b is None:
        return None

    try:
        if a == b:
            return True
    except Exception:
        pass

    try:
        diff = simplify(a - b)
        # For non-symmetric operations like subtraction between sets
        diff_rev = simplify(b - a)

        if hasattr(diff, "__iter__") and hasattr(
                diff_rev, "__iter__"
        ):  # If diff is iterable (e.g. Matrix)
            if all(element == 0 for element in diff) and all(
                    element == 0 for element in diff_rev
            ):
                return True
        else:
            if (
                    not diff and not diff_rev
            ):  # use `not` for non-zero values like `sympy.EmptySet`
                return True
    except Exception:
        pass

    try:
        v_a, v_b = (N(eval(str(v))) for v in [a, b])
        num_eq = is_num_eq(v_a, v_b)
        if num_eq:
            return True
    except Exception:
        pass

    return None
def norm_pm( s: str) -> str:
    """Replaces the LaTeX symbols '$1\\pm$2' or '$1\\mp$2' with '$1-$2,$1+$2'."""

    def replace_pm(match):
        # Extracts the first and second parts of the match.
        first_part, second_part = match.groups()
        # Creates the replacement string as specified.
        return f"{first_part}-{second_part},{first_part}+{second_part}"

    _s = remove_out_paren(s)
    # Define the pattern that matches '$1\\pm$2' or '$1\\mp$2'.
    # We use non-greedy matching (.*?) to capture the parts before and after \pm or \mp.
    # The pattern is corrected to include the '$' signs and to capture the expressions correctly.
    pattern = r"([\w\.\\{}\+\-\*\^]+?)(?:\\pm|\\mp)([\w\.\\{}\+\-\*\^]+)"

    if re.search(pattern, _s):
        # Use re.sub to replace all occurrences of the pattern in the input string.
        return re.sub(pattern, replace_pm, _s)
    else:
        return s
def fix_sqrt(
        s: str,
) -> str:
    """Fixes the formatting of square root expressions in a given string."""
    _s = re.sub(r"\\?sqrt[\(\{\[](\w+)[\)\}\]]", r"\\sqrt{\1}", s)
    _s = re.sub(r"\\?sqrt\s*(\d+)", r"\\sqrt{\1}", _s)
    return _s
def fix_a_slash_b(s: str) -> str:
    """
    Fixes the formatting of fractions in a given string using regular expressions.
    """
    # Define a regular expression to match fractions. Here we match two parts: the numerator (a) and the denominator (b).
    # The numerator and denominator can be numbers (\d+) or expressions containing sqrt (sqrt\(.*?\)).
    fraction_pattern = r"(\b\d+|sqrt\(.*?\))\/(\d+|sqrt\(.*?\)\b)"

    # Use `re.sub` to replace the matched fractions with properly formatted fractions.
    result = re.sub(
        fraction_pattern, lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", s
    )

    return result
def norm_math_str(string: str):
        # delay logics for multi-choice to after extraction from model output
        # lower_str = string.lower()
        # for choice in ALL_CHOICES:
        #     choice_lower = choice.lower()
        #     if lower_str == choice_lower or lower_str == f"({choice_lower})":
        #         return choice

        # Replacement-based normalization

        string = str(string).strip()
        string = clean(string)

        # Simple removals
        for rm_str in SIMPLE_RM_STRS:
            string = string.replace(rm_str, "")

        # Simple replacements
        for k, v in SIMPLE_REPLACE_MAP.items():
            string = string.replace(k, v)
        if "\\infty" not in string:
            string = string.replace("inf", "\\infty")

        # Remove spaces after all space-related operations
        string = string.replace(" ", "")

        for latex_cmd in LATEX_CMDS:
            string = remove_latex_cmd(string, latex_cmd)

        for env in LATEX_FMT_ENVS + LATEX_LIST_ENVS:
            string = rm_latex_env(string, env)

        # Normalize local expressions
        string = norm_deg(string)  # Normalize degrees
        string = re.sub(
            rf"(?<!\\)(pi\b|{'|'.join(BASIC_FN_NAMES)})", r"\\\1", string
        )  # Fix backslashes
        string = norm_basic_fn(string)  # Normalize basic functions

        # Normalize matrix and array
        string = re.sub(r"{[a-z]?matrix}", r"{array}", string)
        string = re.sub(r"\\begin{array}{[lcr]*}", r"\\begin{array}{}", string)
        # NOTE: the substituion str should alse obey the regex syntax, like r"\\begin{array}"
        if "\\begin{array}" not in string:
            string = string.replace("\\\\", "")

        # i, j
        if "j" in string and "i" not in string:
            string = string.replace("j", "i")

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
        string = re.sub(r"(\d+)\.0+$", r"\1", string)

        # remove units
        for unit in UNITS:
            string = re.sub(f"([-\d\.\*\^{{}}]+){unit}e?s?$", "\\1", string)

        # Check if empty before splitting
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # Splitting-based normalization

        # Process complex expressions without parentheses
        s_is_set = is_set(string)
        # if s_is_set:
        #     raw_strings = extract_set(string)
        # else:
        raw_strings = [string]

        strings = []
        for string in raw_strings:
            string = fix_sqrt(string)

            if string.startswith("frac"):
                string = "\\" + string
            # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
            string = fix_fracs(string)

            # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
            string = fix_a_slash_b(string)

            string = re.sub(r"^[a-z]\\in", "", string)

            if "," not in string:
                string = remove_out_paren(string)

            if "\\begin{array}" not in string:
                # to consider: get rid of chain of equalities like "a = b = c = d"
                if len(string.split("=")) > 2:
                    string = string.split("=")[-1]

                # to consider: get rid of e.g. "k = " or "q = " at beginning
                if len(string.split("=")) == 2:
                    first_part = string.split("=")[0].strip()
                    if (
                            re.match(
                                r"^([a-z]|[A-Z]{2}|\\?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|sin|cos|sec|csc|tan|cot|sinh|cosh|sech|csch|tanh|coth|log|ln|exp))\^?{?-?('|\\prime|\d)*}?(\(-?([\d\.]+|[a-z])?\))?$",
                                first_part,
                            )
                            is not None
                    ):
                        string = string.split("=")[1]

                # to consider: get rid of equalities but not equations
                if len(string.split("=")) == 2:
                    if len(re.findall(r"[a-zA-Z]", string.split("=")[0].strip())) == 0:
                        string = string.split("=")[1]
            # replace \pm with +,-
            # string = re.sub(r"(.*?)\\pm(.+?)", r"\1-\2,\1+\2", string)
            string = norm_pm(string)  # might add comma ","

            string = re.sub(r"^0+([1-9])", r"\1", string)

            strings.append(string)
        string = ",".join(strings)

        if "," not in string:
            string = remove_out_paren(string)

        if STR2NUM.get(string):
            string = str(STR2NUM[string])

        # add space
        string = re.sub(r"\\mid([a-z])", r"\\mid \1", string)
        string = clean(string)

        # If there are multiple same inequality signs and no commas
        for ineq in ["<", ">"]:
            if len(re.findall(f"{ineq}=?", string)) > 1 and not any(
                    delim in string.lower() for delim in [",", "and", "or"]
            ):
                string = string.replace(ineq, ",")

        return string
def parse(parser: Callable, s_to_parse: str, parse_errs: list[Exception]) -> Any | None:
    try:
        return parser(s_to_parse)
    except Exception as e:
        parse_errs.append(e)
    return None
def find_box(pred_str: str):
    ans = pred_str.split('boxed')[-1]
    if not ans:
        return ""
    if (ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if (c == '{'):
                stack += 1
                a += c
            elif (c == '}'):
                stack -= 1
                if (stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split('$')[0].strip()
    a = a.strip('?')
    return a
def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string
def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
    string = new_str
    return string
def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string
def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    string = string.strip("?")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.strip('$')
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    # if string == "0.5":
    #    string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
def extract_math_answer(pred_str: str, answer_flag: bool):
    if 'boxed' in pred_str:
        pred = find_box(pred_str)
    elif answer_flag:
        pred_str = pred_str.split('=')[-1].strip()
        if re.match(r'[\d\.]+\s\D+$', pred_str):
            pred_str = pred_str.split(' ')[0]
        pred = pred_str
    else:
        # desparate search over the last number
        preds = re.findall(r'-?\d*\.?\d+', pred_str)
        if(len(preds) >= 1):
            pred = preds[-1]
        else:
            pred = ''

    pred=_strip_string(pred)
    # pred = _fix_2sqrt(pred)
    return pred