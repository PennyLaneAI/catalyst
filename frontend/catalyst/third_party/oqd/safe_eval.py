# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Safe Expression Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides a lightweight tool, `safe_eval`, for evaluating mathematical expressions. The
usage of `safe_eval` should be preferred wherever possible over Python's builtin `eval()` function,
whose ability to perform arbitrary code execution of a user's input makes it inherently unsafe.
The functionality of `safe_eval` is deliberately limited to basic mathematical operations to prevent
malicious code, intentional or not, from being evaluated.
"""

import ast
import math
import operator

# Supported operators and their corresponding functions
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Mathematical constants and functions from the math module are supported.

    Parameters:
        expr (str): The arithmetic expression to evaluate.

    Returns:
        float: The result of the evaluated expression.

    Raises:
        ValueError: If the expression is invalid or contains unsupported elements.

    Examples:

        >>> safe_eval("1 + 1e-1")
        1.1
        >>> safe_eval("1 + (2 * math.sin(math.pi / 2))")
        3.0
    """

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        elif isinstance(node, ast.BinOp):  # Binary operations (e.g., 1 + 2)
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in OPERATORS:
                return OPERATORS[op_type](left, right)
            else:
                raise ValueError(f"Unsupported operator: {op_type}")

        elif isinstance(node, ast.UnaryOp):  # Unary operations (e.g., -1)
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in OPERATORS:
                return OPERATORS[op_type](operand)
            else:
                raise ValueError(f"Unsupported unary operator: {op_type}")

        elif isinstance(node, ast.Call):  # Function calls (e.g., math.sin(0.5))
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module = node.func.value.id
                func = node.func.attr
                if module == "math" and hasattr(math, func):
                    args = [_eval(arg) for arg in node.args]
                    return getattr(math, func)(*args)
                else:
                    raise ValueError(f"Unsupported function: {module}.{func}")
            else:
                raise ValueError("Unsupported function call structure")

        elif isinstance(node, ast.Attribute):  # Accessing attributes (e.g., math.pi)
            if isinstance(node.value, ast.Name):
                module = node.value.id
                attr = node.attr
                if module == "math" and hasattr(math, attr):
                    return getattr(math, attr)
                else:
                    raise ValueError(f"Unsupported attribute: {module}.{attr}")
            else:
                raise ValueError("Unsupported attribute structure")

        elif isinstance(node, ast.Constant):  # Python 3.8+ literal
            return node.value

        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    try:
        parsed_expr = ast.parse(expr, mode="eval")
        return _eval(parsed_expr.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")
