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

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=missing-module-docstring,missing-function-docstring

from catalyst import qjit
from catalyst.debug import get_compilation_stage


# CHECK-LABEL: public @jit_a_plus_b_times_2
@qjit(keep_intermediate=True)
def a_plus_b_times_2(a, b):
    # CHECK: %extracted
    # CHECK: tensor.extract
    # CHECK: arith.addf
    # CHECK-NOT: linalg.generic
    c = a + b
    return c + c


a_plus_b_times_2(1.0, 2.0)
print(get_compilation_stage(a_plus_b_times_2, "HLOLoweringPass"))
a_plus_b_times_2.workspace.cleanup()


# CHECK-LABEL: public @jit_f_with_cond
@qjit(autograph=True, keep_intermediate=True)
def f_with_cond(a, b):
    # CHECK: %extracted
    # CHECK: tensor.extract
    # CHECK: arith.addf
    # CHECK-NOT: linalg.generic
    a2 = a + a
    if a2 > b:
        # CHECK: arith.subf
        a = a - 2.0
        # CHECK: arith.mulf
        b = b * 2.0
        c = a + b
    else:
        # CHECK: arith.mulf
        a = a * 2.0
        # CHECK: arith.subf
        b = b - 2.0
        c = a + b
    return c * 2.0


f_with_cond(1.0, 2.0)
print(get_compilation_stage(f_with_cond, "HLOLoweringPass"))
f_with_cond.workspace.cleanup()


# CHECK-LABEL: public @jit_f_with_for_loop
@qjit(autograph=True, keep_intermediate=True)
def f_with_for_loop(a, b):
    # CHECK: %extracted
    # CHECK: tensor.extract
    # CHECK-NOT: linalg.generic
    b = a + b
    for _ in range(10):
        if b % 2:
            # CHECK: arith.subf
            b = b - b
        else:
            b = b + 1
    # CHECK: arith.mulf
    c = a * b
    return c


f_with_for_loop(1.0, 2.0)
print(get_compilation_stage(f_with_for_loop, "HLOLoweringPass"))
f_with_for_loop.workspace.cleanup()


# CHECK-LABEL: public @jit_f_with_nested_ifs
@qjit(autograph=True, keep_intermediate=True)
def f_with_nested_ifs(a, b, c):
    # CHECK: %extracted
    # CHECK: tensor.extract
    # CHECK-NOT: linalg.generic
    if a < b:
        c = c + 1.0
        if a > 2 * b:
            b = 2 * a
        else:
            b = a / 2
    else:
        c = c - 1.0
        if b > 2 * a:
            a = 2 * b
        else:
            a = b / 2

    def floor_divide(x, y):
        return x - x // y

    c = c + floor_divide(a, b)
    return a * b * c + 2 * a**2 * b**2


f_with_nested_ifs(1.0, 2.0, 3.0)
print(get_compilation_stage(f_with_nested_ifs, "HLOLoweringPass"))
f_with_nested_ifs.workspace.cleanup()


# CHECK-LABEL: public @jit_f_with_nested_scf
@qjit(autograph=True, keep_intermediate=True)
def f_with_nested_scf(a, b, c):
    # CHECK: %extracted
    # CHECK: tensor.extract
    # CHECK-NOT: linalg.generic
    if a < b:
        for i in range(5):
            c = c + i
            if c > 4:
                b = 2 * a
            else:
                b = a / 2
        if a > 2 * b:
            b = 2 * a
        else:
            b = a / 2
    else:
        if c > 4:
            for i in range(3):
                c = c - 1.0
        if b > 2 * a:
            a = 2 * b
        else:
            a = b / 2

    def floor_divide(x, y):
        return x - x // y

    c = c + floor_divide(a, b)
    return a * b * c + 2 * a**2 * b**2


f_with_nested_scf(1.0, 2.0, 3.0)
print(get_compilation_stage(f_with_nested_scf, "HLOLoweringPass"))
f_with_nested_scf.workspace.cleanup()
