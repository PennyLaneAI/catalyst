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
    # CHECK: tensor.from_elements
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
    # CHECK: tensor.from_elements
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
    # CHECK: arith.addf
    b = a + b
    for _ in range(10):
        if b % 2:
            # CHECK: arith.subf
            b = b - b
        else:
            b = b + 1
    # CHECK: tensor.from_elements
    # CHECK: arith.mulf
    c = a * b
    return c


f_with_for_loop(1.0, 2.0)
print(get_compilation_stage(f_with_for_loop, "HLOLoweringPass"))
f_with_for_loop.workspace.cleanup()
