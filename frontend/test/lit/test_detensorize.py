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
from catalyst.debug import print_compilation_stage


# CHECK-LABEL: public @jit_a_plus_b_times_2
@qjit(keep_intermediate=True)
def a_plus_b_times_2(a, b):
    # CHECK: %extracted
    # CHECK: tensor.extract
    c = a + b
    # CHECK: tensor.from_elements
    return c + c


a_plus_b_times_2(1.0, 2.0)
print_compilation_stage(a_plus_b_times_2, "HLOLoweringPass")


# CHECK-LABEL: public @jit_f_with_cond
@qjit(autograph=True, keep_intermediate=True)
def f_with_cond(a, b):
    # CHECK: %extracted
    # CHECK: tensor.extract
    a2 = a + a
    if a2 > b:
        a = a - 2.0
        b = b * 2.0
        # CHECK: tensor.from_elements
        c = a + b
    else:
        a = a * 2.0
        b = b - 2.0
        c = a + b
    return c * 2.0


f_with_cond(1.0, 2.0)
print_compilation_stage(f_with_cond, "HLOLoweringPass")


# CHECK-LABEL: public @jit_f_with_for_loop
@qjit(autograph=True, keep_intermediate=True)
def f_with_for_loop(a, b):
    # CHECK: %extracted
    # CHECK: tensor.extract
    # CHECK: tensor.from_elements
    b = a + b
    for _ in range(10):
        if b % 2:
            b = b - b
        else:
            b = b + 1
    c = a * b
    return c


f_with_for_loop(1.0, 2.0)
print_compilation_stage(f_with_for_loop, "HLOLoweringPass")
