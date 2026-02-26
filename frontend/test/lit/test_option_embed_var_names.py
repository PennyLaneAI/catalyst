# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for embed variable names option."""

# RUN: %PYTHON %s | FileCheck %s

from utils import print_mlir, print_mlir_opt

from catalyst import qjit


# CHECK-LABEL: @jit_f
@qjit(embed_var_names=True)
def f(x: float, y: float):
    """Check that MLIR module contains name location information, and MLIR code uses that name
    location information.
    """
    # CHECK: %x: tensor<f64>, %y: tensor<f64>
    return x * y


assert str(f.mlir_module.body.operations[0].arguments[0].location) == 'loc("x")'
assert str(f.mlir_module.body.operations[0].arguments[1].location) == 'loc("y")'

print_mlir(f, 0.3, 0.4)


# CHECK-LABEL: @jit_f_opt
@qjit(embed_var_names=True)
def f_opt(x: float, y: float):
    """Check that MLIR module contains name location information, and MLIR code uses that name
    location information.
    Same test as before, but now we exercise mlir_opt property.
    """
    # CHECK: %x: !llvm.ptr, {{.*}} %y: !llvm.ptr
    return x * y


assert str(f_opt.mlir_module.body.operations[0].arguments[0].location) == 'loc("x")'
assert str(f_opt.mlir_module.body.operations[0].arguments[1].location) == 'loc("y")'

print_mlir_opt(f_opt, 0.3, 0.4)
