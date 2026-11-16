# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

import numpy as np
import pennylane as qp

from catalyst import qjit


@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=2))
# CHECK-LABEL: public @f(
def f(arg0: float, arg1: int, arg2: int):
    # CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[wire0:%.+]] = tensor.extract %arg1
    # CHECK: [[qubit0:%.+]] = qref.get [[reg]][[[wire0]]]
    # CHECK: qref.custom "RZ"({{%.+}}) [[qubit0]]
    # CHECK: [[wire1:%.+]] = tensor.extract %arg2
    # CHECK: [[qubit1:%.+]] = qref.get [[reg]][[[wire1]]]
    # CHECK: [[m:%.+]] = qref.measure [[qubit1]] : i1
    # CHECK: qref.dealloc [[reg]]
    # CHECK: return
    qp.RZ(arg0, wires=[arg1])
    qp.measure(wires=arg2)
    return qp.expval(qp.Z(arg2))


print(f.mlir)

# -----


# old_frontend: tests elimination of redundant quantum.insert operations.
@qjit(target="mlir", capture=False)
@qp.qnode(qp.device("null.qubit", wires=2))
# CHECK-LABEL: public @g()
def g():
    """
    Test no extraneous insert ops.
    """
    wires = np.arange(2, dtype=int)

    # CHECK: quantum.custom "PauliX"
    qp.X(wires[0])

    # CHECK-NOT: quantum.insert

    # CHECK: quantum.custom "PauliY"
    qp.Y(wires[1])

    return


print(g.mlir)
