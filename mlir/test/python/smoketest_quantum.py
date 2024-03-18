# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s

from mlir_quantum.dialects import quantum as quantum_d
from mlir_quantum.ir import *

with Context() as ctx:
    quantum_d.register_dialect()
    module = Module.parse(
        """
        %c0 = "arith.constant"() {value = 0 : i64} : () -> i64
        %0 = "quantum.alloc"() {nqubits_attr = 2} : () -> !quantum.reg
        %1 = "quantum.extract"(%0, %c0) : (!quantum.reg, i64) -> !quantum.bit
        %2 = "quantum.custom"(%1) { gate_name = "Hadamard", operand_segment_sizes = array<i32: 0, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0> } : (!quantum.bit) -> !quantum.bit
        %3, %4 = "quantum.measure"(%1) : (!quantum.bit) -> (i1, !quantum.bit)
        %5 = "quantum.insert"(%0, %c0, %4) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
        "quantum.dealloc"(%5) : (!quantum.reg) -> ()
        """
    )

    print(module)
