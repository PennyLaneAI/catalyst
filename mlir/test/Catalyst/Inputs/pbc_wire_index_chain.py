# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Emit a dependency chain longer than the former recursive evaluation limit."""

print("func.func @long_local_classical_chain(%unknown: tensor<i64>) {")
print("  %zero = stablehlo.constant dense<0> : tensor<i64>")
print("  %one = stablehlo.constant dense<1> : tensor<i64>")
print("  %true = stablehlo.constant dense<true> : tensor<i1>")
previous = "%zero"
for index in range(300):
    current = f"%value{index}"
    print(f"  {current} = stablehlo.add {previous}, %one : tensor<i64>")
    previous = current
print(f"  %selected = stablehlo.select %true, {previous}, %unknown : tensor<i1>, tensor<i64>")
print("  %wire = tensor.extract %selected[] : tensor<i64>")
print("  %reg = quantum.alloc(301) : !quantum.reg")
print("  %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit")
print("  %q1 = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit")
print('  %a = pbc.ppr ["X"](8) %q0 : !quantum.bit')
print('  %b = pbc.ppr ["X"](8) %q1 : !quantum.bit')
print("  return")
print("}")
