# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file hosts a set of mock pennylane.core.Operator2 subclasses to be used for testing."""

# pylint: disable = missing-class-docstring

import pennylane as qp


class NoParams(qp.core.Operator2):
    wire_argnames = ("reg",)

    def __init__(self, reg):
        super().__init__(reg=reg)


class NoParamsCustomOp(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)


class SingleParam(qp.core.Operator2):

    dynamic_argnames = ("x",)
    wire_argnames = ("reg",)

    def __init__(self, x, reg):
        super().__init__(x, reg=reg)


class SingleParamCustomOp(qp.core.Operator2):

    dynamic_argnames = ("x",)

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


class CompilableData(qp.core.Operator2):

    compilable_argnames = ("a", "b", "thing")

    def __init__(self, a, b, thing, wires):
        super().__init__(a=a, b=b, thing=thing, wires=wires)


class MultipleRegisters(qp.core.Operator2):

    wire_argnames = ("reg1", "reg2")

    def __init__(self, reg1, reg2):
        super().__init__(reg1=reg1, reg2=reg2)


class MultiParams(qp.core.Operator2):

    dynamic_argnames = ("a", "b", "c")
    wire_argnames = ("reg",)

    def __init__(self, reg, a, b, c):
        super().__init__(reg, a, b, c)


class MultiParamsCustom(qp.core.Operator2):

    dynamic_argnames = ("a", "b", "c")

    def __init__(self, wires, a, b, c):
        super().__init__(wires, a, b, c)


class MultiRZ(qp.core.Operator2):

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires)


class PauliRot(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    compilable_argnames = ("pauli_word",)

    def __init__(self, phi, pauli_word, wires):
        super().__init__(phi, pauli_word, wires)


class GlobalPhase(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    wire_argnames = ()

    def __init__(self, phi):
        super().__init__(phi=phi)


class QubitUnitary(qp.core.Operator2):

    dynamic_argnames = ("matrix",)

    def __init__(self, matrix, wires):
        super().__init__(matrix, wires)


class PCPhase(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    compilable_argnames = ("dim",)

    def __init__(self, phi, dim, wires):
        super().__init__(phi, dim, wires)


class StaticData(qp.core.Operator2):

    static_argnames = ("label",)
    wire_argnames = ("reg",)

    def __init__(self, label, reg):
        super().__init__(label=label, reg=reg)


class HybridWires(qp.core.Operator2):

    hybrid_argnames = ("cwires",)
    wire_argnames = ("cwires",)

    def __init__(self, cwires):
        super().__init__(cwires=cwires)


class HybridNoOpArg(qp.core.Operator2):

    hybrid_argnames = ("angles",)

    def __init__(self, angles, wires):
        super().__init__(angles, wires)


class HybridOpArg(qp.core.Operator2):

    dynamic_argnames = ("angle",)
    hybrid_argnames = ("op",)
    wire_argnames = ("cwires",)
    static_argnames = ("n_iters",)

    def __init__(self, angle, op, cwires, n_iters=1):
        super().__init__(angle, op, cwires, n_iters)
