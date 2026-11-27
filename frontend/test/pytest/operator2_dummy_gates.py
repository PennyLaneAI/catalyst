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

from collections import Counter

import pennylane as qp

Operator = getattr(qp.core, "Operator2", None) or qp.core.Operator


class NoParams(Operator):
    wire_argnames = ("reg",)

    def __init__(self, reg):
        super().__init__(reg=reg)


class NoParamsCustomOp(Operator):

    def __init__(self, wires):
        super().__init__(wires=wires)


class SingleParam(Operator):

    dynamic_argnames = ("x",)
    wire_argnames = ("reg",)

    def __init__(self, x, reg):
        super().__init__(x, reg=reg)


class SingleParamCustomOp(Operator):

    dynamic_argnames = ("x",)

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


class CompilableData(Operator):

    compilable_argnames = ("a", "b", "thing")

    def __init__(self, a, b, thing, wires):
        super().__init__(a=a, b=b, thing=thing, wires=wires)


class ArrayData(Operator):

    compilable_argnames = ("angles",)

    def __init__(self, angles, wires):
        super().__init__(angles=angles, wires=wires)


class MultipleRegisters(Operator):

    wire_argnames = ("reg1", "reg2")

    def __init__(self, reg1, reg2):
        super().__init__(reg1=reg1, reg2=reg2)


class MultiParams(Operator):

    dynamic_argnames = ("a", "b", "c")
    wire_argnames = ("reg",)

    def __init__(self, reg, a, b, c):
        super().__init__(reg, a, b, c)


class MultiParamsCustom(Operator):

    dynamic_argnames = ("a", "b", "c")

    def __init__(self, wires, a, b, c):
        super().__init__(wires, a, b, c)


class TestQubitUnitary(Operator):

    dynamic_argnames = ("matrix",)

    def __init__(self, matrix, wires):
        super().__init__(matrix, wires)


class StaticData(Operator):

    static_argnames = ("label",)
    wire_argnames = ("reg",)

    def __init__(self, label, reg):
        super().__init__(label=label, reg=reg)


class HybridWires(Operator):

    hybrid_argnames = ("cwires",)
    wire_argnames = ("cwires",)

    def __init__(self, cwires):
        super().__init__(cwires=cwires)


class HybridNoOpArg(Operator):

    hybrid_argnames = ("angles",)

    def __init__(self, angles, wires):
        super().__init__(angles, wires)


class HybridOpArg(Operator):

    dynamic_argnames = ("angle",)
    hybrid_argnames = ("op",)
    wire_argnames = ("cwires",)
    static_argnames = ("n_iters",)

    def __init__(self, angle, op, cwires, n_iters=1):
        super().__init__(angle, op, cwires, n_iters)


class HybridOpArgs(Operator):
    """A sequence of three operator-valued arguments."""

    hybrid_argnames = ("first_op", "second_op", "third_op")
    wire_argnames = ()

    def __init__(self, first_op, second_op, third_op):
        super().__init__(first_op, second_op, third_op)


@qp.register_resources(
    lambda first_op, second_op, third_op: Counter([first_op, second_op, third_op])
)
def hybrid_op_args_decomp(first_op, second_op, third_op):
    """Apply each operator argument in sequence, including repeated operators."""
    qp.apply(first_op)
    qp.apply(second_op)
    qp.apply(third_op)
