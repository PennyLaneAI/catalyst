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


class SingleParamNoCustomOpBadOrder(qp.core.Operator2):

    dynamic_argnames = ("x",)

    def __init__(self, wires, x):
        super().__init__(wires, x)


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

    def __init__(self, a, b, c, wires):
        super().__init__(a, b, c, wires)


class StaticData(qp.core.Operator2):

    static_argnames = ("label",)
    wire_argnames = ("reg",)

    def __init__(self, label, reg):
        super().__init__(label=label, reg=reg)


class StaticDataMultiReg(qp.core.Operator2):

    static_argnames = ("label",)
    wire_argnames = ("reg", "reg2")
    dynamic_argnames = ("theta",)

    def __init__(self, label, reg, reg2, theta):
        super().__init__(label=label, reg=reg, reg2=reg2, theta=theta)


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


class MultipleFullArgs(qp.core.Operator2):

    wire_argnames = ("reg1", "reg2", "hwires1", "hwires2")
    dynamic_argnames = ("angles1", "angles2")
    static_argnames = ("pytree1", "pytree2")
    hybrid_argnames = ("op1", "op2", "hwires1", "hwires2")

    def __init__(
        self,
        reg1,
        reg2,
        angles1,
        angles2,
        pytree1,
        pytree2,
        op1,
        op2,
        hwires1,
        hwires2,
    ):
        super().__init__(reg1, reg2, angles1, angles2, pytree1, pytree2, op1, op2, hwires1, hwires2)
