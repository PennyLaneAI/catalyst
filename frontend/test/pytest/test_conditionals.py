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

import numpy as np
import pennylane as qml
import pytest

from catalyst import cond, measure, qjit

# pylint: disable=missing-function-docstring


class TestCondToJaxpr:
    """Run tests on the generated JAXPR of conditionals."""

    def test_basic_cond_to_jaxpr(self):
        """Check the JAXPR of simple conditional function."""

        expected = """{ lambda ; a:i64[]. let
    b:bool[] = eq a 5
    c:i64[] = cond[
      branch_jaxprs=[{ lambda ; a:i64[] b:i64[]. let c:i64[] = integer_pow[y=2] a in (c,) },
                     { lambda ; a:i64[] b:i64[]. let c:i64[] = integer_pow[y=3] b in (c,) }]
    ] b a a
  in (c,) }"""

        @qjit
        def circuit(n: int):
            @cond(n == 5)
            def cond_fn():
                return n**2

            @cond_fn.otherwise
            def cond_fn():
                return n**3

            out = cond_fn()
            return out

        def asline(text):
            return " ".join(map(lambda x: x.strip(), str(text).split("\n")))

        assert asline(expected) == asline(circuit.jaxpr)


class TestCond:
    """Test suite for the Cond functionality in Catalyst."""

    def test_simple_cond(self, backend):
        """Test basic function with conditional."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @cond(n > 4)
            def cond_fn():
                return n**2

            @cond_fn.otherwise
            def else_fn():
                return n

            return cond_fn()

        assert circuit(0) == 0
        assert circuit(1) == 1
        assert circuit(2) == 2
        assert circuit(3) == 3
        assert circuit(4) == 4
        assert circuit(5) == 25
        assert circuit(6) == 36

    def test_cond_one_else_if(self, backend):
        """Test a cond with one else_if branch"""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            @cond(x > 2.7)
            def cond_fn():
                return x * 4

            @cond_fn.else_if(x > 1.4)
            def cond_elif():
                return x * 2

            @cond_fn.otherwise
            def cond_else():
                return x

            return cond_fn()

        assert circuit(4) == 16
        assert circuit(2) == 4
        assert circuit(1) == 1

    def test_cond_many_else_if(self, backend):
        """Test a cond with multiple else_if branches"""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            @cond(x > 4.8)
            def cond_fn():
                return x * 8

            @cond_fn.else_if(x > 2.7)
            def cond_elif():
                return x * 4

            @cond_fn.else_if(x > 1.4)
            def cond_elif2():
                return x * 2

            @cond_fn.otherwise
            def cond_else():
                return x

            return cond_fn()

        assert circuit(5) == 40
        assert circuit(3) == 12
        assert circuit(2) == 4
        assert circuit(-3) == -3

    def test_cond_else_if_classical(self):
        """Test a cond with multiple else_if branches using the classical compilation path."""

        @qjit
        def circuit(x):
            @cond(x > 4.8)
            def cond_fn():
                return x * 16

            @cond_fn.else_if(x > 2.7)
            def cond_elif():
                return x * 8

            @cond_fn.else_if(x > 1.4)
            def cond_elif2():
                return x * 4

            @cond_fn.otherwise
            def cond_else():
                return x

            return cond_fn()

        assert circuit(5) == 80
        assert circuit(3) == 24
        assert circuit(2) == 8
        assert circuit(-3) == -3

    def test_qubit_manipulation_cond(self, backend):
        """Test conditional with quantum operation."""

        @qjit()
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            @cond(x > 4)
            def cond_fn():
                qml.PauliX(wires=0)

            cond_fn()

            return measure(wires=0)

        assert circuit(3) == False
        assert circuit(6) == True

    def test_branch_return_pytree_mismatch(self, backend):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """

        def circuit():
            @cond(True)
            def cond_fn():
                return (1, 1)

            @cond_fn.otherwise
            def cond_fn():
                return [2, 2]

            return cond_fn()

        with pytest.raises(
            TypeError, match="Conditional requires consistent return types across all branches"
        ):
            qjit(qml.qnode(qml.device(backend, wires=1))(circuit))

    def test_branch_return_mismatch(self, backend):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """

        def circuit():
            @cond(True)
            def cond_fn():
                return measure(wires=0)

            return cond_fn()

        with pytest.raises(
            TypeError, match="Conditional requires consistent return types across all branches"
        ):
            qjit(qml.qnode(qml.device(backend, wires=1))(circuit))

    def test_branch_multi_return_type_unification(self, backend):
        """Test that an exception is not raised when the return types of all branches do not match
        but could be unified."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            @cond(True)
            def cond_fn():
                return measure(wires=0)

            @cond_fn.else_if(False)
            def cond_elif():
                return 0

            @cond_fn.otherwise
            def cond_else():
                return measure(wires=0)

            return cond_fn()

        assert 0 == circuit()

    def test_branch_return_mismatch_classical(self):
        """Test that an exception is raised when the true branch returns a different pytree-shape
        than the else branch, given a classical tracing context (no QNode).
        """

        def circuit():
            @cond(True)
            def cond_fn():
                return (1, 1)

            @cond_fn.otherwise
            def cond_fn():
                return 2

            return cond_fn()

        with pytest.raises(
            TypeError, match="Conditional requires consistent return types across all branches"
        ):
            qjit(circuit)

    def test_branch_return_promotion_classical(self):
        """Test that an exception is raised when the true branch returns a different type than the
        else branch, given a classical tracing context (no QNode).
        """

        def circuit():
            @cond(True)
            def cond_fn():
                return 1

            @cond_fn.otherwise
            def cond_fn():
                return 2.0

            return cond_fn()

        assert 1.0 == qjit(circuit)()

    def test_branch_with_arg(self):
        """Test that an exception is raised when an 'else if' branch function contains an arg"""

        def circuit(pred: bool):
            @cond(pred)
            def cond_fn():
                qml.PauliX(0)

            @cond_fn.else_if(pred)
            def cond_elif(x):
                qml.PauliX(x)

            return measure(wires=0)

        with pytest.raises(
            TypeError, match="Conditional 'else if' function is not allowed to have any arguments"
        ):
            qjit(qml.qnode(qml.device("lightning.qubit", wires=1))(circuit))

    def test_identical_branch_names(self, backend):
        """Test that branches of the conditional can carry the same function name."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(pred: bool):
            @cond(pred)
            def conditional_flip():
                qml.PauliX(0)

            @conditional_flip.otherwise
            def conditional_flip():
                qml.Identity(0)

            conditional_flip()

            return measure(wires=0)

        assert circuit(False) == 0
        assert circuit(True) == 1


class TestInterpretationConditional:
    """Test that the conditional operation's execution is semantically equivalent
    when compiled and interpreted."""

    def test_conditional_interpreted_and_compiled(self):
        """Test that a compiled and interpreted conditional have the same output."""

        def arithi(x: int, y: int, op: int):
            @cond(op == 0)
            def branch():
                return x - y

            @branch.otherwise
            def branch():
                return x + y

            return branch()

        arithc = qjit(arithi)
        assert arithc(0, 0, 0) == arithi(0, 0, 0)
        assert arithc(0, 0, 1) == arithi(0, 0, 1)

    def test_conditional_interpreted_and_compiled_single_if(self, backend):
        """Test that a compiled and interpreted conditional with no else branch match."""

        num_wires = 2
        device = qml.device(backend, wires=num_wires)

        @qml.qnode(device)
        def interpreted_circuit(n):
            @cond(n == 0)
            def branch():
                qml.RX(np.pi, wires=0)

            branch()
            return qml.state()

        compiled_circuit = qjit(interpreted_circuit)
        assert np.allclose(compiled_circuit(0), interpreted_circuit(0))
        assert np.allclose(compiled_circuit(1), interpreted_circuit(1))


class TestClassicalCompilation:
    """Test suite for the Catalyst Cond functionality outside of QNode contexts."""

    @pytest.mark.parametrize("x,y,op", [(1, 1, 0), (1, 1, 1)])
    def test_conditional(self, x, y, op):
        """Test basic conditional in classical context."""

        @qjit
        def arithc(x: int, y: int, op: int):
            @cond(op == 0)
            def branch():
                return x - y

            @branch.otherwise
            def branch():
                return x + y

            return branch()

        assert arithc.mlir

        def arithi(x, y, op):
            if op == 0:
                return x - y
            else:
                return x + y

        assert arithi(x, y, op) == arithc(x, y, op)

    @pytest.mark.parametrize(
        "x,y,op1,op2", [(2, 2, 0, 0), (2, 2, 1, 0), (2, 2, 0, 1), (2, 2, 1, 1)]
    )
    def test_nested_conditional(self, x, y, op1, op2):
        """Test nested conditional in classical context."""

        @qjit
        def arithc(x: int, y: int, op1: int, op2: int):
            @cond(op1 == 0)
            def branch():
                @cond(op2 == 0)
                def branch2():
                    return x - y

                @branch2.otherwise
                def branch2():
                    return x + y

                return branch2()

            @branch.otherwise
            def branch():
                @cond(op2 == 0)
                def branch3():
                    return x * y

                @branch3.otherwise
                def branch3():
                    return x // y

                return branch3()

            return branch()

        assert arithc.mlir

        def arithi(x, y, op1, op2):
            if op1 == 0:
                if op2 == 0:
                    return x - y
                else:
                    return x + y
            else:
                if op2 == 0:
                    return x * y
                else:
                    return x // y

        assert arithi(x, y, op1, op2) == arithc(x, y, op1, op2)

    def test_no_true_false_parameters(self):
        """Test non-empty parameter detection in conditionals"""

        def arithc2():
            @cond(True)
            def branch(_):
                return 1

            @branch.otherwise
            def branch():
                return 0

            return branch()

        with pytest.raises(TypeError, match="Conditional 'True'"):
            qjit(arithc2)

        def arithc1():
            @cond(True)
            def branch():
                return 1

            @branch.otherwise
            def branch(_):
                return 0

            return branch()  # pylint: disable=no-value-for-parameter

        with pytest.raises(TypeError, match="Conditional 'False'"):
            qjit(arithc1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
