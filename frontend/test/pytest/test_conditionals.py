# Copyright 2022-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-lines

from textwrap import dedent

import re
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from pennylane import cond

import catalyst
from catalyst import api_extensions
from catalyst import cond as catalyst_cond
from catalyst import measure as catalyst_measure
from catalyst import qjit
from catalyst.utils.exceptions import PlxprCaptureCFCompatibilityError

# pylint: disable=missing-function-docstring
pytestmark = pytest.mark.usefixtures("use_both_frontend")


def measure(*args, **kwargs):
    if qml.capture.enabled():
        return qml.measure(*args, **kwargs)
    return catalyst_measure(*args, **kwargs)


class TestCondToJaxpr:
    """Run tests on the generated JAXPR of conditionals."""

    def test_basic_cond_to_jaxpr(self):
        """Check the JAXPR of simple conditional function."""
        # pylint: disable=line-too-long

        expected = dedent(
            """
            { lambda ; a:i64[]. let
                b:bool[] = eq a 5:i64[]
                c:i64[] = cond[
                  branch_jaxprs=[{ lambda ; a:i64[] b:i64[]. let c:i64[] = integer_pow[y=2] a in (c,) },
                                 { lambda ; a:i64[] b:i64[]. let c:i64[] = integer_pow[y=3] b in (c,) }]
                  num_implicit_outputs=0
                ] b a a
              in (c,) }
            """
        )

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
            return " ".join(map(lambda x: re.sub(r"\033\[[0-9;]*m", "", x).strip(), str(text).split("\n"))).strip()

        result = circuit.jaxpr
        assert asline(expected) == asline(result)


# pylint: disable=too-many-public-methods,too-many-lines
class TestCond:
    """Test suite for the Cond functionality in Catalyst."""

    def test_simple_cond(self, backend):
        """Test basic function with conditional."""

        if qml.capture.enabled():
            pytest.xfail("Capture does not support returning classical values from qnodes")

        @qjit
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

        if qml.capture.enabled():
            pytest.xfail("Capture does not support returning classical values from qnodes")

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

        if qml.capture.enabled():
            pytest.xfail("Capture does not support returning classical values from qnodes")

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

        if qml.capture.enabled():
            pytest.xfail("Capture does not support returning mcms")

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            @cond(x > 4)
            def cond_fn():
                qml.PauliX(wires=0)

            cond_fn()
            return measure(wires=0)

        assert not circuit(3)
        assert circuit(6)

    def test_branch_return_pytree_mismatch(self):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """

        if qml.capture.enabled():
            pytest.xfail("We forgot about this case and will fix it in pl-core.")  # [sc-97385]

        def circuit():
            @cond(True)
            def cond_fn():
                return (1, 1)

            @cond_fn.otherwise
            def cond_fn():
                return [2, 2]

            return cond_fn()

        with pytest.raises(
            TypeError,
            match="Control flow requires a consistent return structure across all branches",
        ):
            qjit(circuit)

    def test_branch_return_no_else(self, backend):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """

        def circuit(pred: bool):
            @cond(pred)
            def cond_fn():
                return measure(wires=0)

            return cond_fn()

        if qml.capture.enabled():
            with pytest.raises(
                ValueError, match="false branch must be provided if the true branch"
            ):
                qjit(qml.qnode(qml.device(backend, wires=1))(circuit))
        else:
            with pytest.raises(
                TypeError,
                match="Control flow requires a consistent return structure across all branches",
            ):
                qjit(qml.qnode(qml.device(backend, wires=1))(circuit))

    def test_branch_return_shape_mismatch_classical(self):
        """Test that an exception is raised when the array shapes across branches don't match."""

        def circuit(x: bool):
            @cond(x)
            def cond_fn():
                return True

            @cond_fn.otherwise
            def cond_fn():
                return jnp.array([True, False])

            return cond_fn()

        if qml.capture.enabled():
            # [sc-97387] improve error message
            with pytest.raises(
                ValueError,
                match=r"argument 2 is shorter than argument 1",
            ):
                qjit(circuit)
        else:
            m = "Control flow requires a consistent array shape per result across all branches"
            with pytest.raises(
                TypeError,
                match=m,
            ):
                qjit(circuit)

    def test_branch_return_shape_mismatch_quantum(self, backend):
        """Test that an exception is raised when the array shapes across branches don't match."""

        def circuit(pred: bool):
            @cond(pred)
            def cond_fn():
                return measure(wires=0)

            @cond_fn.otherwise
            def cond_fn():
                return jnp.array([True, False])

            return cond_fn()

        if qml.capture.enabled():
            # [sc-97387] improve error message
            with pytest.raises(
                ValueError,
                match="Mismatch in output abstract values in false branch",
            ):
                qjit(qml.qnode(qml.device(backend, wires=1))(circuit))
        else:
            m = "Control flow requires a consistent array shape per result across all branches"
            with pytest.raises(
                TypeError,
                match=m,
            ):
                qjit(qml.qnode(qml.device(backend, wires=1))(circuit))

    def test_branch_multi_return_type_unification_qnode_1(self, backend):
        """Test that an exception is not raised when the return types of all branches do not match
        but could be unified."""

        if qml.capture.enabled():
            pytest.xfail("capture does not allow returning mcm's or classical values")

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

    def test_branch_multi_return_type_unification_qjit(self):
        """Test that unification happens before the results of the cond primitve is available."""

        if qml.capture.enabled():
            pytest.xfail("capture requires same dtype across all branches")  # [sc-97050]

        @qjit
        def circuit():
            @cond(True)
            def cond_fn():
                return 0

            @cond_fn.otherwise
            def cond_else():
                return True

            r = cond_fn()
            assert r.dtype is jnp.dtype("int")  # pylint: disable=no-member
            return r

        assert 0 == circuit()

    def test_branch_multi_return_type_unification_qjit_2(self):
        """Test that unification happens before the results of the cond primitve is available."""

        if qml.capture.enabled():
            pytest.xfail("capture requires same dtype across all branches")  # [sc-97050]

        @qjit
        def circuit(cond1, cond2):
            @cond(cond1)
            def cond_fn():
                return False

            @cond_fn.else_if(cond2)
            def cond_fn_2():
                return 0.5

            @cond_fn.otherwise
            def cond_fn_3():
                return False

            r = cond_fn()
            assert r.dtype is jnp.dtype(  # pylint: disable=no-member
                "float64" if jax.config.values["jax_enable_x64"] else "float32"
            )
            return r

        assert 0.5 == circuit(False, True)

    def test_branch_multi_return_type_unification_qjit_3(self):
        """Test that unification happens before the results of the cond primitve is available."""

        if qml.capture.enabled():
            pytest.xfail("capture requires same dtype across all branches")  # [sc-97050]

        @qjit
        def circuit(cond1, cond2):
            @cond(cond1)
            def cond_fn():
                return False

            @cond_fn.else_if(cond2)
            def cond_fn_2():
                return False

            @cond_fn.otherwise
            def cond_fn_3():
                return 0.5

            r = cond_fn()
            assert r.dtype is jnp.dtype(  # pylint: disable=no-member
                "float64" if jax.config.values["jax_enable_x64"] else "float32"
            )
            return r

        assert 0.0 == circuit(False, True)

    def test_branch_multi_return_type_unification_qjit_4(self):
        """Test that unification happens before the results of the cond primitve is available."""

        if qml.capture.enabled():
            pytest.xfail("capture requires same dtype across all branches")  # [sc-97050]

        @qjit
        def circuit(cond1, cond2):
            @cond(cond1)
            def cond_fn():
                return {0: True, 1: 0.5}

            @cond_fn.else_if(cond2)
            def cond_fn_2():
                return {0: 0.7, 1: True}

            @cond_fn.otherwise
            def cond_fn_3():
                return {0: True, 1: False}

            r = cond_fn()
            expected_dtype = jnp.dtype(
                "float64" if jax.config.values["jax_enable_x64"] else "float32"
            )
            assert all(v.dtype is expected_dtype for _, v in r.items())  # pylint: disable=no-member
            return r

        assert {0: 0.7, 1: 1.0} == circuit(False, True)

    def test_qnode_cond_inconsistent_return_types(self, backend):
        """Test that catalyst raises an error when the conditional has inconsistent return types."""

        @qjit
        @qml.qnode(qml.device(backend, wires=4))
        def f(flag, sz):
            a = jnp.ones([sz], dtype=float)
            b = jnp.zeros([3], dtype=float)

            @cond(flag)
            def case():
                return a

            @case.otherwise
            def case():
                return b

            c = case()
            return c

        if qml.capture.enabled():
            with pytest.raises(ValueError, match="Mismatch in number of output variables"):
                f(True, 3)
        else:
            with pytest.raises(
                TypeError,
                match="Control flow requires a consistent number of results across all branches",
            ):
                f(True, 3)

    def test_branch_multi_return_type_unification_qnode_2(self, backend):
        """Test that unification happens before the results of the cond primitive is available.
        See the FIXME in the ``CondCallable._call_with_quantum_ctx`` function.
        """
        if qml.capture.enabled():
            pytest.xfail(reason="unification not working with capture")

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            @cond(True)
            def cond_fn():
                return 0

            @cond_fn.otherwise
            def cond_else():
                return True

            r = cond_fn()
            assert r.dtype is jnp.dtype("int")  # pylint: disable=no-member
            return r

        assert 0 == circuit()

    def test_branch_return_mismatch_classical(self):
        """Test that an exception is raised when the true branch returns a different pytree-shape
        than the else branch, given a classical tracing context (no QNode).
        """

        def circuit(pred: bool):
            @cond(pred)
            def cond_fn():
                return (1, 1)

            @cond_fn.otherwise
            def cond_fn():
                return 2

            return cond_fn()

        if qml.capture.enabled():
            with pytest.raises(ValueError, match="Mismatch in number of output variables"):
                qjit(circuit)
        else:
            with pytest.raises(
                TypeError,
                match="Control flow requires a consistent return structure across all branches",
            ):
                qjit(circuit)

    def test_branch_return_promotion_classical(self):
        """Test that an exception is raised when the true branch returns a different type than the
        else branch, given a classical tracing context (no QNode).
        """
        if qml.capture.enabled():
            pytest.xfail("capture requires matching dtypes.")

        def circuit():
            @cond(True)
            def cond_fn():
                return 1

            @cond_fn.otherwise
            def cond_fn():
                return 2.0

            return cond_fn()

        assert 1.0 == qjit(circuit)()

    def test_branch_with_arg(self, backend):
        """Test that we support conditional functions with arguments."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(pred: bool, phi: float):
            @cond(pred)
            def conditional(x):
                qml.RY(x, 0)

            @conditional.otherwise
            def conditional(x):
                qml.RY(x, 1)

            conditional(phi)

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        assert circuit(True, np.pi) == (-1.0, 1.0)
        assert circuit(False, np.pi) == (1.0, -1.0)

    def test_return_type_errors_with_callables(self):
        """Test for errors when branches have mismatched return behaviour."""

        def f(x: int):
            res = qml.cond(x < 5, lambda z: z + 1)(0)

            return res

        if qml.capture.enabled():
            with pytest.raises(ValueError, match="false branch must be provided"):
                qjit(f)
        else:
            with pytest.raises(TypeError, match="requires a consistent return structure"):
                qjit(f)

        def g(x: int):
            res = qml.cond(x < 5, qml.Hadamard, lambda z: z + 1)(0)

            return res

        if qml.capture.enabled():
            with pytest.raises(ValueError, match="Mismatch in output abstract values"):
                qjit(g)
        else:
            with pytest.raises(
                TypeError, match="requires a consistent return structure across all branches"
            ):
                qjit(g)

        def h(x: int):
            res = qml.cond(x < 5, qml.Hadamard, qml.Hadamard, ((x < 6, lambda z: z + 1),))(0)

            return res

        if qml.capture.enabled():
            with pytest.raises(ValueError, match="Mismatch in output abstract values"):
                qjit(h)
        else:
            with pytest.raises(
                TypeError, match="requires a consistent return structure across all branches"
            ):
                qjit(h)

    @pytest.mark.usefixtures("disable_capture")
    def test_cond_raises_compatibility_error_with_capture(self):
        """Test that cond raises PlxprCaptureCFCompatibilityError when capture mode is enabled."""
        if not qml.capture.enabled():
            pytest.skip("capture only test")

        qml.capture.enable()

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @catalyst_cond(True)
            def cond_fn():
                return 1

        # Verify the error message is specific and helpful
        error_msg = str(exc_info.value)
        assert "catalyst.cond is not supported with PennyLane's capture enabled" in error_msg

    @pytest.mark.usefixtures("disable_capture")
    def test_cond_raises_compatibility_error_with_capture_integration(self):
        """Test that cond raises PlxprCaptureCFCompatibilityError when capture mode is enabled."""
        if not qml.capture.enabled():
            pytest.skip("capture only test")

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @qml.qjit
            @qml.qnode(qml.device("lightning.qubit", wires=3))
            def test(n):
                @catalyst_cond(n < 5)
                def loop(n):
                    qml.X(n)

                loop()  # pylint: disable=no-value-for-parameter

            test(4)

        # Verify the error message is specific and helpful
        error_msg = str(exc_info.value)
        assert "catalyst.cond is not supported with PennyLane's capture enabled" in error_msg


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

        def arithc2(pred: bool):
            @cond(pred)
            def branch(_):
                return 1

            @branch.otherwise
            def branch():
                return 0

            return branch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qjit(arithc2)

        def arithc1(pred: bool):
            @cond(pred)
            def branch():
                return 1

            @branch.otherwise
            def branch(_):
                return 0

            return branch()  # pylint: disable=no-value-for-parameter

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qjit(arithc1)


class TestCondOperatorAccess:
    """Test suite for accessing the Cond operation in quantum contexts in Catalyst."""

    def test_cond_access_quantum(self, backend):
        """Test Cond operation access in quantum context."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(n):
            @cond(n > 4)
            def cond_fn():
                qml.PauliZ(0)
                return 1

            @cond_fn.otherwise
            def else_fn():
                qml.PauliX(0)
                return 0

            cond_fn()
            if not qml.capture.enabled():
                assert isinstance(cond_fn.operation, api_extensions.control_flow.Cond)

            return qml.probs()

        assert circuit(2)[0] == 0
        assert circuit(2)[1] == 1
        assert circuit(5)[0] == 1
        assert circuit(5)[1] == 0

    def test_cond_access_classical(self):
        """Test Cond operation access in classical context."""

        c = cond if qml.capture.enabled() else catalyst.cond

        @qjit
        def circuit(x):
            @c(x > 4.8)
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

            cond_fn()
            if not qml.capture.enabled():
                with pytest.raises(
                    AttributeError,
                    match=r"""and thus has no associated quantum operation.""",
                ):
                    isinstance(cond_fn.operation, api_extensions.control_flow.Cond)

            return cond_fn()

        assert circuit(5) == 80
        assert circuit(3) == 24
        assert circuit(2) == 8
        assert circuit(-3) == -3

    def test_cond_access_interpreted(self):
        """Test Cond operation access in interpreted context."""

        c = cond if qml.capture.enabled() else catalyst.cond

        def func(flag: bool):
            @c(flag)
            def branch_t():
                return 1

            @branch_t.otherwise
            def branch_f():
                return 0

            branch_t()
            if not qml.capture.enabled():
                with pytest.raises(
                    AttributeError,
                    match=r"""and thus has no associated quantum operation.""",
                ):
                    isinstance(branch_t.operation, api_extensions.control_flow.Cond)

            return branch_t()

        assert func(True) == 1
        assert func(False) == 0

    def test_cond_single_gate(self, backend):
        """
        Test standard pennylane qml.cond usage on single quantum gates.
        Fixes https://github.com/PennyLaneAI/catalyst/issues/449
        """

        @qml.qnode(qml.device(backend, wires=2))
        def func(x, y):
            qml.cond(x == 42, qml.Hadamard, qml.PauliX)(wires=0)
            qml.cond(x == 42, qml.RY, qml.RZ)(1.5, wires=0)
            qml.cond(x == 42, qml.CNOT)(wires=[1, 0])
            qml.cond(y == 37, qml.PauliX)(wires=1)
            qml.cond(
                y == 36,
                qml.RZ,
                qml.RY,
                (
                    (x == 42, qml.RX),
                    (x == 41, qml.RZ),
                ),
            )(5.1, wires=0)
            qml.cond(y == 37, qml.Rot)(1.2, 3.4, 5.6, wires=1)

            return qml.probs()

        expected_0 = func(42, 37)
        expected_1 = func(0, 37)
        expected_2 = func(42, 0)
        expected_3 = func(41, 0)

        jitted_func = qjit(func)

        observed_0 = jitted_func(42, 37)
        observed_1 = jitted_func(0, 37)
        observed_2 = jitted_func(42, 0)
        observed_3 = jitted_func(41, 0)

        assert np.allclose(expected_0, observed_0)
        assert np.allclose(expected_1, observed_1)
        assert np.allclose(expected_2, observed_2)
        assert np.allclose(expected_3, observed_3)


class TestCondPredicateConversion:
    """Test suite for checking predicate conversion to bool."""

    def test_conversion_integer(self):
        """Test entry predicate conversion from integer to bool."""

        @qjit
        def workflow(x):
            n = 1

            # n is an integer but it gets converted to bool
            @cond(n)
            def cond_fn():
                return x**2

            @cond_fn.otherwise
            def else_fn():
                return x

            return cond_fn()

        assert workflow(3) == 9

    def test_conversion_float(self):
        """Test entry predicate conversion from float to bool."""

        @qjit
        def workflow(x):
            n = 2.0

            # n is a float but it gets converted to bool
            @cond(n)
            def cond_fn():
                return x**2

            @cond_fn.otherwise
            def else_fn():
                return x

            return cond_fn()

        assert workflow(3) == 9

    def test_jax_bool(self):
        """Test entry predicate with a JAX bool."""

        @qjit
        def workflow(x):
            n = jnp.bool_(True)

            # n is a JAX bool and does not need conversion
            @cond(n)
            def cond_fn():
                return x**2

            @cond_fn.otherwise
            def else_fn():
                return x

            return cond_fn()

        assert workflow(3) == 9

    def test_else_if_conversion_integer(self):
        """Test else_if predicate conversion from integer to bool."""

        @qjit
        def workflow(x):
            n = 1

            @cond(n < 0)
            def cond_fn():
                return -x

            # n is an integer but it gets converted to bool
            @cond_fn.else_if(n)
            def else_if_fn():
                return x**2

            @cond_fn.otherwise
            def else_fn():
                return x

            return cond_fn()

        assert workflow(3) == 9

    def test_conversion_int_autograph(self):
        """Test entry predicate conversion from integer to bool using Autograph."""

        @qjit(autograph=True)
        def workflow(x):
            n = 1

            if n:
                y = x**2
            else:
                y = x

            return y

        assert workflow(3) == 9

    def test_conversion_int_autograph_elif(self):
        """Test elif predicate conversion from integer to bool using Autograph."""

        @qjit(autograph=True)
        def workflow(x):
            n = 1

            if n < 0:
                y = -x
            elif n:
                y = x**2
            else:
                y = 0

            return y

        assert workflow(3) == 9

    def test_string_conversion_failed(self):
        """Test failure at converting string to bool using Autograph."""

        if qml.capture.enabled():
            pytest.skip("works with program capture.")

        @qjit(autograph=True)
        def workflow(x):
            n = "fail"

            if n:
                y = x**2
            else:
                y = 0

            return y

        with pytest.raises(
            TypeError,
            match="Conditional predicates are required to be of bool, integer or float type",
        ):
            workflow(3)

    def test_string_conversion_capture_works(self):
        """Test that truthy values in conditionals work when capture is enabled."""

        if not qml.capture.enabled():
            pytest.skip("only works with program capture.")

        @qjit(autograph=True)
        def workflow(x):
            n = "fail"

            if n:
                y = x**2
            else:
                y = 0

            return y

        out = workflow(0.5)
        assert qml.math.allclose(out, 0.25)

    def test_array_conversion_failed(self):
        """Test failure at converting array to bool using Autograph."""

        @qjit(autograph=True)
        def workflow(x):
            n = jnp.array([[1], [2]])

            if n:
                y = x**2
            else:
                y = 0

            return y

        if qml.capture.enabled():
            with pytest.raises(ValueError, match="Condition predicate must be a scalar"):
                workflow(3)
        else:
            with pytest.raises(
                TypeError, match="Array with multiple elements is not a valid predicate"
            ):
                workflow(3)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
