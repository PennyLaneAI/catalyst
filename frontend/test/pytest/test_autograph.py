# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTests for the AutoGraph source-to-source transformation feature."""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import for_loop, measure, qjit
from catalyst.ag_utils import AutoGraphError, autograph_source, check_cache

# pylint: disable=missing-function-docstring
# pylint: disable=unnecessary-lambda-assignment


def test_unavailable(mocker):
    """Check the error produced in the absence of tensorflow."""
    mocker.patch.dict("sys.modules", {"tensorflow": None})

    def fn(x):
        return x**2

    with pytest.raises(ImportError, match="AutoGraph feature in Catalyst requires TensorFlow"):
        qjit(autograph=True)(fn)


# Keep this class first due to hardcoded line numbers.
@pytest.mark.tf
class TestSourceCodeInfo:
    """Unit tests for exception utilities that retrieves traceback information for the original
    source code."""

    def test_qjit(self):
        """Test source info retrieval for a qjit function."""

        def main():
            for _ in range(5):
                raise RuntimeError("Test failure")
            return 0

        with pytest.warns(
            UserWarning,
            match=(
                f'  File "{__file__}", line {"51"}, in {main.__name__}\n'
                r"    for _ in range\(5\):"
            ),
        ):
            try:
                qjit(autograph=True)(main)
            except RuntimeError as e:
                assert e.args == ("Test failure",)

    def test_qnode(self):
        """Test source info retrieval for a qnode function."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def main():
            for _ in range(5):
                raise RuntimeError("Test failure")
            return 0

        with pytest.warns(
            UserWarning,
            match=(
                f'  File "{__file__}", line {"72"}, in {main.__name__}\n'
                r"    for _ in range\(5\):"
            ),
        ):
            try:
                qjit(autograph=True)(main)
            except RuntimeError as e:
                assert e.args == ("Test failure",)

    def test_func(self):
        """Test source info retrieval for a nested function."""

        def inner():
            for _ in range(5):
                raise RuntimeError("Test failure")

        def main():
            inner()
            return 0

        with pytest.warns(
            UserWarning,
            match=(
                f'  File "{__file__}", line {"92"}, in {inner.__name__}\n'
                r"    for _ in range\(5\):"
            ),
        ):
            try:
                qjit(autograph=True)(main)
            except RuntimeError as e:
                assert e.args == ("Test failure",)


@pytest.mark.tf
class TestIntegration:
    """Test that the autograph transformations trigger correctly in different settings."""

    def test_unsupported_object(self):
        """Check the error produced when attempting to convert an unsupported object (neither of
        QNode, function, or method)."""

        class FN:
            """Test object."""

            def __call__(self, x):
                return x**2

        fn = FN()

        with pytest.raises(AutoGraphError, match="Unsupported object for transformation"):
            qjit(autograph=True)(fn)

    def test_lambda(self):
        """Test autograph on a lambda function."""

        fn = lambda x: x**2
        fn = qjit(autograph=True)(fn)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert fn(4) == 16

    def test_classical_function(self):
        """Test autograph on a purely classical function."""

        @qjit(autograph=True)
        def fn(x):
            return x**2

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert fn(4) == 16

    def test_nested_function(self):
        """Test autograph on nested classical functions."""

        def inner(x):
            return x**2

        @qjit(autograph=True)
        def fn(x: int):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner)
        assert fn(4) == 16

    def test_qnode(self):
        """Test autograph on a QNode."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function.func)
        assert fn(np.pi) == -1

    def test_indirect_qnode(self):
        """Test autograph on a QNode called from within a classical function."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner.func)
        assert fn(np.pi) == -1

    def test_multiple_qnode(self):
        """Test autograph on multiple QNodes called from different classical functions."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner1(x) + inner2(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner1.func)
        assert check_cache(inner2.func)
        assert fn(np.pi) == -2

    def test_nested_qnode(self):
        """Test autograph on a QNode called from within another QNode."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            y = inner1(x) * np.pi
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: int):
            return inner2(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner1.func)
        assert check_cache(inner2.func)
        # Unsupported by the runtime:
        # assert fn(np.pi) == -2

    def test_nested_qjit(self):
        """Test autograph on a QJIT function called from within the compilation entry point."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner.user_function.func)
        assert fn(np.pi) == -1


@pytest.mark.tf
class TestCodePrinting:
    """Test that the transformed source code can be printed in different settings."""

    def test_unconverted(self):
        """Test printing on an unconverted function."""

        @qjit(autograph=False)
        def fn(x):
            return x**2

        with pytest.raises(AutoGraphError, match="function was not converted by AutoGraph"):
            autograph_source(fn)

    def test_lambda(self):
        """Test printing on a lambda function."""

        fn = lambda x: x**2
        qjit(autograph=True)(fn)

        assert autograph_source(fn)

    def test_classical_function(self):
        """Test printing on a purely classical function."""

        @qjit(autograph=True)
        def fn(x):
            return x**2

        assert autograph_source(fn)

    def test_nested_function(self):
        """Test printing on nested classical functions."""

        def inner(x):
            return x**2

        @qjit(autograph=True)
        def fn(x: int):
            return inner(x)

        assert autograph_source(fn)
        assert autograph_source(inner)

    def test_qnode(self):
        """Test printing on a QNode."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert autograph_source(fn)

    def test_indirect_qnode(self):
        """Test printing on a QNode called from within a classical function."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert autograph_source(fn)
        assert autograph_source(inner)

    def test_multiple_qnode(self):
        """Test printing on multiple QNodes called from different classical functions."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner1(x) + inner2(x)

        assert autograph_source(fn)
        assert autograph_source(inner1)
        assert autograph_source(inner2)

    def test_nested_qnode(self):
        """Test printing on a QNode called from within another QNode."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            y = inner1(x) * np.pi
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: int):
            return inner2(x)

        assert autograph_source(fn)
        assert autograph_source(inner1)
        assert autograph_source(inner2)

    def test_nested_qjit(self):
        """Test printing on a QJIT function called from within the compilation entry point."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert autograph_source(fn)
        assert autograph_source(inner)


@pytest.mark.tf
class TestConditionals:
    """Test that the autograph transformations produce correct results on conditionals.
    These tests are adapted from the test_conditionals.TestCond class of tests."""

    def test_simple_cond(self):
        """Test basic function with conditional."""

        @qjit(autograph=True)
        def circuit(n):
            if n > 4:
                res = n**2
            else:
                res = n

            return res

        assert circuit(0) == 0
        assert circuit(1) == 1
        assert circuit(2) == 2
        assert circuit(3) == 3
        assert circuit(4) == 4
        assert circuit(5) == 25
        assert circuit(6) == 36

    def test_cond_one_else_if(self):
        """Test a cond with one else_if branch"""

        @qjit(autograph=True)
        def circuit(x):
            if x > 2.7:
                res = x * 4
            elif x > 1.4:
                res = x * 2
            else:
                res = x

            return res

        assert circuit(4) == 16
        assert circuit(2) == 4
        assert circuit(1) == 1

    def test_cond_many_else_if(self):
        """Test a cond with multiple else_if branches"""

        @qjit(autograph=True)
        def circuit(x):
            if x > 4.8:
                res = x * 8
            elif x > 2.7:
                res = x * 4
            elif x > 1.4:
                res = x * 2
            else:
                res = x

            return res

        assert circuit(5) == 40
        assert circuit(3) == 12
        assert circuit(2) == 4
        assert circuit(-3) == -3

    def test_qubit_manipulation_cond(self, backend):
        """Test conditional with quantum operation."""

        @qjit(autograph=True)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            if x > 4:
                qml.PauliX(wires=0)

            return measure(wires=0)

        assert circuit(3) == False
        assert circuit(6) == True

    def test_branch_return_mismatch(self, backend):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """

        def circuit():
            if True:
                res = measure(wires=0)

            return res

        with pytest.raises(
            AutoGraphError, match="Some branches did not define a value for variable 'res'"
        ):
            qjit(autograph=True)(qml.qnode(qml.device(backend, wires=1))(circuit))

    def test_branch_multi_return_mismatch(self, backend):
        """Test that an exception is raised when the return types of all branches do not match."""

        def circuit():
            if True:
                res = measure(wires=0)
            elif False:
                res = 0
            else:
                res = measure(wires=0)

            return res

        with pytest.raises(
            TypeError, match="Conditional requires consistent return types across all branches"
        ):
            qjit(autograph=True)(qml.qnode(qml.device(backend, wires=1))(circuit))


class TestForLoops:
    """Test that the autograph transformations produce correct results on for loops."""

    def test_for_in_array(self):
        """Test for loop over JAX array."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(params):
            for x in params:
                qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f(jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]))
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    def test_for_in_array_unpack(self):
        """Test for loop over a 2D JAX array unpacking the inner dimension."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(params):
            for x1, x2 in params:
                qml.RY(x1, wires=0)
                qml.RY(x2, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f(jnp.array([[0.0, 1 / 4 * jnp.pi], [2 / 4 * jnp.pi, jnp.pi]]))
        assert np.allclose(result, jnp.sqrt(2) / 2)

    def test_for_in_numeric_list(self):
        """Test for loop over a Python list that is convertible to an array."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for x in params:
                qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f()
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    def test_for_in_numeric_list_of_list(self):
        """Test for loop over a nested Python list that is convertible to an array."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = [[0.0, 1 / 4 * jnp.pi], [2 / 4 * jnp.pi, jnp.pi]]
            for xx in params:
                for x in xx:
                    qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f()
        assert np.allclose(result, jnp.sqrt(2) / 2)

    def test_for_in_object_list(self):
        """Test for loop over a Python list that is *not* convertible to an array.
        The behaviour should fall back to standard Python."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = ["0", "1", "2"]
            for x in params:
                qml.RY(int(x) / 4 * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f()
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    def test_for_in_static_range(self):
        """Test for loop over a Python range with static bounds."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f():
            for i in range(3):
                qml.Hadamard(i)
            return qml.probs()

        result = f()
        assert np.allclose(result, [1 / 8] * 8)

    def test_for_in_static_range_indexing_array(self):
        """Test for loop over a Python range with static bounds that is used to index an array."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
            for i in range(3):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f()
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    # With conversion always taking place, the user needs to be careful to manually wrap
    # objects accessed via loop iteration indices into arrays (see test case above).
    # The warning here is actionable.
    def test_for_in_static_range_indexing_numeric_list(self):
        """Test for loop over a Python range with static bounds that is used to index an
        array-compatible Python list. This should fall back to Python with a warning."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i in range(3):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            qjit(autograph=True)(f)

    # This case is slightly problematic because there is no way for the user to compile this for
    # loop correctly. Fallback to a Python loop is always necessary, and will result in a warning.
    # The warning here is not actionable.
    def test_for_in_static_range_indexing_object_list(self):
        """Test for loop over a Python range with static bounds that is used to index an
        array-incompatible Python list. This should fall back to Python with a warning."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = ["0", "1", "2"]
            for i in range(3):
                qml.RY(int(params[i]) / 4 * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            qjit(autograph=True)(f)

    def test_for_in_dynamic_range(self):
        """Test for loop over a Python range with dynamic bounds."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f(n: int):
            for i in range(n):
                qml.Hadamard(i)
            return qml.probs()

        result = f(3)
        assert np.allclose(result, [1 / 8] * 8)

    def test_for_in_dynamic_range_indexing_array(self):
        """Test for loop over a Python range with dynamic bounds that is used to index an array."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(n: int):
            params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
            for i in range(n):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f(3)
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    # This case will fail even without autograph conversion, since dynamic iteration bounds are not
    # allowed in Python ranges. Here, AutoGraph improves the situation by allowing this test case
    # with a slight modification of the user code (see test case above).
    # Raising the warning is vital here to notify the user that this use case is actually supported,
    # but requires a modification. Without it, the user may simply conclude it is unsupported.
    def test_for_in_dynamic_range_indexing_numeric_list(self):
        """Test for loop over a Python range with dynamic bounds that is used to index an
        array-compatible Python list. The fallback to Python will first raise a warning,
        then an error."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(n: int):
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i in range(n):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            with pytest.raises(jax.errors.TracerIntegerConversionError, match="__index__"):
                qjit(autograph=True)(f)

    # This use case is never possible, regardless of whether AutoGraph is used or not.
    def test_for_in_dynamic_range_indexing_object_list(self):
        """Test for loop over a Python range with dynamic bounds that is used to index an
        array-incompatible Python list. The fallback to Python will first raise a warning,
        then an error."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(n: int):
            params = ["0", "1", "2"]
            for i in range(n):
                qml.RY(int(params[i]) * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            with pytest.raises(jax.errors.TracerIntegerConversionError, match="__index__"):
                qjit(autograph=True)(f)

    def test_for_in_enumerate_array(self):
        """Test for loop over a Python enumeration on an array."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f(params):
            for i, x in enumerate(params):
                qml.RY(x, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        result = f(jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]))
        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_array_no_unpack(self):
        """Test for loop over a Python enumeration with delayed unpacking."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f(params):
            for v in enumerate(params):
                qml.RY(v[1], wires=v[0])
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        result = f(jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]))
        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_nested_unpack(self):
        """Test for loop over a Python enumeration with nested unpacking."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f(params):
            for i, (x1, x2) in enumerate(params):
                qml.RY(x1, wires=i)
                qml.RY(x2, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        result = f(
            jnp.array(
                [[0.0, 1 / 4 * jnp.pi], [2 / 4 * jnp.pi, 3 / 4 * jnp.pi], [jnp.pi, 2 * jnp.pi]]
            )
        )
        assert np.allclose(result, [jnp.sqrt(2) / 2, -jnp.sqrt(2) / 2, -1.0])

    def test_for_in_enumerate_start(self):
        """Test for loop over a Python enumeration with offset indices."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=5))
        def f(params):
            for i, x in enumerate(params, start=2):
                qml.RY(x, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(5)]

        result = f(jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]))
        assert np.allclose(result, [1.0, 1.0, 1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_numeric_list(self):
        """Test for loop over a Python enumeration on a list that is convertible to an array."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i, x in enumerate(params):
                qml.RY(x, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        result = f()
        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_object_list(self):
        """Test for loop over a Python enumeration on a list that is *not* convertible to an array.
        The behaviour should fall back to standard Python."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def f():
            params = ["0", "1", "2"]
            for i, x in enumerate(params):
                qml.RY(int(x) / 4 * jnp.pi, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        result = f()
        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_other_iterable_object(self):
        """Test for loop over arbitrary iterable Python objects.
        The behaviour should fall back to standard Python."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f():
            params = {"a": 0.0, "b": 1 / 4 * jnp.pi, "c": 2 / 4 * jnp.pi}
            for k, v in params.items():
                print(k)
                qml.RY(v, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = f()
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    def test_no_python_loops(self):
        """Test AutoGraph behaviour on function with Catalyst loops."""

        @qjit(autograph=True)
        def f():
            @for_loop(0, 3, 1)
            def loop(i, sum):
                return sum + i

            return loop(0)

        assert f() == 3


if __name__ == "__main__":
    pytest.main(["-x", __file__])
