# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module tests the from_plxpr conversion function.
"""

import numpy as np
import pennylane as qml
import pytest

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")

# needs to be below the importorskip calls
# pylint: disable=wrong-import-position
from catalyst.from_plxpr import from_plxpr
from catalyst.jax_primitives import get_call_jaxpr


def catalyst_execute_jaxpr(jaxpr):
    """Create a function capable of executing the provided catalyst-variant jaxpr."""

    # pylint: disable=too-few-public-methods
    class JAXPRRunner(catalyst.QJIT):
        """A variant of catalyst.QJIT with a pre-constructed jaxpr."""

        # pylint: disable=missing-function-docstring
        def capture(self, args):

            result_treedef = jax.tree_util.tree_structure((0,) * len(jaxpr.out_avals))
            arg_signature = catalyst.tracing.type_signatures.get_abstract_signature(args)

            return jaxpr, None, result_treedef, arg_signature

    return JAXPRRunner(fn=lambda: None, compile_options=catalyst.CompileOptions())


def compare_call_jaxprs(jaxpr1, jaxpr2, skip_eqns=()):
    """Compares two call jaxprs and validates that they are essentially equal."""
    for inv1, inv2 in zip(jaxpr1.invars, jaxpr2.invars):
        assert inv1.aval == inv2.aval, f"{inv1.aval}, {inv2.aval}"
    for ov1, ov2 in zip(jaxpr1.outvars, jaxpr2.outvars):
        assert ov1.aval == ov2.aval
    assert len(jaxpr1.eqns) == len(jaxpr2.eqns)

    for i, (eqn1, eqn2) in enumerate(zip(jaxpr1.eqns, jaxpr2.eqns)):
        if i not in skip_eqns:
            compare_eqns(eqn1, eqn2)


def compare_eqns(eqn1, eqn2):
    """Compare two jaxpr equations."""
    assert eqn1.primitive == eqn2.primitive
    if "shots" not in eqn1.params and "shape" not in eqn1.params:
        assert eqn1.params == eqn2.params

    assert len(eqn1.invars) == len(eqn2.invars)
    for inv1, inv2 in zip(eqn1.invars, eqn2.invars):
        assert type(inv1) == type(inv2)  # pylint: disable=unidiomatic-typecheck
        assert inv1.aval == inv2.aval, f"{eqn1}, {inv1.aval}, {inv2.aval}"
        if hasattr(inv1, "val"):
            assert inv1.val == inv2.val, f"{eqn1}, {inv1.val}, {inv2.val}"

    assert len(eqn1.outvars) == len(eqn2.outvars)
    for ov1, ov2 in zip(eqn1.outvars, eqn2.outvars):
        assert type(ov1) == type(ov2)  # pylint: disable=unidiomatic-typecheck
        assert ov1.aval == ov2.aval


class TestErrors:
    """Test that errors are raised in unsupported situations."""

    def test_dynamic_shots(self):
        """Test that a NotImplementedError is raised is shots do not match device shots."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(wires=0)

        def f():
            return circuit(shots=1000)

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(f)()

        with pytest.raises(
            NotImplementedError, match="catalyst does not yet support dynamic shots"
        ):
            from_plxpr(jaxpr)()
        qml.capture.disable()

    def test_observable_without_n_wires(self):
        """Test that a NotImplementedError is raised for an observable without n_wires."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.X(0) + qml.Y(0))

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()

        with pytest.raises(
            NotImplementedError, match="operator arithmetic not yet supported for conversion."
        ):
            from_plxpr(jaxpr)()
        qml.capture.disable()

    def test_measuring_eigvals_not_supported(self):
        """Test that a NotImplementedError is raised for converting a measurement
        specified via eigvals and wires."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            return qml.measurements.SampleMP(
                wires=qml.wires.Wires((0, 1)), eigvals=np.array([-1.0, -1.0, 1.0, 1.0])
            )

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()
        with pytest.raises(NotImplementedError, match="does not yet support measurements with"):
            from_plxpr(jaxpr)()
        qml.capture.disable()

    def test_measuring_measurement_values(self):
        """Test that measuring a MeasurementValue raises a NotImplementedError."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.measurements.ExpectationMP(
                obs=2
            )  # classical value like will be used for mcms

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()

        with pytest.raises(NotImplementedError, match=r"not yet supported"):
            from_plxpr(jaxpr)()
        qml.capture.disable()

    def test_unsupported_measurement(self):
        """Test that a NotImplementedError is raised if a measurement
        is not yet supported for conversion."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0)

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()

        with pytest.raises(NotImplementedError, match="not yet supported"):
            from_plxpr(jaxpr)()
        qml.capture.disable()


class TestCatalystCompareJaxpr:
    """Test comparing catalyst and pennylane jaxpr for a variety of situations."""

    def test_qubit_unitary(self):
        """Test that qubit unitary can be converted."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(U):
            qml.QubitUnitary(U, wires=0)
            return qml.expval(qml.Z(0))

        x = qml.X.compute_matrix()
        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x)
        converted = from_plxpr(plxpr)(x)
        qml.capture.disable()

        catalyst_res = catalyst_execute_jaxpr(converted)(x)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], -1)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(x)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)
        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_globalphase(self):
        """Test conversion of a global phase."""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(phi):
            qml.GlobalPhase(phi)
            return qml.state()

        phi = 0.5
        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(phi)
        converted = from_plxpr(plxpr)(phi)
        qml.capture.disable()
        catalyst_res = catalyst_execute_jaxpr(converted)(phi)
        assert qml.math.allclose(catalyst_res, np.exp(-0.5j) * np.array([1.0, 0.0]))

        qjit_obj = qml.qjit(circuit)
        qjit_obj(0.5)

        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)
        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_expval(self):
        """Test comparison and execution of the jaxpr for a simple qnode."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)
        converted = from_plxpr(plxpr)(0.5)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(0.5)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], jax.numpy.cos(0.5))

        qjit_obj = qml.qjit(circuit)
        qjit_obj(0.5)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_probs(self):
        """Test comparison and execution of a jaxpr containing a probability measurement."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)

        converted = from_plxpr(plxpr)(0.5)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(0.5)
        assert len(catalyst_res) == 1
        expected = np.array([np.cos(0.5 / 2) ** 2, np.sin(0.5 / 2) ** 2])
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(0.5)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_state(self):
        """Test that the state can be converted to catalxpr."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.Hadamard(0)
            qml.IsingXX(phi, wires=(0, 1))
            return qml.state()

        phi = np.array(-0.6234)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(phi)

        converted = from_plxpr(plxpr)(phi)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(phi)
        assert len(catalyst_res) == 1

        x1 = np.cos(phi / 2) / np.sqrt(2)
        x2 = -1j * np.sin(phi / 2) / np.sqrt(2)
        expected = np.array([x1, x2, x1, x2])

        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(phi)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)

        # confused by the weak_types error here
        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_variance(self):
        """Test comparison and execution of a jaxpr containing a variance measurement."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.Y(0))

        x = np.array(0.724)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x)

        converted = from_plxpr(plxpr)(np.array(0.724))
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(x)
        assert len(catalyst_res) == 1
        expected = 1 - np.sin(x) ** 2
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(x)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_sample(self):
        """Test comparison and execution of a jaxpr returning samples."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.sample()

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)()

        converted = from_plxpr(plxpr)()
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)()
        assert len(catalyst_res) == 1
        expected = np.transpose(np.vstack([np.ones(50), np.zeros(50)]))
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj()
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    @pytest.mark.xfail(reason="CountsMP returns a dictionary, which is not compatible with capture")
    def test_counts(self):
        """Test comparison and execution of a jaxpr returning counts."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.counts()

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)()
        converted = from_plxpr(plxpr)()
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)()
        assert len(catalyst_res) == 1
        expected = np.transpose(np.vstack([np.ones(50), np.zeros(50)]))
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj()
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[1].params["call_jaxpr"]

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_multiple_measurements(self):
        """Test that we can convert a circuit with multiple measurement returns."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.probs(wires=0)

        x, y, z = 0.9, 0.2, 0.5

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x, y, z)

        converted = from_plxpr(plxpr)(x, y, z)
        qml.capture.disable()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.quantum_kernel_p
        assert converted.eqns[0].params["qnode"] is circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(x, y, z)
        assert len(catalyst_res) == 3

        a = np.cos(y / 2) * np.exp(-0.5j * (x + z))
        b = np.sin(y / 2) * np.exp(-0.5j * (x - z))
        state = np.array([a, b])
        expected_probs = np.abs(state) ** 2
        expected_expval_x = np.conj(state) @ qml.X.compute_matrix() @ state
        expected_expval_y = np.conj(state) @ qml.Y.compute_matrix() @ state
        assert qml.math.allclose(catalyst_res[0], expected_expval_x)
        assert qml.math.allclose(catalyst_res[1], expected_expval_y)
        assert qml.math.allclose(catalyst_res[2], expected_probs)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(x, y, z)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(catalxpr)

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)


class TestHybridPrograms:
    """from_plxpr conversion tests for hybrid programs."""

    def test_pre_post_processing(self):
        """Test converting a workflow with pre and post processing."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(3 * y + 1, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.X(1)), qml.expval(qml.Y(0))

        def workflow(z):
            a, b = circuit(z, 2 * z)
            return a + b

        qml.capture.enable()
        plxpr = jax.make_jaxpr(workflow)(0.5)

        converted = from_plxpr(plxpr)(0.5)
        qml.capture.disable()

        res = catalyst_execute_jaxpr(converted)(0.5)

        x = 0.5
        y = 3 * 2 * 0.5 + 1

        expval_x1 = np.sin(y)
        expval_y0 = -np.sin(x) * np.sin(y)
        expected = expval_x1 + expval_y0

        assert qml.math.allclose(expected, res[0])

        qjit_obj = qml.qjit(workflow)
        qjit_obj(0.5)

        call_jaxpr_pl = get_call_jaxpr(converted)
        call_jaxpr_c = get_call_jaxpr(qjit_obj.jaxpr)

        # qubit extraction and classical equations in a slightly different order
        # thus cant check specific equations and have to discard comparing counts
        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c, skip_eqns=(4, 5, 6))
        compare_eqns(call_jaxpr_pl.eqns[4], call_jaxpr_c.eqns[5])
        compare_eqns(call_jaxpr_pl.eqns[5], call_jaxpr_c.eqns[6])
        compare_eqns(call_jaxpr_pl.eqns[6], call_jaxpr_c.eqns[4])

    def test_multiple_qnodes(self):
        """Test that a workflow with multiple qnodes can be converted."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Y(0))

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def g(y):
            qml.Hadamard(0)
            qml.IsingXX(y, wires=(0, 1))
            return qml.expval(qml.PauliZ(1))

        def workflow(x, y):
            return f(x) + g(y)

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(workflow)(0.5, 1.2)
        catalxpr = from_plxpr(jaxpr)(0.5, 1.2)
        qml.capture.disable()
        results = catalyst_execute_jaxpr(catalxpr)(0.5, 1.2)

        expected = -np.sin(0.5) + np.cos(1.2)

        assert qml.math.allclose(results, expected)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
