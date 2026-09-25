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
"""Tests for executing qjit programs on PennyLane devices implemented in Python, through the
PennyLane Python device bridge (``runtime/lib/backend/pennylane_python``)."""

# pylint: disable=missing-function-docstring,unused-argument

import os
import textwrap

import numpy as np
import pennylane as qp
import pytest
from pennylane.core.operator import Operator2
from pennylane.devices import Device
from pennylane.devices.preprocess import decompose
from pennylane.exceptions import DeviceError
from pennylane.ops import MidMeasure

from catalyst.debug import intercept_runtime_tape
from catalyst.device import python_device
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.capture

qjit = qp.qjit(capture=True, collect_decomp_rules=False)

TOML = """
schema = 3

[operators.gates]
RX = { properties = ["invertible", "controllable", "differentiable"] }
RY = { properties = ["invertible", "controllable", "differentiable"] }
RZ = { properties = ["invertible", "controllable", "differentiable"] }
CNOT = { properties = ["invertible"] }
GlobalPhase = { properties = ["invertible"] }
NotABridgeGate = { }

[operators.observables]
PauliZ = { }

[measurement_processes]
ExpectationMP = { }
SampleMP = { }

[compilation]
qjit_compatible = true
supported_mcm_methods = [ "one-shot" ]
"""

GATE_SET = {"RX", "RY", "RZ", "CNOT", "GlobalPhase"}


@pytest.fixture(scope="module")
def toml_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("python_device") / "my_device.toml"
    path.write_text(textwrap.dedent(TOML))
    return str(path)


@pytest.fixture
def my_device(toml_file):
    """A minimal PennyLane Python device with its own TOML file and preprocessing."""

    class MyDevice(Device):
        """A device that only supports RX, RY, RZ, CNOT and GlobalPhase."""

        name = "my.device"
        config_filepath = toml_file

        def __init__(self, wires=None, decompose_ops=True):
            super().__init__(wires=wires)
            self.decompose_ops = decompose_ops
            self.executed = []

        def preprocess_transforms(self, execution_config=None):
            program = qp.CompilePipeline()
            if self.decompose_ops:
                program.add_transform(
                    decompose, stopping_condition=lambda op: op.name in GATE_SET, name=self.name
                )
            else:

                @qp.transform
                def validate(tape):
                    for op in tape.operations:
                        if op.name not in GATE_SET:
                            raise DeviceError(f"Operator {op.name} is not supported by my.device")
                    return (tape,), lambda r: r[0]

                program.add_transform(validate)
            return program

        def execute(self, circuits, execution_config=None):
            self.executed.extend(circuits)
            return qp.device("default.qubit", wires=self.wires).execute(circuits, execution_config)

    return MyDevice


class TestDetection:
    """Python devices are demarcated from devices with a C interface at compile time."""

    def test_is_python_device(self, my_device):
        assert python_device.is_python_device(qp.device("default.qubit", wires=1))
        assert python_device.is_python_device(my_device(wires=1))
        assert not python_device.is_python_device(qp.device("lightning.qubit", wires=1))
        assert not python_device.is_python_device(qp.device("null.qubit", wires=1))

    def test_metadata_in_ir(self):
        """The QNode is marked in the IR and bound to the bridge runtime backend."""

        @qp.qjit(capture=True, collect_decomp_rules=False, target="mlir")
        @qp.qnode(qp.device("default.qubit", wires=1))
        def f():
            qp.H(0)
            return qp.expval(qp.Z(0))

        assert "catalyst.python_device" in f.mlir
        assert "PLPythonDevice" in f.mlir and "librtd_pennylane_python" in f.mlir
        assert "PLPythonDevice" in f.mlir_opt  # still present after all compilation stages

    def test_capabilities_intersect_device_toml(self, my_device):
        """The capabilities are the bridge capabilities intersected with the device TOML."""
        caps = python_device.python_device_capabilities(my_device(wires=1))
        assert set(caps.operations) == GATE_SET  # "NotABridgeGate" is dropped
        assert set(caps.observables) == {"PauliZ"}
        assert set(caps.measurement_processes) == {"ExpectationMP", "SampleMP"}
        assert caps.supported_mcm_methods == ["one-shot"]

        bridge = python_device.python_device_capabilities(qp.device("default.qubit", wires=1))
        assert "Hadamard" in bridge.operations and "StateMP" in bridge.measurement_processes


class TestExecution:
    """Results of the runtime pathway."""

    def test_results_match_pennylane(self):
        """All terminal measurements are computed from a single execution of the tape."""
        dev = qp.device("default.qubit", wires=2)

        def circuit(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            qp.QubitUnitary(np.array([[0, 1], [1, 0]]), 1)
            qp.H(0)
            return qp.expval(qp.Z(1)), qp.probs(wires=[0]), qp.var(qp.X(0) @ qp.Z(1)), qp.state()

        compiled = qjit(qp.qnode(dev)(circuit))
        expected = qp.qnode(dev)(circuit)(0.3)
        for res, exp in zip(compiled(0.3), expected, strict=True):
            assert np.allclose(res, exp)

    def test_single_execution_per_call(self, my_device):
        """The device receives one tape holding all the terminal measurements."""
        dev = my_device(wires=2)

        @qjit
        @qp.set_shots(20)
        @qp.qnode(dev)
        def f(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0)), qp.sample(wires=[0, 1])

        expval, samples = f(0.0)
        assert np.isclose(expval, 1.0) and samples.shape == (20, 2) and not samples.any()
        assert len(dev.executed) == 1
        assert len(dev.executed[0].measurements) == 2

    def test_device_instance_is_used(self, my_device):
        """The user's device instance (and its configuration) executes the tape."""
        dev = my_device(wires=1)

        @qjit
        @qp.qnode(dev)
        def f(x):
            qp.RY(x, 0)
            return qp.expval(qp.Z(0))

        assert np.isclose(f(0.5), np.cos(0.5))
        assert len(dev.executed) == 1
        assert type(dev).__name__ == "MyDevice"  # the device class is not modified

    def test_device_preprocessing(self, my_device):
        """Device preprocessing (here, decomposition to its gate set) is applied at runtime."""
        dev = my_device(wires=2)

        @qjit
        @qp.qnode(dev)
        def f(x):
            qp.H(1)
            qp.PauliRot(x, "XY", wires=[0, 1])
            return qp.expval(qp.Z(1))

        expected = qp.qnode(qp.device("default.qubit", wires=2))(f.original_function.func)(0.3)
        assert np.isclose(f(0.3), expected)
        assert {op.name for op in dev.executed[0].operations} <= GATE_SET

    def test_unsupported_instruction_error(self, my_device):
        """The error of a device for an unsupported instruction is raised to the user."""

        @qjit
        @qp.qnode(my_device(wires=1, decompose_ops=False))
        def f():
            qp.H(0)
            return qp.expval(qp.Z(0))

        with pytest.raises(DeviceError, match="Operator Hadamard is not supported by my.device"):
            f()

    def test_counts_and_samples(self):
        dev = qp.device("default.qubit", wires=2)

        @qjit
        @qp.set_shots(100)
        @qp.qnode(dev)
        def f():
            qp.X(0)
            return qp.counts(wires=[0, 1]), qp.sample()

        (basis, counts), samples = f()
        assert list(basis) == [0, 1, 2, 3] and list(counts) == [0, 0, 100, 0]
        assert samples.shape == (100, 2) and np.all(np.asarray(samples) == [1, 0])


class TestRuntimeTape:
    """The tape generated at runtime can be intercepted by developers."""

    def test_intercept_interrupts_execution(self):
        dev = qp.device("default.qubit", wires=2)

        @qjit
        @qp.set_shots(5)
        @qp.qnode(dev)
        def f(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(1)), qp.probs(wires=[0])

        reached = []
        with intercept_runtime_tape() as intercepted:
            f(0.5)
            reached.append(True)  # pragma: no cover

        assert not reached and intercepted.interrupted
        tape = intercepted.tape
        assert len(intercepted.tapes) == 1
        assert all(isinstance(op, Operator2) for op in tape.operations)
        assert qp.equal(tape.operations[0], qp.RX(0.5, 0))
        assert qp.equal(tape.operations[1], qp.CNOT([0, 1]))
        assert tape.measurements == [qp.expval(qp.Z(1)), qp.probs(wires=[0])]
        assert tape.shots == qp.measurements.Shots(5)

    def test_intercept_without_interrupting(self):
        @qjit
        @qp.qnode(qp.device("default.qubit", wires=1))
        def f(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        seen = []
        with intercept_runtime_tape(interrupt=False, callback=seen.append) as intercepted:
            res = f(0.2)
        assert np.isclose(res, np.cos(0.2))
        assert seen == intercepted.tapes and len(seen) == 1

    def test_classical_processing_is_separated(self):
        """Only quantum instructions reach the tape; classical processing stays in the
        program."""

        @qjit
        def f(x):
            @qp.qnode(qp.device("default.qubit", wires=1))
            def circuit(y):
                qp.RX(y, 0)
                return qp.expval(qp.Z(0))

            return 2 * circuit(qp.math.sin(x)) + 1

        with intercept_runtime_tape(interrupt=False) as intercepted:
            res = f(0.4)
        assert np.isclose(res, 2 * np.cos(np.sin(0.4)) + 1)
        assert qp.equal(intercepted.tape.operations[0], qp.RX(np.sin(0.4), 0), atol=1e-12)


class TestMidCircuitMeasurements:
    """MCM-conditioned branches reach the device as Conditional operations."""

    def test_mcm_example(self):
        """The runtime tape has the same structure as the non-qjit tape."""

        @qp.set_shots(10)
        @qp.qnode(qp.device("default.qubit", wires=3))
        def f():
            qp.H(0)
            m = qp.measure(0)
            qp.cond(m, qp.X)(0)
            return qp.sample()

        f()
        expected = f._tape  # pylint: disable=protected-access

        with intercept_runtime_tape(interrupt=False) as intercepted:
            samples = qjit(f)()
        assert samples.shape == (10, 3) and not samples.any()

        tape = intercepted.tape
        h, mcm, cond = tape.operations
        assert qp.equal(h, qp.H(0))
        assert isinstance(mcm, MidMeasure) and mcm.wires == qp.wires.Wires(0)
        assert isinstance(cond, qp.ops.Conditional) and qp.equal(cond.base, qp.X(0))
        assert cond.meas_val.measurements == [mcm] and not cond.meas_val.has_processing
        assert tape.measurements == expected.measurements
        assert tape.shots == expected.shots

    @pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
    def test_mcm_statistics(self, mcm_method):
        """Results with MCMs, reset and multi-MCM predicates match the non-qjit pathway."""

        def circuit(x):
            qp.RY(x, 0)
            m0 = qp.measure(0, reset=True)
            m1 = qp.measure(1)
            qp.cond(m0 == 1, qp.RX)(0.5, 1)
            qp.cond(m0 & ~m1, qp.X, qp.Z)(2)
            return qp.expval(qp.Z(2)), qp.expval(qp.Z(0))

        dev = qp.device("default.qubit", wires=3, seed=1234)
        qnode = qp.set_shots(qp.qnode(dev, mcm_method=mcm_method)(circuit), 4000)

        with intercept_runtime_tape(interrupt=False) as intercepted:
            res = qjit(qnode)(1.2)
        expected = qnode(1.2)
        assert np.allclose(res, expected, atol=0.1)
        assert np.isclose(res[1], 1.0)  # wire 0 was reset

        ops = intercepted.tape.operations
        assert [type(op).__name__ for op in ops] == [
            "RY",
            "MidMeasure",
            "MidMeasure",
            "Conditional",
            "Conditional",
            "Conditional",
        ]
        assert ops[1].reset and not ops[2].reset
        for bits, value in ops[5].meas_val.items():  # the false branch: not (m0 and not m1)
            assert bool(value) == (bits != (1, 0))

    def test_mcm_in_loop(self):
        """MCMs inside loops are recorded once per iteration."""

        @qjit
        @qp.set_shots(10)
        @qp.qnode(qp.device("default.qubit", wires=2), mcm_method="one-shot")
        def f():
            @qp.for_loop(0, 3)
            def loop(i):
                qp.X(0)
                m = qp.measure(0)
                qp.cond(m, qp.X)(1)

            loop()
            return qp.sample(wires=[1])

        with intercept_runtime_tape(interrupt=False) as intercepted:
            samples = f()
        # the outcomes are 1, 0, 1: wire 1 is flipped twice
        assert np.all(np.asarray(samples) == 0)
        names = [type(op).__name__ for op in intercepted.tape.operations]
        assert names == ["PauliX", "MidMeasure", "Conditional"] * 3

    def test_mcm_dependent_parameter_error(self):
        """Classical uses of MCM outcomes other than branches are rejected at compile time."""

        @qp.qnode(qp.device("default.qubit", wires=2))
        def f():
            m = qp.measure(0)
            qp.RX(m * 0.1, 1)
            return qp.expval(qp.Z(0))

        with pytest.raises(CompileError, match="gate parameter or wire depends on the outcome"):
            qjit(f)()


class TestDifferentiation:
    """Differentiation goes through Enzyme and parameter-shift."""

    def test_parameter_shift_gradient(self):
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(1))

        @qjit
        def grad(x):
            return qp.grad(circuit)(x)

        assert np.isclose(grad(0.4), -np.sin(0.4))

    def test_adjoint_rejected_at_compile_time(self):
        @qp.qnode(qp.device("default.qubit", wires=1), diff_method="adjoint")
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        with pytest.raises(CompileError, match="diff_method='adjoint' is not supported"):
            qjit(circuit)(0.1)


@pytest.mark.old_frontend
class TestLegacyFrontend:
    """The bridge also works without program capture, without MCM support."""

    def test_expval(self):
        @qp.qjit
        @qp.qnode(qp.device("default.qubit", wires=1))
        def f(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        assert np.isclose(f(0.3), np.cos(0.3))

    def test_mcm_rejected(self):
        @qp.qnode(qp.device("default.qubit", wires=1))
        def f(x):
            qp.RX(x, 0)
            qp.measure(0)
            return qp.expval(qp.Z(0))

        with pytest.raises(CompileError, match="only supported with qjit\\(capture=True\\)"):
            qp.qjit(f)(0.3)


@pytest.mark.skipif(not hasattr(qp, "flatten"), reason="requires qp.flatten")
def test_flatten_with_python_device():
    """qp.flatten works with Python devices; no bridge rewrite is applied to the tape."""

    @qjit
    @qp.qnode(qp.device("default.qubit", wires=2))
    def f():
        qp.H(0)
        m = qp.measure(0)
        qp.cond(m, qp.X)(1)
        return qp.expval(qp.Z(1))

    tape = qp.flatten(f)()
    assert [type(op).__name__ for op in tape.operations] == [
        "Hadamard",
        "MidMeasure",
        "Conditional",
    ]
    assert tape.wires == qp.wires.Wires([0, 1])


if __name__ == "__main__":
    pytest.main(["-x", os.path.abspath(__file__)])
