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

"""Test program verification routines"""

import platform
from copy import deepcopy
from unittest.mock import patch

import pennylane as qp
import pytest
from pennylane.devices.capabilities import OperatorProperties
from pennylane.measurements import ExpectationMP, VarianceMP
from pennylane.ops import Adjoint, Controlled

from catalyst import (
    CompileError,
    DifferentiableCompileError,
    adjoint,
    cond,
    ctrl,
    for_loop,
    grad,
    qjit,
)
from catalyst.api_extensions import HybridAdjoint, HybridCtrl
from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities
from catalyst.device.qjit_device import RUNTIME_OPERATIONS, get_qjit_device_capabilities
from catalyst.device.verification import validate_measurements

# pylint: disable = unused-argument, unnecessary-lambda-assignment, unnecessary-lambda


def get_custom_device(
    non_differentiable_gates=frozenset(),
    non_differentiable_obs=frozenset(),
    non_invertible_gates=frozenset(),
    non_controllable_gates=frozenset(),
    native_gates=frozenset(),
    **kwargs,
):
    """Generate a custom device where certain gates are marked as non-differensiable."""

    lightning_device = qp.device("lightning.qubit", wires=0)

    class CustomDevice(qp.devices.Device):
        """Custom Gate Set Device"""

        name = lightning_device.name
        author = "Tester"

        config = None
        backend_name = "default"
        backend_lib = "default"
        backend_kwargs = {}

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            lightning_capabilities = get_device_capabilities(lightning_device, shots=shots)
            custom_capabilities = deepcopy(lightning_capabilities)
            for gate in native_gates:
                custom_capabilities.operations[gate] = OperatorProperties(True, True, True)
            for gate in non_differentiable_gates:
                custom_capabilities.operations[gate].differentiable = False
            for gate in non_invertible_gates:
                custom_capabilities.operations[gate].invertible = False
            for gate in non_controllable_gates:
                custom_capabilities.operations[gate].controllable = False
            for obs in non_differentiable_obs:
                custom_capabilities.observables[obs].differentiable = False
            self.qjit_capabilities = custom_capabilities

        def preprocess(self, execution_config=None):
            """Device preprocessing function."""
            return lightning_device.preprocess(execution_config)

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """

            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            # Borrowing the NullQubit library:
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self, _circuits, _execution_config):
            """
            Raises: RuntimeError
            """
            raise RuntimeError("QJIT devices cannot execute tapes.")

        def supports_derivatives(self, config, circuit=None):  # pylint: disable=unused-argument
            """Pretend we support any derivatives"""
            return True

    return CustomDevice(**kwargs)


@qp.transform
def null_transform(tape, *args, **kwargs):
    """A null transform that passes on the tape and the null post processing function.
    Used to overwrite transforms in the device preprocess with mocker when we want to
    skip them for testing purproses"""

    return (tape,), lambda x: x[0]


class PauliX2(qp.PauliX):
    """Test operation without the analytic gradient"""

    name = "PauliX2"
    grad_method = "F"

    def __repr__(self):
        return "PauliX2"


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
def test_unsupported_ops_raise_an_error():
    """Test that an unsupported op raises an error"""

    class MyOp(qp.operation.Operator):
        """An unsupported operation"""

        @property
        def name(self):
            """name of MyOp"""
            return "UnsupportedOp"

        def decomposition(self):
            """No decomposition is implemented"""
            raise NotImplementedError()

    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def f(_):
        MyOp(wires=0)
        return qp.expval(qp.PauliX(0))

    with pytest.raises(CompileError, match="UnsupportedOp is not supported"):
        qjit(f)(1.2)


def queue_ops(x, wires):
    """Queue two operators. To be used to create HybridAdjoint
    and HybridCtrl instances for testing"""
    qp.RX(x, wires=wires)
    qp.Z(wires)


# callables to return adjoint ops via different contruction methods
adj_operator = lambda x, wires: adjoint(qp.RX(x, wires))  # instantiated op (Adjoint)
adj_op_callable = lambda x, wires: adjoint(qp.RX)(x, wires)  # op callable (Adjoint)
adj_op_multiple = lambda x, wires: adjoint(queue_ops)(x, wires)  # op queue (HybridAdjoint)

# callables to return controlled ops via different construction methods
ctrl_operator = lambda x, wires: ctrl(qp.Z(wires), control=[1, 2, 3])  # (Controlled)
ctrl_op_callable = lambda x, wires: ctrl(qp.Z, control=[1, 2, 3])(wires)  # (Controlled)
ctrl_op_multiple = lambda x, wires: ctrl(queue_ops, control=[1, 2, 3])(x, wires)  # (HybridCtrl)


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
class TestHybridOpVerification:
    """Test that the verification catches situations where a HybridOp subtape contains
    an operation the given device can't support inside that HybridOp"""

    @pytest.mark.parametrize(
        "op_fn, op_type",
        [(adj_operator, Adjoint), (adj_op_callable, Adjoint), (adj_op_multiple, HybridAdjoint)],
    )
    def test_non_invertible_gate_simple(self, op_fn, op_type):
        """Emulate a device with a non-invertible gate applied inside an Adjoint HybridOp."""

        dev = get_custom_device(non_invertible_gates={"RX"}, wires=1)

        @qp.qnode(dev)
        def f(x):
            op = op_fn(x, wires=0)
            assert isinstance(op, op_type), f"op expected to be {op_type} but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize(
        "op_fn, op_type",
        [(adj_operator, Adjoint), (adj_op_callable, Adjoint), (adj_op_multiple, HybridAdjoint)],
    )
    def test_non_invertible_gate_nested_cond(self, op_fn, op_type):
        """Emulate a device with a non-invertible gate inside an Adjoint that
        is further nested in a Cond operation."""

        @qp.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
        def f(x):
            @cond(True)
            def true_path():
                op = op_fn(x, wires=0)
                assert isinstance(op, op_type), f"op expected to be {op_type} but got {type(op)}"

            @true_path.otherwise
            def false_path():
                op = op_fn(x, wires=0)
                assert isinstance(op, op_type), f"op expected to be {op_type} but got {type(op)}"

            true_path()

            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize(
        "op_fn, op_type",
        [(adj_operator, Adjoint), (adj_op_callable, Adjoint), (adj_op_multiple, HybridAdjoint)],
    )
    def test_non_invertible_gate_nested_for(self, op_fn, op_type):
        """Emulate a device with a non-invertible gate inside an Adjoint that
        is further nested in a For operation."""

        @qp.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
        def f(x):
            @for_loop(0, 10, 1)
            def loop(_i):
                op = op_fn(x, wires=0)
                assert isinstance(op, op_type), f"op expected to be {op_type} but got {type(op)}"

            loop()
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize("op_fn", [ctrl_operator, ctrl_op_callable])
    def test_non_controllable_gate_pennylane(self, op_fn):
        """Emulate a device with a non-controllable gate applied inside a PL control."""

        @qp.qnode(get_custom_device(non_controllable_gates={"PauliZ"}, wires=4))
        def f(x: float):
            op = op_fn(x, wires=0)
            assert isinstance(
                op, Controlled
            ), f"op expected to be qp.ops.Controlled but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="PauliZ is not controllable"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="PauliZ is not controllable"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_controllable_gate_hybridctrl(self):
        """Test that a non-controllable gate applied inside a HybridCtrl raises an error."""

        # Note: The HybridCtrl operator is not currently supported with the QJIT device, but the
        # verification structure is in place, so we test the verification of its nested operators by
        # adding HybridCtrl to the list of native gates for the custom base device and by patching
        # the list of RUNTIME_OPERATIONS for the QJIT device to include HybridCtrl for this test.

        @qp.qnode(
            get_custom_device(
                native_gates={"HybridCtrl"}, non_controllable_gates={"PauliZ"}, wires=4
            )
        )
        def f(x: float):
            op = ctrl_op_multiple(x, wires=0)
            assert isinstance(op, HybridCtrl), f"op expected to be HybridCtrl but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        runtime_ops_with_qctrl = deepcopy(RUNTIME_OPERATIONS)
        runtime_ops_with_qctrl["HybridCtrl"] = OperatorProperties(
            invertible=True, controllable=True, differentiable=True
        )

        with patch("catalyst.device.qjit_device.RUNTIME_OPERATIONS", runtime_ops_with_qctrl):
            with pytest.raises(CompileError, match="PauliZ is not controllable"):
                qjit(f)(1.2)

            with pytest.raises(CompileError, match="PauliZ is not controllable"):

                @qjit
                def cir(x: float):
                    return grad(f)(x)

    def test_hybridctrl_raises_error(self):
        """Test that a HybridCtrl operator is rejected by the verification."""

        # TODO: If you are deleting this test because HybridCtrl support has been added, consider
        # updating the tests that patch RUNTIME_OPERATIONS to inclue HybridCtrl accordingly

        @qp.qnode(get_custom_device(non_controllable_gates={"PauliZ"}, wires=4))
        def f(x: float):
            op = ctrl_op_multiple(x, wires=0)
            assert isinstance(op, HybridCtrl), f"op expected to be HybridCtrl but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="HybridCtrl is not supported"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="HybridCtrl is not supported"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    def test_pennylane_ctrl_of_hybridop_raises_error(self):
        """Test that a PennyLane Controlled op with a HybridOp as its base is
        caught in verification"""

        @qp.qnode(get_custom_device(wires=4))
        def f(x: float):
            op = Controlled(adj_op_multiple(x, wires=0), control_wires=[1, 2, 3])
            assert isinstance(op, Controlled), f"op expected to be Controlled but got {type(op)}"
            assert isinstance(
                op.base, HybridAdjoint
            ), f"base op expected to be HybridAdjoint but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="Cannot compile PennyLane control of the hybrid op"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="Cannot compile PennyLane control of the hybrid op"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    def test_pennylane_adj_of_hybridop_raises_error(self):
        """Test that a PennyLane Controlled op with a HybridOp as its base is caught
        in verification"""

        @qp.qnode(get_custom_device(wires=4))
        def f(x: float):
            op = Adjoint(adj_op_multiple(x, wires=0))
            assert isinstance(op, Adjoint), f"op expected to be Adjoint but got {type(op)}"
            assert isinstance(
                op.base, HybridAdjoint
            ), f"base op expected to be HybridAdjoint but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match="Cannot compile PennyLane inverse of the hybrid op"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match="Cannot compile PennyLane inverse of the hybrid op"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize("adjoint_type", [Adjoint, HybridAdjoint])
    @pytest.mark.parametrize("unsupported_gate_attribute", ["controllable", "invertible"])
    def test_hybrid_ctrl_containing_adjoint(self, adjoint_type, unsupported_gate_attribute):
        """Test that verification catches a non-invertible or non-controllable base that
        is in an Adjoint or HybridAdjoint inside a HybridCtrl"""

        # Note: The HybridCtrl operator is not currently supported with the QJIT device, but the
        # verification structure is in place, so we test the verification of its nested operators by
        # adding HybridCtrl to the list of native gates for the custom base device and by patching
        # the list of RUNTIME_OPERATIONS for the QJIT device to include HybridCtrl for this test.

        def _ops(x, wires):
            if adjoint_type == HybridAdjoint:
                adj_op_multiple(x, wires)
            else:
                adjoint(qp.Z(0))
            qp.Z(1)

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qp.qnode(get_custom_device(native_gates={"HybridCtrl"}, wires=4, **device_kwargs))
        def f(x: float):
            op = ctrl(_ops, control=[2, 3, 4])(x, wires=0)
            assert isinstance(op, HybridCtrl), f"expected HybridCtrl but got {type(op)}"
            base = op.regions[0].quantum_tape.operations[0]
            assert isinstance(base, adjoint_type), f"expected {adjoint_type} but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        runtime_ops_with_qctrl = deepcopy(RUNTIME_OPERATIONS)
        runtime_ops_with_qctrl["HybridCtrl"] = OperatorProperties(
            invertible=True, controllable=True, differentiable=True
        )

        with patch("catalyst.device.qjit_device.RUNTIME_OPERATIONS", runtime_ops_with_qctrl):
            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
                qjit(f)(1.2)

            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

                @qjit
                def cir(x: float):
                    return grad(f)(x)

    @pytest.mark.parametrize("ctrl_type", [Controlled, HybridCtrl])
    @pytest.mark.parametrize("unsupported_gate_attribute", ["controllable", "invertible"])
    def test_hybrid_adjoint_containing_hybrid_ctrl(self, ctrl_type, unsupported_gate_attribute):
        """Test that verification catches a non-invertible or non-controllable base that
        is in a HybridCtrl or Controlled inside a HybridAdjoint"""

        # Note: The HybridCtrl operator is not currently supported with the QJIT device, but the
        # verification structure is in place, so we test the verification of its nested operators by
        # adding HybridCtrl to the list of native gates for the custom base device and by patching
        # the list of RUNTIME_OPERATIONS for the QJIT device to include HybridCtrl for this test.

        def _ops(x, wires):
            if ctrl_type == HybridCtrl:
                ctrl_op_multiple(x, wires)
            else:
                ctrl_operator(x, wires)
            qp.Z(1)

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qp.qnode(get_custom_device(native_gates={"HybridCtrl"}, wires=4, **device_kwargs))
        def f(x: float):
            op = adjoint(_ops)(x, wires=0)
            base = op.regions[0].quantum_tape.operations[0]
            assert isinstance(op, HybridAdjoint), f"expected HybridAdjoint but got {type(op)}"
            assert isinstance(base, ctrl_type), f"expected {ctrl_type} but got {type(op)}"
            return qp.expval(qp.PauliX(0))

        runtime_ops_with_qctrl = deepcopy(RUNTIME_OPERATIONS)
        runtime_ops_with_qctrl["HybridCtrl"] = OperatorProperties(
            invertible=True, controllable=True, differentiable=True
        )

        with patch("catalyst.device.qjit_device.RUNTIME_OPERATIONS", runtime_ops_with_qctrl):
            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
                qjit(f)(1.2)

            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

                @qjit
                def cir(x: float):
                    return grad(f)(x)

    @pytest.mark.parametrize("unsupported_gate_attribute", ["controllable", "invertible"])
    def test_pennylane_ctrl_containing_adjoint(self, unsupported_gate_attribute):
        """stuff"""

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qp.qnode(get_custom_device(wires=4, **device_kwargs))
        def f(x: float):
            Controlled(Adjoint(qp.Z(0)), control_wires=[1, 2, 3])
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize("unsupported_gate_attribute", ["controllable", "invertible"])
    def test_pennylane_adjoint_containing_controlled(self, unsupported_gate_attribute):
        """stuff"""

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qp.qnode(get_custom_device(wires=4, **device_kwargs))
        def f(x: float):
            Adjoint(Controlled(qp.Z(0), control_wires=[1, 2, 3]))
            return qp.expval(qp.PauliX(0))

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
            qjit(f)(1.2)

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

            @qjit
            def cir(x: float):
                return grad(f)(x)


class TestObservableValidation:
    """Tests the general validation of observables (independent of gradient method)"""

    def test_unsupported_observable_raises_error(self):
        """Test that including an unsupported observable in a measurement raises an
        error when jitting the circuit"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def f():
            qp.RX(1.23, 0)
            return qp.expval(qp.RX(1.2, 0))

        with pytest.raises(CompileError, match="RX.*not supported as an observable"):
            qjit(f)()

    @pytest.mark.parametrize(
        "measurements, invalid_op",
        [
            ([qp.expval(qp.X(0))], None),  # single obs
            ([qp.expval(qp.RX(1.2, 0))], "RX"),
            ([qp.var(qp.X(0) @ qp.Y(2))], None),  # prod
            ([qp.var(qp.X(0) @ qp.RY(1.23, 2))], "RY"),
            (
                [qp.var((2 * qp.X(0) @ qp.Y(2)) @ (2 * qp.X(3)) @ qp.Y(1))],
                None,
            ),  # nested prod+sprod
            ([qp.var((2 * qp.X(0) @ qp.Y(2)) @ (2 * qp.RY(1.2, 3)) @ qp.Y(1))], "RY"),
            ([qp.var((2 * qp.X(0) @ qp.RY(1.2, 2)) @ (2 * qp.X(3)) @ qp.Y(1))], "RY"),
            ([qp.var(qp.X(1) + qp.Y(2))], None),  # sum
            ([qp.var(qp.RX(1.23, 1) + qp.Y(2))], "RX"),
            ([qp.expval(2 * qp.Z(1))], None),  # sprod
            ([qp.expval(2 * qp.RZ(1.23, 1))], "RZ"),
            ([qp.expval(qp.Hamiltonian([2, 3], [qp.X(0), qp.Y(1)]))], None),  # hamiltonian
            ([qp.expval(qp.Hamiltonian([2, 3], [qp.X(0), qp.RY(2.3, 1)]))], "RY"),
            ([qp.expval(qp.Hamiltonian([2, 3], [qp.Y(0), PauliX2(1)]))], "PauliX2"),
            ([qp.sample(), qp.expval(qp.X(0))], None),  # with empty sample
            ([qp.sample(), qp.expval(qp.RX(1.2, 0))], "RX"),
            # sample with observable is currently unsupported
            # ([qp.sample(qp.X(0)), qp.expval(qp.X(0))], None),
            # ([qp.sample(qp.RX(1.2, 0)), qp.expval(qp.X(0))], "RX"),
            ([qp.probs(wires=0), qp.var(qp.X(1) + qp.Y(2))], None),  # with probs
            ([qp.probs(wires=0), qp.var(qp.RX(1.23, 1) + qp.Y(2))], "RX"),
            ([qp.counts(), qp.expval(qp.X(0))], None),  # with empty counts
            ([qp.counts(), qp.expval(qp.RX(1.2, 0))], "RX"),
            # counts with observable is currently unsupported
            # ([qp.counts(qp.Y(0)), qp.expval(qp.X(0))], None),  # with counts with observable
            # ([qp.counts(qp.RX(1.23, 0)), qp.expval(qp.X(0))], "RX"),
        ],
    )
    def test_validate_measurements_transform(self, backend, measurements, invalid_op):
        """Test that the validate_measurements transform raises an error (or not) as expected
        for different base observables."""

        dev = qp.device(backend, wires=3)
        qjit_capabilities = get_device_capabilities(dev, shots=2048)

        tape = qp.tape.QuantumScript([], measurements=measurements, shots=2048)

        if invalid_op:
            with pytest.raises(CompileError, match=f"{invalid_op}.*not supported as an observable"):
                validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)
        else:
            validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)

    @pytest.mark.parametrize(
        "obs, obs_type",
        [
            (qp.X(0) @ qp.Y(1), "Prod"),
            (2 * qp.Y(1), "SProd"),
            (qp.Hamiltonian([2, 3], [qp.X(0), qp.Y(1)]), "LinearCombination"),
            (qp.X(0) + 2 * qp.Y(1), "Sum"),
        ],
    )
    def test_arithmetic_ops_validation(self, obs, obs_type, backend):
        """Test that the validate_measurements transform raises an error (or not) as expected
        for different observables composed of other base observables, when the overall observable
        type is supported/unsupported."""

        dev = qp.device(backend, wires=1)
        dev_capabilities = get_device_capabilities(dev, shots=None)
        qjit_capabilities = get_qjit_device_capabilities(dev_capabilities)

        tape = qp.tape.QuantumScript([], measurements=[qp.expval(obs)])

        # all good
        validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)

        del qjit_capabilities.observables[obs_type]
        with pytest.raises(CompileError, match="not supported as an observable"):
            validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)

    def test_non_qjit_observables_raise_error(self, backend):
        """Test that an observable that is supported by the backend according to the
        TOML file, but is not supported by Catalyst, raises an error in validation"""

        dev = qp.device(backend, wires=1)
        dev_capabilities = get_device_capabilities(dev)

        dev_capabilities.observables.update(
            {"PauliX2": OperatorProperties(invertible=True, controllable=True, differentiable=True)}
        )

        qjit_capabilities = get_qjit_device_capabilities(dev_capabilities)

        tape = qp.tape.QuantumScript([], measurements=[qp.expval(PauliX2(0))])

        with pytest.raises(CompileError, match="PauliX2 is not supported as an observable"):
            validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)

    @pytest.mark.parametrize(
        "measurement", [qp.expval(qp.X(0)), qp.var(qp.X(0)), qp.sample(qp.X(0))]
    )
    def test_only_expval_and_var_allow_observables(self, measurement):
        """Test that the validate_measurements transform catches measurements other
        than expval and var that include observables, and raises an error"""

        dev = qp.device("lightning.qubit", wires=1)
        dev_capabilities = get_device_capabilities(dev, shots=None)
        qjit_capabilities = get_qjit_device_capabilities(dev_capabilities)

        tape = qp.tape.QuantumScript([], measurements=[measurement])

        if isinstance(measurement, (ExpectationMP, VarianceMP)):
            validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)
        else:
            with pytest.raises(
                CompileError,
                match="Only expectation value and variance measurements can accept observables",
            ):
                validate_measurements(tape, qjit_capabilities, dev.name, tape.shots)


class TestMeasurementTypeValidation:
    """Test validation of measurement processes versus the a device's supported
    measurement types"""

    @pytest.mark.parametrize(
        "measurement, shots, msg",
        [
            (qp.state(), 100, "Please specify shots=None."),
            (qp.sample(), None, "Please specify a finite number of shots."),
            (qp.expval(qp.X(0)), 100, "is not a supported measurement process"),
            (qp.expval(qp.X(0)), None, "is not a supported measurement process"),
        ],
    )
    def test_validate_measurements_works_on_measurement_processes(self, measurement, shots, msg):
        """Test that the validate_measurements transform raises a CompileError as
        expected for an unsupported MeasurementProcess"""

        dev = qp.device("lightning.qubit", wires=1)
        tape = qp.tape.QuantumScript([], measurements=[measurement])

        qjit_capabilities = get_device_capabilities(dev)
        qjit_capabilities.measurement_processes.pop("ExpectationMP")

        with pytest.raises(CompileError, match=msg):
            validate_measurements(tape, qjit_capabilities, dev.name, shots)

    def test_state_measurements_rejected_with_shots(self):
        """Test that trying to measure a state on a device with finite shots
        raises a CompileError informing the user that shots must be None for
        state based measurements"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.set_shots(100)
        @qp.qnode(dev)
        def f():
            qp.RX(1.23, 0)
            return qp.state()

        with pytest.raises(CompileError, match="Please specify shots=None."):
            qjit(f)()

    @pytest.mark.parametrize("measurement", [qp.sample, qp.counts])
    def test_sample_measurements_rejected_without_shots(self, measurement):
        """Test that trying to take a sample-based measurement on a device
        without shots raises a CompileError informing the user that a
        finite number of shots is needed for sampling"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.set_shots(None)
        @qp.qnode(dev)
        def f():
            qp.RX(1.23, 0)
            return measurement()

        with pytest.raises(CompileError, match="Please specify a finite number of shots."):
            qjit(f)()

    def test_unsupported_measurement_types_rejected(self):
        """Test that trying to use a measurement type that is generally unsupported by
        the device raises a CompileError"""

        dev = qp.device("lightning.qubit", wires=1)

        class MyMeasurement(qp.measurements.SampleMeasurement):
            """A custom measurement (not supported on lightning.qubit)"""

            def __init__(self, obs=None, wires=None):
                super().__init__(obs=obs, wires=wires, eigvals=None)

            def process_samples(self, samples, wire_order, shot_range, bin_size):
                """overwrite ABC method"""
                raise NotImplementedError

            def process_counts(self, counts, wire_order):
                """overwrite ABC method"""
                raise NotImplementedError

        @qp.set_shots(100)
        @qp.qnode(dev)
        def f():
            qp.RX(1.23, 0)
            return MyMeasurement()

        with pytest.raises(CompileError, match="is not a supported measurement process"):
            qjit(f)()


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
class TestAdjointMethodVerification:
    """Test the verification of operators and observables when the adjoint diff method
    is used for differentiation"""

    def test_non_differentiable_gate_simple(self):
        """Emulate a device with a non-differentiable gate."""

        @qp.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=[0]), diff_method="adjoint"
        )
        def f(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize(
        "observable",
        [
            qp.PauliX(0),  # single observable
            qp.PauliX(0) @ qp.PauliZ(1),  # prod
            qp.PauliX(0) + qp.PauliY(1),  # sum
            qp.Hamiltonian([2, 3], [qp.X(0), qp.Y(1)]),  # linearcombination
            2 * qp.PauliX(0),  # sprod
            (2 * qp.X(0) @ qp.Y(2)) @ (2 * qp.X(3)) @ qp.Y(1),  # nested prod+sprod
        ],
    )
    def test_non_differentiable_observable(self, observable):
        """Test that taking the adjoint diff of a circuit with an observable that doesn't support
        adjoint differentiation raises an error."""

        @qp.qnode(
            get_custom_device(non_differentiable_obs={"PauliX"}, wires=[0, 1]),
            diff_method="adjoint",
        )
        def f(x):
            qp.RX(x, wires=0)
            return qp.expval(observable)

        with pytest.raises(DifferentiableCompileError, match="PauliX.*non-differentiable"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    def test_empty_observable(self):
        """Test that taking the adjoint diff of a circuit with an empyt observable with adjoint
        adjoint passes the validation."""

        @qp.qnode(
            get_custom_device(non_differentiable_obs={"PauliX"}, wires=[0]), diff_method="adjoint"
        )
        def f(x):
            qp.RX(x, wires=0)
            return qp.probs()

        qjit(f)(1.2)

    def test_non_differentiable_gate_nested_cond(self):
        """Test that taking the adjoint diff of a tape containing a parameterized operation
        that doesn't support adjoint differentiation raises an error."""

        @qp.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint"
        )
        def f(x):
            @cond(True)
            def true_path():
                qp.RX(x, wires=0)

            @true_path.otherwise
            def false_path():
                qp.RX(x, wires=0)

            true_path()

            return qp.expval(qp.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_differentiable_gate_nested_adjoint(self):
        """Test that taking the adjoint diff of a tape containing a HybridOp with a
        parameterized operation that doesn't support adjoint differentiation raises
        an error."""

        @qp.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint"
        )
        def f(x):
            adjoint(qp.RX(x, wires=[0]))
            return qp.expval(qp.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qjit
            def cir(x: float):
                return grad(f)(x)


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
class TestParameterShiftMethodVerification:
    """Test the verification of operators and observables when the parameter shift method
    is used for differentiation"""

    @pytest.mark.parametrize(
        "observable",
        [
            qp.PauliX(0),
            # qp.PauliX(0) @ qp.PauliZ(1),
            # qp.PauliX(0)+qp.PauliY(1)
        ],
    )
    @patch.object(qp.PauliX, "grad_method", "F")
    def test_paramshift_obs_simple(self, observable):
        """Test that taking a parameter-shift gradient of an observable that doesn't support
        analytic differentiation raises an error."""

        assert qp.PauliX.grad_method != "A"

        @qp.qnode(get_custom_device(wires=2), diff_method="parameter-shift")
        def f(x):
            qp.PauliY(wires=1)
            qp.RX(x, wires=0)
            return qp.expval(observable)

        with pytest.raises(
            DifferentiableCompileError, match="PauliX does not support analytic differentiation"
        ):

            @qjit
            def cir(x: float):
                return grad(f)(x)


def test_no_state_returns():
    """Test state returns are rejected in gradients."""

    @qp.qnode(get_custom_device(wires=1))
    def f(_):
        qp.PauliX(wires=0)
        return qp.state()

    with pytest.raises(DifferentiableCompileError, match="State returns.*forbidden"):

        @qjit
        def cir(x: float):
            return grad(f)(x)


def test_no_variance_returns():
    """Test variance returns are rejected in gradients."""

    @qp.qnode(get_custom_device(wires=1))
    def f(_):
        qp.PauliX(wires=0)
        return qp.var(qp.PauliX(0))

    with pytest.raises(DifferentiableCompileError, match="Variance returns.*forbidden"):

        @qjit
        def cir(x: float):
            return grad(f)(x)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
