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

""" Test program verification routines """

from copy import deepcopy
from unittest.mock import patch

import platform

import pennylane as qml
import pytest
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
    while_loop,
)
from catalyst.api_extensions import HybridAdjoint, HybridCtrl
from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities
from catalyst.device.qjit_device import RUNTIME_OPERATIONS, get_qjit_device_capabilities
from catalyst.device.verification import validate_measurements
from catalyst.utils.toml import OperationProperties

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

    lightning_device = qml.device("lightning.qubit", wires=0)

    class CustomDevice(qml.devices.Device):
        """Custom Gate Set Device"""

        name = lightning_device.name
        author = "Tester"

        config = None
        backend_name = "default"
        backend_lib = "default"
        backend_kwargs = {}

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            lightning_capabilities = get_device_capabilities(lightning_device)
            custom_capabilities = deepcopy(lightning_capabilities)
            for gate in native_gates:
                custom_capabilities.native_ops[gate] = OperationProperties(True, True, True)
            for gate in non_differentiable_gates:
                custom_capabilities.native_ops[gate].differentiable = False
            for gate in non_invertible_gates:
                custom_capabilities.native_ops[gate].invertible = False
            for gate in non_controllable_gates:
                custom_capabilities.native_ops[gate].controllable = False
            for obs in non_differentiable_obs:
                custom_capabilities.native_obs[obs].differentiable = False
            self.qjit_capabilities = custom_capabilities

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


@qml.transform
def null_transform(tape, *args, **kwargs):
    """A null transform that passes on the tape and the null post processing function.
    Used to overwrite transforms in the device preprocess with mocker when we want to
    skip them for testing purproses"""

    return (tape,), lambda x: x[0]


class PauliX2(qml.PauliX):
    """Test operation without the analytic gradient"""

    name = "PauliX2"
    grad_method = "F"

    def __repr__(self):
        return "PauliX2"


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
def test_unsupported_ops_raise_an_error():
    """Test that an unsupported op raises an error"""

    class MyOp(qml.operation.Operator):
        """An unsupported operation"""

        @property
        def name(self):
            """name of MyOp"""
            return "UnsupportedOp"

        def decomposition(self):
            """No decomposition is implemented"""
            raise NotImplementedError()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(_):
        MyOp(wires=0)
        return qml.expval(qml.PauliX(0))

    with pytest.raises(CompileError, match="UnsupportedOp is not supported"):
        qml.qjit(f)(1.2)


def queue_ops(x, wires):
    """Queue two operators. To be used to create HybridAdjoint
    and HybridCtrl instances for testing"""
    qml.RX(x, wires=wires)
    qml.Z(wires)


# callables to return adjoint ops via different contruction methods
adj_operator = lambda x, wires: adjoint(qml.RX(x, wires))  # instantiated op (Adjoint)
adj_op_callable = lambda x, wires: adjoint(qml.RX)(x, wires)  # op callable (Adjoint)
adj_op_multiple = lambda x, wires: adjoint(queue_ops)(x, wires)  # op queue (HybridAdjoint)

# callables to return controlled ops via different construction methods
ctrl_operator = lambda x, wires: ctrl(qml.Z(wires), control=[1, 2, 3])  # (Controlled)
ctrl_op_callable = lambda x, wires: ctrl(qml.Z, control=[1, 2, 3])(wires)  # (Controlled)
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

        @qml.qnode(dev)
        def f(x):
            op = op_fn(x, wires=0)
            assert isinstance(op, op_type), f"op expected to be {op_type} but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize(
        "op_fn, op_type",
        [(adj_operator, Adjoint), (adj_op_callable, Adjoint), (adj_op_multiple, HybridAdjoint)],
    )
    def test_non_invertible_gate_nested_cond(self, op_fn, op_type):
        """Emulate a device with a non-invertible gate inside an Adjoint that
        is further nested in a Cond operation."""

        @qml.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
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

            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize(
        "op_fn, op_type",
        [(adj_operator, Adjoint), (adj_op_callable, Adjoint), (adj_op_multiple, HybridAdjoint)],
    )
    def test_non_invertible_gate_nested_for(self, op_fn, op_type):
        """Emulate a device with a non-invertible gate inside an Adjoint that
        is further nested in a For operation."""

        @qml.qnode(get_custom_device(non_invertible_gates={"RX"}, wires=1))
        def f(x):
            @for_loop(0, 10, 1)
            def loop(_i):
                op = op_fn(x, wires=0)
                assert isinstance(op, op_type), f"op expected to be {op_type} but got {type(op)}"

            loop()
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="RX.*not invertible"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="RX.*not invertible"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize("op_fn", [ctrl_operator, ctrl_op_callable])
    def test_non_controllable_gate_pennylane(self, op_fn):
        """Emulate a device with a non-controllable gate applied inside a PL control."""

        @qml.qnode(get_custom_device(non_controllable_gates={"PauliZ"}, wires=4))
        def f(x: float):
            op = op_fn(x, wires=0)
            assert isinstance(
                op, Controlled
            ), f"op expected to be qml.ops.Controlled but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="PauliZ is not controllable"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="PauliZ is not controllable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_controllable_gate_hybridctrl(self):
        """Test that a non-controllable gate applied inside a HybridCtrl raises an error."""

        # Note: The HybridCtrl operator is not currently supported with the QJIT device, but the
        # verification structure is in place, so we test the verification of its nested operators by
        # adding HybridCtrl to the list of native gates for the custom base device and by patching
        # the list of RUNTIME_OPERATIONS for the QJIT device to include HybridCtrl for this test.

        @qml.qnode(
            get_custom_device(
                native_gates={"HybridCtrl"}, non_controllable_gates={"PauliZ"}, wires=4
            )
        )
        def f(x: float):
            op = ctrl_op_multiple(x, wires=0)
            assert isinstance(op, HybridCtrl), f"op expected to be HybridCtrl but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        runtime_ops_with_qctrl = deepcopy(RUNTIME_OPERATIONS)
        runtime_ops_with_qctrl["HybridCtrl"] = OperationProperties(
            invertible=True, controllable=True, differentiable=True
        )

        with patch("catalyst.device.qjit_device.RUNTIME_OPERATIONS", runtime_ops_with_qctrl):
            with pytest.raises(CompileError, match="PauliZ is not controllable"):
                qml.qjit(f)(1.2)

            with pytest.raises(CompileError, match="PauliZ is not controllable"):

                @qml.qjit
                def cir(x: float):
                    return grad(f)(x)

    def test_hybridctrl_raises_error(self):
        """Test that a HybridCtrl operator is rejected by the verification."""

        # TODO: If you are deleting this test because HybridCtrl support has been added, consider
        # updating the tests that patch RUNTIME_OPERATIONS to inclue HybridCtrl accordingly

        @qml.qnode(get_custom_device(non_controllable_gates={"PauliZ"}, wires=4))
        def f(x: float):
            op = ctrl_op_multiple(x, wires=0)
            assert isinstance(op, HybridCtrl), f"op expected to be HybridCtrl but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="HybridCtrl is not supported"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="HybridCtrl is not supported"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_pennylane_ctrl_of_hybridop_raises_error(self):
        """Test that a PennyLane Controlled op with a HybridOp as its base is
        caught in verification"""

        @qml.qnode(get_custom_device(wires=4))
        def f(x: float):
            op = Controlled(adj_op_multiple(x, wires=0), control_wires=[1, 2, 3])
            assert isinstance(op, Controlled), f"op expected to be Controlled but got {type(op)}"
            assert isinstance(
                op.base, HybridAdjoint
            ), f"base op expected to be HybridAdjoint but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="Cannot compile PennyLane control of the hybrid op"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="Cannot compile PennyLane control of the hybrid op"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_pennylane_adj_of_hybridop_raises_error(self):
        """Test that a PennyLane Controlled op with a HybridOp as its base is caught
        in verification"""

        @qml.qnode(get_custom_device(wires=4))
        def f(x: float):
            op = Adjoint(adj_op_multiple(x, wires=0))
            assert isinstance(op, Adjoint), f"op expected to be Adjoint but got {type(op)}"
            assert isinstance(
                op.base, HybridAdjoint
            ), f"base op expected to be HybridAdjoint but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match="Cannot compile PennyLane inverse of the hybrid op"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match="Cannot compile PennyLane inverse of the hybrid op"):

            @qml.qjit
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
                adjoint(qml.Z(0))
            qml.Z(1)

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qml.qnode(get_custom_device(native_gates={"HybridCtrl"}, wires=4, **device_kwargs))
        def f(x: float):
            op = ctrl(_ops, control=[2, 3, 4])(x, wires=0)
            assert isinstance(op, HybridCtrl), f"expected HybridCtrl but got {type(op)}"
            base = op.regions[0].quantum_tape.operations[0]
            assert isinstance(base, adjoint_type), f"expected {adjoint_type} but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        runtime_ops_with_qctrl = deepcopy(RUNTIME_OPERATIONS)
        runtime_ops_with_qctrl["HybridCtrl"] = OperationProperties(
            invertible=True, controllable=True, differentiable=True
        )

        with patch("catalyst.device.qjit_device.RUNTIME_OPERATIONS", runtime_ops_with_qctrl):
            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
                qml.qjit(f)(1.2)

            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

                @qml.qjit
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
            qml.Z(1)

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qml.qnode(get_custom_device(native_gates={"HybridCtrl"}, wires=4, **device_kwargs))
        def f(x: float):
            op = adjoint(_ops)(x, wires=0)
            base = op.regions[0].quantum_tape.operations[0]
            assert isinstance(op, HybridAdjoint), f"expected HybridAdjoint but got {type(op)}"
            assert isinstance(base, ctrl_type), f"expected {ctrl_type} but got {type(op)}"
            return qml.expval(qml.PauliX(0))

        runtime_ops_with_qctrl = deepcopy(RUNTIME_OPERATIONS)
        runtime_ops_with_qctrl["HybridCtrl"] = OperationProperties(
            invertible=True, controllable=True, differentiable=True
        )

        with patch("catalyst.device.qjit_device.RUNTIME_OPERATIONS", runtime_ops_with_qctrl):
            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
                qml.qjit(f)(1.2)

            with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

                @qml.qjit
                def cir(x: float):
                    return grad(f)(x)

    @pytest.mark.parametrize("unsupported_gate_attribute", ["controllable", "invertible"])
    def test_pennylane_ctrl_containing_adjoint(self, unsupported_gate_attribute):
        """stuff"""

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qml.qnode(get_custom_device(wires=4, **device_kwargs))
        def f(x: float):
            Controlled(Adjoint(qml.Z(0)), control_wires=[1, 2, 3])
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize("unsupported_gate_attribute", ["controllable", "invertible"])
    def test_pennylane_adjoint_containing_controlled(self, unsupported_gate_attribute):
        """stuff"""

        device_kwargs = {f"non_{unsupported_gate_attribute}_gates": {"PauliZ"}}

        @qml.qnode(get_custom_device(wires=4, **device_kwargs))
        def f(x: float):
            Adjoint(Controlled(qml.Z(0), control_wires=[1, 2, 3]))
            return qml.expval(qml.PauliX(0))

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):
            qml.qjit(f)(1.2)

        with pytest.raises(CompileError, match=f"PauliZ is not {unsupported_gate_attribute}"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)


class TestObservableValidation:
    """Tests the general validation of observables (independent of gradient method)"""

    def test_unsupported_observable_raises_error(self):
        """Test that including an unsupported observable in a measurement raises an
        error when jitting the circuit"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def f():
            qml.RX(1.23, 0)
            return qml.expval(qml.RX(1.2, 0))

        with pytest.raises(CompileError, match="RX.*not supported as an observable"):
            qml.qjit(f)()

    @pytest.mark.parametrize(
        "measurements, invalid_op",
        [
            ([qml.expval(qml.X(0))], None),  # single obs
            ([qml.expval(qml.RX(1.2, 0))], "RX"),
            ([qml.var(qml.X(0) @ qml.Y(2))], None),  # prod
            ([qml.var(qml.X(0) @ qml.RY(1.23, 2))], "RY"),
            (
                [qml.var((2 * qml.X(0) @ qml.Y(2)) @ (2 * qml.X(3)) @ qml.Y(1))],
                None,
            ),  # nested prod+sprod
            ([qml.var((2 * qml.X(0) @ qml.Y(2)) @ (2 * qml.RY(1.2, 3)) @ qml.Y(1))], "RY"),
            ([qml.var((2 * qml.X(0) @ qml.RY(1.2, 2)) @ (2 * qml.X(3)) @ qml.Y(1))], "RY"),
            ([qml.var(qml.operation.Tensor(qml.X(0), qml.Y(2)))], None),  # tensor
            ([qml.var(qml.operation.Tensor(qml.X(0), PauliX2(2)))], "PauliX2"),
            ([qml.var(qml.X(1) + qml.Y(2))], None),  # sum
            ([qml.var(qml.RX(1.23, 1) + qml.Y(2))], "RX"),
            ([qml.expval(2 * qml.Z(1))], None),  # sprod
            ([qml.expval(2 * qml.RZ(1.23, 1))], "RZ"),
            ([qml.expval(qml.Hamiltonian([2, 3], [qml.X(0), qml.Y(1)]))], None),  # hamiltonian
            ([qml.expval(qml.Hamiltonian([2, 3], [qml.X(0), qml.RY(2.3, 1)]))], "RY"),
            ([qml.expval(qml.Hamiltonian([2, 3], [qml.Y(0), PauliX2(1)]))], "PauliX2"),
            ([qml.sample(), qml.expval(qml.X(0))], None),  # with empty sample
            ([qml.sample(), qml.expval(qml.RX(1.2, 0))], "RX"),
            # sample with observable is currently unsupported
            # ([qml.sample(qml.X(0)), qml.expval(qml.X(0))], None),
            # ([qml.sample(qml.RX(1.2, 0)), qml.expval(qml.X(0))], "RX"),
            ([qml.probs(wires=0), qml.var(qml.X(1) + qml.Y(2))], None),  # with probs
            ([qml.probs(wires=0), qml.var(qml.RX(1.23, 1) + qml.Y(2))], "RX"),
            ([qml.counts(), qml.expval(qml.X(0))], None),  # with empty counts
            ([qml.counts(), qml.expval(qml.RX(1.2, 0))], "RX"),
            # counts with observable is currently unsupported
            # ([qml.counts(qml.Y(0)), qml.expval(qml.X(0))], None),  # with counts with observable
            # ([qml.counts(qml.RX(1.23, 0)), qml.expval(qml.X(0))], "RX"),
        ],
    )
    def test_validate_measurements_transform(self, backend, measurements, invalid_op):
        """Test that the validate_measurements transform raises an error (or not) as expected
        for different base observables."""

        dev = qml.device(backend, wires=3, shots=2048)
        qjit_capabilities = get_device_capabilities(dev)

        tape = qml.tape.QuantumScript([], measurements=measurements)

        if invalid_op:
            with pytest.raises(CompileError, match=f"{invalid_op}.*not supported as an observable"):
                validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)
        else:
            validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)

    @pytest.mark.parametrize(
        "obs, obs_type",
        [
            (qml.X(0) @ qml.Y(1), "Prod"),
            (2 * qml.Y(1), "SProd"),
            (qml.Hamiltonian([2, 3], [qml.X(0), qml.Y(1)]), "LinearCombination"),
            (qml.X(0) + 2 * qml.Y(1), "Sum"),
        ],
    )
    def test_arithmetic_ops_validation(self, obs, obs_type, backend):
        """Test that the validate_measurements transform raises an error (or not) as expected
        for different observables composed of other base observables, when the overall observable
        type is supported/unsupported."""

        dev = qml.device(backend, wires=1)
        dev_capabilities = get_device_capabilities(dev)
        qjit_capabilities = get_qjit_device_capabilities(dev_capabilities)

        tape = qml.tape.QuantumScript([], measurements=[qml.expval(obs)])

        # all good
        validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)

        del qjit_capabilities.native_obs[obs_type]
        with pytest.raises(CompileError, match="not supported as an observable"):
            validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)

    def test_non_qjit_observables_raise_error(self, backend):
        """Test that an observable that is supported by the backend according to the
        TOML file, but is not supported by Catalyst, raises an error in validation"""

        dev = qml.device(backend, wires=1)
        dev_capabilities = get_device_capabilities(dev)

        dev_capabilities.native_obs.update(
            {
                "PauliX2": OperationProperties(
                    invertible=True, controllable=True, differentiable=True
                )
            }
        )

        qjit_capabilities = get_qjit_device_capabilities(dev_capabilities)

        tape = qml.tape.QuantumScript([], measurements=[qml.expval(PauliX2(0))])

        with pytest.raises(CompileError, match="PauliX2 is not supported as an observable"):
            validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)

    @pytest.mark.parametrize(
        "measurement", [qml.expval(qml.X(0)), qml.var(qml.X(0)), qml.sample(qml.X(0))]
    )
    def test_only_expval_and_var_allow_observables(self, measurement):
        """Test that the validate_measurements transform catches measurements other
        than expval and var that include observables, and raises an error"""

        dev = qml.device("lightning.qubit", wires=1)
        dev_capabilities = get_device_capabilities(dev)
        qjit_capabilities = get_qjit_device_capabilities(dev_capabilities)

        tape = qml.tape.QuantumScript([], measurements=[measurement])

        if isinstance(measurement, (ExpectationMP, VarianceMP)):
            validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)
        else:
            with pytest.raises(
                CompileError,
                match="Only expectation value and variance measurements can accept observables",
            ):
                validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)


class TestMeasurementTypeValidation:
    """Test validation of measurement processes versus the a device's supported
    measurement types"""

    @pytest.mark.parametrize(
        "measurement, shots, msg",
        [
            (qml.state(), 100, "Please specify shots=None."),
            (qml.sample(), None, "Please specify a finite number of shots."),
            (qml.expval(qml.X(0)), 100, "is not a supported measurement process"),
            (qml.expval(qml.X(0)), None, "is not a supported measurement process"),
        ],
    )
    def test_validate_measurements_works_on_measurement_processes(self, measurement, shots, msg):
        """Test that the validate_measurements transform raises a CompileError as
        expected for an unsupported MeasurementProcess"""

        dev = qml.device("lightning.qubit", wires=1, shots=shots)
        tape = qml.tape.QuantumScript([], measurements=[measurement])

        qjit_capabilities = get_device_capabilities(dev)
        qjit_capabilities.measurement_processes.remove("Expval")

        with pytest.raises(CompileError, match=msg):
            validate_measurements(tape, qjit_capabilities, dev.name, dev.shots)

    def test_state_measurements_rejected_with_shots(self):
        """Test that trying to measure a state on a device with finite shots
        raises a CompileError informing the user that shots must be None for
        state based measurements"""

        dev = qml.device("lightning.qubit", wires=1, shots=100)

        @qml.qnode(dev)
        def f():
            qml.RX(1.23, 0)
            return qml.state()

        with pytest.raises(CompileError, match="Please specify shots=None."):
            qml.qjit(f)()

    @pytest.mark.parametrize("measurement", [qml.sample, qml.counts])
    def test_sample_measurements_rejected_without_shots(self, measurement):
        """Test that trying to take a sample-based measurement on a device
        without shots raises a CompileError informing the user that a
        finite number of shots is needed for sampling"""

        dev = qml.device("lightning.qubit", wires=1, shots=None)

        @qml.qnode(dev)
        def f():
            qml.RX(1.23, 0)
            return measurement()

        with pytest.raises(CompileError, match="Please specify a finite number of shots."):
            qml.qjit(f)()

    def test_unsupported_measurement_types_rejected(self):
        """Test that trying to use a measurement type that is generally unsupported by
        the device raises a CompileError"""

        dev = qml.device("lightning.qubit", wires=1, shots=100)

        class MyMeasurement(qml.measurements.SampleMeasurement):
            """A custom measurement (not supported on lightning.qubit)"""

            def __init__(self, obs=None, wires=None):
                super().__init__(obs=obs, wires=wires, eigvals=None, id=None)

            def process_samples(self, samples, wire_order, shot_range, bin_size):
                """overwrite ABC method"""
                raise NotImplementedError

            def process_counts(self, counts, wire_order):
                """overwrite ABC method"""
                raise NotImplementedError

        @qml.qnode(dev)
        def f():
            qml.RX(1.23, 0)
            return MyMeasurement()

        with pytest.raises(CompileError, match="is not a supported measurement process"):
            qml.qjit(f)()


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
class TestAdjointMethodVerification:
    """Test the verification of operators and observables when the adjoint diff method
    is used for differentiation"""

    def test_non_differentiable_gate_simple(self):
        """Emulate a device with a non-differentiable gate."""

        @qml.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=[0]), diff_method="adjoint"
        )
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.PauliX(0),  # single observable
            qml.PauliX(0) @ qml.PauliZ(1),  # prod
            qml.operation.Tensor(qml.X(0), qml.Z(1)),  # tensor
            qml.PauliX(0) + qml.PauliY(1),  # sum
            qml.Hamiltonian([2, 3], [qml.X(0), qml.Y(1)]),  # hamiltonian
            2 * qml.PauliX(0),  # sprod
            (2 * qml.X(0) @ qml.Y(2)) @ (2 * qml.X(3)) @ qml.Y(1),  # nested prod+sprod
        ],
    )
    def test_non_differentiable_observable(self, observable):
        """Test that taking the adjoint diff of a circuit with an observable that doesn't support
        adjoint differentiation raises an error."""

        @qml.qnode(
            get_custom_device(non_differentiable_obs={"PauliX"}, wires=[0, 1]),
            diff_method="adjoint",
        )
        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(observable)

        with pytest.raises(DifferentiableCompileError, match="PauliX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_empty_observable(self):
        """Test that taking the adjoint diff of a circuit with an empyt observable with adjoint
        adjoint passes the validation."""

        @qml.qnode(
            get_custom_device(non_differentiable_obs={"PauliX"}, wires=[0]), diff_method="adjoint"
        )
        def f(x):
            qml.RX(x, wires=0)
            return qml.probs()

        qml.qjit(f)(1.2)

    def test_non_differentiable_gate_nested_cond(self):
        """Test that taking the adjoint diff of a tape containing a parameterized operation
        that doesn't support adjoint differentiation raises an error."""

        @qml.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint"
        )
        def f(x):
            @cond(True)
            def true_path():
                qml.RX(x, wires=0)

            @true_path.otherwise
            def false_path():
                qml.RX(x, wires=0)

            true_path()

            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    def test_non_differentiable_gate_nested_adjoint(self):
        """Test that taking the adjoint diff of a tape containing a HybridOp with a
        parameterized operation that doesn't support adjoint differentiation raises
        an error."""

        @qml.qnode(
            get_custom_device(non_differentiable_gates={"RX"}, wires=1), diff_method="adjoint"
        )
        def f(x):
            adjoint(qml.RX(x, wires=[0]))
            return qml.expval(qml.PauliX(0))

        with pytest.raises(DifferentiableCompileError, match="RX.*non-differentiable"):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)


@patch("catalyst.device.qjit_device.catalyst_decompose", null_transform)
class TestParameterShiftMethodVerification:
    """Test the verification of operators and observables when the parameter shift method
    is used for differentiation"""

    @pytest.mark.parametrize(
        "observable",
        [
            qml.PauliX(0),
            # qml.PauliX(0) @ qml.PauliZ(1),
            qml.operation.Tensor(qml.X(0), qml.Z(1)),
            # qml.PauliX(0)+qml.PauliY(1)
        ],
    )
    @patch.object(qml.PauliX, "grad_method", "F")
    def test_paramshift_obs_simple(self, observable):
        """Test that taking a parameter-shift gradient of an observable that doesn't support
        analytic differentiation raises an error."""

        assert qml.PauliX.grad_method != "A"

        @qml.qnode(get_custom_device(wires=2), diff_method="parameter-shift")
        def f(x):
            qml.PauliY(wires=1)
            qml.RX(x, wires=0)
            return qml.expval(observable)

        with pytest.raises(
            DifferentiableCompileError, match="PauliX does not support analytic differentiation"
        ):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @patch.object(qml.RX, "grad_method", "F")
    def test_paramshift_gate_simple(self):
        """Test that taking a parameter-shift gradient of a tape containing a parameterized
        operation that doesn't support analytic differentiation raises an error."""

        @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method="parameter-shift")
        def f(_):
            qml.RX(1.23, 0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(
            DifferentiableCompileError, match="RX does not support analytic differentiation"
        ):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)

    @patch.object(qml.RX, "grad_method", "F")
    def test_paramshift_gate_while(self):
        """Test that taking a parameter-shift gradient of a tape containing a WhileLoop HybridOp
        containing a parameterized operation that doesn't support analytic differentiation raises
        an error."""

        @qml.qnode(qml.device("lightning.qubit", wires=1), diff_method="parameter-shift")
        def f(_):
            @while_loop(lambda s: s > 0)
            def loop(s):
                qml.RX(1.23, 0)
                return s + 1

            loop(0)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(
            DifferentiableCompileError, match="RX does not support analytic differentiation"
        ):

            @qml.qjit
            def cir(x: float):
                return grad(f)(x)


def test_no_state_returns():
    """Test state returns are rejected in gradients."""

    @qml.qnode(get_custom_device(wires=1))
    def f(_):
        qml.PauliX(wires=0)
        return qml.state()

    with pytest.raises(DifferentiableCompileError, match="State returns.*forbidden"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


def test_no_variance_returns():
    """Test variance returns are rejected in gradients."""

    @qml.qnode(get_custom_device(wires=1))
    def f(_):
        qml.PauliX(wires=0)
        return qml.var(qml.PauliX(0))

    with pytest.raises(DifferentiableCompileError, match="Variance returns.*forbidden"):

        @qml.qjit
        def cir(x: float):
            return grad(f)(x)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
