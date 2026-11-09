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

"""Tests for delegated feed-forward: ``mcm_method="device"`` under ``qjit(capture=True)``."""

import pennylane as qp
import pytest

from catalyst.debug import get_compilation_stage
from catalyst.passes import ppm_compilation, ppr_to_ppm, to_ppr, verify_pbc_gadgets
from catalyst.utils.exceptions import CompileError


def clifford_t():
    """A small Clifford+T circuit."""
    qp.H(0)
    qp.T(0)
    qp.CNOT([0, 1])
    qp.adjoint(qp.T(1))


def test_device_method_delegates_feedforward():
    """A PBC program on a device that supports "device" runs once, with its feed-forward as
    data."""

    @qp.qjit(capture=True, keep_intermediate=True)
    @ppm_compilation(emit_gadgets=True)
    @qp.qnode(qp.device("null.qubit", wires=2), shots=100, mcm_method="device")
    def delegated_circuit():
        clifford_t()
        return qp.sample(wires=[0, 1])

    samples = delegated_circuit()
    assert qp.math.shape(samples) == (100, 2)
    ir = delegated_circuit.mlir_opt
    assert "__catalyst__rt__device_require" in ir
    assert "__catalyst__qis__BeginDelegatedProgram" in ir
    assert "__catalyst__qis__PushPredicate" in ir
    assert "__catalyst__qis__BeginGadget" in ir
    # Compiled code no longer reads measurement bits.
    assert "llvm.load" not in "\n".join(l for l in ir.splitlines() if "PauliMeasure" in l)
    delegated_circuit.workspace.cleanup()


def test_device_method_needs_the_capability():
    """mcm_method="device" fails at compile time on a device that does not list it."""

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1), shots=10, mcm_method="device")
    def circuit():
        qp.H(0)
        return qp.sample(wires=[0])

    # lightning.qubit lists "device" for its Python path; under qjit its library has no
    # Delegated extension, so either the compiler or the load check refuses it.
    with pytest.raises((CompileError, RuntimeError)):
        circuit()


def test_gadget_verdicts():
    """Gadgets carry the verdict of the compiler's own check."""

    @qp.qjit(capture=True, keep_intermediate=True)
    @verify_pbc_gadgets
    @ppr_to_ppm(emit_gadgets=True)
    @to_ppr
    @qp.qnode(qp.device("null.qubit", wires=2))
    def verified_circuit():
        clifford_t()
        return qp.expval(qp.Z(0))

    verified_circuit()
    ir = get_compilation_stage(verified_circuit, "QuantumCompilationStage")
    assert ir.count("pbc.gadget") > 0
    assert ir.count("verdict(verified)") == ir.count("pbc.gadget")
    verified_circuit.workspace.cleanup()


@pytest.mark.parametrize(
    "method, avoid_y_measure",
    [("auto-corrected", False), ("clifford-corrected", False), ("pauli-corrected", True)],
)
def test_strict_verification_refuses_wrong_gadgets(method, avoid_y_measure):
    """Catalyst's known pi/8 decomposition bugs fail strict verification."""

    @qp.qjit(capture=True, target="mlir")
    @verify_pbc_gadgets(strict=True)
    @ppr_to_ppm(decompose_method=method, avoid_y_measure=avoid_y_measure, emit_gadgets=True)
    @to_ppr
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        qp.adjoint(qp.T(0))
        return qp.expval(qp.Z(0))

    with pytest.raises(CompileError, match="does not implement its source rotation"):
        _ = circuit.mlir_opt


def test_pauli_measure_postselect_reaches_the_compiler():
    """The postselect of qp.pauli_measure is kept under capture."""

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        qp.pauli_measure("XZ", wires=[0, 1], postselect=1)
        return qp.expval(qp.Z(0))

    assert "postselect 1" in circuit.mlir
