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
import pennylane as qp
import pytest
from pennylane.tape import QuantumTape
from pennylane_lightning.lightning_qubit.lightning_qubit import OperatorProperties
from utils import get_custom_qjit_device

from catalyst import cond, for_loop, qjit, while_loop
from catalyst.api_extensions.control_flow import Cond, ForLoop, WhileLoop
from catalyst.jax_tracer import HybridOp, has_nested_tapes
from catalyst.utils.exceptions import CompileError


def test_no_parameters(backend):
    """Test no-param operations."""

    def circuit():
        qp.Identity(wires=0)
        qp.Identity(wires=[0, 1])

        qp.PauliX(wires=1)
        qp.PauliY(wires=2)
        qp.PauliZ(wires=0)

        qp.Hadamard(wires=0)
        qp.Hadamard(wires=1)
        qp.Hadamard(wires=2)

        qp.S(wires=0)
        qp.T(wires=0)

        qp.CNOT(wires=[0, 1])
        qp.CNOT(wires=[1, 0])

        qp.CY(wires=[0, 1])
        qp.CY(wires=[0, 2])

        qp.CZ(wires=[0, 1])
        qp.CZ(wires=[0, 2])

        qp.SWAP(wires=[0, 1])
        qp.SWAP(wires=[0, 2])
        qp.SWAP(wires=[1, 2])
        qp.ISWAP(wires=[0, 1])

        qp.CSWAP(wires=[0, 1, 2])

        U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
        qp.QubitUnitary(U1, wires=0)

        # To check if the generated qubit out of `wires=0` can be reused by another gate
        # and `quantum-opt` doesn't fail to materialize conversion for the result
        qp.Hadamard(wires=0)

        U2 = np.array(
            [
                [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
            ]
        )
        qp.QubitUnitary(U2, wires=[1, 2])

        # To check if the generated qubits out of `wires=[1, 2]` can be reused by other gates
        # and `quantum-opt` doesn't fail to materialize conversion for the result
        qp.CZ(wires=[1, 2])

        qp.MultiControlledX(wires=[0, 1, 2, 3])

        qp.BlockEncode(np.array([[1, 1, 1], [0, 1, 0]]), wires=[0, 1, 2])

        # Unsupported:
        # qp.SX(wires=0)
        # qp.ECR(wires=[0,1])
        # qp.SISWAP(wires=[0,1])
        # qp.Toffoli(wires=[0,1,2])

        return qp.state()

    qjit_fn = qjit(qp.qnode(qp.device(backend, wires=4))(circuit))
    qp_fn = qp.qnode(qp.device("default.qubit", wires=4))(circuit)

    assert np.allclose(qjit_fn(), qp_fn())


def test_param(backend):
    """Test param operations."""

    def circuit(x: float, y: float):
        qp.Rot(x, y, x + y, wires=0)

        qp.RX(x, wires=0)
        qp.RY(y, wires=1)
        qp.RZ(x, wires=2)

        qp.RZ(y, wires=0)
        qp.RY(x, wires=1)
        qp.RX(y, wires=2)

        qp.PhaseShift(x, wires=0)
        qp.PhaseShift(y, wires=1)

        qp.IsingXX(x, wires=[0, 1])
        qp.IsingXX(y, wires=[1, 2])

        qp.IsingYY(x, wires=[0, 1])
        qp.IsingYY(y, wires=[1, 2])

        qp.IsingXY(x, wires=[0, 1])
        qp.IsingXY(y, wires=[1, 2])

        qp.IsingZZ(x, wires=[0, 1])
        qp.IsingZZ(y, wires=[1, 2])

        qp.SingleExcitation(x, wires=[0, 1])
        qp.SingleExcitation(y, wires=[1, 2])

        qp.DoubleExcitation(x, wires=[0, 1, 2, 3])
        qp.DoubleExcitation(y, wires=[2, 3, 0, 1])

        qp.CRX(x, wires=[0, 1])
        qp.CRY(x, wires=[0, 1])
        qp.CRZ(x, wires=[0, 1])

        qp.CRX(y, wires=[1, 2])
        qp.CRY(y, wires=[1, 2])
        qp.CRZ(y, wires=[1, 2])

        qp.PSWAP(x, wires=[0, 1])

        qp.MultiRZ(x, wires=[0, 1, 2, 3])

        qp.PCPhase(x, dim=2, wires=[0, 1, 2, 3])

        # Unsupported:
        # qp.PauliRot(x, 'IXYZ', wires=[0,1,2,3])
        # qp.U1(x, wires=0)
        # qp.U2(x, x, wires=0)
        # qp.U3(x, x, x, wires=0)

        return qp.state()

    qjit_fn = qjit(qp.qnode(qp.device(backend, wires=4))(circuit))
    qp_fn = qp.qnode(qp.device("default.qubit", wires=4))(circuit)

    assert np.allclose(qjit_fn(3.14, 0.6), qp_fn(3.14, 0.6))


def test_hybrid_op_repr(backend):
    """Test hybrid operation representation"""

    def circuit(n):
        quantum_tape = QuantumTape()
        with qp.QueuingManager.stop_recording(), quantum_tape:

            @for_loop(0, 1, 1)
            def loop0(_):
                qp.RX(np.pi, wires=0)
                return ()

            loop0()

            @while_loop(lambda v: v < 1)
            def loop1(v):
                qp.RY(np.pi, wires=0)
                return v + 1

            loop1(0)

            @cond(n == 1)
            def cond_fn():
                qp.RZ(np.pi, wires=0)
                return 0

            @cond_fn.otherwise
            def cond_fn():
                qp.RZ(np.pi, wires=0)
                return 1

            cond_fn()

        for term in ["ForLoop", "WhileLoop", "Cond", "RX", "RY", "RZ"]:
            assert term in str(quantum_tape.operations)
        for op in quantum_tape.operations:
            if isinstance(op, (ForLoop, WhileLoop, Cond)):
                assert (
                    isinstance(op, HybridOp)
                    and len(op.regions) > 0
                    and any(r.quantum_tape is not None for r in op.regions)
                )
                assert has_nested_tapes(op)
            else:
                assert not has_nested_tapes(op)
        return qp.state()

    qjit(qp.qnode(qp.device(backend, wires=4))(circuit))(1)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_qubitunitary_complex(inp, backend):
    """Test qubitunitary with complex matrix."""

    def f(x):
        qp.RX(x, wires=0)
        U1 = np.array([[0.5 + 0.5j, -0.5 - 0.5j], [0.5 - 0.5j, 0.5 - 0.5j]], dtype=complex)
        qp.QubitUnitary(U1, wires=0)
        return qp.expval(qp.PauliY(0))

    @qjit
    def compiled(x: float):
        g = qp.qnode(qp.device(backend, wires=1))(f)
        return g(x)

    def interpreted(x):
        device = qp.device("default.qubit", wires=1)
        g = qp.QNode(f, device)
        return g(x)

    assert np.allclose(compiled(inp), interpreted(inp))


def test_multicontrolledx_via_paulix():
    """Test that lightning executes multicontrolled x via paulix rather than qubit unitary."""

    dev = qp.device("lightning.qubit", wires=4)

    @qjit
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.Hadamard(1)
        qp.Hadamard(2)
        qp.MultiControlledX(wires=[0, 1, 2, 3], control_values=[True, False, True])
        return qp.state()

    assert "QubitUnitary" not in str(circuit.jaxpr)
    assert "PauliX" in str(circuit.jaxpr)

    assert np.allclose(circuit(), circuit.original_function())


def test_to_matrix_ops():
    """Test that devices with ``to_matrix_ops`` should have support for ``QubitUnitary``."""
    dev = get_custom_qjit_device(
        num_wires=1,
        discards=("QubitUnitary",),
        additions={"Rot": OperatorProperties(True, True, False)},
        to_matrix_ops={"Rot"},
    )

    def qfunc(x, y, z):
        qp.Rot(x, y, z, wires=[0])
        return qp.state()

    circuit = qjit(qp.qnode(dev)(qfunc))

    with pytest.raises(
        CompileError, match="The device that specifies to_matrix_ops must support QubitUnitary"
    ):
        circuit(0.3, 0.4, 0.5)

    # Test that if we pass `None`` for `to_matrix_ops`, and exclude `QubitUnitary` from the
    # capabilities, the device can successfully compile the circuit by decomposing to the
    # target gateset.
    # Related to https://github.com/PennyLaneAI/pennylane-lightning/pull/1348
    dev = get_custom_qjit_device(
        num_wires=1, discards=("QubitUnitary", "Rot"), additions=set(), to_matrix_ops=None
    )

    circuit = qjit(qp.qnode(dev)(qfunc))
    circuit(0.3, 0.4, 0.5)  # should compile successfully


if __name__ == "__main__":
    pytest.main(["-x", __file__])
