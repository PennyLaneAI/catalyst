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
from pennylane.tape import QuantumTape

from catalyst import cond, for_loop, qjit, while_loop
from catalyst.api_extensions.control_flow import Cond, ForLoop, WhileLoop
from catalyst.jax_tracer import HybridOp, has_nested_tapes


def test_no_parameters(backend):
    """Test no-param operations."""

    def circuit():
        qml.Identity(wires=0)

        qml.PauliX(wires=1)
        qml.PauliY(wires=2)
        qml.PauliZ(wires=0)

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.S(wires=0)
        qml.T(wires=0)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 0])

        qml.CY(wires=[0, 1])
        qml.CY(wires=[0, 2])

        qml.CZ(wires=[0, 1])
        qml.CZ(wires=[0, 2])

        qml.SWAP(wires=[0, 1])
        qml.SWAP(wires=[0, 2])
        qml.SWAP(wires=[1, 2])
        qml.ISWAP(wires=[0, 1])

        qml.CSWAP(wires=[0, 1, 2])

        U1 = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
        qml.QubitUnitary(U1, wires=0)

        # To check if the generated qubit out of `wires=0` can be reused by another gate
        # and `quantum-opt` doesn't fail to materialize conversion for the result
        qml.Hadamard(wires=0)

        U2 = np.array(
            [
                [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
            ]
        )
        qml.QubitUnitary(U2, wires=[1, 2])

        # To check if the generated qubits out of `wires=[1, 2]` can be reused by other gates
        # and `quantum-opt` doesn't fail to materialize conversion for the result
        qml.CZ(wires=[1, 2])

        qml.MultiControlledX(wires=[0, 1, 2, 3])

        qml.BlockEncode(np.array([[1, 1, 1], [0, 1, 0]]), wires=[0, 1, 2])

        # Unsupported:
        # qml.SX(wires=0)
        # qml.ECR(wires=[0,1])
        # qml.SISWAP(wires=[0,1])
        # qml.Toffoli(wires=[0,1,2])

        return qml.state()

    qjit_fn = qjit()(qml.qnode(qml.device(backend, wires=4))(circuit))
    qml_fn = qml.qnode(qml.device("default.qubit", wires=4))(circuit)

    assert np.allclose(qjit_fn(), qml_fn())


def test_param(backend):
    """Test param operations."""

    def circuit(x: float, y: float):
        qml.Rot(x, y, x + y, wires=0)

        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(x, wires=2)

        qml.RZ(y, wires=0)
        qml.RY(x, wires=1)
        qml.RX(y, wires=2)

        qml.PhaseShift(x, wires=0)
        qml.PhaseShift(y, wires=1)

        qml.IsingXX(x, wires=[0, 1])
        qml.IsingXX(y, wires=[1, 2])

        qml.IsingYY(x, wires=[0, 1])
        qml.IsingYY(y, wires=[1, 2])

        qml.IsingXY(x, wires=[0, 1])
        qml.IsingXY(y, wires=[1, 2])

        qml.IsingZZ(x, wires=[0, 1])
        qml.IsingZZ(y, wires=[1, 2])

        qml.CRX(x, wires=[0, 1])
        qml.CRY(x, wires=[0, 1])
        qml.CRZ(x, wires=[0, 1])

        qml.CRX(y, wires=[1, 2])
        qml.CRY(y, wires=[1, 2])
        qml.CRZ(y, wires=[1, 2])

        qml.PSWAP(x, wires=[0, 1])

        qml.MultiRZ(x, wires=[0, 1, 2, 3])

        # Unsupported:
        # qml.PauliRot(x, 'IXYZ', wires=[0,1,2,3])
        # qml.U1(x, wires=0)
        # qml.U2(x, x, wires=0)
        # qml.U3(x, x, x, wires=0)

        return qml.state()

    qjit_fn = qjit()(qml.qnode(qml.device(backend, wires=4))(circuit))
    qml_fn = qml.qnode(qml.device("default.qubit", wires=4))(circuit)

    assert np.allclose(qjit_fn(3.14, 0.6), qml_fn(3.14, 0.6))


def test_hybrid_op_repr(backend):
    """Test hybrid operation representation"""

    def circuit(n):
        quantum_tape = QuantumTape()
        with qml.QueuingManager.stop_recording(), quantum_tape:

            @for_loop(0, 1, 1)
            def loop0(_):
                qml.RX(np.pi, wires=0)
                return ()

            loop0()

            @while_loop(lambda v: v < 1)
            def loop1(v):
                qml.RY(np.pi, wires=0)
                return v + 1

            loop1(0)

            @cond(n == 1)
            def cond_fn():
                qml.RZ(np.pi, wires=0)
                return 0

            @cond_fn.otherwise
            def cond_fn():
                qml.RZ(np.pi, wires=0)
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
        return qml.state()

    qjit()(qml.qnode(qml.device(backend, wires=4))(circuit))(1)


@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_qubitunitary_complex(inp, backend):
    """Test qubitunitary with complex matrix."""

    def f(x):
        qml.RX(x, wires=0)
        U1 = np.array([[0.5 + 0.5j, -0.5 - 0.5j], [0.5 - 0.5j, 0.5 - 0.5j]], dtype=complex)
        qml.QubitUnitary(U1, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit()
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1))(f)
        return g(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device)
        return g(x)

    assert np.allclose(compiled(inp), interpreted(inp))


def test_multicontrolledx_via_paulix():
    """Test that lightning executes multicontrolled x via paulix rather than qubit unitary."""

    dev = qml.device("lightning.qubit", wires=4)

    @qjit
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.Hadamard(2)
        qml.MultiControlledX(wires=[0, 1, 2, 3], control_values=[True, False, True])
        return qml.state()

    assert "QubitUnitary" not in str(circuit.jaxpr)
    assert "PauliX" in str(circuit.jaxpr)

    assert np.allclose(circuit(), circuit.original_function())


if __name__ == "__main__":
    pytest.main(["-x", __file__])
