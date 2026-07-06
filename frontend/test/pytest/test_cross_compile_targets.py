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

"""Integration test for the target cross-compilation + remote-dispatch chain.
"""

import os

import numpy as np
import pennylane as qp

from catalyst import qjit, target
from catalyst.compiler import CompileOptions
from catalyst.debug import get_compilation_stage
from catalyst.jit import QJIT


def test_remote_dispatch_full_chain():
    """A ``target(..., address=...)`` QNode is cross-compiled to its own object and its host call is
    rewritten for remote dispatch."""
    dev = target(qp.device("null.qubit", wires=2), address="ADDR:PORT")

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.state()

    jitted = QJIT(circuit, CompileOptions(keep_intermediate=True, link=False))
    try:
        target_object = os.path.join(str(jitted.workspace), "module_circuit", "module_circuit.o")
        assert os.path.exists(target_object)

        ir = get_compilation_stage(jitted, "CrossCompileTargets")
        assert "remote.open" in ir
        assert "remote.send_binary" in ir
        assert "remote.launch" in ir
        # Teardown close is handled by the runtime at process exit, not emitted as an op.
        assert "remote.close" not in ir
        assert "ADDR:PORT" in ir
        assert target_object in ir
        assert "module @module_circuit" not in ir
    finally:
        jitted.workspace.cleanup()


def test_local_target_compiles_links_and_runs():
    """A ``target``-tagged (non-remote) QNode is cross-compiled to its own object, statically linked,
    and executed in-process — its host launch_kernel is flattened to a flat call into the object."""
    # No ``address`` and no explicit triple -> host triple -> local static-link path.
    dev = target(qp.device("lightning.qubit", wires=2))

    @qjit
    @qp.qnode(dev)
    def circuit(x: float):
        qp.RX(x, wires=0)
        qp.CNOT(wires=[0, 1])
        return qp.expval(qp.PauliZ(0))

    assert np.isclose(circuit(0.0), 1.0)
    assert np.isclose(circuit(np.pi), -1.0)
