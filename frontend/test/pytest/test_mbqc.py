# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test cases for capturing and compiling circuits with parametric arbitrary-basis mid-circuit
measurements from PennyLane's ftqc module in Catalyst.
"""

from functools import partial, reduce

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.ftqc as plft
import pytest

from catalyst import qjit
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.usefixtures("disable_capture")

mbqc_pipeline = [
    (
        "default-pipeline",
        [
            "enforce-runtime-invariants-pipeline",
            "hlo-lowering-pipeline",
            "quantum-compilation-pipeline",
            "bufferization-pipeline",
        ],
    ),
    (
        "mbqc-pipeline",
        [
            "convert-mbqc-to-llvm",
        ],
    ),
    (
        "llvm-dialect-lowering-pipeline",
        [
            "llvm-dialect-lowering-pipeline",
        ],
    ),
]


def test_measure_x():
    """Test the compilation of the qml.ftqc.measure_x function, which performs a mid-circuit
    measurement in the Pauli X basis.

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end. We expect expval(Z) to always return
    0.0 for the null.qubit device, but this is not guaranteed in future releases. The test therefore
    only asserts that the result is mathematically valid given this final measurement process, i.e.
    that it is in the range [-1, +1].
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        _ = plft.measure_x(0)
        return qml.expval(qml.Z(0))

    result = workload()
    qml.capture.disable()

    assert -1.0 <= result <= 1.0


def test_measure_y():
    """Test the compilation of the qml.ftqc.measure_y function, which performs a mid-circuit
    measurement in the Pauli Y basis.

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end. We expect expval(Z) to always return
    0.0 for the null.qubit device, but this is not guaranteed in future releases. The test therefore
    only asserts that the result is mathematically valid given this final measurement process, i.e.
    that it is in the range [-1, +1].
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        _ = plft.measure_y(0)
        return qml.expval(qml.Z(0))

    result = workload()
    qml.capture.disable()

    assert -1.0 <= result <= 1.0


def test_measure_z():
    """Test the compilation of the qml.ftqc.measure_z function, which performs a mid-circuit
    measurement in the Pauli Z basis. Including for completeness; measure_z() dispatches directly to
    qml.measure().

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end. We expect expval(Z) to always return
    0.0 for the null.qubit device, but this is not guaranteed in future releases. The test therefore
    only asserts that the result is mathematically valid given this final measurement process, i.e.
    that it is in the range [-1, +1].
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        _ = plft.measure_z(0)
        return qml.expval(qml.Z(0))

    result = workload()
    qml.capture.disable()

    assert -1.0 <= result <= 1.0


@pytest.mark.parametrize("angle", [-np.pi / 2, 0.0, np.pi / 2])
@pytest.mark.parametrize("plane", ["XY", "ZX", "YZ"])
def test_measure_measure_arbitrary_basis(angle, plane):
    """Test the compilation of the qml.ftqc.measure_arbitrary_basis function, which performs a
    mid-circuit measurement in an arbitrary basis defined by a plane and rotation angle about that
    plane on the supplied qubit.

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end. We expect expval(Z) to always return
    0.0 for the null.qubit device, but this is not guaranteed in future releases. The test therefore
    only asserts that the result is mathematically valid given this final measurement process, i.e.
    that it is in the range [-1, +1].
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        _ = plft.measure_arbitrary_basis(wires=0, angle=angle, plane=plane)
        return qml.expval(qml.Z(0))

    result = workload()
    qml.capture.disable()

    assert -1.0 <= result <= 1.0


@pytest.mark.parametrize("postselect", [0, 1])
def test_measure_measure_arbitrary_basis_postselect(postselect):
    """Test the compilation of the qml.ftqc.measure_arbitrary_basis function with a postselect
    argument.

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        _ = plft.measure_arbitrary_basis(wires=0, angle=0.1, plane="XY", postselect=postselect)
        return qml.expval(qml.Z(0))

    result = workload()
    qml.capture.disable()

    assert -1.0 <= result <= 1.0


def test_measure_measure_arbitrary_basis_invalid_plane():
    """Test that inputting an invalid ``plane`` parameter to qml.ftqc.measure_arbitrary_basis raises
    a ValueError.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    with pytest.raises(ValueError, match=r"Measurement plane must be one of \['XY', 'YZ', 'ZX'\]"):

        @qjit(pipelines=mbqc_pipeline)
        @qml.qnode(dev)
        def workload():
            _ = plft.measure_arbitrary_basis(wires=0, angle=0.1, plane="YX")
            return qml.expval(qml.Z(0))

        _ = workload()

    qml.capture.disable()


@pytest.mark.parametrize("postselect", [-1, 2])
def test_measure_measure_arbitrary_basis_invalid_postselect(postselect):
    """Test that inputting an invalid ``postselect`` parameter to qml.ftqc.measure_arbitrary_basis
    raises a CompileError.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    with pytest.raises(
        CompileError, match="op attribute 'postselect' failed to satisfy constraint"
    ):

        @qjit(pipelines=mbqc_pipeline)
        @qml.qnode(dev)
        def workload():
            _ = plft.measure_arbitrary_basis(wires=0, angle=0.1, plane="XY", postselect=postselect)
            return qml.expval(qml.Z(0))

        _ = workload()

    qml.capture.disable()


# ---------------------------------------------------------------------------- #
# Workloads implementing gates explicitly in the MBQC representation
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("rz_angle", [0.5])
def test_explicit_rz_in_mbqc(rz_angle):
    """Test the compilation of a circuit implementing the RZ gate in the MBQC representation
    following the pattern from

        Raussendorf, Browne & Briegel (2003), https://arxiv.org/abs/quant-ph/0301052

    Note that while the authors use qubit indices starting at 1, we index them starting at 0 for
    Catalyst compatibility.

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end.
    """
    dev = qml.device("null.qubit", wires=5)

    # Define the graph structure for the RZ cluster state (omit node 0 for input):
    # 0 -- 1 -- 2 -- 3 -- 4
    lattice = plft.generate_lattice([4], "chain")

    qml.capture.enable()

    # RZ circuit in the MBQC representation
    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev, mcm_method="tree-traversal")
    def circuit_mbqc(start_state, angle):
        # prep input node
        qml.StatePrep(start_state, wires=[0])

        # prep graph state
        qml.ftqc.make_graph_state(lattice.graph, wires=[1, 2, 3, 4])

        # entangle input and graph state
        qml.CZ([0, 1])

        # RZ measurement pattern
        m0 = qml.ftqc.measure_x(0)
        m1 = qml.ftqc.measure_x(1)
        m2 = qml.ftqc.cond_measure(
            m1,
            partial(qml.ftqc.measure_arbitrary_basis, angle=angle),
            partial(qml.ftqc.measure_arbitrary_basis, angle=-angle),
        )(plane="XY", wires=2)
        m3 = qml.ftqc.measure_x(3)

        # by-product corrections based on measurement outcomes
        qml.cond(m0 ^ m2, qml.Z, qml.I)(4)
        qml.cond(m1 ^ m3, qml.X, qml.I)(4)

        return qml.expval(qml.X(4)), qml.expval(qml.Y(4)), qml.expval(qml.Z(4))

    initial_state = np.array([1, 0], dtype=np.complex128)

    expval_x, expval_y, expval_z = circuit_mbqc(initial_state, rz_angle)

    qml.capture.disable()

    # We only assert that the expectation-value results are mathematically valid
    assert -1.0 <= expval_x <= 1.0
    assert -1.0 <= expval_y <= 1.0
    assert -1.0 <= expval_z <= 1.0


def test_cnot_in_mbqc_representation():
    """Test the compilation of a circuit implementing the CNOT gate in the MBQC representation
    following the pattern from

        Raussendorf, Browne & Briegel (2003), https://arxiv.org/abs/quant-ph/0301052

    Note that while the authors use qubit indices starting at 1, we index them starting at 0 for
    Catalyst compatibility.

    Executes on the null.qubit device. This test does not check the correctness of the result, only
    that the workload can be compiled and executed end-to-end.
    """
    dev = qml.device("null.qubit", wires=15)

    # Define the graph structure for the CNOT cluster state (omit nodes 0 and 8 for input)
    # 0 -- 1 -- 2 -- 3 -- 4 -- 5 -- 6
    #                |
    #                7
    #                |
    # 8 -- 9 -- 10 - 11 - 12 - 13 - 14
    aux_wires = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    g = nx.Graph()
    g.add_nodes_from(aux_wires)
    g.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (3, 7),
            (7, 11),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
        ]
    )

    def parity(*args):
        """Helper function to check the parity of the sum of a sequence of bools.

        Example:

        >>> parity(False, False, False)
        False
        >>> parity(False, False, True)
        True
        """
        return reduce(lambda a, b: a ^ b, args)

    qml.capture.enable()

    # Equivalent CNOT circuit in the MBQC representation
    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev, mcm_method="tree-traversal")
    def circuit_mbqc(start_state):
        # prep input nodes
        qml.StatePrep(start_state, wires=[1, 9])

        # prep graph state
        qml.ftqc.make_graph_state(g, wires=aux_wires)

        # entangle
        qml.CZ([0, 1])
        qml.CZ([8, 9])

        # CNOT measurement pattern
        m0 = qml.ftqc.measure_x(0)
        m1 = qml.ftqc.measure_y(1)
        m2 = qml.ftqc.measure_y(2)
        m3 = qml.ftqc.measure_y(3)
        m4 = qml.ftqc.measure_y(4)
        m5 = qml.ftqc.measure_y(5)
        m7 = qml.ftqc.measure_y(7)
        m8 = qml.ftqc.measure_x(8)
        m9 = qml.ftqc.measure_x(9)
        m10 = qml.ftqc.measure_x(10)
        m11 = qml.ftqc.measure_y(11)
        m12 = qml.ftqc.measure_x(12)
        m13 = qml.ftqc.measure_x(13)

        # corrections on controls
        x_cor = parity(m1, m2, m4, m5)
        z_cor = parity(m0, m2, m3, m4, m7, m8, m10, True)
        qml.cond(z_cor, qml.Z, qml.I)(7)
        qml.cond(x_cor, qml.X, qml.I)(7)

        # corrections on target
        x_cor = parity(m1, m2, m7, m9, m11, m13)
        z_cor = parity(m8, m10, m12)
        qml.cond(z_cor, qml.Z, qml.I)(14)
        qml.cond(x_cor, qml.X, qml.I)(14)

        return qml.expval(qml.Z(6)), qml.expval(qml.Z(14))

    initial_state = np.array([1, 0], dtype=np.complex128)

    expval_z_0, expval_z_1 = circuit_mbqc(initial_state)

    qml.capture.disable()

    assert -1.0 <= expval_z_0 <= 1.0
    assert -1.0 <= expval_z_1 <= 1.0


if __name__ == "__main__":
    pytest.main(["-x", __file__])
