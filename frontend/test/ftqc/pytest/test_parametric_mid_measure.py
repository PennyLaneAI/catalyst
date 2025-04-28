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

"""Test cases for capturing and compiling circuits with parametric mid-circuit measurements from
PennyLane's ftqc module in Catalyst.
"""

import numpy as np
import pennylane as qml
import pennylane.ftqc as plft
import pytest

from catalyst import qjit

mbqc_pipeline = [
    (
        "device-agnostic-pipeline",
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

    Executes on the null.qubit device; does not check correctness of the result.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        m0 = plft.measure_x(0)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    result = workload()
    assert result == 0.0


def test_measure_y():
    """Test the compilation of the qml.ftqc.measure_y function, which performs a mid-circuit
    measurement in the Pauli Y basis.

    Executes on the null.qubit device; does not check correctness of the result.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        m0 = plft.measure_y(0)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    result = workload()
    assert result == 0.0


@pytest.mark.xfail(reason="qml.ftqc.measure_z is not yet supported with program capture")
def test_measure_z():
    """Test the compilation of the qml.ftqc.measure_z function, which performs a mid-circuit
    measurement in the Pauli Z basis. Including for completeness; measure_z() dispatches directly to
    qml.measure().

    Executes on the null.qubit device; does not check correctness of the result.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        m0 = plft.measure_z(0)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    result = workload()
    assert result == 0.0


@pytest.mark.parametrize("angle", [-np.pi / 2, 0.0, np.pi / 2])
@pytest.mark.parametrize("plane", ["XY", "ZX", "YZ"])
def test_measure_measure_arbitrary_basis(angle, plane):
    """Test the compilation of the qml.ftqc.measure_arbitrary_basis function, which performs a
    mid-circuit measurement in an arbitrary basis defined by a plane and rotation angle about that
    plane on the supplied qubit.

    Executes on the null.qubit device; does not check correctness of the result.
    """
    dev = qml.device("null.qubit", wires=1)

    qml.capture.enable()

    @qjit(pipelines=mbqc_pipeline)
    @qml.qnode(dev)
    def workload():
        m0 = plft.measure_arbitrary_basis(wires=0, angle=angle, plane=plane)
        return qml.expval(qml.Z(0))

    qml.capture.disable()

    result = workload()
    assert result == 0.0


if __name__ == "__main__":
    pytest.main(["-x", __file__])
