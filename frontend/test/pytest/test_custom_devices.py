# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test for custom device integration with Catalyst.
"""
import platform

import pennylane as qml
import pytest

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path
from catalyst.device import extract_backend_info, get_device_capabilities
from catalyst.utils.exceptions import CompileError

# These have to match the ones in the configuration file.
OPERATIONS = [
    "QubitUnitary",
    "PauliX",
    "PauliY",
    "PauliZ",
    "MultiRZ",
    "Hadamard",
    "S",
    "T",
    "CNOT",
    "SWAP",
    "CSWAP",
    "Toffoli",
    "CY",
    "CZ",
    "PhaseShift",
    "ControlledPhaseShift",
    "RX",
    "RY",
    "RZ",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "Identity",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "IsingXY",
    "SX",
    "ISWAP",
    "PSWAP",
    "SISWAP",
    "SQISW",
    "BasisState",
    "QubitStateVector",
    "StatePrep",
    "ControlledQubitUnitary",
    "DiagonalQubitUnitary",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "QubitCarry",
    "QubitSum",
    "OrbitalRotation",
    "QFT",
    "ECR",
    "Adjoint(S)",
    "Adjoint(T)",
    "Adjoint(SX)",
    "Adjoint(ISWAP)",
    "Adjoint(SISWAP)",
    "MultiControlledX",
    "SISWAP",
    "ControlledPhaseShift",
    "C(QubitUnitary)",
    "C(PauliY)",
    "C(RY)",
    "C(PauliX)",
    "C(RX)",
    "C(IsingXX)",
    "C(Hadamard)",
    "C(SWAP)",
    "C(IsingYY)",
    "C(S)",
    "C(MultiRZ)",
    "C(PhaseShift)",
    "C(T)",
    "C(IsingXY)",
    "C(PauliZ)",
    "C(Rot)",
    "C(IsingZZ)",
    "C(RZ)",
    "C(SingleExcitationPlus)",
    "C(GlobalPhase)",
    "C(DoubleExcitationPlus)",
    "C(SingleExcitationMinus)",
    "C(DoubleExcitation)",
    "GlobalPhase",
    "C(SingleExcitation)",
    "C(DoubleExcitationMinus)",
    "BlockEncode",
]
OBSERVABLES = [
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Hermitian",
    "Identity",
    "Projector",
    "Hamiltonian",
    "SparseHamiltonian",
    "Sum",
    "SProd",
    "Prod",
    "Exp",
]

RUNTIME_LIB_PATH = get_lib_path("runtime", "RUNTIME_LIB_DIR")


def test_custom_device_load():
    """Test that custom device can run using Catalyst."""

    class NullQubit(qml.devices.QubitDevice):
        """Null Qubit"""

        name = "Null Qubit"
        short_name = "null.qubit"
        pennylane_requires = "0.33.0"
        version = "0.0.1"
        author = "Dummy"

        operations = OPERATIONS
        observables = OBSERVABLES

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
            self._option1 = 42

        def apply(self, operations, **kwargs):
            """Unused"""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """
            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR")
                + "/librtd_null_qubit"
                + system_extension
            )
            return "NullQubit", lib_path

    device = NullQubit(wires=1)
    capabilities = get_device_capabilities(device)
    backend_info = extract_backend_info(device, capabilities)
    assert backend_info.kwargs["option1"] == 42
    assert "option2" not in backend_info.kwargs

    @qjit
    @qml.qnode(device)
    def f():
        """This function would normally return False.
        However, NullQubit as defined in librtd_null_qubit.so
        has been implemented to always return True."""
        return measure(0)

    assert True == f()


def test_custom_device_bad_directory():
    """Test that custom device error."""

    class DummyDevice(qml.devices.QubitDevice):
        """Dummy Device"""

        name = "Null Qubit"
        short_name = "null.qubit"
        pennylane_requires = "0.33.0"
        version = "0.0.1"
        author = "Dummy"

        operations = OPERATIONS
        observables = OBSERVABLES

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Unused."""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """

            return "DummyDevice", "this-file-does-not-exist.so"

    with pytest.raises(
        CompileError, match="Device at this-file-does-not-exist.so cannot be found!"
    ):

        @qjit
        @qml.qnode(DummyDevice(wires=1))
        def f():
            return measure(0)


def test_custom_device_no_c_interface():
    """Test that custom device error."""

    class DummyDevice(qml.devices.QubitDevice):
        """Dummy Device"""

        name = "Null Qubit"
        short_name = "null.qubit"
        pennylane_requires = "0.33.0"
        version = "0.0.1"
        author = "Dummy"

        operations = OPERATIONS
        observables = OBSERVABLES

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            """Unused."""
            raise RuntimeError("Dummy device")

    with pytest.raises(
        CompileError, match="The null.qubit device does not provide C interface for compilation."
    ):

        @qjit
        @qml.qnode(DummyDevice(wires=1))
        def f():
            return measure(0)
