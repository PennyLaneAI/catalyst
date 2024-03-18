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
"""Test for the OQC device.
"""
# pylint: disable=unused-argument,import-outside-toplevel
import pathlib
import sys

import pennylane as qml
import pytest

from catalyst.compiler import get_lib_path


# TODO: replace when the OQC CPP layer is available.
@pytest.mark.skipif(
    not pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so").is_file(),
    reason="lib_dummydevice.so was not found.",
)
def test_unavailable(monkeypatch, set_dummy_oqc_env):
    """Check the error produced in the absence of qcaas."""
    monkeypatch.setitem(sys.modules, "qcaas_client", None)
    with pytest.raises(
        ImportError,
        match="Oqc qcaas client not found. Please install: pip install oqc-qcaas-client",
    ):
        from catalyst.oqc import OQCDevice

        OQCDevice(backend="lucy", shots=1000, wires=8)


# TODO: replace when the OQC CPP layer is available.
@pytest.mark.skipif(
    not pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so").is_file(),
    reason="lib_dummydevice.so was not found.",
)
class TestOQCDevice:
    """Test the OQC device python layer for Catalyst."""

    def test_initialization(self, set_dummy_oqc_env):
        """Test the initialization."""
        from catalyst.oqc import OQCDevice

        device = OQCDevice(backend="lucy", shots=1000, wires=8)

        assert device.backend == "lucy"
        assert device.shots == qml.measurements.Shots(1000)
        assert device.wires == qml.wires.Wires(range(0, 8))

        device = OQCDevice(backend="toshiko", shots=1000, wires=32)

        assert device.backend == "toshiko"
        assert device.shots == qml.measurements.Shots(1000)
        assert device.wires == qml.wires.Wires(range(0, 32))

    def test_wrong_backend(self, set_dummy_oqc_env):
        """Test the backend check."""
        from catalyst.oqc import OQCDevice

        with pytest.raises(ValueError, match="The backend falcon is not supported."):
            OQCDevice(backend="falcon", shots=1000, wires=8)

    def test_execute_not_implemented(self, set_dummy_oqc_env):
        """Test the python execute is not implemented."""
        from catalyst.oqc import OQCDevice

        with pytest.raises(NotImplementedError, match="The OQC device only supports Catalyst."):
            dev = OQCDevice(backend="lucy", shots=1000, wires=8)
            dev.execute([], [])

    def test_preprocess(self, set_dummy_oqc_env):
        """Test the device preprocessing"""
        from catalyst.oqc import OQCDevice

        dev = OQCDevice(backend="lucy", shots=1000, wires=8)
        tranform_program, _ = dev.preprocess()
        assert tranform_program == qml.transforms.core.TransformProgram()

    def test_get_c_interface(self, set_dummy_oqc_env):
        """Test the device get_c_interface method."""
        from catalyst.oqc import OQCDevice

        dev = OQCDevice(backend="lucy", shots=1000, wires=8)
        name, _ = dev.get_c_interface()
        assert name == "oqc"

    def test_no_envvar(self):
        """Test the device get_c_interface method."""
        from catalyst.oqc import OQCDevice

        with pytest.raises(
            ValueError, match="You must set url, email and password as environment variables."
        ):
            OQCDevice(backend="lucy", shots=1000, wires=8)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
