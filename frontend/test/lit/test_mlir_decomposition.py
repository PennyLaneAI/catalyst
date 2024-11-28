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

"""
This file performs the frontend lit tests that the peephole transformations are correctly lowered.

We check the transform jax primitives for each pass is correctly injected
during tracing, and these transform primitives are correctly lowered to the mlir before
running -apply-transform-sequence. 
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import os
import pathlib
import platform
import shutil

import pennylane as qml
from lit_util_printers import print_jaxpr
from pennylane.devices import NullQubit

from catalyst import qjit
from catalyst.debug import get_compilation_stage
from catalyst.utils.runtime_environment import get_lib_path

TEST_PATH = os.path.dirname(__file__)
CONFIG_CUSTOM_DEVICE = pathlib.Path(f"{TEST_PATH}/../custom_device/custom_device.toml")


def flush_peephole_opted_mlir_to_iostream(func):
    """
    The QJIT compiler does not offer a direct interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace
    """
    print(get_compilation_stage(func, "QuantumCompilationPass"))
    shutil.rmtree(func.__name__)


class CustomDevice(NullQubit):
    """Custom Gate Set Device"""

    name = "oqd.cloud"

    config_filepath = CONFIG_CUSTOM_DEVICE

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = (
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
        )
        return "NullQubit", lib_path


def test_decomposition_lowering():
    """
    Basic pipeline lowering on one qnode.
    """

    @qjit(keep_intermediate=True)
    @qml.qnode(CustomDevice(2))
    def test_decomposition_lowering_workflow(x):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   pass_name=ions-decomposition
    # CHECK: ]
    print_jaxpr(test_decomposition_lowering_workflow, 1.2)
    # CHECK: quantum.custom "RX"
    # CHECK-NOT: quantum.custom "Hadamard"
    # CHECK: quantum.custom "RX"
    # CHECK: quantum.custom "RY"
    # CHECK: quantum.custom "RX"
    # CHECK: quantum.custom "RX"
    # CHECK: quantum.custom "RY"
    # CHECK: quantum.custom "RX"
    flush_peephole_opted_mlir_to_iostream(test_decomposition_lowering_workflow)


test_decomposition_lowering()
