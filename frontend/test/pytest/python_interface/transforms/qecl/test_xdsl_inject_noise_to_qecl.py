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

"""Test module for the inject-noise-to-qecl transform."""

import pennylane as qml
import pytest

from catalyst.python_interface.transforms.qecl import (
    InjectNoiseToQECLPass,
    convert_quantum_to_qecl_pass,
    inject_noise_to_qecl_pass,
)

pytestmark = pytest.mark.xdsl


class TestInjectNoiseToQECLPass:
    """Tests for the InjectNoiseToQECLPass."""

    @pytest.mark.usefixtures("use_capture")
    def test_inject_noise_to_qecl_pass(self, run_filecheck):
        """Test that the inject_noise_to_qecl_pass correctly injects noise into a QECL circuit."""
        program = """
            func.func @test_program() {
            // CHECK: [[cb0:%.+]] = "test.op"() : () -> !qecl.codeblock<1>
            %0 = "test.op"() : () -> !qecl.codeblock<1>

            // CHECK: [[cb1:%.+]] = qecl.noise [[cb0:%.+]] : !qecl.codeblock<1>
            // CHECK: qecl.qec [[cb1:%.+]] : !qecl.codeblock<1>
            %1 = qecl.qec %0 : !qecl.codeblock<1>

            // CHECK: [[mres:%.+]], [[cb2:%.+]] = qecl.measure [[cb1]][0] : i1, !qecl.codeblock<1>
            %3, %2 = qecl.measure %0[0] : i1, !qecl.codeblock<1>

            return
        }
        """

        pipeline = (InjectNoiseToQECLPass(),)

        run_filecheck(program, pipeline)


class TestInjectNoiseToQECLPassIntegration:
    """Integration lit tests for the inject-noise-to-qecl pass"""

    @pytest.mark.usefixtures("use_capture")
    def test_inject_noise_to_qecl_pass_integration(self, run_filecheck_qjit):
        """Test the inject-noise-to-qecl pass on the simplest possible, non-trivial circuit."""
        dev = qml.device("null.qubit", wires=1)

        @qml.qjit(target="mlir")
        @inject_noise_to_qecl_pass
        @convert_quantum_to_qecl_pass(k=1)
        @qml.qnode(dev, shots=1)
        def circuit():
            # CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            # CHECK: qecl.extract_block {{%.+}}[0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.insert_block {{%.+}}[0], {{%.+}}
            # CHECK: qecl.extract_block
            # CHECK: qecl.noise
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard {{%.+}}[0]
            # CHECK: qecl.noise
            # CHECK: qecl.qec
            # CHECK: qecl.measure {{%.+}}[0]
            # CHECK: quantum.mcmobs
            # CHECK: quantum.sample
            # CHECK: qecl.insert_block
            # CHECK: qecl.dealloc
            qml.H(0)
            m0 = qml.measure(0)
            return qml.sample([m0])

        run_filecheck_qjit(circuit)
