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

"""Test module for the convert-qecl-to-qecp dialect-conversion transform."""

import pennylane as qp
import pytest

from catalyst.python_interface.transforms.qecl import (
    convert_quantum_to_qecl_pass,
    inject_noise_to_qecl_pass,
)
from catalyst.python_interface.transforms.qecp import (
    ConvertQecLogicalToQecPhysicalPass,
    convert_qecl_to_qecp_pass,
)

pytestmark = pytest.mark.xdsl


class TestQECLNoiseLoweringPassIntegration:
    """Integration lit tests for the convert-qecl-noise-to-qecp-noise pass"""

    @pytest.mark.usefixtures("use_capture")
    def test_convert_qecl_noise_to_qecp_noise_pass_integration(self, run_filecheck_qjit):
        """Test the convert-qecl-noise-to-qecp-noise pass on the simplest possible, non-trivial circuit."""
        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(target="mlir", keep_intermediate=True)
        @convert_qecl_to_qecp_pass(qec_code="steane[[7,1,3]]", number_errors=1)
        @inject_noise_to_qecl_pass
        @convert_quantum_to_qecl_pass(k=1)
        @qp.qnode(dev, shots=1)
        def circuit():
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecl.codeblock<1> to !qecp.codeblock<1 x 7>
            # CHECK: arith.constant dense
            # CHECK: arith.constant dense
            # CHECK: func.call @noise_subroutine_code
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecp.codeblock<1 x 7> to !qecl.codeblock<1>
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard
            qp.H(0)
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecl.codeblock<1> to !qecp.codeblock<1 x 7>
            # CHECK: arith.constant dense
            # CHECK: arith.constant dense
            # CHECK: func.call @noise_subroutine_code
            # CHECK: builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecp.codeblock<1 x 7> to !qecl.codeblock<1>
            # CHECK: qecl.qec
            m0 = qp.measure(0)
            return qp.sample([m0])

        run_filecheck_qjit(circuit)
