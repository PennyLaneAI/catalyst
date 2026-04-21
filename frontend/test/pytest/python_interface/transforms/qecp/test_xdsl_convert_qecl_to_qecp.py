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

import pytest

from catalyst.python_interface.transforms.qecp import (
    ConvertQecLogicalToQecPhysicalPass,
    convert_qecl_to_qecp,
)

pytestmark = pytest.mark.xdsl


@pytest.mark.filterwarnings("ignore:Unable to remove cast UnrealizedConversionCastOp")
class TestLoweringEncode():
    """Test lowering the qecl.EncodeOp to a subroutine of qecp gates"""

    def test_1(self, run_filecheck):
        """Test that a qecl.encode operation raises an error if we are not encoding to zero"""

        # 

        program = """
            builtin.module @module_circuit {
                func.func @test_func() attributes {quantum.node} {
                    // CHECK: [[codeblock:%.*]] = "test.op"() : () -> !qecl.codeblock<1>
                    // CHECK-NEXT: [[casted_codeblock:%.*]] = builtin.unrealized_conversion_cast [[codeblock:%.*]] : !qecl.codeblock<1> to !qecp.codeblock<1 x 7>
                    %0 = "test.op"() : () -> !qecl.codeblock<1>
                    %1 = qecl.encode ["zero"] %0 : !qecl.codeblock<1>
                    return
                }
            }
            """

        pipeline = (ConvertQecLogicalToQecPhysicalPass(qec_code="steane[[7,1,3]]"),)
        run_filecheck(program, pipeline)