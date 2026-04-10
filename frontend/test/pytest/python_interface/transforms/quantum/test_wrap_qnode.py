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

"""Unit and integration tests for the unified compiler `measurements_from_samples` transform."""

# pylint: disable=line-too-long

import pytest

from catalyst.python_interface.transforms import (
    MeasurementsFromSamplesPass,
)

pytestmark = pytest.mark.xdsl


class TestWrapQNodePass:
    """Unit tests for the wrap-qnode pass."""

    def test_wrapping_single_quantum_node(self, run_filecheck):
        """Test the wrap_qnode pass works as expected on a module containing a single quantum.node
        """

        raise RuntimeError
    
        program = """
        builtin.module @module_circuit {
            // CHECK-LABEL: circuit
            // CHECK-SAME: (tensor<5x1xf64>) -> tensor<f64>
            // CHECK: func.func public @circuit.some_pass_name{{.*}} attributes {quantum.node}
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                %0 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]

                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.namedobs
                // CHECK: quantum.expval
                %2 = "test.op"() : () -> !quantum.bit
                %3 = quantum.namedobs %2[PauliZ] : !quantum.obs
                %4 = quantum.expval %3 : f64
                %5 = "tensor.from_elements"(%4) : (f64) -> tensor<f64>
                func.return %5 : tensor<f64>
            }
        }
        """

        pipeline = (WrapQNodePass(),)
        run_filecheck(program, pipeline)

    def test_wrapping_multiple_quantum_nodes(self, run_filecheck):
        """Test the wrap_qnode pass works as expected on a module containing multiple quantum.nodes
        """

        raise RuntimeError

        program = """
        builtin.module @module_circuit {
        }
        """

        pipeline = (WrapQNodePass(),)
        run_filecheck(program, pipeline)

    def test_composability(self, run_filecheck):
        """Test that we can apply the pass repeatedly to add nested classical functions 
        for different post-processing steps"""

        raise RuntimeError

        program = """
        builtin.module @module_circuit {
        }
        """

        pipeline = (WrapQNodePass(),)
        run_filecheck(program, pipeline)

    def test_composability(self, run_filecheck):
        """Test that we can apply the pass repeatedly to add nested classical functions 
        for different post-processing steps"""

        raise RuntimeError

        program = """
        builtin.module @module_circuit {
        }
        """

        pipeline = (WrapQNodePass(),)
        run_filecheck(program, pipeline)

class TestGetCallOp:

    def test_single_quantum_node(self):
        raise RuntimeError
    
    def test_multiple_quantum_nodes(self):
        raise RuntimeError
    
    def test_multiple_call_ops_raises_error(self):
        raise RuntimeError
    
    def test_no_module_raises_error(self):
        raise RuntimeError


class TestIntegrationWrapQNodePass:

    def test_before_split_non_commuting():
        """Test that the pass works as expected when post-processing is added 
        before split-non-commuting"""

        raise RuntimeError
    
    def test_after_split_non_commuting():
        """Test that the pass works as expected when post-processing is added 
        before split-non-commuting"""

        raise RuntimeError
    
    def test_adding_postprocessing():
        """Test that we can use the pass and the get_call_op function to add post-processing"""

        raise RuntimeError
    
        
    
if __name__ == "__main__":
    pytest.main(["-x", __file__])
