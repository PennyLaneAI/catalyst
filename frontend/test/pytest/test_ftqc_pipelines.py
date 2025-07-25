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

"""Test suite for the Catalyst FTQC built-in compilation pipelines."""

from catalyst.ftqc import mbqc_pipeline
from catalyst.pipelines import default_pipeline


def test_mbqc_pipeline():
    """Loosely check that the MBQC pipeline is the same as the default Catalyst pipeline, but with
    the MBQC-to-LLVM dialect-conversion pass inserted before the Quantum-to-LLVM dialect-conversion
    pass in the ``MLIRToLLVMDialect`` pipeline stage.
    """
    default_stages = default_pipeline()
    mbqc_stages = mbqc_pipeline()

    # Check that the stage names are the same
    default_stage_names = [item[0] for item in default_stages]
    mbqc_stage_names = [item[0] for item in mbqc_stages]

    assert default_stage_names == mbqc_stage_names
    assert mbqc_stage_names[-1] == "MLIRToLLVMDialect"

    # Check that the conversion pass(es) are in the mbqc pipeline
    _, mbqc_llvm_conversion_pipeline = mbqc_stages[-1]

    assert "convert-quantum-to-llvm" in mbqc_llvm_conversion_pipeline
    assert "convert-mbqc-to-llvm" in mbqc_llvm_conversion_pipeline

    # Check that the mbqc-to-llvm pass comes *before* the quantum-to-llvm pass
    quantum_to_llvm_index = mbqc_llvm_conversion_pipeline.index("convert-quantum-to-llvm")
    mbqc_to_llvm_index = mbqc_llvm_conversion_pipeline.index("convert-mbqc-to-llvm")

    assert mbqc_to_llvm_index < quantum_to_llvm_index
