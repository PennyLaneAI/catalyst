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

"""
This module contains builtin-in compilation pipelines for FTQC workloads.
"""

from ..pipelines import PipelineStages, default_pipeline, insert_pass_before


def mbqc_pipeline() -> PipelineStages:
    """Return the pipeline stages for MBQC workloads.

    The MBQC pipeline is identical to the default Catalyst pipeline, but with

      * the MBQC-to-LLVM dialect-conversion pass inserted immediately before the Quantum-to-LLVM
        dialect-conversion pass in the ``MLIRToLLVMDialectConversion`` pipeline stage, and
      * array-backed registers enabled via the ``use-array-backed-registers`` option of the
        ``convert-quantum-to-llvm`` pass.

    Returns:
        PipelineStages: The list of pipeline stages.

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qml
        from catalyst.ftqc import mbqc_pipeline

        qml.capture.enable()

        dev = qml.device("null.qubit", wires=1)

        @qml.qjit(pipelines=mbqc_pipeline())
        @qml.qnode(dev, mcm_method="tree-traversal")
        def workload():
            m0 = qml.ftqc.measure_arbitrary_basis(wires=0, angle=np.pi/4, plane="XY")
            qml.cond(m0, qml.X, qml.I)(0)
            return qml.expval(qml.Z(0))
    """
    stages = default_pipeline()

    # Find the MLIR-to-LLVM dialect-conversion stage, 'MLIRToLLVMDialectConversion'
    stage_names = [item[0] for item in stages]
    llvm_dialect_conversion_stage_name = "MLIRToLLVMDialectConversion"

    assert (
        llvm_dialect_conversion_stage_name in stage_names
    ), "Stage 'MLIRToLLVMDialectConversion' not found in default pipeline stages"

    llvm_dialect_conversion_stage_index = stage_names.index(llvm_dialect_conversion_stage_name)

    _, pipeline = stages[llvm_dialect_conversion_stage_index]

    assert "convert-quantum-to-llvm" in pipeline, (
        "Pipeline for stage 'MLIRToLLVMDialectConversion' missing required pass "
        "'convert-quantum-to-llvm'.\n"
        f"The pipeline for this stage is: {pipeline}"
    )

    # Insert (in-place) the "convert-mbqc-to-llvm" pass immediately before the
    # "convert-quantum-to-llvm" pass in the MLIRToLLVMDialectConversion pipeline
    insert_pass_before(
        pipeline, ref_pass="convert-quantum-to-llvm", new_pass="convert-mbqc-to-llvm"
    )

    # Enable array-backed registers via the "convert-quantum-to-llvm" pass
    quantum_to_llvm_pass = pipeline.index("convert-quantum-to-llvm")
    pipeline[quantum_to_llvm_pass] = "convert-quantum-to-llvm{use-array-backed-registers=true}"

    return stages
