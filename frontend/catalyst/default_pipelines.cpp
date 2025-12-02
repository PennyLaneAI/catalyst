// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "Driver/DefaultPipelines.h"

namespace nb = nanobind;

NB_MODULE(default_pipelines, m)
{
    m.doc() = "Bindings for Catalyst default pipelines.";
    m.def("get_pipeline_names", &catalyst::driver::getPipelineNames,
          "Returns the list of pipeline names.");
    m.def("get_enforce_runtime_invariants_stage",
          &catalyst::driver::getEnforceRuntimeInvariantsPipeline,
          "Returns the list of pass names for the Enforce Runtime Invariants pipeline.");
    m.def("get_hlo_lowering_stage", &catalyst::driver::getHLOLoweringPipeline,
          "Returns the list of pass names for the HLO Lowering pipeline.");
    m.def("get_quantum_compilation_stage", &catalyst::driver::getQuantumCompilationPipeline,
          "Returns the list of pass names for the Quantum Compilation pipeline.");
    m.def("get_bufferization_stage", &catalyst::driver::getBufferizationPipeline,
          "Returns the list of pass names for the Bufferization pipeline.");
    m.def("get_convert_to_llvm_stage", &catalyst::driver::getLLVMDialectLoweringPipeline,
          "Returns the list of pass names for the LLVM Dialect Lowering pipeline.");
}
