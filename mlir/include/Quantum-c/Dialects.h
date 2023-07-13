// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Quantum, quantum);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Gradient, gradient);

/// The possible IRs the compiler supports generation from.
enum SourceType { SourceMLIR, SourceLLVMIR };

enum CatalystCReturnCode {
    ReturnOk,
    ReturnUnrecognizedSourceType,
    ReturnParsingFailed,
    ReturnLoweringFailed,
    ReturnTranslationFailed,
    /// The JIT function was not found within the module
    ReturnFunctionNotFound,
    ReturnObjectCompilationFailed,
};

/// Run a given set of passes on an MLIR module.
///
/// The IR is supplied in textual form while the passes are expected in MLIR's command line
/// interface form. This allocates a buffer for the resulting textual IR that the caller must
/// take ownership of freeing.
CatalystCReturnCode RunPassPipeline(const char *source, const char *passes, char **dest);

/// Data about the JIT function that is optionally inferred and returned to the caller.
///
/// This is important for calling a function when invoking the compiler on an MLIR or LLVM textual
/// representation intead of from Python.
struct FunctionData {
    char *functionName;
    char *returnType;
};

/// Entry point to the MLIR portion of the compiler.
CatalystCReturnCode QuantumDriverMain(const char *source, const char *dest, const char *sourceType,
                                      FunctionData *functionData = nullptr);

#ifdef __cplusplus
}
#endif
