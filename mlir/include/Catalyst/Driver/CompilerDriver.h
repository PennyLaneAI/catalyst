// Copyright 2023 Xanadu Quantum Technologies Inc.

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

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

/// Run a given set of passes on an MLIR module.
///
/// The IR is supplied in textual form while the passes are expected in MLIR's command line
/// interface form. This allocates a buffer for the resulting textual IR that the caller must
/// take ownership of freeing.
mlir::FailureOr<std::string> RunPassPipeline(mlir::StringRef source, mlir::StringRef passes);

/// Data about the JIT function that is optionally inferred and returned to the caller.
///
/// This is important for calling a function when invoking the compiler on an MLIR or LLVM textual
/// representation intead of from Python.
struct FunctionAttributes {
    std::string functionName;
    std::string returnType;
};

struct CompilerOptions {
    mlir::MLIRContext *ctx;
    mlir::StringRef source;
    mlir::StringRef dest;
    mlir::StringRef moduleName;
    llvm::raw_ostream &diagnosticStream;
};

/// Entry point to the MLIR portion of the compiler.
mlir::LogicalResult QuantumDriverMain(const CompilerOptions &options,
                                      FunctionAttributes *inferredData);
