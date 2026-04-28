// Copyright 2026 Xanadu Quantum Technologies Inc.

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

#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

namespace catalyst {
namespace test {

using PyArg = std::variant<int, double, std::string>;
using PyWires = std::vector<int>;

std::string python_circuit_execution(llvm::StringRef module_name, llvm::StringRef function_name,
                                     std::vector<PyArg> args, PyWires wires);

mlir::OwningOpRef<mlir::Operation *> get_op_from_python(mlir::ModuleOp module,
                                                        llvm::StringRef module_name,
                                                        llvm::StringRef function_name,
                                                        std::vector<PyArg> args, PyWires wires);
} // namespace test
} // namespace catalyst
