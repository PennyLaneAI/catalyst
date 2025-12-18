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

#pragma once

#include "llvm/Support/JSON.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

namespace catalyst {

/// Convert a JSON value to an MLIR Attribute.
/// Handles strings, integers, floats, booleans, arrays, objects, and null.
mlir::Attribute jsonToAttribute(mlir::MLIRContext *ctx, const llvm::json::Value &json);

/// Load a JSON file and convert it to a DictionaryAttr.
/// Returns failure if the file cannot be read or parsed, or if the root is not an object.
mlir::FailureOr<mlir::DictionaryAttr> loadJsonFileAsDict(mlir::MLIRContext *ctx,
                                                         llvm::StringRef filePath);

} // namespace catalyst
