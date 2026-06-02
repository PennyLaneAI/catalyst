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

#include <variant>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace catalyst {
namespace quantum {

/// A struct to represent qubit indices in quantum operations.
///
/// This struct provides a way to handle qubit indices that can be either:
/// - A runtime Value (for dynamic indices computed at runtime)
/// - An IntegerAttr (for compile-time constant indices)
/// - Invalid/uninitialized (represented by std::monostate)
/// And a qreg value to represent the qreg that the index belongs to
///
/// The struct uses std::variant to ensure only one type is active at a time,
/// preventing invalid states.
///
/// Example usage:
///   QubitIndex dynamicIdx(operandValue, qreg);         // Runtime qubit index
///   QubitIndex staticIdx(IntegerAttr::get(...), qreg); // Compile-time constant
///   QubitIndex invalidIdx;                             // Uninitialized state
///
///   if (dynamicIdx) {                                  // Check if valid
///     if (dynamicIdx.isValue()) {                      // Check if runtime value
///       Value idx = dynamicIdx.getValue();             // Get the Value
///     }
///   }
///
/// Typical construction site: building a QubitIndex from a quantum.extract op,
/// where the index may be either a dynamic SSA Value or a static IntegerAttr:
///
///   if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
///       if (Value idx = extractOp.getIdx()) {
///           return QubitIndex(idx, extractOp.getQreg());
///       }
///       if (IntegerAttr idxAttr = extractOp.getIdxAttrAttr()) {
///           return QubitIndex(idxAttr, extractOp.getQreg());
///       }
///   }
class QubitIndex {
  private:
    // use monostate to represent the invalid index
    std::variant<std::monostate, mlir::Value, mlir::IntegerAttr> index;
    mlir::Value qreg;

  public:
    QubitIndex() : index(std::monostate()), qreg(nullptr) {}
    QubitIndex(mlir::Value val, mlir::Value qreg) : index(val), qreg(qreg) {}
    QubitIndex(mlir::IntegerAttr attr, mlir::Value qreg) : index(attr), qreg(qreg) {}

    bool isValue() const { return std::holds_alternative<mlir::Value>(index); }
    bool isAttr() const { return std::holds_alternative<mlir::IntegerAttr>(index); }
    operator bool() const { return isValue() || isAttr(); }
    mlir::Value getReg() const { return qreg; }
    mlir::Value getValue() const { return isValue() ? std::get<mlir::Value>(index) : nullptr; }
    mlir::IntegerAttr getAttr() const
    {
        return isAttr() ? std::get<mlir::IntegerAttr>(index) : nullptr;
    }
};

} // namespace quantum
} // namespace catalyst
