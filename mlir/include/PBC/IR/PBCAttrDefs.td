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

#ifndef PBC_ATTR_DEFS
#define PBC_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

include "PBC/IR/PBCDialect.td"

//===----------------------------------------------------------------------===//
// PBC dialect enums and attributes.
//===----------------------------------------------------------------------===//

// Constraint for rotation_kind: must be one of ±1, ±2, ±4, ±8
// The power-of-2 check works because powers of 2 have exactly one bit set.
def RotationKindConstraint : AttrConstraint<
    CPred<"[&]() {"
          "  auto val = ::llvm::cast<::mlir::IntegerAttr>($_self).getSInt();"
          "  auto absVal = val < 0 ? -val : val;"
          "  return absVal >= 1 && absVal <= 8 && (absVal & (absVal - 1)) == 0;"
          "}()">,
    "whose value is ±1, ±2, ±4, or ±8">;

#endif // PBC_ATTR_DEFS
