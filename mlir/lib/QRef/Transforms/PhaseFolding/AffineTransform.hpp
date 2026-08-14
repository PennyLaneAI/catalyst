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

#include "llvm/ADT/ArrayRef.h"

#include "AffineBase.hpp"

namespace catalyst::phase_folding {

class AffineTransform : public AffineBase<TransformSchema> {
  public:
    // Constructors
    AffineTransform() = default;
    AffineTransform(BinaryMatrix matrix, TransformSchema schema)
        : AffineBase<TransformSchema>(std::move(matrix), std::move(schema)) {}
    explicit AffineTransform(size_t numQubits)
        : AffineTransform(BinaryMatrix::Identity(numQubits), TransformSchema(numQubits)) {
    } // Identity matrix by default

    // Prints
    [[nodiscard]] std::string toString() const;
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const AffineTransform &affTransform) {
        os << affTransform.toString();
        return os;
    }

    // Methods
    void extendQubitsBy(size_t addQubitNum);
    void prepareQubit(size_t wire, bool basisState);
    void applyGateX(size_t wire);
    void applyGateCNOT(size_t controlWire, size_t targetWire);
    void applyGateSWAP(size_t wire1, size_t wire2);
    void applyGateH(size_t wire);
    void applyGateU(llvm::ArrayRef<size_t> wires);
};

inline void AffineTransform::applyGateX(size_t wire) {
    matrix.flipBitAtRowAtLoc(wire, schema.affVal);
}

inline void AffineTransform::applyGateCNOT(size_t controlWire, size_t targetWire) {
    matrix.addRowToRow(controlWire, targetWire);
}

inline void AffineTransform::applyGateSWAP(size_t wire1, size_t wire2) {
    matrix.swapRows(wire1, wire2);
}

inline void AffineTransform::applyGateH(size_t wire) {
    matrix.setRowToBasis(wire, schema.allocAuxVar());
}

} // namespace catalyst::phase_folding
