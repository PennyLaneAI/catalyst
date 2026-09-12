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
#include "AffineTransform.hpp"

namespace catalyst::phase_folding {

class AffineRelation : public AffineBase<RelationSchema> {
  public:
    // Constructors
    AffineRelation() = default;
    AffineRelation(BinaryMatrix matrix, RelationSchema schema)
        : AffineBase<RelationSchema>(std::move(matrix), std::move(schema)) {}
    AffineRelation(size_t numConstraints, size_t numQubits, std::optional<size_t> numAuxVars = 0)
        : AffineRelation(BinaryMatrix(numConstraints), RelationSchema(numQubits, numAuxVars)) {}
    explicit AffineRelation(AffineTransform &&affTrans)
        : AffineBase<RelationSchema>(std::move(affTrans.getMatrixMutable()),
                                     RelationSchema(std::move(affTrans.getSchemaMutable()))) {
        const size_t maxBlocks = schema.maxBlock();
        const size_t numQubits = schema.numQubits();

        for (size_t i = 0; i < numQubits; ++i) {
            Parity &row = matrix.getRowMutableAt(i);

            row.extendBitsFor(maxBlocks);
            row.setBitAtLoc(schema.postVars[i]);
        }
    }

    // Static Factories:
    static AffineRelation Identity(size_t numQubits); // 1
    static AffineRelation Trivial(size_t numQubits);  // \top
    static AffineRelation Unsat(size_t numQubits);    // 0 = \bot

    // Prints
    [[nodiscard]] std::string toString() const;
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineRelation &affRelation) {
        os << affRelation.toString();
        return os;
    }

    // Checks & Inspections
    [[nodiscard]] bool isTrivial();
    [[nodiscard]] bool isUnsat();

    // Methods
    const AffineRelation &joinWith(const AffineRelation &rhs);
    const AffineRelation &meetWith(const AffineRelation &rhs);
    const AffineRelation &composeWith(const AffineRelation &rhs);
    const AffineRelation &propagateThrough(const AffineRelation &rhs);
    const AffineRelation &applyKleeneStar();
    [[nodiscard]] AffineRelation meet(const AffineRelation &rhs) const;
    [[nodiscard]] AffineRelation join(const AffineRelation &rhs) const;
    [[nodiscard]] AffineRelation compose(const AffineRelation &rhs) const;
    [[nodiscard]] AffineRelation kleeneStar() const;
    [[nodiscard]] AffineTransform solveRelation();
    [[nodiscard]] Parity reduce(const Parity &par, const AffineSchema &parSchm,
                                bool isAgainstPrecond = false,
                                bool isProjectOutAuxVars = false) const;

  private:
    void opPreProcess(const AffineRelation &rhs);
    void embedInto(BinaryMatrix &trgtMat, const RelationSchemaView &trgtSchm) const;
    static void mapRowBits(const Parity &srcRow, const RelationSchemaView &srcSchm, Parity &trgtRow,
                           const RelationSchemaView &trgtSchm);
    static BinaryMatrix concretizer(const PropagateSchema &propSchm);
};

inline void AffineRelation::opPreProcess(const AffineRelation &rhs) {
    assert(schema.numQubits() == rhs.schema.numQubits()); // mostly is not, how to handle?
    matrix.reserveRowsFor(rhs.matrix.getNumRows());
}

inline void AffineRelation::mapRowBits(const Parity &srcRow, const RelationSchemaView &srcSchm,
                                       Parity &trgtRow, const RelationSchemaView &trgtSchm) {
    trgtRow.extendBitsFor(trgtSchm.maxBlock);

    trgtRow.mapBitsFrom(srcRow, srcSchm.postVars, trgtSchm.postVars);
    trgtRow.mapBitsFrom(srcRow, srcSchm.preVars, trgtSchm.preVars);
    trgtRow.mapBitsFrom(srcRow, srcSchm.auxVars, trgtSchm.auxVars);
    trgtRow.mapBitFrom(srcRow, srcSchm.affVal, trgtSchm.affVal);
}

inline bool AffineRelation::isTrivial() {
    return matrix.isEmpty() || (*this == Trivial(numQubits()));
}

// inline bool AffineRelation::isUnsat()
// {
//   // return matrix.contains(\unsat parity\)
//   // or reduce matrix so it keeps only unsat parity or what
// }

inline const AffineRelation &AffineRelation::applyKleeneStar() {
    *this = kleeneStar();
    return *this;
}

inline AffineRelation AffineRelation::meet(const AffineRelation &rhs) const {
    AffineRelation res = *this;
    return res.meetWith(rhs);
}

inline AffineRelation AffineRelation::join(const AffineRelation &rhs) const {
    AffineRelation res = *this;
    return res.joinWith(rhs);
}

inline AffineRelation AffineRelation::compose(const AffineRelation &rhs) const {
    AffineRelation res = *this;
    return res.composeWith(rhs);
}

} // namespace catalyst::phase_folding
