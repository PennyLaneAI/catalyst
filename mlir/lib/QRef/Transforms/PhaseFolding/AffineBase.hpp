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

#include "AffineSchema.hpp"
#include "BinaryMatrix.hpp"

namespace catalyst::phase_folding {

template <typename SchemaT> class AffineBase {
  public:
    // Constructors
    AffineBase() = default;
    AffineBase(BinaryMatrix matrix, SchemaT schema)
        : matrix(std::move(matrix)), schema(std::move(schema)) {}

    // Operators
    bool operator==(AffineBase<SchemaT> &rhs);

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const AffineBase<SchemaT> &affBase) {
        auto colOrder = affBase.schema.getOrder();
        for (size_t i = 0; i < affBase.matrix.getNumRows(); ++i) {
            os << affBase.matrix.getRowAt(i).toStringWithOrder(colOrder) << "\n";
        }
        os << "Schema:\n" << affBase.schema;
        return os;
    }

    // Getters
    [[nodiscard]] const SchemaT &getSchema() const;
    [[nodiscard]] SchemaT &getSchemaMutable();
    [[nodiscard]] const BinaryMatrix &getMatrix() const &;
    [[nodiscard]] BinaryMatrix &&getMatrix() &&;
    [[nodiscard]] BinaryMatrix &getMatrixMutable();
    [[nodiscard]] const Parity &getExpr(size_t row) const;
    [[nodiscard]] Parity &getExprMutable(size_t row);

    // Stats
    [[nodiscard]] size_t numQubits() const;

    // Methods
    template <typename ColOrderRange> void projectOutVars(ColOrderRange projRange);
    void projectOutAuxVars();
    void applySchema(SchemaT newSchm);

  protected:
    BinaryMatrix matrix;
    SchemaT schema;
};

template <typename SchemaT> bool AffineBase<SchemaT>::operator==(AffineBase<SchemaT> &rhs) {
    projectOutAuxVars();
    rhs.projectOutAuxVars();

    rhs.applySchema(schema);

    matrix.normalize(schema.getOrder());
    rhs.getMatrixMutable().normalize(rhs.getSchema().getOrder());

    return matrix == rhs.matrix;
}

// Getters
template <typename SchemaT> inline const SchemaT &AffineBase<SchemaT>::getSchema() const {
    return schema;
}

template <typename SchemaT> inline SchemaT &AffineBase<SchemaT>::getSchemaMutable() {
    return schema;
}

template <typename SchemaT> inline const BinaryMatrix &AffineBase<SchemaT>::getMatrix() const & {
    return matrix;
}

template <typename SchemaT> inline BinaryMatrix &&AffineBase<SchemaT>::getMatrix() && {
    return std::move(matrix);
}

template <typename SchemaT> inline BinaryMatrix &AffineBase<SchemaT>::getMatrixMutable() {
    return matrix;
}

template <typename SchemaT> inline const Parity &AffineBase<SchemaT>::getExpr(size_t row) const {
    return matrix.getRowAt(row);
}

template <typename SchemaT> inline Parity &AffineBase<SchemaT>::getExprMutable(size_t row) {
    return matrix.getRowMutableAt(row);
}

// Stats
template <typename SchemaT> inline size_t AffineBase<SchemaT>::numQubits() const {
    return schema.numQubits();
}

// Methods
template <typename SchemaT>
template <typename ColOrderRange>
void AffineBase<SchemaT>::projectOutVars(ColOrderRange projRange) {
    // matrix.semiNormalize(schema.getProjOrder(projRange));
    matrix.normalize(schema.getProjOrder(projRange));
    size_t newSt = matrix.firstTrivialInRangeRow(projRange);

    if (newSt <= matrix.getNumRows()) {
        matrix.dropTopRows(newSt);
    }

    schema.recycleLocs(projRange);
}

template <typename SchemaT> void AffineBase<SchemaT>::projectOutAuxVars() {
    projectOutVars(schema.auxVars);
    schema.auxVars.clear();
}

template <typename SchemaT> void AffineBase<SchemaT>::applySchema(SchemaT newSchm) {
    auto curOrder = schema.getOrder();
    auto newOrder = newSchm.getOrder();

    Parity scratch(newSchm.maxBlock());

    for (Parity &row : matrix.getRowsMutable()) {
        scratch.mapBitsFrom(row, curOrder, newOrder);
        std::swap(row, scratch);
    }
    schema = std::move(newSchm);
}

} // namespace catalyst::phase_folding
