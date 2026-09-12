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

#include "AffineRelation.hpp"

using namespace catalyst::phase_folding;

/*
    Static Factories
*/
AffineRelation AffineRelation::Identity(size_t numQubits) // 1 = <X' = X>
{
    AffineRelation affRel(numQubits, numQubits);

    for (size_t i = 0; i < numQubits; ++i) {
        Parity &curRow = affRel.matrix.allocRow();
        curRow.mkBasis(affRel.schema.postVars[i], affRel.schema.maxBlock());
        curRow.setBitAtLoc(affRel.schema.preVars[i]);
    }

    return affRel;
}

AffineRelation AffineRelation::Trivial(size_t numQubits) // 0 = <>
{
    AffineRelation affRel(1, numQubits);

    Parity &curRow = affRel.matrix.allocRow();
    curRow.mkTrivial(affRel.getSchema().maxBlock()); // [0 0 0]

    return affRel;
}

AffineRelation AffineRelation::Unsat(size_t numQubits) // \top = <0 = 1>
{
    AffineRelation affRel(1, numQubits);

    Parity &curRow = affRel.matrix.allocRow();
    curRow.mkUnsat(affRel.getSchema().affVal); // [0 0 1]

    return affRel;
}

BinaryMatrix AffineRelation::concretizer(const PropagateSchema &propSchm) // <X' = Y>
{
    const size_t numQubits = propSchm.numQubits();
    BinaryMatrix concretizerMat(numQubits);

    for (size_t i = 0; i < numQubits; ++i) {
        Parity &curRow = concretizerMat.allocRow();
        curRow.mkBasis(propSchm.postVars[i], propSchm.maxBlock());
        curRow.setBitAtLoc(propSchm.concretizerVars[i]);
    }

    return concretizerMat;
}

/*
    Print:
*/
std::string AffineRelation::toString() const {
    std::string res = "";

    if (matrix.getNumRows() > 0) {
        res += "X' X";
        if (schema.auxVars.size() > 0) {
            res += " Y";
        }
        res += " | c\n";
    }

    for (size_t i = 0; i < matrix.getNumRows(); ++i) {
        std::string post = matrix.getRowAt(i).toStringWithOrder(schema.postVars);
        std::string pre = matrix.getRowAt(i).toStringWithOrder(schema.preVars);
        std::string aux = matrix.getRowAt(i).toStringWithOrder(schema.auxVars);
        bool affVal = matrix.getRowAt(i).getBitAtLoc(schema.affVal);

        res += (post + " " + pre + " " + aux + " | " + std::to_string(affVal)) + "\n";
    }
    return res;
}

/*
    In-place Op.s
*/
void AffineRelation::embedInto(BinaryMatrix &trgtMat,
                               const RelationSchemaView &trgtSchmView) const {
    const RelationSchemaView srcSchmView(schema);

    for (const Parity &row : matrix.getRows()) {
        Parity &newRow = trgtMat.allocRow();
        mapRowBits(row, srcSchmView, newRow, trgtSchmView);
    }
}

const AffineRelation &AffineRelation::meetWith(const AffineRelation &rhs) {
    AffineRelation auxFreeRhs = rhs;
    auxFreeRhs.projectOutAuxVars();
    this->projectOutAuxVars();

    opPreProcess(auxFreeRhs);
    MeetSchema meetSchm(std::move(schema), auxFreeRhs.schema);

    auxFreeRhs.embedInto(matrix, RelationSchemaView(meetSchm));

    this->schema = std::move(meetSchm);
    matrix.semiNormalize(schema.getOrder());
    return *this;
}

const AffineRelation &AffineRelation::joinWith(const AffineRelation &rhs) {
    // if (rhs.isTrivial()) return *this;
    // if (this->isTrivial()) return rhs;

    AffineRelation auxFreeRhs = rhs;
    auxFreeRhs.projectOutAuxVars();
    this->projectOutAuxVars();

    opPreProcess(auxFreeRhs);

    AffineBase<JoinSchema> affJoin(std::move(matrix),
                                   JoinSchema(std::move(schema), auxFreeRhs.schema));
    BinaryMatrix &joinMat = affJoin.getMatrixMutable();
    JoinSchema &joinSchm = affJoin.getSchemaMutable();

    RelationSchemaView rJoinSchmView(joinSchm.postVars, joinSchm.preVars, {}, joinSchm.affVal,
                                     joinSchm.maxBlock());
    RelationSchemaView lJoinSchmView(joinSchm.lPostVars, joinSchm.lPreVars, {}, joinSchm.lAffVal,
                                     joinSchm.maxBlock());

    for (Parity &row : joinMat.getRowsMutable()) {
        mapRowBits(row, rJoinSchmView, row, lJoinSchmView);
    }
    auxFreeRhs.embedInto(joinMat, lJoinSchmView);

    affJoin.projectOutVars(joinSchm.getProjRange());

    this->schema = std::move(joinSchm);
    this->matrix = std::move(joinMat);
    return *this;
}

const AffineRelation &AffineRelation::composeWith(const AffineRelation &rhs) {
    opPreProcess(rhs);

    AffineBase<CompositionSchema> affCmps(std::move(matrix),
                                          CompositionSchema(std::move(schema), rhs.schema));
    CompositionSchema &cmpsSchm = affCmps.getSchemaMutable();

    rhs.embedInto(affCmps.getMatrixMutable(),
                  RelationSchemaView(cmpsSchm.postVars, cmpsSchm.projVars,
                                     IdxView(cmpsSchm.auxVars).take_back(rhs.schema.numAuxVars()),
                                     cmpsSchm.affVal, cmpsSchm.maxBlock()));
    affCmps.projectOutVars(cmpsSchm.projVars);

    this->schema = std::move(cmpsSchm);
    this->matrix = std::move(affCmps.getMatrixMutable());
    return *this;
}

/*
    Out-of-place Op.s
*/
AffineRelation AffineRelation::kleeneStar() const {
    AffineRelation rel = *this;
    rel.projectOutAuxVars();

    AffineRelation cur = AffineRelation::Identity(numQubits()); // 1
    AffineRelation sum = cur;
    AffineRelation prevSum = AffineRelation::Trivial(numQubits()); // 0

    while (prevSum != sum) {
        // llvm::outs() << "\nkleeneStar:\n";
        // llvm::outs() << "\nsum:\n" << sum << "\n";
        // llvm::outs() << "\ncur:\n" << cur << "\n";
        prevSum = sum;
        cur.composeWith(rel);
        sum.joinWith(cur);
    }
    return sum;
}

const AffineRelation &AffineRelation::propagateThrough(const AffineRelation &rhs) {
    assert(rhs.schema.numAuxVars() == 0);

    PropagateSchema propagateSchm(std::move(schema), rhs.schema);
    BinaryMatrix propagateMat = (std::move(this->matrix));

    propagateMat.appendRows(concretizer(propagateSchm));

    // BinaryMatrix propagateMat = concretizer(propagateSchm);
    // propagateMat.appendRows(this->matrix);

    rhs.embedInto(propagateMat,
                  RelationSchemaView(propagateSchm.postVars, propagateSchm.projVars, {},
                                     propagateSchm.affVal, propagateSchm.maxBlock()));

    AffineBase<CompositionSchema> affPropag(std::move(propagateMat), propagateSchm);
    affPropag.projectOutVars(propagateSchm.projVars);

    this->matrix = std::move(affPropag.getMatrixMutable());
    this->schema = std::move(affPropag.getSchemaMutable());
    return *this;
}

AffineTransform AffineRelation::solveRelation() {
    const size_t qubitNum = numQubits();
    assert(qubitNum <= matrix.getNumRows());

    for (size_t i = 0; i < qubitNum; ++i) {
        matrix.getRowMutableAt(i).clearBitAtLoc(
            schema.postVars[i]); // is it logically correct? i.e. does it make the postVars empty?
    }

    schema.recycleLocs(schema.postVars);
    schema.postVars.clear();
    TransformSchema solvedSchm = schema.toTransformSchema();

    return AffineTransform(std::move(matrix), std::move(solvedSchm));
}

Parity AffineRelation::reduce(const Parity &par, const AffineSchema &parSchm, bool isAgainstPrecond,
                              bool isProjectOutAuxVars) const {
    BinaryMatrix redMat = matrix;
    RelationSchema relSchm = schema;

    // is it necessary? would there be any case where the auxVars are not the same?
    const int diffAuxVars = parSchm.numAuxVars() - relSchm.numAuxVars();
    if (diffAuxVars > 0) {
        relSchm.growAuxVars(diffAuxVars);
    }

    Parity &lastRow = redMat.allocRow();
    size_t reducingRow = redMat.getNumRows() - 1;

    lastRow.extendBitsFor(relSchm.maxBlock());

    lastRow.mapBitsFrom(par, parSchm.preVars,
                        isAgainstPrecond ? relSchm.postVars : relSchm.preVars);
    lastRow.mapBitsFrom(par, parSchm.auxVars, relSchm.auxVars);
    lastRow.mapBitFrom(par, parSchm.affVal, relSchm.affVal);

    reducingRow = redMat.toREF(isProjectOutAuxVars ? relSchm.getProjOrder(relSchm.auxVars)
                                                   : relSchm.getOrder(),
                               reducingRow); // tracking row is the last row
    return redMat.getRowAt(reducingRow);
}
