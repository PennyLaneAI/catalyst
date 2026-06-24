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
#include "llvm/Support/raw_ostream.h"

#include "BinaryMatrix.h"

#include <vector>
#include <cassert>

struct TransformLayout {
  std::vector<size_t> inVars;
  std::vector<size_t> auxVars;

  TransformLayout() = default;
  explicit TransformLayout(size_t n);
};

class AffineTransform {
  public:
    // Constructors
    AffineTransform() = default;
    explicit AffineTransform(size_t n) : layout(TransformLayout(n)), mat(BinaryMatrix::identity(n)) {} // Identity matrix by default
    
    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineTransform &trans);
    std::string algebraicView() const;

    // Getters
    [[nodiscard]] size_t getQubitNum() const;
    [[nodiscard]] size_t getAuxVarNum() const;
    [[nodiscard]] size_t getVarNum() const;
    [[nodiscard]] const Parity &getExpr(size_t qubitInd) const;
    [[nodiscard]] Parity &getExprMutable(size_t qubitInd) const;

    // Setters
    
    // Methods
    void extendQubitsTo(size_t newQubitNum);
    void initQubit(size_t qubitIndex, bool basisState);
    void applyGateX(size_t qubitIndex);
    void applyGateCNOT(size_t controlIndex, size_t targetIndex);
    void applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2);
    void applyGateH(size_t qubitIndex);
    void applyGateU(llvm::ArrayRef<size_t> qubitIndices);
    
  private:
    TransformLayout layout;
    BinaryMatrix mat;
};

inline size_t AffineTransform::getQubitNum() const
{
  assert(layout.inVars.size() == mat.getRowNum());
  return layout.inVars.size();
}

inline size_t AffineTransform::getAuxVarNum() const
{
  return layout.auxVars.size();
}

inline size_t AffineTransform::getVarNum() const
{
  return getQubitNum() + getAuxVarNum();
}

inline const Parity &AffineTransform::getExpr(size_t qubitInd) const
{
    return mat.getRow(qubitInd);
}

inline Parity &AffineTransform::getExprMutable(size_t qubitInd) const
{
    return mat.getRowMutable(qubitInd);
}
