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
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "Parity.h"
#include "PhaseBucket.h"

struct PhasePolynomial {
    llvm::DenseMap<Parity, PhaseBucket> terms;
    // bots are not seperated and duplicate!

    // Constructors
    PhasePolynomial() = default;
    PhasePolynomial(llvm::DenseMap<Parity, PhaseBucket> terms) : terms(std::move(terms)) {}

    // Operators
    PhasePolynomial &operator+=(const PhasePolynomial &rhs);
    PhasePolynomial operator+(const PhasePolynomial &rhs) const;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const PhasePolynomial &pp);
    std::string algebraicView(size_t qubitNum) const;

    // Methods
    void insertContributor(const Parity &parity, const PhaseBucket &contributor);
};