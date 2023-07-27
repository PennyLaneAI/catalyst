// Copyright 2023 Xanadu Quantum Technologies Inc.

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

namespace catalyst {
namespace quantum {

/// A collection of the data required to reconstruct a deterministic hybrid quantum program with
/// classical preprocessing and arbitrary classical control flow.
struct QuantumCache {
    mlir::TypedValue<ArrayListType> paramVector;
    mlir::TypedValue<ArrayListType> wireVector;
    mlir::DenseMap<mlir::Operation *, mlir::TypedValue<ArrayListType>> controlFlowTapes;

    // Initialize the quantum cache to traverse and store the necessary parameters for the given
    // `region`.
    static QuantumCache initialize(mlir::Region &region, mlir::OpBuilder &builder,
                                   mlir::Location loc);
};

/// Given a `region` containing classical preprocessing and quantum operations, clone the
/// preprocessing
void cloneClassical(mlir::Region &region, mlir::IRMapping &oldToCloned,
                    mlir::PatternRewriter &rewriter, QuantumCache &cache);

} // namespace quantum
} // namespace catalyst
