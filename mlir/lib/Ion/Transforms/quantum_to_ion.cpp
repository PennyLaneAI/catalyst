// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Passes.h"
#include "Ion/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::ion;

namespace catalyst {
namespace ion {

#define GEN_PASS_DECL_QUANTUMTOIONPASS
#define GEN_PASS_DEF_QUANTUMTOIONPASS
#include "Ion/Transforms/Passes.h.inc"

struct QuantumToIonPass : impl::QuantumToIonPassBase<QuantumToIonPass> {
    using QuantumToIonPassBase::QuantumToIonPassBase;

    void runOnOperation() final
    {
        RewritePatternSet ionPatterns(&getContext());
        populateQuantumToIonPatterns(ionPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(ionPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace ion

std::unique_ptr<Pass> createQuantumToIonPass() { return std::make_unique<ion::QuantumToIonPass>(); }

} // namespace catalyst
