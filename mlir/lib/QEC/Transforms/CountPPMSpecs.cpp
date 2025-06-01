// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "ppm_specs"

#include "llvm/Support/Debug.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_COUNTPPMSPECSPASS
#define GEN_PASS_DECL_COUNTPPMSPECSPASS
#include "QEC/Transforms/Passes.h.inc"


struct CountPPMSpecsPass : public impl::CountPPMSpecsPassBase<CountPPMSpecsPass> {
    using CountPPMSpecsPassBase::CountPPMSpecsPassBase;

    void runOnOperation() final
    {
        llvm::outs() << "TEST for Count PPM Specs\n";
    }
};

} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createCountPPMSpecsPass() { return std::make_unique<CountPPMSpecsPass>(); }

} // namespace catalyst
