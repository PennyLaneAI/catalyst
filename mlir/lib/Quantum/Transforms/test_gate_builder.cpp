// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "test-gate-builder"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_GATEBUILDERTESTERPASS
#define GEN_PASS_DECL_GATEBUILDERTESTERPASS
#include "Quantum/Transforms/Passes.h.inc"

struct GateBuilderTesterPass
    : impl::GateBuilderTesterPassBase<GateBuilderTesterPass> {
    using GateBuilderTesterPassBase::GateBuilderTesterPassBase;

    void runOnOperation() final
    {
        //llvm::errs() << "test gate builder\n";
        /*
        Tests are of the form
        module {
          %0 = quantum.alloc( 2) : !quantum.reg
          %bit0 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        }
        */

        Operation *module = getOperation();
        MLIRContext *ctx = module->getContext();
        mlir::IRRewriter builder(ctx);
        Location loc = module->getLoc();

        Operation &bit0 = module->getRegion(0).front().back();
        Value inQubit = bit0.getResult(0);

        builder.setInsertionPointAfter(&bit0);

        builder.create<quantum::CustomOp>(loc, mlir::ValueRange({inQubit}), "PauliZ");
        builder.create<quantum::CustomOp>(loc, mlir::ValueRange({inQubit}), "PauliY", true);
    }
};

} // namespace quantum

std::unique_ptr<Pass> createGateBuilderTesterPass()
{
    return std::make_unique<quantum::GateBuilderTesterPass>();
}

} // namespace catalyst
