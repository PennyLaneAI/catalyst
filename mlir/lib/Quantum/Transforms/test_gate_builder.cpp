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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
          %true = llvm.mlir.constant (1 : i1) :i1
          %angle = arith.constant 37.420000e+00 : f64  // to be used as parameter
          %0 = quantum.alloc( 2) : !quantum.reg
          %bit0 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        }
        */

        Operation *module = getOperation();
        MLIRContext *ctx = module->getContext();
        mlir::IRRewriter builder(ctx);
        Location loc = module->getLoc();

        Operation *bit0;
        Operation *angleOp;
        Operation *trueOp;
        Block &b = module->getRegion(0).front();
        for (const auto &[idx, op] : llvm::enumerate(b.getOperations())) {
            if (idx == 0){
                trueOp = &op;
            }
            if (idx == 1){
                angleOp = &op;
            }
            if (idx == 3){
                bit0 = &op;
            }
        }

        Value inQubit = bit0->getResult(0);
        Value angle = angleOp->getResult(0);
        Value trueVal = trueOp->getResult(0);

        builder.setInsertionPointAfter(bit0);

        auto pz = builder.create<quantum::CustomOp>(loc, "PauliZ", inQubit);
        builder.create<quantum::CustomOp>(loc, "PauliY", inQubit, true);
        auto rx = builder.create<quantum::CustomOp>(loc, "RX", inQubit, mlir::ValueRange({angle}));
        builder.create<quantum::CustomOp>(loc, "SWAP", mlir::ValueRange({inQubit, pz->getResult(0)}));

        builder.create<quantum::CustomOp>(loc, "Rot",
            mlir::ValueRange({inQubit, pz->getResult(0)}),
            mlir::ValueRange({rx->getResult(0)}),
            mlir::ValueRange({trueVal}),
            mlir::ValueRange({angle})
        );

        builder.create<quantum::CustomOp>(loc, "my_controlled_U",
            mlir::ValueRange({inQubit, pz->getResult(0)}),
            mlir::ValueRange({rx->getResult(0)}),
            mlir::ValueRange({trueVal})
        );
    }
};

} // namespace quantum

std::unique_ptr<Pass> createGateBuilderTesterPass()
{
    return std::make_unique<quantum::GateBuilderTesterPass>();
}

} // namespace catalyst
