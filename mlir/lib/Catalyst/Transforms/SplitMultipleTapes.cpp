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

#define DEBUG_TYPE "splitmultipletapes"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_SPLITMULTIPLETAPESPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct SplitMultipleTapesPass : public impl::SplitMultipleTapesPassBase<SplitMultipleTapesPass> {
    using impl::SplitMultipleTapesPassBase<SplitMultipleTapesPass>::SplitMultipleTapesPassBase;

    bool isProgram(func::FuncOp func){
        bool result = false;
        WalkResult _ = func->walk([&](Operation *op){
            llvm::errs() << op->getName() << "\n";
            if (op->getName().getStringRef() == "quantum.device"){
                result = true;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        return result;
    }

    void runOnOperation() override { 
        Operation *module = getOperation();
        llvm::errs() << *module << "Hello world!\n"; 

        // 1. Identify the function with the circuits
        // Usually it is the only private function
        // And its name is the same as the top level python qnode name
        // e.g. if the qnode name is "circuit", then the entry function
        // is "public @jit_circuit", and it calls the "private @circuit"
        // that actually has the tapes

        // Another way to identify the target function is 
        // just walk through it and find that it is the function with
        // the device. This won't be costly since quantum.device 
        // is usually one of the first instructions
        func::FuncOp MultiTapeFunc;
        WalkResult result = module->walk([&](func::FuncOp Func){
            llvm::errs() << "visiting " << Func.getSymName() << "\n";
            bool IsProgram = isProgram(Func);
            llvm::errs() << IsProgram << "\n";
            return WalkResult::advance();
        });
    }
};

std::unique_ptr<Pass> createSplitMultipleTapesPass() { return std::make_unique<SplitMultipleTapesPass>(); }

} // namespace catalyst