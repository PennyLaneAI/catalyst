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

    unsigned int countTapes(func::FuncOp func){
        // Count the number of quantum.device operations in a function
        unsigned int count = 0;
        func->walk([&](Operation *op){
            //llvm::errs() << op->getName() << "\n";
            if (op->getName().getStringRef() == "quantum.device"){
                count++;
            }
        });
        return count;
    }

    void runOnOperation() override { 
        Operation *module = getOperation();
        //llvm::errs() << *module << "Hello world!\n"; 

        // 1. Identify the functions with multiple tapes
        // Walk through each function and count the number of devices.
        // In frontend when tracing (jax_tracers.py/trace_quantum_function), 
        // each tape has its own quantum.device operation attached to it
        SmallVector<func::FuncOp> MultitapePrograms;
        module->walk([&](func::FuncOp func){
            //llvm::errs() << "visiting " << func.getSymName() << "\n";
            if (countTapes(func) >= 2){
                MultitapePrograms.push_back(func);
            }
        });

        // Do nothing and exit for classical and single-tape programs
        if (MultitapePrograms.empty()){
            return;
        }

        llvm::errs() << "program function is: \n";
        for (auto _ : MultitapePrograms){
            llvm::errs() << _;
        }

        // 2. Count the number of tapes



    }
};

std::unique_ptr<Pass> createSplitMultipleTapesPass() { return std::make_unique<SplitMultipleTapesPass>(); }

} // namespace catalyst