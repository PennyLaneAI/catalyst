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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
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

    unsigned int countTapes(func::FuncOp func)
    {
        // Count the number of quantum.device operations in a function
        unsigned int count = 0;
        func->walk([&](Operation *op) {
            if (op->getName().getStringRef() == "quantum.device") {
                count++;
            }
        });
        return count;
    }

    void CollectOperationsForEachTape(func::FuncOp func,
                                      SmallVector<SmallVector<Operation *> *> &OpsEachTape)
    {
        // During tracing, each tape starts with a qdevice_p primitive
        // This means each tape starts with a quantum.device
        // and ends with a quantum.device_release

        // The structure of the FuncOp looks like the following:
        // func.func @circuit(...) -> ... {...} {
        // preprocessing
        // quantum.device[...]
        // tape 1
        // quantum.device_release
        // quantum.device[...]
        // tape 2
        // quantum.device_release
        // ... more tapes
        // post processing and return
        // }

        // This function parses the operations in the funcop and puts them into
        // the container as 
        // OpsEachTape = [[preprocessing ops], [tape1 ops], [tape2 ops], ..., [postprocessing ops]]

        for (size_t i = 0; i < OpsEachTape.size(); i++) {
            OpsEachTape[i] =
                new SmallVector<Operation *>; // todo: use smart ptrs instead of manually
        }

        // special: [0] for pre and [-1] for post processing
        unsigned int cur_tape = 0;

        func->walk([&](Operation *op) {
            if (op == func) {
                return; // don't visit the funcop itself
            }
            if (op->getName().getStringRef() == "quantum.device") {
                cur_tape++;
            }
            OpsEachTape[cur_tape]->push_back(op);
            if ((op->getName().getStringRef() == "quantum.device_release") &&
                cur_tape == OpsEachTape.size() - 2) {
                // reached post processing
                cur_tape++;
            }
        });
    }

    void runOnOperation() override
    {
        Operation *module = getOperation();
        // llvm::errs() << *module << "Hello world!\n";

        // 1. Identify the functions with multiple tapes
        // Walk through each function and count the number of devices.
        // In frontend when tracing (jax_tracers.py/trace_quantum_function),
        // each tape has its own quantum.device operation attached to it
        SmallVector<func::FuncOp> MultitapePrograms;
        module->walk([&](func::FuncOp func) {
            if (countTapes(func) >= 2) {
                MultitapePrograms.push_back(func);
            }
        });

        // Do nothing and exit for classical and single-tape programs
        if (MultitapePrograms.empty()) {
            return;
        }

        llvm::errs() << "program function is: \n";
        for (auto _ : MultitapePrograms) {
            llvm::errs() << _;
        }

        // 2. Parse the function into tapes
        // total number of operation lists is number of tapes +2
        // for classical pre and post processing

        // for (auto func : MultitapePrograms){
        func::FuncOp func = MultitapePrograms[0]; // temporary!
        SmallVector<SmallVector<Operation *> *> OpsEachTape(countTapes(func) + 2, nullptr);

        CollectOperationsForEachTape(func, OpsEachTape);
        //}

        for (auto _ : OpsEachTape) {
            llvm::errs() << "############################\n";
            for (auto op : *_) {
                llvm::errs() << *op << "\n";
            }
        }

        // 3. Get the SSA values needed by the post processing
        // These need to be returned by the tapes
        SmallVector<Value *> NecessaryValuesForPostProcessing;
        for (Operation *op : *(OpsEachTape.back())) {
            llvm::errs() << *op << " visited in pp\n";
        }

        // TODO: use smart ptrs instead of manually
        for (auto _ : OpsEachTape) {
            delete _;
        }
    }
};

std::unique_ptr<Pass> createSplitMultipleTapesPass()
{
    return std::make_unique<SplitMultipleTapesPass>();
}

} // namespace catalyst