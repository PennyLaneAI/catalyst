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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
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

    void CreateTapeFunction(SmallVector<Operation *> *TapeOps,
                            SmallVector<Value> &NecessaryValuesForPostProcessing,
                            IRRewriter &builder, Operation *module, unsigned int tape_number,
                            StringRef OriginalMultitapeFuncName)
    {
        assert(TapeOps); // nullptr protection

        // 1. Identify the necessary return values
        SmallVector<Value> RetValues;
        SmallVector<mlir::Type> RetTypes;

        for (Value v : NecessaryValuesForPostProcessing) {
            Operation *VDefOp = v.getDefiningOp();
            if (std::find(TapeOps->begin(), TapeOps->end(), VDefOp) != TapeOps->end()) {
                // This Value needed for PP is in this tape!
                RetValues.push_back(v);
                RetTypes.push_back(v.getType());
            }
        }

        // We process the tapes in reverse order
        // Hence from the viewpoint of the earlier tapes,
        // a processed later tape (which has become a call op) is also their "post processing"
        // The later tape's arguments need to be returned by the earlier tapes
        // Hence when processing a tape, it needs to ask for earlier tapes to return
        // values it needs, just like how post processing asks for all tapes to return
        // values it needs.
        for (Operation *op : *TapeOps) {
            for (auto operand : op->getOperands()) {
                Operation *OperandSource = operand.getDefiningOp();
                if (std::find(TapeOps->begin(), TapeOps->end(), OperandSource) == TapeOps->end()) {
                    // A tape operand not produced in this tape itself,
                    // must be from the tapes/preprocessing!
                    // Need to be replaced by the previous tape functions' return values
                    NecessaryValuesForPostProcessing.push_back(operand);
                }
            }
        }

        // 2. Create a scf.execute_region and wrap it around the tape ops
        // The scf.execute_region needs to yield (aka return) the values identified
        builder.setInsertionPoint(TapeOps->front());
        scf::ExecuteRegionOp executeRegionOp =
            builder.create<scf::ExecuteRegionOp>(module->getLoc(), ArrayRef(RetTypes));

        builder.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
        mlir::Block::iterator it = executeRegionOp.getRegion().front().end();
        for (auto op : *TapeOps) {
            op->moveBefore(&executeRegionOp.getRegion().front(), it);
        }

        builder.setInsertionPointAfter(&executeRegionOp.getRegion().front().back());
        Operation *y = builder.create<scf::YieldOp>(module->getLoc(), ArrayRef(RetValues));

        // 3. The scf outliner automatically captures values not defined in the
        // to-be-outlined function body and transforms them to the outlined function's
        // arguments, but it does not automatically propagate the returned value
        // downstream to replace the previously used values. This we need to do manually.

        // At this stage, the scf.yield values ARE the ones to be replaced by
        // the scf region's result values, since we never replaced anything yet
        // Of course, don't replace the use in the region itself!

        SmallPtrSet<Operation *, 16> exceptions;
        exceptions.insert(y);
        for (auto op : *TapeOps) {
            exceptions.insert(op);
        }

        for (size_t i = 0; i < RetValues.size(); i++) {
            RetValues[i].replaceAllUsesExcept(executeRegionOp->getResults()[i], exceptions);
        }

        // 4. Outline the region
        func::CallOp call;
        FailureOr<func::FuncOp> outlined = outlineSingleBlockRegion(
            builder, module->getLoc(), executeRegionOp.getRegion(),
            OriginalMultitapeFuncName.str() + "_tape_" + std::to_string(tape_number), &call);
    }

    void runOnOperation() override
    {
        Operation *module = getOperation();
        mlir::IRRewriter builder(module->getContext());

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

        // 2. Parse the function into tapes
        // total number of operation lists is number of tapes +2
        // for classical pre and post processing

        // for (auto func : MultitapePrograms){
        func::FuncOp func =
            MultitapePrograms[0]; // temporary! TODO: process all multitape functions
        SmallVector<SmallVector<Operation *> *> OpsEachTape(countTapes(func) + 2, nullptr);

        CollectOperationsForEachTape(func, OpsEachTape);
        //}

        // 3. Get the SSA values needed by the post processing (PP)
        // These need to be returned by the tapes
        SmallVector<Value> NecessaryValuesForPostProcessing;

        // Go through all the operands of all the PP ops
        // Find the ones that are not produced in PP itself
        SmallVector<Operation *> PPOps = *(OpsEachTape.back());

        for (Operation *op : PPOps) {
            for (auto operand : op->getOperands()) {
                Operation *OperandSource = operand.getDefiningOp();
                if (std::find(PPOps.begin(), PPOps.end(), OperandSource) == PPOps.end()) {
                    // A PP operand not produced in PP itself, must be from the tapes/preprocessing!
                    // Need to be replaced by the tape functions' return values
                    NecessaryValuesForPostProcessing.push_back(operand);
                }
            }
        }

        // 4. Generate the functions for each tape
        unsigned int NumTapes = countTapes(func);
        for (unsigned int i = 0; i < NumTapes; i++) {

            CreateTapeFunction(OpsEachTape[OpsEachTape.size() - 2 - i],
                               NecessaryValuesForPostProcessing, builder, module,
                               OpsEachTape.size() - 3 - i, func.getSymName());
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