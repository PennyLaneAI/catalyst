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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
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

    unsigned int whichResult(Operation *op, Value v)
    {
        // Returns the index at which op's result is v
        for (unsigned int i = 0; i < op->getNumResults(); i++) {
            if (op->getResult(i) == v) {
                return i;
            }
        }
        assert(false); // failure
    }

    void CreateTapeFunction(SmallVector<Operation *> *TapeOps,
                            SmallVector<Operation *> &NecessaryOpsForPostProcessing,
                            const ArrayRef<NamedAttribute> &FullOriginalFuncAttrs,
                            OpBuilder &builder, Operation *module,
                            func::FuncOp OriginalMultitapeFunc, unsigned int tape_number,
                            StringRef OriginalMultitapeFuncName)
    {
        SmallVector<Operation *> ArgOps;
        SmallVector<mlir::Type> ArgOpsTypes;
        SmallVector<Value> CallArgs;
        SmallVector<Value> VisitedOperands;
        llvm::DenseMap<Value, unsigned int> FuncArg2BlockArg;
        SmallVector<Operation *> RetOps;
        SmallVector<mlir::Type> RetOpsTypes;

        unsigned int p = 0;
        for (auto op : *TapeOps) {

            // If an op in tape uses something not defined in the tape,
            // it must have come from preprocessing or a previous tape.
            // In either case, take in it as an argument
            // The i-th argument of the tapefunc will correspond to the i-th op in Argops
            for (auto operand : op->getOperands()) {
                if (std::find(VisitedOperands.begin(), VisitedOperands.end(), operand) !=
                    VisitedOperands.end()) {
                    // already recorded this value as an argument
                    continue;
                }
                if (!isa<BlockArgument>(operand)) {
                    Operation *OperandSource = operand.getDefiningOp();
                    if (std::find(TapeOps->begin(), TapeOps->end(), OperandSource) ==
                        TapeOps->end()) {
                        ArgOps.push_back(OperandSource);
                        ArgOpsTypes.push_back(operand.getType());
                        CallArgs.push_back(operand);
                        FuncArg2BlockArg[operand] = p;
                        p++;
                    }
                }
                else {
                    ArgOpsTypes.push_back(operand.getType());
                    CallArgs.push_back(operand);
                    FuncArg2BlockArg[operand] = p;
                    p++;
                }
            }

            // If an op is needed by post processing, it should be returned
            if (std::find(NecessaryOpsForPostProcessing.begin(),
                          NecessaryOpsForPostProcessing.end(),
                          op) != NecessaryOpsForPostProcessing.end()) {
                RetOps.push_back(op);
                for (size_t i = 0; i < op->getNumResults(); i++) {
                    RetOpsTypes.push_back(op->getResultTypes()[i]);
                }
            }
        }

        // We process the tapes in reverse order
        // Hence from the viewpoint of the earlier tapes,
        // a processed later tape (which has become a call op) is also its "post processing"
        // The later tape's arguments need to be returned by the earlier tapes
        for (auto op : ArgOps) {
            NecessaryOpsForPostProcessing.push_back(op);
        }

        FunctionType fType = FunctionType::get(module->getContext(), ArgOpsTypes, RetOpsTypes);

        // Create the tape function.
        // Keep the original function's attributes
        // We do not want the name and the type of the original multitape function:
        // The type could be different, and we want different names for each tape
        SmallVector<NamedAttribute> FuncAttrs;
        for (auto attr : FullOriginalFuncAttrs) {
            StringRef attrname = attr.getName();
            if ((attrname != "sym_name") && (attrname != "function_type")) {
                FuncAttrs.push_back(attr);
            }
        }

        // The arguments don't need to carry attributes
        builder.setInsertionPointAfter(OriginalMultitapeFunc);
        func::FuncOp TapeFunc = builder.create<func::FuncOp>(
            module->getLoc(),
            OriginalMultitapeFuncName.str() + "_tape_" + std::to_string(tape_number), fType,
            FuncAttrs);

        // Call the newly made tape function in the original function
        // The i-th argument corresponds to the i-th op in Argops
        // The callop results should replace the retops in their users

        builder.setInsertionPoint(TapeOps->front());
        func::CallOp callOp = builder.create<func::CallOp>(module->getLoc(), TapeFunc, CallArgs);

        // Inject the tape's operations into the tape function
        Block *entrybb = TapeFunc.addEntryBlock();
        builder.setInsertionPointToEnd(entrybb);

        //<< entrybb->getArgument(0) << entrybb->getArgument(1) << "\n";

        std::map<Operation *, Operation *> ClonedOpShadows;
        SmallVector<Value> ClonedRetValues;
        for (auto op : *TapeOps) {
            ClonedOpShadows[op] = op->clone();
            builder.insert(ClonedOpShadows[op]);
        }

        // for (auto op : *TapeOps) {
        //     op->replaceAllUsesWith(ClonedOpShadows[op]);
        // }

        for (auto it = ClonedOpShadows.begin(); it != ClonedOpShadows.end(); it++) {
            for (size_t i = 0; i < it->second->getNumOperands(); i++) {
                if (!isa<BlockArgument>(it->second->getOperand(i))) {
                    Operation *OperandSource = it->second->getOperand(i).getDefiningOp();
                    if (std::find(TapeOps->begin(), TapeOps->end(), OperandSource) !=
                        TapeOps->end()) {
                        it->second->replaceUsesOfWith(
                            it->first->getOperand(i),
                            ClonedOpShadows[OperandSource]->getResult(
                                whichResult(OperandSource, it->first->getOperand(i))));
                    }
                }
            }
        }

        llvm::DenseMap<Value, Value> ResultMap;
        unsigned int j = 0;
        for (auto op : *TapeOps) {
            // Return the values needed
            if (std::find(RetOps.begin(), RetOps.end(), op) != RetOps.end()) {
                for (size_t i = 0; i < op->getNumResults(); i++) {
                    ClonedRetValues.push_back(ClonedOpShadows[op]->getResult(i));
                    ResultMap[op->getResult(i)] = callOp->getResult(j);
                    j++;
                }
            }
        }
        builder.create<func::ReturnOp>(module->getLoc(), ArrayRef(ClonedRetValues));

        for (auto op : RetOps) {
            for (size_t i = 0; i < op->getNumResults(); i++) {
                op->getResult(i).replaceAllUsesWith(ResultMap[op->getResult(i)]);
            }
        }

        // Replace the cloned operation's operands
        // for (auto op : *TapeOps) {
        //    op->replaceAllUsesWith(ClonedOpShadows[op]);
        //}

        // If an operand is an argument, the i-th argument corresponds to the i-th op in Argops
        for (auto it = ClonedOpShadows.begin(); it != ClonedOpShadows.end(); it++) {
            for (size_t i = 0; i < it->first->getNumOperands(); i++) {
                if (!isa<BlockArgument>(it->first->getOperand(i))) {
                    Operation *OperandSource = it->first->getOperand(i).getDefiningOp();
                    if (!(std::find(ArgOps.begin(), ArgOps.end(), OperandSource) == ArgOps.end())) {
                        // An argop
                        Value operand = it->first->getOperand(i);

                        it->second->replaceUsesOfWith(
                            it->second->getOperand(i),
                            // entrybb->getArgument(
                            //     std::find(ArgOps.begin(), ArgOps.end(), OperandSource) -
                            //     ArgOps.begin()));
                            entrybb->getArgument(FuncArg2BlockArg[operand]));
                    }
                }
                else {
                    Value operand = it->first->getOperand(i);

                    it->second->replaceUsesOfWith(it->second->getOperand(i),
                                                  entrybb->getArgument(FuncArg2BlockArg[operand]));
                }
            }
        }

        // Injection done, erase the original operations
        // Note that some op in TapeOps are in NecessaryOpsForPostProcessing
        // so remove them first
        for (auto op : *TapeOps) {
            auto where = std::find(NecessaryOpsForPostProcessing.begin(),
                                   NecessaryOpsForPostProcessing.end(), op);
            if (where != NecessaryOpsForPostProcessing.end()) {
                NecessaryOpsForPostProcessing.erase(where);
            }
            op->dropAllUses();
            op->erase();
        }
    }

    void runOnOperation() override
    {
        Operation *module = getOperation();
        OpBuilder builder(module->getContext());

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
        SmallVector<Operation *> NecessaryOpsForPostProcessing;

        // Go through all the operands of all the PP ops
        // Find the ones that are not produced in PP itself
        SmallVector<Operation *> PPOps = *(OpsEachTape.back());
        for (Operation *op : PPOps) {
            for (auto operand : op->getOperands()) {
                Operation *OperandSource = operand.getDefiningOp();
                if (std::find(PPOps.begin(), PPOps.end(), OperandSource) == PPOps.end()) {
                    // A PP operand not produced in PP itself, must be from the tapes/preprocessing!
                    // Need to be replaced by the tape functions' return values
                    NecessaryOpsForPostProcessing.push_back(OperandSource);
                }
            }
        }

        // 4. Generate the functions for each tape
        unsigned int NumTapes = countTapes(func);
        for (unsigned int i = 0; i < NumTapes; i++) {
            CreateTapeFunction(OpsEachTape[OpsEachTape.size() - 2 - i],
                               NecessaryOpsForPostProcessing, func->getAttrs(), builder, module,
                               func, OpsEachTape.size() - 3 - i, func.getSymName());
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