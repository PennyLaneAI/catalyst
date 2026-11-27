// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "StringConstants.h"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DEF_ANNOTATEBUFFEROWNERSHIPPASS
#include "Catalyst/Transforms/Passes.h.inc"

} // namespace catalyst

namespace {

/// Function-local information that stays unchanged while callee summaries converge.
struct FunctionBufferOwnership {
    func::FuncOp function;
    BufferViewFlowAnalysis origins;
    BufferViewFlowAnalysis::ValueMapT reverseOrigins;
    SmallVector<std::pair<func::CallOp, func::FuncOp>> calls;
    SmallVector<func::ReturnOp> returns;

    FunctionBufferOwnership(func::FuncOp function, SymbolTableCollection &symbolTables)
        : function(function), origins(function) {
        for (const auto &[source, destinations] : origins) {
            for (Value destination : destinations) {
                reverseOrigins[destination].insert(source);
            }
        }
        function.walk([&](Operation *operation) {
            if (auto call = dyn_cast<func::CallOp>(operation)) {
                calls.emplace_back(call, symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
                                             call, call.getCalleeAttr()));
            } else if (auto ret = dyn_cast<func::ReturnOp>(operation);
                       ret && ret->getParentOfType<func::FuncOp>() == function) {
                returns.push_back(ret);
            }
        });
    }
};

/// Infer one function's owned results using the current callee summaries.
static SmallVector<bool>
inferOwnedBufferResults(const FunctionBufferOwnership &analysis,
                        const DenseMap<Operation *, SmallVector<bool>> &ownedResults) {
    func::FuncOp function = analysis.function;
    SetVector<Value> freshCallResults;
    for (auto [call, callee] : analysis.calls) {
        auto it = ownedResults.find(callee);
        if (it == ownedResults.end()) {
            continue;
        }
        for (auto [i, isOwned] : llvm::enumerate(it->second)) {
            if (isOwned) {
                freshCallResults.insert(call.getResult(i));
            }
        }
    }
    auto resolveOrigins = [&](Value value) {
        BufferViewFlowAnalysis::ValueSetT visited;
        SmallVector<Value> pending{value};
        while (!pending.empty()) {
            Value current = pending.pop_back_val();
            if (!visited.insert(current).second) {
                continue;
            }
            // A proven fresh call result cannot alias its operands. Stop at it
            // instead of removing edges from the cached graph.
            if (freshCallResults.contains(current)) {
                continue;
            }
            auto it = analysis.reverseOrigins.find(current);
            if (it != analysis.reverseOrigins.end()) {
                llvm::append_range(pending, it->second);
            }
        }
        return visited;
    };
    SmallVector<BufferViewFlowAnalysis::ValueSetT> resultOrigins(function.getNumResults());
    SmallVector<bool> fresh(function.getNumResults(), true);
    for (func::ReturnOp ret : analysis.returns) {
        for (auto [index, value] : llvm::enumerate(ret.getOperands())) {
            if (!isa<BaseMemRefType>(value.getType())) {
                fresh[index] = false;
                continue;
            }
            bool hasOrigin = false;
            for (Value source : resolveOrigins(value)) {
                if (!isa<BaseMemRefType>(source.getType())) {
                    continue;
                }
                // Deallocating a returned subview is not ownership transfer of
                // a base buffer. Only casts are accepted.
                if (source.getDefiningOp() && isa<ViewLikeOpInterface>(source.getDefiningOp()) &&
                    !source.getDefiningOp<memref::CastOp>()) {
                    fresh[index] = false;
                }
                if (!analysis.origins.mayBeTerminalBuffer(source)) {
                    continue;
                }
                hasOrigin = true;
                resultOrigins[index].insert(source);
                if (!source.getDefiningOp<memref::AllocOp>() &&
                    !freshCallResults.contains(source)) {
                    fresh[index] = false;
                }
            }
            fresh[index] = fresh[index] && hasOrigin;
        }
    }
    if (analysis.returns.empty()) {
        return SmallVector<bool>(function.getNumResults(), false);
    }
    // Include unowned results too: they can alias an otherwise fresh result.
    DenseMap<Value, size_t> firstResult;
    for (auto [index, origins] : llvm::enumerate(resultOrigins)) {
        for (Value origin : origins) {
            auto [it, inserted] = firstResult.try_emplace(origin, index);
            if (!inserted) {
                fresh[index] = false;
                fresh[it->second] = false;
            }
        }
    }
    return fresh;
}

/// Infer ownership before running function passes: inspecting sibling functions
/// from the nested deallocation pass would race with their transformations.
struct AnnotateBufferOwnershipPass
    : public catalyst::impl::AnnotateBufferOwnershipPassBase<AnnotateBufferOwnershipPass> {
    void runOnOperation() override {
        SmallVector<func::FuncOp> functions;
        getOperation().walk([&](func::FuncOp function) {
            if (!function.isExternal()) {
                functions.push_back(function);
            }
        });
        DenseMap<Operation *, SmallVector<bool>> ownedResults;
        for (auto function : functions) {
            ownedResults[function] = SmallVector<bool>(function.getNumResults(), false);
        }
        SymbolTableCollection symbolTables;

        DenseMap<Operation *, FunctionBufferOwnership> analyses;
        DenseMap<Operation *, SetVector<Operation *>> callers;
        SetVector<Operation *> worklist;
        for (auto function : functions) {
            // These functions cannot return owned buffers. Calls in their bodies
            // are still annotated by the final module walk.
            if (llvm::none_of(function.getResultTypes(),
                              [](Type type) { return isa<BaseMemRefType>(type); })) {
                continue;
            }
            auto [it, inserted] = analyses.try_emplace(function, function, symbolTables);
            for (const auto &call : it->second.calls) {
                if (call.second) {
                    callers[call.second].insert(function);
                }
            }
            worklist.insert(function);
        }

        // Revisit only callers whose callee summaries changed. Recursive calls
        // participate in the same fixed point and cannot bootstrap ownership.
        while (!worklist.empty()) {
            Operation *function = worklist.pop_back_val();
            SmallVector<bool> fresh =
                inferOwnedBufferResults(analyses.find(function)->second, ownedResults);
            auto &owned = ownedResults[function];
            bool changed = false;
            for (unsigned i = 0; i < fresh.size(); ++i) {
                if (fresh[i] && !owned[i]) {
                    owned[i] = true;
                    changed = true;
                }
            }
            if (!changed) {
                continue;
            }
            auto it = callers.find(function);
            if (it != callers.end()) {
                for (Operation *caller : it->second) {
                    worklist.insert(caller);
                }
            }
        }

        getOperation().walk([&](func::CallOp call) {
            call->removeAttr(catalyst::OWNED_BUFFER_RESULTS_ATTR);
            auto callee =
                symbolTables.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
            auto it = ownedResults.find(callee);
            if (it == ownedResults.end()) {
                return;
            }
            SmallVector<int32_t> ownedResultIndices;
            for (auto [i, isOwned] : llvm::enumerate(it->second)) {
                if (isOwned) {
                    ownedResultIndices.push_back(i);
                }
            }
            if (!ownedResultIndices.empty()) {
                call->setAttr(catalyst::OWNED_BUFFER_RESULTS_ATTR,
                              DenseI32ArrayAttr::get(&getContext(), ownedResultIndices));
            }
        });
    }
};

} // namespace
