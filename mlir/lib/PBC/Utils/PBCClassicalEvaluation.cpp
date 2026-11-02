// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "PBC/Utils/PBCClassicalEvaluation.h"

#include <dlfcn.h>
#include <string>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace catalyst::pbc {
namespace {

constexpr int64_t maxElements = 100000;
constexpr unsigned maxOperations = 10000;
constexpr unsigned maxFunctions = 256;
using Evaluate = std::vector<std::string> (*)(const std::string &,
                                              const std::vector<std::string> &);

bool boundedTensor(Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getNumElements() > maxElements) {
        return false;
    }
    Type element = tensor.getElementType();
    return isa<IntegerType, FloatType, ComplexType>(element);
}

// Restrict native evaluation to numerical tensor IR. In particular, user-defined
// custom calls cannot turn a resource analysis into an arbitrary runtime action.
class PureTensorGraph {
    llvm::DenseSet<Operation *> visited;
    unsigned operations = 0;

  public:
    llvm::SmallVector<func::FuncOp> functions;

    bool inspect(Operation *root) {
        return !root->walk([&](Operation *op) {
                        if (++operations > maxOperations) {
                            return WalkResult::interrupt();
                        }
                        if (auto function = dyn_cast<func::FuncOp>(op)) {
                            return function.isDeclaration() ? WalkResult::interrupt()
                                                            : WalkResult::advance();
                        }
                        if (auto call = dyn_cast<func::CallOp>(op)) {
                            auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                                call, call.getCalleeAttr());
                            if (!callee || callee.isDeclaration() || callee.getName() == "main") {
                                return WalkResult::interrupt();
                            }
                            if (visited.insert(callee).second) {
                                if (functions.size() >= maxFunctions) {
                                    return WalkResult::interrupt();
                                }
                                functions.push_back(callee);
                                if (!inspect(callee)) {
                                    return WalkResult::interrupt();
                                }
                            }
                            return WalkResult::advance();
                        }
                        if (isa<func::ReturnOp>(op)) {
                            return WalkResult::advance();
                        }
                        if (!llvm::all_of(op->getResultTypes(), boundedTensor)) {
                            return WalkResult::interrupt();
                        }
                        StringRef name = op->getName().getStringRef();
                        if (!name.starts_with("stablehlo.") && !name.starts_with("chlo.")) {
                            return WalkResult::interrupt();
                        }
                        if (name == "stablehlo.custom_call") {
                            auto target = op->getAttrOfType<StringAttr>("call_target_name");
                            auto effects = op->getAttrOfType<BoolAttr>("has_side_effect");
                            if (!target || (effects && effects.getValue()) ||
                                (target.getValue() != "lapack_dgetrf_ffi" &&
                                 target.getValue() != "lapack_sgetrf_ffi" &&
                                 target.getValue() != "lapack_zgetrf_ffi" &&
                                 target.getValue() != "lapack_cgetrf_ffi")) {
                                return WalkResult::interrupt();
                            }
                        }
                        // These operations communicate with the host or other processes.
                        if (name == "stablehlo.infeed" || name == "stablehlo.outfeed" ||
                            name == "stablehlo.send" || name == "stablehlo.recv" ||
                            name == "stablehlo.all_reduce" || name == "stablehlo.all_gather" ||
                            name == "stablehlo.all_to_all" ||
                            name == "stablehlo.collective_broadcast" ||
                            name == "stablehlo.collective_permute" ||
                            name == "stablehlo.reduce_scatter" || name == "stablehlo.rng" ||
                            name == "stablehlo.rng_bit_generator" ||
                            name == "stablehlo.replica_id" || name == "stablehlo.partition_id") {
                            return WalkResult::interrupt();
                        }
                        return WalkResult::advance();
                    })
                    .wasInterrupted();
    }
};

} // namespace

Attribute evaluatePBCClassicalValue(Value value, llvm::function_ref<Attribute(Value)> resolve) {
    Operation *op = value.getDefiningOp();
    if (!op || !llvm::all_of(op->getResultTypes(), boundedTensor)) {
        return {};
    }
    auto getter = reinterpret_cast<void *(*)()>(
        dlsym(RTLD_DEFAULT, "getPythonPBCClassicalEvaluationFunction"));
    auto evaluate = getter ? reinterpret_cast<Evaluate>(getter()) : nullptr;
    if (!evaluate) {
        return {};
    }
    PureTensorGraph graph;
    if (!graph.inspect(op)) {
        return {};
    }

    llvm::SetVector<Value> inputs(op->operand_begin(), op->operand_end());
    for (Region &region : op->getRegions()) {
        getUsedValuesDefinedAbove(region, inputs);
    }
    llvm::SmallVector<Type> inputTypes;
    std::vector<std::string> inputAttributes;
    for (Value input : inputs) {
        if (!boundedTensor(input.getType())) {
            return {};
        }
        auto attribute = dyn_cast_or_null<DenseElementsAttr>(resolve(input));
        if (!attribute || attribute.getType() != input.getType()) {
            return {};
        }
        inputTypes.push_back(input.getType());
        std::string text;
        llvm::raw_string_ostream stream(text);
        attribute.print(stream);
        inputAttributes.push_back(std::move(text));
    }

    OpBuilder builder(op->getContext());
    OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
    for (func::FuncOp function : graph.functions) {
        func::FuncOp cloned = function.clone();
        cloned->removeAttr("llvm.linkage");
        cloned.setPrivate();
        module->push_back(cloned);
    }
    builder.setInsertionPointToEnd(module->getBody());
    auto entry = func::FuncOp::create(builder, builder.getUnknownLoc(), "main",
                                      builder.getFunctionType(inputTypes, op->getResultTypes()));
    Block *body = entry.addEntryBlock();
    IRMapping mapping;
    for (auto [input, argument] : llvm::zip(inputs, body->getArguments())) {
        mapping.map(input, argument);
    }
    builder.setInsertionPointToStart(body);
    Operation *copy = builder.clone(*op, mapping);
    func::ReturnOp::create(builder, builder.getUnknownLoc(), copy->getResults());

    std::string text;
    llvm::raw_string_ostream stream(text);
    module->print(stream, OpPrintingFlags().useLocalScope());
    std::vector<std::string> outputs = evaluate(text, inputAttributes);
    unsigned index = cast<OpResult>(value).getResultNumber();
    if (outputs.size() != op->getNumResults()) {
        return {};
    }
    auto result =
        dyn_cast_or_null<DenseElementsAttr>(parseAttribute(outputs[index], op->getContext()));
    return result && result.getType() == value.getType() ? result : Attribute{};
}

} // namespace catalyst::pbc
