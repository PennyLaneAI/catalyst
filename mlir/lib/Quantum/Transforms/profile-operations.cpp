// Copyright 2025 Xanadu Quantum Technologies Inc.
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

#define DEBUG_TYPE "profile-operations"

#include "Quantum/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;

namespace {

func::FuncOp getOrCreateTimestampFunction(ModuleOp module, OpBuilder &builder)
{
    if (!module) {
        return func::FuncOp();
    }

    if (auto existingFunc =
            module.lookupSymbol<func::FuncOp>("__catalyst__rt__profiler_get_timestamp")) {
        return existingFunc;
    }

    auto i64Type = mlir::IntegerType::get(builder.getContext(), 64);
    auto funcType = builder.getFunctionType({}, {i64Type});

    OpBuilder moduleBuilder(module.getBodyRegion());
    auto func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), "__catalyst__rt__profiler_get_timestamp", funcType);

    func.setPrivate();
    return func;
}

func::FuncOp getOrCreateRecordFunction(ModuleOp module, OpBuilder &builder)
{
    if (!module) {
        return func::FuncOp();
    }

    auto existingFunc = module.lookupSymbol<func::FuncOp>("__catalyst__rt__profiler_record");
    if (existingFunc) {
        return existingFunc;
    }

    auto i32Type = mlir::IntegerType::get(builder.getContext(), 32);
    auto i64Type = mlir::IntegerType::get(builder.getContext(), 64);
    auto stringType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    auto funcType = builder.getFunctionType({stringType, i32Type, i32Type, i64Type, i64Type}, {});

    OpBuilder moduleBuilder(module.getBodyRegion());
    auto func = moduleBuilder.create<func::FuncOp>(module.getLoc(),
                                                   "__catalyst__rt__profiler_record", funcType);

    func.setPrivate();
    return func;
}

func::FuncOp getOrCreatePrintStatsFunction(ModuleOp module, OpBuilder &builder)
{
    if (!module) {
        return func::FuncOp();
    }

    auto existingFunc = module.lookupSymbol<func::FuncOp>("__catalyst__rt__profiler_print_stats");
    if (existingFunc) {
        return existingFunc;
    }

    auto funcType = builder.getFunctionType({}, {});

    OpBuilder moduleBuilder(module.getBodyRegion());
    auto func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), "__catalyst__rt__profiler_print_stats", funcType);

    func.setPrivate();
    return func;
}

/// Helper function to recursively extract file location information from any location type
void extractLocationInfo(Location loc, std::string &file_name, uint32_t &line, uint32_t &column,
                         std::string &op_name)
{
    if (auto fileLoc = mlir::dyn_cast<FileLineColLoc>(loc)) {
        file_name = fileLoc.getFilename().str();
        line = fileLoc.getLine();
        column = fileLoc.getColumn();
    }
    else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
        // Check each sub-location in the fused location
        for (Location subLoc : fusedLoc.getLocations()) {
            extractLocationInfo(subLoc, file_name, line, column, op_name);
            if (!file_name.empty() || !op_name.empty())
                break; // Found a location
        }
    }
    else if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
        // First try to extract from the callee
        extractLocationInfo(callSiteLoc.getCallee(), file_name, line, column, op_name);
        // If not found, try the caller
        if (file_name.empty() && op_name.empty()) {
            extractLocationInfo(callSiteLoc.getCaller(), file_name, line, column, op_name);
        }
    }
    else if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
        // Extract the operation name from NameLoc
        std::string currentName = nameLoc.getName().str();

        // HACKY: Look for quantum operation names in the jit circuit path
        if (currentName.find("jit(circuit)/") != std::string::npos) {
            // Extract the operation name after "jit(circuit)/"
            size_t pos = currentName.find("jit(circuit)/");
            if (pos != std::string::npos) {
                std::string afterCircuit =
                    currentName.substr(pos + 13); // length of "jit(circuit)/"
                // Find the next "/" or end of string
                size_t nextSlash = afterCircuit.find("/");
                if (nextSlash != std::string::npos) {
                    op_name = afterCircuit.substr(0, nextSlash);
                }
                else {
                    op_name = afterCircuit;
                }
            }
        }
        else if (op_name.empty()) {
            op_name = currentName;
        }

        // Also try to extract from the child location for file info
        extractLocationInfo(nameLoc.getChildLoc(), file_name, line, column, op_name);
    }
}

/// Add profiling instrumentation around an operation
void addProfilingInstrumentation(Operation *op, OpBuilder &builder, const std::string &file_name,
                                 uint32_t line, uint32_t column)
{
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto timestampFunc = getOrCreateTimestampFunction(module, builder);
    if (!timestampFunc) {
        return;
    }

    OpBuilder beforeBuilder(op);
    auto startTime = beforeBuilder.create<func::CallOp>(op->getLoc(), timestampFunc, ValueRange{});

    OpBuilder afterBuilder(op);
    afterBuilder.setInsertionPointAfter(op);
    auto endTime = afterBuilder.create<func::CallOp>(op->getLoc(), timestampFunc, ValueRange{});

    // Create global string constant for file name
    std::string globalName =
        "__catalyst_profiler_filename_" + std::to_string(llvm::hash_value(file_name));

    auto existingGlobal = module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName);
    if (!existingGlobal) {
        OpBuilder moduleBuilder(module.getBodyRegion());

        auto i8Type = mlir::IntegerType::get(afterBuilder.getContext(), 8);
        auto stringLength = file_name.length() + 1;
        auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, stringLength);

        std::string nullTerminatedString = file_name + '\0';
        auto stringAttr = afterBuilder.getStringAttr(nullTerminatedString);

        existingGlobal = moduleBuilder.create<mlir::LLVM::GlobalOp>(
            module.getLoc(), arrayType, true, mlir::LLVM::Linkage::Private, globalName, stringAttr);
    }

    // Get pointer to the global
    auto i8PtrType = mlir::LLVM::LLVMPointerType::get(afterBuilder.getContext());
    auto stringConst = afterBuilder.create<mlir::LLVM::AddressOfOp>(op->getLoc(), i8PtrType,
                                                                    existingGlobal.getSymName());

    auto lineConst =
        afterBuilder.create<arith::ConstantOp>(op->getLoc(), afterBuilder.getI32IntegerAttr(line));
    auto columnConst = afterBuilder.create<arith::ConstantOp>(
        op->getLoc(), afterBuilder.getI32IntegerAttr(column));
    auto recordFunc = getOrCreateRecordFunction(module, afterBuilder);
    if (!recordFunc) {
        return;
    }

    SmallVector<mlir::Value> args;
    args.push_back(stringConst);            // const char* file_name
    args.push_back(lineConst);              // uint32_t line
    args.push_back(columnConst);            // uint32_t column
    args.push_back(startTime.getResult(0)); // int64_t start_time
    args.push_back(endTime.getResult(0));   // int64_t end_time
    afterBuilder.create<func::CallOp>(op->getLoc(), recordFunc, args);
}

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_PROFILINGPASS
#define GEN_PASS_DECL_PROFILINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct ProfilingPass : impl::ProfilingPassBase<ProfilingPass> {
    using ProfilingPassBase::ProfilingPassBase;

    void runOnOperation() final
    {
        Operation *module = getOperation();

        LLVM_DEBUG(dbgs() << "profiling pass - adding instrumentation to all operations\n");

        // Walk through all operations in the module hierarchy
        module->walk([&](Operation *op) {
            // Only instrument operations that are within functions
            if (!op->getParentOfType<func::FuncOp>()) {
                return;
            }

            // Skip function declarations and return operations
            if (isa<func::FuncOp>(op) || isa<func::ReturnOp>(op)) {
                return;
            }

            // Skip function calls for now
            if (isa<func::CallOp>(op)) {
                return;
            }

            // Skip control flow operations for now
            if (isa<scf::IfOp>(op) || isa<scf::ForOp>(op) || isa<scf::WhileOp>(op) ||
                isa<scf::ConditionOp>(op) || isa<scf::YieldOp>(op)) {
                return;
            }

            // Skip branch operations for now
            if (isa<cf::CondBranchOp>(op) || isa<cf::BranchOp>(op)) {
                return;
            }

            std::string file_name;
            uint32_t line = 0;
            uint32_t column = 0;
            std::string op_name;

            extractLocationInfo(op->getLoc(), file_name, line, column, op_name);

            // Create a combined identifier: operation_name:file_name:line:column
            std::string identifier;
            if (!op_name.empty()) {
                identifier = op_name;
                if (!file_name.empty()) {
                    identifier += ":" + file_name;
                    identifier += ":" + std::to_string(line) + ":" + std::to_string(column);
                }
            }
            else if (!file_name.empty()) {
                identifier = file_name + ":" + std::to_string(line) + ":" + std::to_string(column);
            }
            else {
                identifier = op->getName().getStringRef().str();
            }

            OpBuilder builder(op);
            addProfilingInstrumentation(op, builder, identifier, line, column);
        });

        // Add print stats call to teardown function
        module->walk([&](func::FuncOp funcOp) {
            if (funcOp.getSymName() == "teardown") {
                funcOp.walk([&](func::ReturnOp returnOp) {
                    OpBuilder beforeReturn(returnOp);
                    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
                    auto printStatsFunc = getOrCreatePrintStatsFunction(moduleOp, beforeReturn);
                    if (printStatsFunc) {
                        beforeReturn.create<func::CallOp>(returnOp.getLoc(), printStatsFunc,
                                                          ValueRange{});
                    }
                });
            }
        });
    }
};

} // namespace quantum

std::unique_ptr<Pass> createProfilingPass() { return std::make_unique<quantum::ProfilingPass>(); }

} // namespace catalyst
