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

/// This pass creates the ARTIQ entry point structure that allows Catalyst-generated
/// kernels to be loaded and executed by ARTIQ
///
/// The pass transforms:
///   @__kernel__(ptr, ptr, i64)
/// Into:
///   @__modinit__(ptr) -> calls @__kernel__(ptr, nullptr, 0)

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "ARTIQRuntimeBuilder.hpp"
#include "RTIO/Transforms/Passes.h"

using namespace mlir;

namespace catalyst {
namespace rtio {

#define GEN_PASS_DEF_RTIOEMITARTIQRUNTIMEPASS
#include "RTIO/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ARTIQ Runtime Constants
//===----------------------------------------------------------------------===//

namespace ARTIQRuntime {
constexpr StringLiteral modinit = "__modinit__";
constexpr StringLiteral artiqPersonality = "__artiq_personality";
} // namespace ARTIQRuntime

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RTIOEmitARTIQRuntimePass
    : public impl::RTIOEmitARTIQRuntimePassBase<RTIOEmitARTIQRuntimePass> {
    using RTIOEmitARTIQRuntimePassBase::RTIOEmitARTIQRuntimePassBase;

    void runOnOperation() override
    {
        ModuleOp module = getOperation();
        MLIRContext *ctx = &getContext();
        OpBuilder builder(ctx);

        // Check if __modinit__ already exists (the entry point of ARTIQ device)
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQRuntime::modinit)) {
            return;
        }

        // Find the kernel function e(could be LLVM func or func.func)
        LLVM::LLVMFuncOp llvmKernelFunc =
            module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQFuncNames::kernel);

        if (!llvmKernelFunc) {
            module.emitError("Cannot find kernel function");
            return signalPassFailure();
        }

        // Create ARTIQ runtime wrapper
        if (failed(emitARTIQRuntimeForLLVMFunc(module, builder, llvmKernelFunc))) {
            return signalPassFailure();
        }
    }

  private:
    /// Emit ARTIQ runtime wrapper for LLVM dialect kernel function
    LogicalResult emitARTIQRuntimeForLLVMFunc(ModuleOp module, OpBuilder &builder,
                                              LLVM::LLVMFuncOp kernelFunc)
    {
        MLIRContext *ctx = builder.getContext();
        Location loc = module.getLoc();

        // Types
        Type voidTy = LLVM::LLVMVoidType::get(ctx);
        Type ptrTy = LLVM::LLVMPointerType::get(ctx);
        Type i32Ty = IntegerType::get(ctx, 32);
        Type i64Ty = IntegerType::get(ctx, 64);

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());

        // Declare __artiq_personality (exception handling)
        declareARTIQPersonality(module, builder, loc);

        // Create entry function: void @__modinit__(ptr %self)
        auto modinitTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy});
        auto modinitFunc = builder.create<LLVM::LLVMFuncOp>(loc, ARTIQRuntime::modinit, modinitTy);
        modinitFunc.setLinkage(LLVM::Linkage::External);

        // Set personality function for exception handling
        modinitFunc.setPersonalityAttr(FlatSymbolRefAttr::get(ctx, ARTIQRuntime::artiqPersonality));

        // Create function body
        Block *entry = modinitFunc.addEntryBlock(builder);
        builder.setInsertionPointToStart(entry);

        // Get the actual kernel function type and create matching arguments
        auto kernelFuncTy = kernelFunc.getFunctionType();
        SmallVector<Value> callArgs;

        for (Type argTy : kernelFuncTy.getParams()) {
            // Create zero/null values for each argument type
            if (isa<LLVM::LLVMPointerType>(argTy)) {
                callArgs.push_back(builder.create<LLVM::ZeroOp>(loc, ptrTy));
            }
            else if (argTy.isInteger(64)) {
                callArgs.push_back(
                    builder.create<LLVM::ConstantOp>(loc, i64Ty, builder.getI64IntegerAttr(0)));
            }
            else if (argTy.isInteger(32)) {
                callArgs.push_back(
                    builder.create<LLVM::ConstantOp>(loc, i32Ty, builder.getI32IntegerAttr(0)));
            }
            else {
                // For other types, use null pointer as fallback
                callArgs.push_back(builder.create<LLVM::ZeroOp>(loc, ptrTy));
            }
        }

        auto callOp = builder.create<LLVM::CallOp>(loc, kernelFunc, callArgs);
        callOp.setTailCallKind(LLVM::TailCallKind::Tail);

        builder.create<LLVM::ReturnOp>(loc, ValueRange{});

        return success();
    }

    /// Declare __artiq_personality function
    void declareARTIQPersonality(ModuleOp module, OpBuilder &builder, Location loc)
    {
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(ARTIQRuntime::artiqPersonality)) {
            return;
        }

        Type i32Ty = IntegerType::get(builder.getContext(), 32);
        auto personalityTy = LLVM::LLVMFunctionType::get(i32Ty, {}, /*isVarArg=*/true);
        builder.create<LLVM::LLVMFuncOp>(loc, ARTIQRuntime::artiqPersonality, personalityTy,
                                         LLVM::Linkage::External);
    }
};

} // namespace
} // namespace rtio
} // namespace catalyst
