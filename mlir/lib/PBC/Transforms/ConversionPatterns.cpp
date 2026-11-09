// Copyright 2025 Xanadu Quantum Technologies Inc.

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

#include "llvm/Support/MathExtras.h" // for llvm::numbers::pi
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumTypes.h"
#include "Quantum/Utils/QIRPauliRot.h"

using namespace mlir;

namespace {

using namespace catalyst::pbc;
using namespace catalyst::quantum;

//////////////////////////////////////////
// Pauli-based Computational Operations //
//////////////////////////////////////////

template <typename T> struct PPRotationBasedPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

    LogicalResult matchAndRewrite(T op, typename T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();

        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        // Get the rotation angle based on the op type
        // Since the qp.PauliRot(phi) == PPR(phi/2), this rotation_kind is multiplied by 2.
        Value thetaValue;
        Value cond;
        if constexpr (std::is_same_v<T, PPRotationOp>) {
            if (op.getCondition()) {
                cond = op.getCondition();
            } else {
                cond = LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(1));
            }
            // Compute the rotation angle: theta = π / rotation_kind
            // rotation_kind can be ±1, ±2, ±4, ±8
            double theta = 2 * (llvm::numbers::pi / static_cast<double>(op.getRotationKind()));
            thetaValue = LLVM::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(theta));
        } else if constexpr (std::is_same_v<T, PPRotationArbitraryOp>) {
            if (op.getCondition()) {
                cond = op.getCondition();
            } else {
                cond = LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(1));
            }
            // multiply by 2 to get the rotation angle
            thetaValue = LLVM::FMulOp::create(
                rewriter, loc, adaptor.getArbitraryAngle(),
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(2.0)));
        } else {
            static_assert(!std::is_same_v<T, T>(), "unexpected type in templated rewrite");
        }

        Value pauliWordPtr = getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct());
        Value modifiersPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
        createPauliRotCall(loc, rewriter, op.getOperation(), pauliWordPtr, thetaValue, modifiersPtr,
                           cond, adaptor.getInQubits());

        // Replace the op with the input qubits
        rewriter.replaceOp(op, adaptor.getInQubits());

        return success();
    }
};

template <typename T> struct PPMeasurementOpPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

    LogicalResult matchAndRewrite(T op, typename T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        const TypeConverter *conv = this->getTypeConverter();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();

        // Create a global string for the Pauli word
        Value selectSwitch;
        Value pauliWordPtr, pauliWordAltPtr;
        Value negated, negatedAlt;
        if constexpr (std::is_same_v<T, PPMeasurementOp>) {
            pauliWordPtr = getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct());
            negated =
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(op.getNegated()));
            pauliWordAltPtr = pauliWordPtr;
            negatedAlt = negated;
            selectSwitch = LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
        } else if constexpr (std::is_same_v<T, SelectPPMeasurementOp>) {
            pauliWordPtr = getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct_0());
            negated =
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(op.getNegated_0()));
            pauliWordAltPtr = getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct_1());
            negatedAlt =
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(op.getNegated_1()));
            selectSwitch = op.getSelectSwitch();
        } else {
            static_assert(!std::is_same_v<T, T>(), "unexpected type in templated rewrite");
        }

        // Postselection on the outcome, after the sign; -1 means none.
        int32_t postselect = -1;
        if constexpr (std::is_same_v<T, PPMeasurementOp>) {
            if (auto value = op.getPostselect()) {
                postselect = static_cast<int32_t>(*value);
            }
        }
        Value postselectValue =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(postselect));

        StringRef qirName = "__catalyst__qis__PauliMeasure";
        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Type qirSignature = LLVM::LLVMFunctionType::get(
            conv->convertType(ResultType::get(ctx)),
            {ptrType, IntegerType::get(ctx, 1), ptrType, IntegerType::get(ctx, 1),
             IntegerType::get(ctx, 1), IntegerType::get(ctx, 32), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, qirName, qirSignature);

        // Build the arguments
        int64_t numQubits = adaptor.getInQubits().size();
        SmallVector<Value> args;
        args.push_back(pauliWordPtr);
        args.push_back(negated);
        args.push_back(pauliWordAltPtr);
        args.push_back(negatedAlt);
        args.push_back(selectSwitch);
        args.push_back(postselectValue);
        args.push_back(
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(numQubits)));
        args.append(adaptor.getInQubits().begin(), adaptor.getInQubits().end());

        // Call the function and get the result pointer
        Value resultPtr = LLVM::CallOp::create(rewriter, loc, fnDecl, args).getResult();

        // Load the measurement result (i1) from the result pointer. On a delegated program the
        // result is a handle, and is not read.
        Value mres = isa<ResultType>(op.getMres().getType())
                         ? resultPtr
                         : LLVM::LoadOp::create(rewriter, loc, IntegerType::get(ctx, 1), resultPtr)
                               .getResult();

        // Replace the op with the measurement result and the input qubits
        SmallVector<Value> values;
        values.push_back(mres);
        values.append(adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        rewriter.replaceOp(op, values);

        return success();
    }
};

/// Lower gadget brackets to calls that carry the source rotation and verdict. The calls are
/// no-ops on a device without the Gadget extension.
struct GadgetBeginOpPattern : public OpConversionPattern<GadgetBeginOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(GadgetBeginOp op, GadgetBeginOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        Type ptrType = LLVM::LLVMPointerType::get(ctx);

        SmallVector<Value> args;
        args.push_back(getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct()));
        args.push_back(LLVM::ConstantOp::create(rewriter, loc, op.getThetaAttr()));
        const GadgetVerdict verdict = op.getVerdict().value_or(GadgetVerdict::unchecked);
        args.push_back(LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(verdict))));
        args.push_back(LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(adaptor.getInQubits().size())));
        args.append(adaptor.getInQubits().begin(), adaptor.getInQubits().end());

        Type signature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {ptrType, Float64Type::get(ctx), IntegerType::get(ctx, 32), IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);
        auto begin = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__qis__BeginGadget", signature);
        LLVM::CallOp::create(rewriter, loc, begin, args);
        rewriter.eraseOp(op);
        return success();
    }
};

struct GadgetEndOpPattern : public OpConversionPattern<GadgetEndOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(GadgetEndOp op, GadgetEndOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        MLIRContext *ctx = getContext();
        Type signature = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});
        auto end = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, "__catalyst__qis__EndGadget", signature);
        LLVM::CallOp::create(rewriter, op.getLoc(), end, ValueRange{});
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace pbc {

void flattenGadgets(Operation *root) {
    IRRewriter rewriter(root->getContext());
    SmallVector<GadgetOp> worklist;
    root->walk<WalkOrder::PostOrder>([&](GadgetOp op) { worklist.push_back(op); });
    for (GadgetOp op : worklist) {
        rewriter.setInsertionPoint(op);
        GadgetBeginOp::create(rewriter, op.getLoc(), op.getPauliProduct(), op.getThetaAttr(),
                              op.getVerdictAttr(), op.getInQubits());
        Block *body = &op.getRegion().front();
        auto yield = cast<YieldOp>(body->getTerminator());
        SmallVector<Value> outQubits(yield.getResults());
        for (Value &qubit : outQubits) {
            // A yielded block argument is replaced by its qubit when the block is inlined.
            if (auto arg = dyn_cast<BlockArgument>(qubit); arg && arg.getOwner() == body) {
                qubit = op.getInQubits()[arg.getArgNumber()];
            }
        }
        rewriter.eraseOp(yield);
        rewriter.inlineBlockBefore(body, op, op.getInQubits());
        GadgetEndOp::create(rewriter, op.getLoc());
        rewriter.replaceOp(op, outQubits);
    }
}

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
    // Pauli-based Computational Operations
    patterns.add<PPRotationBasedPattern<PPRotationOp>>(typeConverter, patterns.getContext());
    patterns.add<PPRotationBasedPattern<PPRotationArbitraryOp>>(typeConverter,
                                                                patterns.getContext());
    patterns.add<PPMeasurementOpPattern<PPMeasurementOp>>(typeConverter, patterns.getContext());
    patterns.add<PPMeasurementOpPattern<SelectPPMeasurementOp>>(typeConverter,
                                                                patterns.getContext());
    patterns.add<GadgetBeginOpPattern>(typeConverter, patterns.getContext());
    patterns.add<GadgetEndOpPattern>(typeConverter, patterns.getContext());
}

} // namespace pbc
} // namespace catalyst
