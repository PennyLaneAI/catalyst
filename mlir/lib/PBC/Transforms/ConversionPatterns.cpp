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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/Utils/EnsureFunctionDeclaration.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;

namespace {

using namespace catalyst::pbc;
using namespace catalyst::quantum;

//////////////////////////////////////////
// Pauli-based Computational Operations //
//////////////////////////////////////////

/// Helper function to convert a Pauli product ArrayAttr to a string
// (e.g., ["X", "I", "Z"] -> "XIZ")
std::string pauliProductToString(ArrayAttr pauliProduct)
{
    std::string result;
    for (auto attr : pauliProduct) {
        result += cast<StringAttr>(attr).getValue().str();
    }
    return result;
}

Value getPauliProductPtr(Location loc, ConversionPatternRewriter &rewriter, ModuleOp mod,
                         ArrayAttr pauliProduct)
{
    std::string pauliWord = pauliProductToString(pauliProduct);
    std::string pauliWordKey = "pauli_word_" + pauliWord;
    return catalyst::quantum::getGlobalString(
        loc, rewriter, pauliWordKey, StringRef(pauliWord.c_str(), pauliWord.length() + 1), mod);
}

template <typename T> struct PPRotationBasedPattern : public OpConversionPattern<T> {
    using OpConversionPattern<T>::OpConversionPattern;

    LogicalResult matchAndRewrite(T op, typename T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = this->getContext();
        ModuleOp mod = op->template getParentOfType<ModuleOp>();

        // Create a global string for the Pauli word
        Value pauliWordPtr = getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct());

        // void __catalyst__qis__PauliRot(const char* pauliStr, double theta,
        //                                 const Modifiers*, int64_t numQubits, ...qubits)
        StringRef qirName = "__catalyst__qis__PauliRot";
        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Type qirSignature = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx),
            {ptrType, Float64Type::get(ctx), ptrType, IntegerType::get(ctx, 64)},
            /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, qirName, qirSignature);

        // Get the rotation angle based on the op type
        // Since the qml.PauliRot(phi) == PPR(phi/2), this rotation_kind is multiplied by 2.
        Value thetaValue;
        if constexpr (std::is_same_v<T, PPRotationOp>) {
            if (op.getCondition()) {
                return op.emitOpError("PPRotationOp with condition is not supported.");
            }
            // Compute the rotation angle: theta = π / rotation_kind
            // rotation_kind can be ±1, ±2, ±4, ±8
            int16_t rotationKind = static_cast<int16_t>(op.getRotationKind());
            double theta = 2 * (llvm::numbers::pi / static_cast<double>(rotationKind));
            thetaValue = LLVM::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(theta));
        }
        else if constexpr (std::is_same_v<T, PPRotationArbitraryOp>) {
            if (op.getCondition()) {
                return op.emitOpError("PPRotationArbitraryOp with condition is not supported.");
            }
            // multiply by 2 to get the rotation angle
            thetaValue = LLVM::FMulOp::create(
                rewriter, loc, adaptor.getArbitraryAngle(),
                LLVM::ConstantOp::create(rewriter, loc, rewriter.getF64FloatAttr(2.0)));
        }
        else if constexpr (std::is_same_v<T, PauliRotOp>) {
            // Use the arbitrary angle directly
            thetaValue = adaptor.getAngle();
        }

        // Build the arguments: pauliStr, theta, modifiers, numQubits, qubits...
        int64_t numQubits = adaptor.getInQubits().size();
        SmallVector<Value> args;
        args.push_back(pauliWordPtr);
        args.push_back(thetaValue);
        args.push_back(LLVM::ZeroOp::create(rewriter, loc, ptrType));
        args.push_back(
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(numQubits)));
        args.append(adaptor.getInQubits().begin(), adaptor.getInQubits().end());

        LLVM::CallOp::create(rewriter, loc, fnDecl, args);

        // Replace the op with the input qubits
        rewriter.replaceOp(op, adaptor.getInQubits());

        return success();
    }
};

struct PPMeasurementOpPattern : public OpConversionPattern<PPMeasurementOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PPMeasurementOp op, PPMeasurementOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        MLIRContext *ctx = getContext();
        const TypeConverter *conv = getTypeConverter();
        ModuleOp mod = op->getParentOfType<ModuleOp>();

        // Create a global string for the Pauli word
        Value pauliWordPtr = getPauliProductPtr(loc, rewriter, mod, op.getPauliProduct());

        // RESULT* __catalyst__qis__PauliMeasure(const char* pauliWord, int64_t numQubits,
        // ...qubits)
        StringRef qirName = "__catalyst__qis__PauliMeasure";
        Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Type qirSignature = LLVM::LLVMFunctionType::get(conv->convertType(ResultType::get(ctx)),
                                                        {ptrType, IntegerType::get(ctx, 64)},
                                                        /*isVarArg=*/true);

        LLVM::LLVMFuncOp fnDecl = catalyst::ensureFunctionDeclaration<LLVM::LLVMFuncOp>(
            rewriter, op, qirName, qirSignature);

        // Build the arguments
        int64_t numQubits = adaptor.getInQubits().size();
        SmallVector<Value> args;
        args.push_back(pauliWordPtr);
        args.push_back(
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(numQubits)));
        args.append(adaptor.getInQubits().begin(), adaptor.getInQubits().end());

        // Call the function and get the result pointer
        Value resultPtr = LLVM::CallOp::create(rewriter, loc, fnDecl, args).getResult();

        // Load the measurement result (i1) from the result pointer
        Value mres = LLVM::LoadOp::create(rewriter, loc, IntegerType::get(ctx, 1), resultPtr);

        // if the uint16_t rotation_sign is -1, we need to negate the measurement result
        if (static_cast<int16_t>(op.getRotationSign()) == -1) {
            Value one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                                 rewriter.getBoolAttr(true));
            mres = LLVM::XOrOp::create(rewriter, loc, mres, one);
        }

        // Replace the op with the measurement result and the input qubits
        SmallVector<Value> values;
        values.push_back(mres);
        values.append(adaptor.getInQubits().begin(), adaptor.getInQubits().end());
        rewriter.replaceOp(op, values);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace pbc {

void populateConversionPatterns(LLVMTypeConverter &typeConverter, RewritePatternSet &patterns)
{
    // Pauli-based Computational Operations
    patterns.add<PPRotationBasedPattern<PauliRotOp>>(typeConverter, patterns.getContext());

    patterns.add<PPRotationBasedPattern<PPRotationOp>>(typeConverter, patterns.getContext());
    patterns.add<PPRotationBasedPattern<PPRotationArbitraryOp>>(typeConverter,
                                                                patterns.getContext());
    patterns.add<PPMeasurementOpPattern>(typeConverter, patterns.getContext());
}

} // namespace pbc
} // namespace catalyst
