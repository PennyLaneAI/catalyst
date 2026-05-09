// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_QUANTUMCONVERSIONPASS
#define GEN_PASS_DEF_QUANTUMCONVERSIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct QIRTypeConverter : public LLVMTypeConverter {

    QIRTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx)
    {
        addConversion([&](QubitType type) { return convertQubitType(type); });
        addConversion([&](QuregType type) { return convertQuregType(type); });
        addConversion([&](ObservableType type) { return convertObservableType(type); });
        addConversion([&](ResultType type) { return convertResultType(type); });
        // DO-QAOA type conversions
        addConversion([&](PartitionType type) { return convertPartitionType(type); });
        addConversion([&](ClusterMapType type) { return convertClusterMapType(type); });
        addConversion([&](CircuitRefType type) { return convertCircuitRefType(type); });
        addConversion([&](ParamsType type) { return convertParamsType(type); });
        addConversion([&](BitstringType type) { return convertBitstringType(type); });
    }

  private:
    Type convertQubitType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(
            &getContext()); // LLVM::LLVMStructType::getOpaque("Qubit", &getContext());
    }

    Type convertQuregType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(
            &getContext()); // LLVM::LLVMStructType::getOpaque("Array", &getContext());
    }

    Type convertObservableType(Type mlirType)
    {
        return this->convertType(IntegerType::get(&getContext(), 64));
    }

    Type convertResultType(Type mlirType)
    {
        return LLVM::LLVMPointerType::get(
            &getContext()); // LLVM::LLVMStructType::getOpaque("Result", &getContext());
    }

    // DO-QAOA: !quantum.partition<N, m> → { i32, i32 }
    // Fields: numQubits (i32), m (i32)
    Type convertPartitionType(PartitionType type)
    {
        auto i32 = IntegerType::get(&getContext(), 32);
        return LLVM::LLVMStructType::getLiteral(&getContext(), {i32, i32});
    }

    // DO-QAOA: !quantum.cluster_map<K> → { i32 }
    // Field: k (i32) — number of landscape clusters
    Type convertClusterMapType(ClusterMapType type)
    {
        auto i32 = IntegerType::get(&getContext(), 32);
        return LLVM::LLVMStructType::getLiteral(&getContext(), {i32});
    }

    // DO-QAOA: !quantum.circuit_ref → i64
    // Opaque index into the representative sub-circuit table
    Type convertCircuitRefType(CircuitRefType type)
    {
        return this->convertType(IntegerType::get(&getContext(), 64));
    }

    // DO-QAOA: !quantum.params → ptr
    // Pointer to a heap-allocated f64 parameter buffer (gamma/beta values)
    Type convertParamsType(ParamsType type) { return LLVM::LLVMPointerType::get(&getContext()); }

    // DO-QAOA: !quantum.bitstring → ptr
    // Pointer to a heap-allocated i8 buffer holding the binary node assignment
    Type convertBitstringType(BitstringType type)
    {
        return LLVM::LLVMPointerType::get(&getContext());
    }
};

struct QuantumConversionPass : impl::QuantumConversionPassBase<QuantumConversionPass> {
    using QuantumConversionPassBase::QuantumConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        QIRTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);
        populateQIRConversionPatterns(typeConverter, patterns, useArrayBackedRegisters);

        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();

        if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
