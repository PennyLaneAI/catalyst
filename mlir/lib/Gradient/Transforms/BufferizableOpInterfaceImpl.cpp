#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

constexpr int64_t UNKNOWN = ShapedType::kDynamic;

LLVM::LLVMFuncOp ensureFunctionDeclaration(RewriterBase &rewriter, Operation *op,
                                           StringRef fnSymbol, Type fnType)
{
    Operation *fnDecl = SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnSymbol));

    if (!fnDecl) {
        RewriterBase::InsertionGuard insertGuard(rewriter);
        ModuleOp mod = op->getParentOfType<ModuleOp>();
        rewriter.setInsertionPointToStart(mod.getBody());

        fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
    }
    else {
        assert(isa<LLVM::LLVMFuncOp>(fnDecl) && "QIR function declaration is not a LLVMFuncOp");
    }

    return cast<LLVM::LLVMFuncOp>(fnDecl);
}

struct AdjointOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<AdjointOpInterface, AdjointOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return false;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        return false;
    }

    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand,
                      const bufferization::AnalysisState &state) const
    {
        return {};
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        llvm::outs() << "Happens?\n";
        SmallVector<Type> resTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resTypes)))
            return failure();

        Location loc = op.getLoc();
        Value gradSize = op.getGradSize();
        SmallVector<Value> memrefValues;
        for (Type resType : resTypes) {
            MemRefType memrefType = cast<MemRefType>(resType);
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
            memrefValues.push_back(memrefValue);
        }

        rewriter.create<AdjointOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getGradSize(),
                                   adaptor.getArgs(), memrefValues);
       bufferization::replaceOpWithBufferizedValues(rewriter, op, memrefValues);
        return success();
    }
};

} // namespace

void catalyst::gradient::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, GradientDialect *dialect) {
        AdjointOp::attachInterface<AdjointOpInterface>(*ctx);
    });
}