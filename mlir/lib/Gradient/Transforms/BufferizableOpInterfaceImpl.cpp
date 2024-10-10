#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {
/**
 * The new bufferization interface requires `bufferizesToMemoryRead`, `bufferizesToMemoryWrite`,
 * and `getAliasingValues`.
 *
 * `bufferizesToMemoryRead`: Return `true` if the buffer of the given tensor OpOperand is read.
 *
 * `bufferizesToMemoryWrite`: Return `true` if the buffer of the given tensor OpOperand is written
 * (if bufferizing in-place).
 *
 * `getAliasingOpOperands`: Return the OpResults that may share the same buffer as the given
 * OpOperand. Note that MLIR documentation does not mention `getAliasingValues` but it seems to
 * serve the same purpose.
 *
 * Bufferizing FunctionOpInterface is also not documented by MLIR. It requires
 * `OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel`, which requires the
 * implementation of `supportsUnstructuredControlFlow`, `hasTensorSemantics`, and
 * `getAliasingOpOperands`.
 *
 * Link: https://mlir.llvm.org/docs/Bufferization/#extending-one-shot-bufferize
 */

static BaseMemRefType
getBufferizedFunctionArgType(FunctionOpInterface funcOp, int64_t index,
                             const bufferization::BufferizationOptions &options)
{
    auto tensorType = dyn_cast<TensorType>(funcOp.getArgument(index).getType());
    assert(tensorType && "expected TensorType");

    BaseMemRefType memrefType = options.functionArgTypeConverterFn(
        tensorType, *options.defaultMemorySpaceFn(tensorType), funcOp, options);

    auto layoutAttr = funcOp.getArgAttrOfType<AffineMapAttr>(
        index, bufferization::BufferizationDialect::kBufferLayoutAttrName);
    if (!layoutAttr)
        return memrefType;

    auto rankedMemrefType = dyn_cast<MemRefType>(memrefType);
    assert(rankedMemrefType && "buffer layout not supported on unranked tensors");
    return MemRefType::get(rankedMemrefType.getShape(), rankedMemrefType.getElementType(),
                           layoutAttr.getValue(), rankedMemrefType.getMemorySpace());
}

static ReturnOp getAssumedUniqueReturnOp(FunctionOpInterface funcOp)
{
    ReturnOp returnOp;
    for (Block &b : funcOp.getFunctionBody()) {
        if (auto candidateOp = dyn_cast<ReturnOp>(b.getTerminator())) {
            if (returnOp)
                return nullptr;
            returnOp = candidateOp;
        }
    }
    return returnOp;
}

Value generateAllocation(OpBuilder &builder, Location loc, Value reference)
{
    auto origMemrefType = cast<MemRefType>(reference.getType());
    // Rebuild MemRefType without memory layout.
    auto memrefType = MemRefType::get(origMemrefType.getShape(), origMemrefType.getElementType());
    // Get dynamic dimension sizes from the provided reference value if necessary.
    SmallVector<Value> dynamicDims;
    if (!memrefType.hasStaticShape()) {
        for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
            if (memrefType.isDynamicDim(dim)) {
                Value dimIndex = builder.create<index::ConstantOp>(loc, dim);
                dynamicDims.push_back(builder.create<memref::DimOp>(loc, reference, dimIndex));
            }
        }
    }

    return builder.create<memref::AllocOp>(loc, memrefType, dynamicDims);
}

/// Helper function to generate a set of memref allocations.
///
/// The allocation size and shape is deduced from a list of existing memref values.
///
void generateAllocations(RewriterBase &rewriter, Location loc, SmallVectorImpl<Value> &allocations,
                         ValueRange referenceValues)
{
    for (Value memref : referenceValues) {
        allocations.push_back(
            generateAllocation(rewriter, loc, cast<TypedValue<MemRefType>>(memref)));
    }
}

struct AdjointOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<AdjointOpInterface, AdjointOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
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
        auto adjointOp = cast<AdjointOp>(op);

        bufferization::BufferizeTypeConverter typeConverter;

        SmallVector<Type> resTypes;
        if (failed(typeConverter.convertTypes(adjointOp.getResultTypes(), resTypes)))
            return failure();

        Location loc = adjointOp.getLoc();
        Value gradSize = adjointOp.getGradSize();
        SmallVector<Value> memrefValues;
        for (Type resType : resTypes) {
            MemRefType memrefType = cast<MemRefType>(resType);
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
            memrefValues.push_back(memrefValue);
        }

        SmallVector<Value> bufferArgs;
        ValueRange operands = adjointOp.getArgs();
        for (Value operand : operands) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer))
                return failure();
            bufferArgs.push_back(*opBuffer);
        }

        rewriter.create<AdjointOp>(loc, TypeRange{}, adjointOp.getCalleeAttr(),
                                   adjointOp.getGradSize(), bufferArgs, memrefValues);
        bufferization::replaceOpWithBufferizedValues(rewriter, op, memrefValues);
        return success();
    }
};

struct BackpropOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<BackpropOpInterface,
                                                                   BackpropOp> {
    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const bufferization::AnalysisState &state) const
    {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const bufferization::AnalysisState &state) const
    {
        return true;
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
        auto backpropOp = cast<BackpropOp>(op);

        Location loc = backpropOp.getLoc();
        SmallVector<Value> gradients;
        SmallVector<Value> argShadows;
        // Conceptually a map from scalar result indices (w.r.t. other scalars) to the position in
        // the overall list of returned gradients.
        // For instance, a backprop op that returns (tensor, f64, tensor, f64, f64) will have
        // scalarIndices = {1, 3, 4}.
        SmallVector<unsigned> scalarIndices;
        SmallVector<Type> scalarReturnTypes;

        SmallVector<Value> bufferArgs;
        ValueRange operands = backpropOp.getArgs();
        for (Value operand : operands) {
            if (isa<TensorType>(operand.getType())) {
                FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
                if (failed(opBuffer))
                    return failure();
                bufferArgs.push_back(*opBuffer);
            }
            else {
                bufferArgs.push_back(operand);
            }
        }

        std::vector<Value> diffArgs =
            computeDiffArgs(bufferArgs, backpropOp.getDiffArgIndicesAttr());

        for (const auto &[idx, diffArg] : llvm::enumerate(diffArgs)) {
            // Allocate buffers to place the differentiation results (gradients) into. Enzyme refers
            // to these as shadow arguments. There is one result for each differentiable MemRef
            // argument, with a matching shape and type.
            if (isa<MemRefType>(diffArg.getType())) {
                Value shadow = generateAllocation(rewriter, loc, diffArg);
                gradients.push_back(shadow);
                argShadows.push_back(shadow);
            }
            else if (isa<FloatType>(diffArg.getType())) {
                scalarReturnTypes.push_back(diffArg.getType());
                scalarIndices.push_back(idx);
                // Put a null placeholder value that will be filled in with the result of the
                // bufferized BackpropOp.
                gradients.push_back(Value());
            }
        }

        // Enzyme requires buffers for the primal outputs as well, even though we don't need their
        // values. We'll mark them dupNoNeed later on to allow Enzyme to optimize away their
        // computation.
        SmallVector<Value> calleeResults, resShadows;
        ValueRange cotangents = backpropOp.getCotangents();
        SmallVector<Value> bufferCotangentsList;
        for (Value operand : cotangents) {
            FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
            if (failed(opBuffer))
                return failure();
            bufferCotangentsList.push_back(*opBuffer);
        }
        mlir::ValueRange bufferCotangents(bufferCotangentsList);

        generateAllocations(rewriter, loc, calleeResults, bufferCotangents);
        // Enzyme mutates the result shadows but the cotangent tensors must be immutable, so we
        // create copies to pass into Enzyme. Concretely, this issue pops up with multiple
        // BackpropOps that have the same cotangent tensor due to a CSE effect from one-shot
        // bufferization.
        generateAllocations(rewriter, loc, resShadows, bufferCotangents);
        for (const auto &[cotangent, resShadow] : llvm::zip(bufferCotangents, resShadows)) {
            rewriter.create<memref::CopyOp>(loc, cotangent, resShadow);
        }

        DenseIntElementsAttr diffArgIndicesAttr = backpropOp.getDiffArgIndices().value_or(nullptr);
        auto bufferizedBackpropOp = rewriter.create<BackpropOp>(
            loc, TypeRange{}, scalarReturnTypes, backpropOp.getCalleeAttr(), bufferArgs, argShadows,
            calleeResults, resShadows, diffArgIndicesAttr, backpropOp.getKeepValueResultsAttr());
        // Fill in the null placeholders.
        for (const auto &[idx, scalarResult] :
             llvm::enumerate(bufferizedBackpropOp.getGradients())) {
            gradients[scalarIndices[idx]] = scalarResult;
        }

        // BackpropOp can return two results for value_and_grad: values and gradients
        // or only one for grad: gradients
        SmallVector<Value> results;
        {
            // If we are lowering a value_and_grad operation, then take values from the
            // calleeResults
            if (!backpropOp.getVals().empty()) {
                results.insert(results.end(), calleeResults.begin(), calleeResults.end());
            }
            results.insert(results.end(), gradients.begin(), gradients.end());
        }

        bufferization::replaceOpWithBufferizedValues(rewriter, op, results);
        return success();
    }
};

struct ForwardOpInterface
    : public bufferization::OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel<
          ForwardOpInterface, ForwardOp> {
    static bool supportsUnstructuredControlFlow() { return true; }

    bool hasTensorSemantics(Operation *op) const
    {
        auto isaTensor = llvm::IsaPred<TensorType>;

        // A function has tensor semantics if it has tensor arguments/results.
        auto forwardOp = cast<ForwardOp>(op);
        bool hasTensorArg = any_of(forwardOp.getArgumentTypes(), isaTensor);
        bool hasTensorResult = any_of(forwardOp.getResultTypes(), isaTensor);
        bool hasTensorFuncInType = any_of(forwardOp.getFunctionType().getInputs(), isaTensor);
        bool hasTensorFuncOutType = any_of(forwardOp.getFunctionType().getResults(), isaTensor);
        if (hasTensorArg || hasTensorResult || hasTensorFuncInType || hasTensorFuncOutType)
            return true;

        return false;
    }

    bufferization::AliasingOpOperandList
    getAliasingOpOperands(Operation *op, Value value,
                          const bufferization::AnalysisState &state) const
    {
        return getAliasingBranchOpOperands(op, cast<BlockArgument>(value), state);
    }

    FailureOr<BaseMemRefType> getBufferType(Operation *op, Value value,
                                            const bufferization::BufferizationOptions &options,
                                            SmallVector<Value> &invocationStack) const
    {
        auto forwardOp = cast<ForwardOp>(op);
        auto bbArg = cast<BlockArgument>(value);

        // Function arguments are special.
        if (bbArg.getOwner() == &forwardOp.getBody().front())
            return getBufferizedFunctionArgType(forwardOp, bbArg.getArgNumber(), options);

        return OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel::getBufferType(
            op, value, options, invocationStack);
    }

    LogicalResult verifyAnalysis(Operation *op, const bufferization::AnalysisState &state) const
    {
        auto forwardOp = cast<ForwardOp>(op);
        // TODO: func.func with multiple returns are not supported.
        if (!getAssumedUniqueReturnOp(forwardOp))
            return op->emitOpError("op without unique func.return is not supported");
        return success();
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        auto forwardOp = cast<ForwardOp>(op);
        FunctionType funcType = forwardOp.getFunctionType();

        // Construct the bufferized function type.
        SmallVector<Type> argTypes;
        for (const auto &it : llvm::enumerate(funcType.getInputs())) {
            Type argType = it.value();
            if (dyn_cast<TensorType>(argType)) {
                argTypes.push_back(getBufferizedFunctionArgType(forwardOp, it.index(), options));
                continue;
            }
            argTypes.push_back(argType);
        }

        ReturnOp returnOp = getAssumedUniqueReturnOp(forwardOp);
        assert(returnOp && "expected func with single return op");
        Location loc = returnOp.getLoc();

        // 1. Bufferize every block.
        for (Block &block : forwardOp.getBody())
            if (failed(bufferization::bufferizeBlockSignature(&block, rewriter, options)))
                return failure();

        // 2. For each result, keep track of which inplace argument it reuses.
        SmallVector<Value> returnValues;
        for (OpOperand &returnOperand : returnOp->getOpOperands()) {
            Value returnVal = returnOperand.get();
            auto tensorType = dyn_cast<TensorType>(returnVal.getType());
            rewriter.setInsertionPoint(returnOp);

            // If not a tensor type just forward it.
            if (!tensorType) {
                returnValues.push_back(returnVal);
                continue;
            }

            // Note: If `inferFunctionResultLayout = true`, cast are later folded
            // away.
            BaseMemRefType resultType = options.functionArgTypeConverterFn(
                tensorType, *options.defaultMemorySpaceFn(tensorType), forwardOp, options);
            Value toMemrefOp =
                rewriter.create<bufferization::ToMemrefOp>(loc, resultType, returnVal);
            returnValues.push_back(toMemrefOp);
        }

        // 3. Rewrite the terminator.
        forwardOp.walk([&](ReturnOp returnOp) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, returnValues, returnOp.getEmpty());
        });

        // 4. Rewrite the FuncOp type to buffer form. Also preserve unused return types.
        SmallVector<Type> returnTypes;
        for (auto retTy : forwardOp.getResultTypes()) {
            auto tensorType = dyn_cast<TensorType>(retTy);
            BaseMemRefType resultType = options.functionArgTypeConverterFn(
                tensorType, *options.defaultMemorySpaceFn(tensorType), forwardOp, options);
            returnTypes.push_back(resultType);
        }
        forwardOp.setType(FunctionType::get(op->getContext(), argTypes, returnTypes));

        return success();
    }
};

struct ReverseOpInterface
    : public bufferization::OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel<
          ReverseOpInterface, ReverseOp> {
    static bool supportsUnstructuredControlFlow() { return true; }

    bool hasTensorSemantics(Operation *op) const
    {
        auto isaTensor = llvm::IsaPred<TensorType>;

        // A function has tensor semantics if it has tensor arguments/results.
        auto reverseOp = cast<ReverseOp>(op);
        bool hasTensorArg = any_of(reverseOp.getArgumentTypes(), isaTensor);
        bool hasTensorResult = any_of(reverseOp.getResultTypes(), isaTensor);
        bool hasTensorFuncInType = any_of(reverseOp.getFunctionType().getInputs(), isaTensor);
        bool hasTensorFuncOutType = any_of(reverseOp.getFunctionType().getResults(), isaTensor);
        if (hasTensorArg || hasTensorResult || hasTensorFuncInType || hasTensorFuncOutType)
            return true;

        return false;
    }

    bufferization::AliasingOpOperandList
    getAliasingOpOperands(Operation *op, Value value,
                          const bufferization::AnalysisState &state) const
    {
        return getAliasingBranchOpOperands(op, cast<BlockArgument>(value), state);
    }

    FailureOr<BaseMemRefType> getBufferType(Operation *op, Value value,
                                            const bufferization::BufferizationOptions &options,
                                            SmallVector<Value> &invocationStack) const
    {
        auto reverseOp = cast<ReverseOp>(op);
        auto bbArg = cast<BlockArgument>(value);

        // Function arguments are special.
        if (bbArg.getOwner() == &reverseOp.getBody().front())
            return getBufferizedFunctionArgType(reverseOp, bbArg.getArgNumber(), options);

        return OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel::getBufferType(
            op, value, options, invocationStack);
    }

    LogicalResult verifyAnalysis(Operation *op, const bufferization::AnalysisState &state) const
    {
        auto reverseOp = cast<ReverseOp>(op);
        // TODO: func.func with multiple returns are not supported.
        if (!getAssumedUniqueReturnOp(reverseOp))
            return op->emitOpError("op without unique func.return is not supported");
        return success();
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const
    {
        auto reverseOp = cast<ReverseOp>(op);
        FunctionType funcType = reverseOp.getFunctionType();

        // Construct the bufferized function type.
        SmallVector<Type> argTypes;
        for (const auto &it : llvm::enumerate(funcType.getInputs())) {
            Type argType = it.value();
            if (dyn_cast<TensorType>(argType)) {
                argTypes.push_back(getBufferizedFunctionArgType(reverseOp, it.index(), options));
                continue;
            }
            argTypes.push_back(argType);
        }

        ReturnOp returnOp = getAssumedUniqueReturnOp(reverseOp);
        assert(returnOp && "expected func with single return op");
        Location loc = returnOp.getLoc();

        // 1. Bufferize every block.
        for (Block &block : reverseOp.getBody())
            if (failed(bufferization::bufferizeBlockSignature(&block, rewriter, options)))
                return failure();

        // 2. For each result, keep track of which inplace argument it reuses.
        SmallVector<Value> returnValues;
        for (OpOperand &returnOperand : returnOp->getOpOperands()) {
            Value returnVal = returnOperand.get();
            auto tensorType = dyn_cast<TensorType>(returnVal.getType());
            rewriter.setInsertionPoint(returnOp);

            // If not a tensor type just forward it.
            if (!tensorType) {
                returnValues.push_back(returnVal);
                continue;
            }

            // Note: If `inferFunctionResultLayout = true`, cast are later folded
            // away.
            BaseMemRefType resultType = options.functionArgTypeConverterFn(
                tensorType, *options.defaultMemorySpaceFn(tensorType), reverseOp, options);
            Value toMemrefOp =
                rewriter.create<bufferization::ToMemrefOp>(loc, resultType, returnVal);
            returnValues.push_back(toMemrefOp);
        }

        // 3. Rewrite the terminator.
        reverseOp.walk([&](ReturnOp returnOp) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, returnValues, returnOp.getEmpty());
        });

        // 4. Rewrite the FuncOp type to buffer form. Also preserve unused return types.
        SmallVector<Type> returnTypes;
        for (auto retTy : reverseOp.getResultTypes()) {
            auto tensorType = dyn_cast<TensorType>(retTy);
            BaseMemRefType resultType = options.functionArgTypeConverterFn(
                tensorType, *options.defaultMemorySpaceFn(tensorType), reverseOp, options);
            returnTypes.push_back(resultType);
        }
        reverseOp.setType(FunctionType::get(op->getContext(), argTypes, returnTypes));

        return success();
    }
};

} // namespace

void catalyst::gradient::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext *ctx, GradientDialect *dialect) {
        AdjointOp::attachInterface<AdjointOpInterface>(*ctx);
        BackpropOp::attachInterface<BackpropOpInterface>(*ctx);
        ForwardOp::attachInterface<ForwardOpInterface>(*ctx);
        ReverseOp::attachInterface<ReverseOpInterface>(*ctx);
    });
}