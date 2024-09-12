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
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Gradient/Utils/GradientShape.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

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
void generateAllocations(RewriterBase &rewriter, Location loc,
                         SmallVectorImpl<Value> &allocations, ValueRange referenceValues)
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


        rewriter.create<AdjointOp>(loc, TypeRange{}, adjointOp.getCalleeAttr(), adjointOp.getGradSize(),
                                   bufferArgs, memrefValues);
        bufferization::replaceOpWithBufferizedValues(rewriter, op, memrefValues);
        return success();
    }
};

struct BackpropOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<BackpropOpInterface, BackpropOp> {
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
            if(isa<TensorType>(operand.getType())) {
                FailureOr<Value> opBuffer = getBuffer(rewriter, operand, options);
                if (failed(opBuffer))
                    return failure();
                bufferArgs.push_back(*opBuffer);
            } else {
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
        if (hasTensorArg || hasTensorResult)
            return true;

        return false;
    }

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
        auto forwardOp = cast<ForwardOp>(op);

        auto argc = forwardOp.getArgc();
        auto resc = forwardOp.getResc();
        SmallVector<Value> inputs;
        SmallVector<Value> differentials;
        SmallVector<Value> outputs;
        SmallVector<Value> cotangents;

        // Update signature
        auto argTys = forwardOp.getArgumentTypes();
        auto retTys = forwardOp.getResultTypes();
        SmallVector<Type> emptyRets;
        SmallVector<Type> args(argTys.begin(), argTys.end());
        args.insert(args.end(), retTys.begin(), retTys.end());
        SmallVector<Type> bufferArgs;
        for (Type ty : args) {
            auto tensorType = dyn_cast<RankedTensorType>(ty);
            if (!tensorType)
                bufferArgs.push_back(ty);
            else
                bufferArgs.push_back(
                    MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
        }
        auto forwardTy = rewriter.getFunctionType(bufferArgs, emptyRets);
        
        Block *block;
        rewriter.modifyOpInPlace(op, [&] { 
            forwardOp.setFunctionType(forwardTy); 
            block = forwardOp.addEntryBlock();
        });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto params = forwardOp.getArguments();

        for (size_t i = 0; i < argc * 2; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? differentials.push_back(val) : inputs.push_back(val);
        }

        auto upperLimit = (argc * 2) + (resc * 2);
        for (size_t i = argc * 2; i < upperLimit; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? cotangents.push_back(val) : outputs.push_back(val);
        }

        auto implAttr = forwardOp.getImplementationAttr();
        auto impl = forwardOp.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = forwardOp.getLoc();

        SmallVector<Value> tensorInputs;
        for (auto input : inputs) {
            Value tensorIn = rewriter.create<bufferization::ToTensorOp>(loc, input);
            tensorInputs.push_back(tensorIn);
        }
        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, tensorInputs);
        SmallVector<Value> tensorOutputs(callOp.getResults());

        for (auto [memrefOutput, tensorOutput] : llvm::zip(outputs, tensorOutputs)) {
            Value castVal = rewriter.create<bufferization::ToMemrefOp>(loc, memrefOutput.getType(),
                                                                       tensorOutput);
            rewriter.create<memref::CopyOp>(loc, castVal, memrefOutput);
        }

        auto tapeCount = forwardOp.getTape();
        SmallVector<Value> tapeOutputs;
        tapeOutputs.insert(tapeOutputs.begin(), tensorOutputs.end() - tapeCount,
                           tensorOutputs.end());

        SmallVector<Value> tapeMemrefOutputs;
        for (auto [tapeTensorOutput, memrefTapeOutput] :
             llvm::zip(tapeOutputs, forwardOp.getResultTypes())) {
            Value castVal =
                rewriter.create<bufferization::ToMemrefOp>(loc, memrefTapeOutput, tapeTensorOutput);
            tapeMemrefOutputs.push_back(castVal);
        }

        auto F = rewriter.getIntegerAttr(rewriter.getI1Type(), 0);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, tapeMemrefOutputs, F);

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
        if (hasTensorArg || hasTensorResult)
            return true;

        return false;
    }

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
        auto reverseOp = cast<ReverseOp>(op);

        auto argc = reverseOp.getArgc();
        auto resc = reverseOp.getResc();
        SmallVector<Value> inputs;
        SmallVector<Value> differentials;
        SmallVector<Value> outputs;
        SmallVector<Value> cotangents;
        SmallVector<Value> tapeElements;

        // Update signature
        auto argTys = reverseOp.getArgumentTypes();
        auto retTys = reverseOp.getResultTypes();
        SmallVector<Type> emptyRets;
        SmallVector<Type> args(argTys.begin(), argTys.end());
        args.insert(args.end(), retTys.begin(), retTys.end());
        SmallVector<Type> bufferArgs;
        for (Type ty : args) {
            auto tensorType = dyn_cast<RankedTensorType>(ty);
            if (!tensorType)
                bufferArgs.push_back(ty);
            else
                bufferArgs.push_back(
                    MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
        }
        auto reverseTy = rewriter.getFunctionType(bufferArgs, emptyRets);
        
        Block *block;
        rewriter.modifyOpInPlace(op, [&] { 
            reverseOp.setFunctionType(reverseTy); 
            block = reverseOp.addEntryBlock();
        });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto params = reverseOp.getArguments();

        for (size_t i = 0; i < argc * 2; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? differentials.push_back(val) : inputs.push_back(val);
        }

        auto upperLimit = (argc * 2) + (resc * 2);
        for (size_t i = argc * 2; i < upperLimit; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? cotangents.push_back(val) : outputs.push_back(val);
        }

        auto tapeCount = reverseOp.getTape();
        auto uppestLimit = upperLimit + tapeCount;
        for (size_t i = upperLimit; i < uppestLimit; i++) {
            tapeElements.push_back(params[i]);
        }

        auto implAttr = reverseOp.getImplementationAttr();
        auto impl = reverseOp.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(reverseOp, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = reverseOp.getLoc();

        SmallVector<Value> tensorInputs;
        for (auto tapeElement : tapeElements) {
            Value tensorIn = rewriter.create<bufferization::ToTensorOp>(loc, tapeElement);
            tensorInputs.push_back(tensorIn);
        }

        for (auto cotangent : cotangents) {
            Value tensorIn = rewriter.create<bufferization::ToTensorOp>(loc, cotangent);
            tensorInputs.push_back(tensorIn);
        }

        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, tensorInputs);
        SmallVector<Value> tensorOutputs(callOp.getResults());

        for (auto [differential, tensorOutput] : llvm::zip(differentials, tensorOutputs)) {
            Value castVal = rewriter.create<bufferization::ToMemrefOp>(loc, differential.getType(),
                                                                       tensorOutput);
            rewriter.create<memref::CopyOp>(loc, castVal, differential);
        }

        auto T = rewriter.getIntegerAttr(rewriter.getI1Type(), 1);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, ValueRange{}, T);

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