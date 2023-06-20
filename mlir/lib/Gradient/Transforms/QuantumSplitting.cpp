/*
What does this interface concretely look like?

Should we match the structured cases explicitly? Or try to implement an interface extension?
Could have two interface methods, one to store metadata and another to load it.

func f(x, n) {
    RX(x, 0)

    for i = 0 to n
        Hadamard(0)
        RY(x, 0)

    if n is odd
        RZ(0.2, 0)
    else
        PauliX(0)

    return expval(PauliZ(0))
}

func augment_f(x, n) {
    bounds.push(0, n, 1)
    for i = 0 to n:
        params.push(x)
        wires.push(0); wires.push(0)

    preds.push(n is odd)
    if n is odd:
        params.push(0.2)
        wires.push(0)
    else:
        wires.push(0)

    wires.push(0)
}

func quantum_f(x, bounds, pred, params, wires) {
    RX(x, 0)
    for i = bounds[0] to bounds[1]:
        Hadamard(wires[widx++])
        RY(params[pidx++], wires[widx++])

    if pred:
        RZ(params[pidx++], wires[widx++])
    else:
        PauliX(wires[widx++])
    return expval(PauliZ(wires[widx++]))
}

---

Needs to be recursive:

func g(x, n) {
    for i = 0 to n:
        if i is odd:
            RZ(0.2, 0)
        else:
            PauliX(0)

}

func quantum_g(x, bounds, preds: list<bool>, params, wires):
    for i = bounds[0] to bounds[1]:
        if preds[i]:
            RZ(...)
        else:
            PauliX(...)

---

func h(x: f64, y: f64) {
    while x < 10.0:
        # need to store gate params, wires can also be dynamic.
        # perhaps useful to detect constant wires.
        RZ(x, int(x) % 2)
        x += y


func augment_h(x, y):
    counter = 0
    while x < 10.0:
        params.push(x)
        wires.push(int(x) % 2)
        x += y
        counter++

func quantum_h(params, n_iters):
    # One idea: convert while loops to for loops
    for i = 0 to n_iters:
        RZ(params[i], 0)
}

This should faithfully execute the same circuit given the same parameters.
By definition, the same control flow will execute in the original, augmented, and quantum-only
programs. Potentially linear memory overhead w.r.t. total number of loop iterations.

Plan:
    - Support for loops with nesting
    - Support if statements, potentially nested
    - See if detecting constant gate parameters/wires is useful. May leverage
SparseConstantPropagation here.
    - Hybrid circuits: don't copy preprocessing and postprocessing.
*/

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/Analysis/QuantumDependenceAnalysis.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/RemoveQuantumMeasurements.h"

using namespace mlir;
using namespace catalyst;

bool catalyst::gradient::shouldCache(Value value)
{
    // As a default, cache every non-null value.
    return value != nullptr;
}

func::FuncOp genAugmentedClassicalFunction(PatternRewriter &rewriter, Location loc,
                                           func::FuncOp callee)
{
    // Define the properties of the classical preprocessing function.
    std::string fnName = callee.getSymName().str() + ".augmented";
    std::vector<Type> fnArgTypes = callee.getArgumentTypes().vec();
    RankedTensorType paramsVectorType =
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type());
    RankedTensorType controlFlowTensorType =
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getIndexType());
    FunctionType fnType =
        rewriter.getFunctionType(fnArgTypes, {paramsVectorType, controlFlowTensorType});

    func::FuncOp augmentedFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, rewriter.getStringAttr(fnName));
    if (!augmentedFn) {
        // First copy the original function as is, then we can replace all quantum ops by
        // collecting their gate parameters in a memory buffer instead. The size of this vector
        // is passed as an input to the new function.
        augmentedFn = rewriter.create<func::FuncOp>(loc, fnName, fnType);
        augmentedFn.setPrivate();
        rewriter.cloneRegionBefore(callee.getBody(), augmentedFn.getBody(), augmentedFn.end());

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&augmentedFn.getBody().front());

        // Allocate the memory for the gate parameters collected at runtime.
        auto arrayListType = ArrayListType::get(rewriter.getContext(), rewriter.getF64Type());
        Value paramsBuffer = rewriter.create<ListInitOp>(loc, arrayListType);
        Value controlFlowTape = rewriter.create<ListInitOp>(
            loc, ArrayListType::get(rewriter.getContext(), rewriter.getIndexType()));
        MemRefType paramsProcessedType = MemRefType::get({}, rewriter.getIndexType());
        Value paramsProcessed = rewriter.create<memref::AllocaOp>(loc, paramsProcessedType);
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        rewriter.create<memref::StoreOp>(loc, cZero, paramsProcessed);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);

        augmentedFn.walk([&](Operation *op) {
            // Insert gate parameters into the params buffer.
            if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gate);

                ValueRange diffParams = gate.getDiffParams();
                if (!diffParams.empty()) {
                    Value paramIdx = rewriter.create<memref::LoadOp>(loc, paramsProcessed);
                    for (auto param : diffParams) {
                        rewriter.create<ListPushOp>(loc, param, paramsBuffer);
                        paramIdx = rewriter.create<index::AddOp>(loc, paramIdx, cOne);
                    }
                    rewriter.create<memref::StoreOp>(loc, paramIdx, paramsProcessed);
                }

                rewriter.replaceOp(op, gate.getQubitOperands());
            }
            // Replace any return statements from the original function with the params vector.
            else if (isa<func::ReturnOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(op);
                Value data = rewriter.create<ListLoadDataOp>(loc, paramsBuffer);
                Value paramsTensor = rewriter.create<bufferization::ToTensorOp>(loc, data);
                Value controlFlowData = rewriter.create<ListLoadDataOp>(loc, controlFlowTape);
                Value controlFlowTensor =
                    rewriter.create<bufferization::ToTensorOp>(loc, controlFlowData);
                op->setOperands({paramsTensor, controlFlowTensor});
            }
            // Erase redundant device specifications.
            else if (isa<quantum::DeviceOp>(op)) {
                rewriter.eraseOp(op);
            }
            // Cache the relevant parameters
            else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(forOp);

                rewriter.create<catalyst::ListPushOp>(loc, forOp.getLowerBound(), controlFlowTape);
                rewriter.create<catalyst::ListPushOp>(loc, forOp.getUpperBound(), controlFlowTape);
                rewriter.create<catalyst::ListPushOp>(loc, forOp.getStep(), controlFlowTape);
            }
            else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(ifOp);

                auto casted = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                                  ifOp.getCondition());
                rewriter.create<catalyst::ListPushOp>(loc, casted, controlFlowTape);
            }
            else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                PatternRewriter::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(whileOp);

                // While ops are converted to for ops
                Value loopCounter = rewriter.create<memref::AllocaOp>(
                    loc, MemRefType::get({}, rewriter.getIndexType()));
                rewriter.create<memref::StoreOp>(loc, cZero, loopCounter);

                // Count the number of times the "after" region iterates.
                // Assume that the loop condition, being a classical value, does not depend on
                // any quantum operations.
                rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
                auto beforeIdx = rewriter.create<memref::LoadOp>(loc, loopCounter);
                auto afterIdx = rewriter.create<index::AddOp>(loc, beforeIdx, cOne);
                rewriter.create<memref::StoreOp>(loc, afterIdx, loopCounter);

                // Store the final iteration count to the tape.
                rewriter.setInsertionPointAfter(whileOp);
                auto finalCount = rewriter.create<memref::LoadOp>(loc, loopCounter);
                rewriter.create<catalyst::ListPushOp>(loc, finalCount, controlFlowTape);
            }
        });

        quantum::removeQuantumMeasurements(augmentedFn);
    }

    return augmentedFn;
}

func::FuncOp genQuantumSplitFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee)
{
    auto fnName = rewriter.getStringAttr(callee.getName() + ".qsplit");
    SmallVector<Type> tapeTypes{
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getF64Type()),
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getIndexType()),
        RankedTensorType::get({ShapedType::kDynamic}, rewriter.getI64Type())};
    SmallVector<Type> argTypes{callee.getArgumentTypes()};
    argTypes.insert(argTypes.end(), tapeTypes.begin(), tapeTypes.end());
    SmallVector<Location> argLocs{tapeTypes.size(), loc};
    FunctionType fnType = rewriter.getFunctionType(argTypes, callee.getResultTypes());
    func::FuncOp qsplitFn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, fnName);
    if (!qsplitFn) {
        qsplitFn = rewriter.create<func::FuncOp>(loc, fnName, fnType);
        qsplitFn.setPrivate();

        rewriter.cloneRegionBefore(callee.getBody(), qsplitFn.getBody(), qsplitFn.end());
        Region &newBody = qsplitFn.getFunctionBody();
        SmallVector<Value, 3> tapes{newBody.addArguments(tapeTypes, argLocs)};
        Value paramTape = tapes[0];
        Value cfTape = tapes[1];
        Value wireTape = tapes[2];

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(&newBody.front());

        Value paramCounter =
            rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, rewriter.getIndexType()));
        Value cfCounter =
            rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, rewriter.getIndexType()));
        Value wireCounter =
            rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, rewriter.getIndexType()));
        Value cZero = rewriter.create<index::ConstantOp>(loc, 0);
        Value cOne = rewriter.create<index::ConstantOp>(loc, 1);
        rewriter.create<memref::StoreOp>(loc, cZero, paramCounter);
        rewriter.create<memref::StoreOp>(loc, cZero, cfCounter);
        rewriter.create<memref::StoreOp>(loc, cZero, wireCounter);

        auto loadThenIncrementCounter = [&](OpBuilder &builder, Value counter,
                                            Value tape) -> Value {
            Value index = builder.create<memref::LoadOp>(loc, counter);
            Value nextIndex = builder.create<index::AddOp>(loc, index, cOne);
            builder.create<memref::StoreOp>(loc, nextIndex, counter);
            return builder.create<tensor::ExtractOp>(loc, tape, index);
        };

        qsplitFn.walk([&](Operation *op) {
            // Cached parameters
            if (auto gateOp = dyn_cast<quantum::DifferentiableGate>(op)) {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(gateOp);

                ValueRange diffParams = gateOp.getDiffParams();
                SmallVector<Value> newParams{diffParams.size()};
                for (const auto [paramIdx, recomputedParam] : llvm::enumerate(diffParams)) {
                    if (gradient::shouldCache(recomputedParam)) {
                        newParams[paramIdx] =
                            loadThenIncrementCounter(rewriter, paramCounter, paramTape);
                    }
                    else {
                        newParams[paramIdx] = recomputedParam;
                    }
                }
                MutableOperandRange range{gateOp, static_cast<unsigned>(gateOp.getDiffOperandIdx()),
                                          static_cast<unsigned>(diffParams.size())};
                range.assign(newParams);
            }
            // Cached wires
            else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
                if (gradient::shouldCache(extractOp.getIdx())) {
                    extractOp.getIdxMutable().assign(
                        loadThenIncrementCounter(rewriter, wireCounter, wireTape));
                }
            }
            else if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
                if (gradient::shouldCache(insertOp.getIdx())) {
                    insertOp.getIdxMutable().assign(
                        loadThenIncrementCounter(rewriter, wireCounter, wireTape));
                }
            }
            // Cached control flow
            else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                if (gradient::shouldCache(ifOp.getCondition())) {
                    ifOp.getConditionMutable().assign(
                        loadThenIncrementCounter(rewriter, cfCounter, cfTape));
                }
            }
            else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                // TODO: If quantum ops and classical ops are jointly returned from a loop, the
                // canonicalization won't handle removing the classical parts.
                if (gradient::shouldCache(forOp.getLowerBound())) {
                    forOp.getLowerBoundMutable().assign(
                        loadThenIncrementCounter(rewriter, cfCounter, cfTape));
                }
                if (gradient::shouldCache(forOp.getUpperBound())) {
                    forOp.getUpperBoundMutable().assign(
                        loadThenIncrementCounter(rewriter, cfCounter, cfTape));
                }
                if (gradient::shouldCache(forOp.getStep())) {
                    forOp.getStepMutable().assign(
                        loadThenIncrementCounter(rewriter, cfCounter, cfTape));
                }
            }
            else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                auto conditionOp = whileOp.getConditionOp();
                if (gradient::shouldCache(conditionOp.getCondition())) {
                    conditionOp.getConditionMutable().assign(
                        loadThenIncrementCounter(rewriter, cfCounter, cfTape));
                }
            }
        });
    }

    return qsplitFn;
}

void catalyst::gradient::splitHybridCircuit(Operation *top)
{
    auto gradOp = cast<gradient::GradOp>(top);
    auto moduleOp = gradOp->getParentOfType<ModuleOp>();
    SymbolTable table(moduleOp);
    auto callee = table.lookup<func::FuncOp>(gradOp.getCallee());

    ConversionPatternRewriter rewriter(top->getContext());
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    genAugmentedClassicalFunction(rewriter, gradOp.getLoc(), callee);
    genQuantumSplitFunction(rewriter, gradOp.getLoc(), callee);

    rewriter.setInsertionPoint(gradOp);
}
