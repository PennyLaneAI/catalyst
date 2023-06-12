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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/Analysis/QuantumDependenceAnalysis.h"
#include "Quantum/IR/QuantumInterfaces.h"

#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"
using llvm::errs;

using namespace mlir;
using namespace catalyst;

void cloneQuantumRegion(OpBuilder &builder, Location loc, Region &region, BlockAndValueMapping &bvm,
                        ValueRange tapes, ValueRange tapeIndexCounters)
{
    assert(region.hasOneBlock() && "Expected only structured control flow (region with one block)");
    Dialect *quantumDialect = builder.getContext()->getLoadedDialect("quantum");
    Value paramTape = tapes[0];
    Value paramCounter = tapeIndexCounters[0];
    Value cfTape = tapes[1];
    Value cfCounter = tapeIndexCounters[1];
    Value cOne = builder.create<index::ConstantOp>(loc, 1);

    for (Operation &oldOp : region.front().without_terminator()) {
        // TODO(jacob): this is a weird case, tensor.from_elements is used to convert the
        // expectation value into a tensor. Should it be considered classical postprocessing?
        bool isTensor = isa<tensor::FromElementsOp>(&oldOp);
        bool isCast = isa<arith::IndexCastOp>(&oldOp);
        bool isConstant = isa<arith::ConstantOp>(&oldOp);
        bool isQuantum = oldOp.getDialect() == quantumDialect;

        if (auto gateOp = dyn_cast<quantum::DifferentiableGate>(&oldOp)) {
            // Load cached parameters.
            Value paramIdx = builder.create<memref::LoadOp>(loc, paramCounter);
            auto newGateOp = builder.clone(*gateOp, bvm);
            for (size_t diffParamIdx = 0; diffParamIdx < gateOp.getDiffParams().size();
                 diffParamIdx++) {
                OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPoint(newGateOp);
                Value cachedParam = builder.create<tensor::ExtractOp>(loc, paramTape, paramIdx);
                newGateOp->setOperand(gateOp.getDiffOperandIdx() + diffParamIdx, cachedParam);

                paramIdx = builder.create<index::AddOp>(loc, paramIdx, cOne);
            }

            builder.create<memref::StoreOp>(loc, paramIdx, paramCounter);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(&oldOp)) {
            // We only want to keep quantum iterArgs, so loopIndexMapping maps the index of the old
            // iteration arguments to the new iteration arguments.
            SmallVector<Value> iterArgs;
            DenseMap<size_t, size_t> loopIndexMapping;
            size_t newIterIdx = 0;
            for (const auto &[iterIdx, oldIterOperand] : llvm::enumerate(forOp.getIterOperands())) {
                if (&oldIterOperand.getType().getDialect() == quantumDialect) {
                    iterArgs.push_back(bvm.lookup(oldIterOperand));
                    loopIndexMapping[iterIdx] = newIterIdx++;
                }
            }

            // Load control flow parameters from the tape, updating the counter index.
            Value lowerBoundIdx = builder.create<memref::LoadOp>(loc, cfCounter);
            Value upperBoundIdx = builder.create<index::AddOp>(loc, lowerBoundIdx, cOne);
            Value stepIdx = builder.create<index::AddOp>(loc, upperBoundIdx, cOne);
            Value newIdxVal = builder.create<index::AddOp>(loc, stepIdx, cOne);
            builder.create<memref::StoreOp>(loc, newIdxVal, cfCounter);

            Value lowerBound = builder.create<tensor::ExtractOp>(loc, cfTape, lowerBoundIdx);
            Value upperBound = builder.create<tensor::ExtractOp>(loc, cfTape, upperBoundIdx);
            Value step = builder.create<tensor::ExtractOp>(loc, cfTape, stepIdx);

            auto updateLoopMapping = [&bvm, &loopIndexMapping](ValueRange oldValues,
                                                               ValueRange newValues) {
                for (const auto &[oldIdx, newIdx] : loopIndexMapping) {
                    bvm.map(oldValues[oldIdx], newValues[newIdx]);
                }
            };
            auto lookupLoopMapping = [&bvm, &loopIndexMapping](ValueRange oldValues,
                                                               MutableArrayRef<Value> newValues) {
                for (const auto &[oldIdx, newIdx] : loopIndexMapping) {
                    newValues[newIdx] = bvm.lookup(oldValues[oldIdx]);
                }
            };

            auto newForOp = builder.create<scf::ForOp>(
                loc, lowerBound, upperBound, step, iterArgs,
                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
                    bvm.map(forOp.getInductionVar(), iv);
                    updateLoopMapping(forOp.getRegionIterArgs(), iterArgs);
                    cloneQuantumRegion(builder, loc, forOp.getLoopBody(), bvm, tapes,
                                       tapeIndexCounters);

                    // Update loop terminator
                    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
                    SmallVector<Value> newYieldOperands{iterArgs.size()};
                    lookupLoopMapping(oldYield.getOperands(), newYieldOperands);
                    builder.create<scf::YieldOp>(loc, newYieldOperands);
                });

            updateLoopMapping(forOp.getResults(), newForOp.getResults());
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(&oldOp)) {
            auto cZero = builder.create<index::ConstantOp>(loc, 0);
            DenseMap<size_t, size_t> loopIndexMapping;
            size_t newIterIdx = 0;
            SmallVector<Value> iterArgs;
            for (const auto &[iterIdx, oldIterOperand] : llvm::enumerate(whileOp.getInits())) {
                if (&oldIterOperand.getType().getDialect() == quantumDialect) {
                    iterArgs.push_back(bvm.lookup(oldIterOperand));
                    loopIndexMapping[iterIdx] = newIterIdx++;
                }
            }

            Value tapeIdx = builder.create<memref::LoadOp>(loc, cfCounter);
            Value newIdx = builder.create<index::AddOp>(loc, tapeIdx, cOne);
            builder.create<memref::StoreOp>(loc, newIdx, cfCounter);
            Value numIters = builder.create<tensor::ExtractOp>(loc, cfTape, tapeIdx);

            auto updateLoopMapping = [&bvm, &loopIndexMapping](ValueRange oldValues,
                                                               ValueRange newValues) {
                for (const auto &[oldIdx, newIdx] : loopIndexMapping) {
                    bvm.map(oldValues[oldIdx], newValues[newIdx]);
                }
            };
            auto lookupLoopMapping = [&bvm, &loopIndexMapping](ValueRange oldValues,
                                                               MutableArrayRef<Value> newValues) {
                for (const auto &[oldIdx, newIdx] : loopIndexMapping) {
                    newValues[newIdx] = bvm.lookup(oldValues[oldIdx]);
                }
            };
            auto newLoop = builder.create<scf::ForOp>(
                loc, cZero, numIters, cOne, iterArgs,
                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
                    updateLoopMapping(whileOp.getAfterArguments(), iterArgs);
                    cloneQuantumRegion(builder, loc, whileOp.getAfter(), bvm, tapes,
                                       tapeIndexCounters);

                    auto oldYield = cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
                    SmallVector<Value> newYieldOperands{iterArgs.size()};
                    lookupLoopMapping(oldYield.getOperands(), newYieldOperands);
                    builder.create<scf::YieldOp>(loc, newYieldOperands);
                });

            updateLoopMapping(whileOp.getResults(), newLoop.getResults());
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(&oldOp)) {
            if (ifOp.getNumResults() == 0) {
                continue;
            }

            DenseMap<size_t, size_t> resultIndexMapping;
            SmallVector<Type> resultTypes;
            size_t newResultIndex = 0;
            for (const auto &[oldResultIndex, resultType] :
                 llvm::enumerate(ifOp.getResultTypes())) {
                if (&resultType.getDialect() == quantumDialect) {
                    resultTypes.push_back(resultType);
                    resultIndexMapping[oldResultIndex] = newResultIndex++;
                }
            }

            Value conditionIdx = builder.create<memref::LoadOp>(loc, cfCounter);
            Value newIdx = builder.create<index::AddOp>(loc, conditionIdx, cOne);
            builder.create<memref::StoreOp>(loc, newIdx, cfCounter);

            Value condition = builder.create<tensor::ExtractOp>(loc, cfTape, conditionIdx);
            Value conditionI1 =
                builder.create<arith::IndexCastOp>(loc, builder.getI1Type(), condition);

            auto createIfOpBuilder = [&](Region &region) {
                return [&](OpBuilder &builder, Location loc) {
                    cloneQuantumRegion(builder, loc, region, bvm, tapes, tapeIndexCounters);
                    auto oldYield = cast<scf::YieldOp>(region.front().getTerminator());
                    SmallVector<Value> newYieldOperands{resultTypes.size()};
                    for (const auto &[oldIdx, newIdx] : resultIndexMapping) {
                        newYieldOperands[newIdx] = bvm.lookup(oldYield.getOperand(oldIdx));
                    }
                    builder.create<scf::YieldOp>(loc, newYieldOperands);
                };
            };

            auto newIfOp = builder.create<scf::IfOp>(loc, resultTypes, conditionI1,
                                                     createIfOpBuilder(ifOp.getThenRegion()),
                                                     createIfOpBuilder(ifOp.getElseRegion()));

            // Update the mapping
            for (const auto &[oldIdx, newIdx] : resultIndexMapping) {
                bvm.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
            }
        }
        else if (isConstant || isQuantum || isTensor || isCast) {
            builder.clone(oldOp, bvm);
        }
    }
}

func::FuncOp genQuantumSplitFunction(OpBuilder &builder, Location loc, func::FuncOp callee)
{
    auto fnName = builder.getStringAttr(callee.getName() + ".qsplit");
    SmallVector<Type> argTypes{
        RankedTensorType::get({ShapedType::kDynamic}, builder.getF64Type()),
        RankedTensorType::get({ShapedType::kDynamic}, builder.getIndexType())};
    SmallVector<Location> argLocs{2, loc};
    FunctionType fnType = builder.getFunctionType(argTypes, callee.getResultTypes());
    func::FuncOp qsplitFn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callee, fnName);
    if (!qsplitFn) {
        qsplitFn = builder.create<func::FuncOp>(loc, fnName, fnType);
        Region &newBody = qsplitFn.getFunctionBody();
        PatternRewriter::InsertionGuard insertGuard(builder);

        builder.createBlock(&newBody, newBody.end(), argTypes, argLocs);
        Value paramIndex =
            builder.create<memref::AllocaOp>(loc, MemRefType::get({}, builder.getIndexType()));
        Value cfIndex =
            builder.create<memref::AllocaOp>(loc, MemRefType::get({}, builder.getIndexType()));
        Value cZero = builder.create<index::ConstantOp>(loc, 0);
        builder.create<memref::StoreOp>(loc, cZero, paramIndex);
        builder.create<memref::StoreOp>(loc, cZero, cfIndex);

        BlockAndValueMapping bvm;

        cloneQuantumRegion(builder, loc, callee.getFunctionBody(), bvm, newBody.getArguments(),
                           {paramIndex, cfIndex});

        auto oldReturn = cast<func::ReturnOp>(callee.getFunctionBody().front().getTerminator());
        builder.clone(*oldReturn, bvm);
    }

    errs() << "qsplit: " << qsplitFn << "\n";
    if (failed(mlir::verify(qsplitFn))) {
        errs() << "fn failed verification\n";
    }
    return qsplitFn;
}

void catalyst::gradient::splitHybridCircuit(Operation *top,
                                            quantum::QuantumDependenceAnalysis &qdepAnalysis)
{
    auto gradOp = cast<gradient::GradOp>(top);
    auto moduleOp = gradOp->getParentOfType<ModuleOp>();
    SymbolTable table(moduleOp);
    auto callee = table.lookup<func::FuncOp>(gradOp.getCallee());

    OpBuilder builder(top);
    builder.setInsertionPointToStart(moduleOp.getBody());

    genQuantumSplitFunction(builder, gradOp.getLoc(), callee);

    builder.setInsertionPoint(gradOp);
}
