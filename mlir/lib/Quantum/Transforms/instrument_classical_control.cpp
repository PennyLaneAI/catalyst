// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "instrument-classical-control"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst;

namespace {

constexpr StringLiteral MeasurementTokenFnName = "__catalyst__rt__measurement_token";
constexpr StringLiteral BeginClassicalControlFnName = "__catalyst__rt__begin_classical_control";
constexpr StringLiteral EndClassicalControlFnName = "__catalyst__rt__end_classical_control";

/// Return the measurement result of `op` if it is a measurement, or a null value otherwise.
Value getMeasurementResult(Operation *op) {
    if (auto measureOp = dyn_cast<quantum::MeasureOp>(op)) {
        return measureOp.getMres();
    }
    if (auto ppmOp = dyn_cast<pbc::PPMeasurementOp>(op)) {
        return ppmOp.getMres();
    }
    if (auto selectPPMOp = dyn_cast<pbc::SelectPPMeasurementOp>(op)) {
        return selectPPMOp.getMres();
    }
    return nullptr;
}

bool isQuantumOp(Operation *op) {
    return isa_and_nonnull<quantum::QuantumDialect, pbc::PBCDialect>(op->getDialect());
}

/// Return whether executing `region` may apply quantum operations.
bool mayApplyQuantumOps(Region &region) {
    return region
        .walk([](Operation *op) {
            return isQuantumOp(op) || isa<CallOpInterface>(op) ? WalkResult::interrupt()
                                                               : WalkResult::advance();
        })
        .wasInterrupted();
}

/// Collects the measurements that classical values depend on.
///
/// Dependencies are traced through the operands of classical operations, and through the values
/// carried by scf control flow. The traversal stops at quantum operations other than
/// measurements, so that values computed from the quantum state (e.g. expectation values) do not
/// depend on earlier measurements through the qubits they act on.
class MeasurementCollector {
  public:
    SetVector<Operation *> collect(ValueRange values) {
        SetVector<Operation *> measurements;
        DenseSet<Value> visited;
        SmallVector<Value> worklist(values.begin(), values.end());

        while (!worklist.empty()) {
            Value value = worklist.pop_back_val();
            if (!visited.insert(value).second) {
                continue;
            }

            if (auto arg = dyn_cast<BlockArgument>(value)) {
                addBlockArgumentSources(arg, worklist);
                continue;
            }

            Operation *def = value.getDefiningOp();
            if (Value mres = getMeasurementResult(def)) {
                if (value == mres) {
                    measurements.insert(def);
                }
                continue;
            }
            if (isQuantumOp(def)) {
                continue;
            }
            addResultSources(cast<OpResult>(value), worklist);
        }
        return measurements;
    }

  private:
    static Operation *getTerminator(Region &region) {
        return region.empty() ? nullptr : region.front().getTerminator();
    }

    static void addYieldedValue(Region &region, unsigned index, SmallVectorImpl<Value> &worklist) {
        if (Operation *terminator = getTerminator(region)) {
            worklist.push_back(terminator->getOperand(index));
        }
    }

    static void addResultSources(OpResult result, SmallVectorImpl<Value> &worklist) {
        Operation *def = result.getOwner();
        unsigned index = result.getResultNumber();

        if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
            worklist.push_back(ifOp.getCondition());
            addYieldedValue(ifOp.getThenRegion(), index, worklist);
            addYieldedValue(ifOp.getElseRegion(), index, worklist);
        } else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(def)) {
            worklist.push_back(switchOp.getArg());
            for (Region &region : switchOp->getRegions()) {
                addYieldedValue(region, index, worklist);
            }
        } else if (auto forOp = dyn_cast<scf::ForOp>(def)) {
            worklist.append({forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()});
            worklist.push_back(forOp.getInitArgs()[index]);
            addYieldedValue(forOp.getRegion(), index, worklist);
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
            scf::ConditionOp conditionOp = whileOp.getConditionOp();
            worklist.push_back(conditionOp.getCondition());
            worklist.push_back(conditionOp.getArgs()[index]);
        } else {
            worklist.append(def->getOperands().begin(), def->getOperands().end());
        }
    }

    static void addBlockArgumentSources(BlockArgument arg, SmallVectorImpl<Value> &worklist) {
        Operation *parent = arg.getOwner()->getParentOp();
        unsigned index = arg.getArgNumber();

        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            if (arg == forOp.getInductionVar()) {
                worklist.append({forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()});
                return;
            }
            unsigned iterArgIndex = index - forOp.getNumInductionVars();
            worklist.push_back(forOp.getInitArgs()[iterArgIndex]);
            addYieldedValue(forOp.getRegion(), iterArgIndex, worklist);
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
            if (arg.getOwner()->getParent() == &whileOp.getBefore()) {
                worklist.push_back(whileOp.getInits()[index]);
                addYieldedValue(whileOp.getAfter(), index, worklist);
            } else {
                worklist.push_back(whileOp.getConditionOp().getArgs()[index]);
            }
        }
        // Function arguments and the arguments of other regions are not traced.
    }
};

/// Instruments classically controlled regions with the runtime calls informing the device of the
/// measurements they depend on.
class ClassicalControlInstrumenter {
  public:
    explicit ClassicalControlInstrumenter(ModuleOp module) : module(module), domInfo(module) {}

    /// Wrap the execution of `region` in a classically controlled region of the runtime.
    void instrumentRegion(Region &region, Operation *controlledOp,
                          const SetVector<Operation *> &measurements) {
        if (region.empty() || !mayApplyQuantumOps(region)) {
            return;
        }

        Block &entry = region.front();
        OpBuilder builder(&entry, entry.begin());
        Value token = buildControlToken(builder, controlledOp, measurements);
        func::CallOp::create(builder, controlledOp->getLoc(), getBeginClassicalControlFn(),
                             ValueRange{token});

        for (Block &block : region) {
            if (block.mightHaveTerminator() &&
                isa<RegionBranchTerminatorOpInterface>(block.getTerminator())) {
                builder.setInsertionPoint(block.getTerminator());
                func::CallOp::create(builder, controlledOp->getLoc(), getEndClassicalControlFn(),
                                     ValueRange{});
            }
        }
    }

    /// Wrap the execution of the operation `controlledOp` in a classically controlled region of
    /// the runtime.
    void instrumentOp(Operation *controlledOp, const SetVector<Operation *> &measurements) {
        OpBuilder builder(controlledOp);
        Value token = buildControlToken(builder, controlledOp, measurements);
        func::CallOp::create(builder, controlledOp->getLoc(), getBeginClassicalControlFn(),
                             ValueRange{token});

        builder.setInsertionPointAfter(controlledOp);
        func::CallOp::create(builder, controlledOp->getLoc(), getEndClassicalControlFn(),
                             ValueRange{});
    }

  private:
    ModuleOp module;
    DominanceInfo domInfo;
    DenseMap<Operation *, Value> measurementTokens;
    func::FuncOp measurementTokenFn;
    func::FuncOp beginClassicalControlFn;
    func::FuncOp endClassicalControlFn;

    func::FuncOp getOrCreateRuntimeFn(func::FuncOp &cache, StringRef name, TypeRange inputs,
                                      TypeRange results) {
        if (cache) {
            return cache;
        }
        cache = module.lookupSymbol<func::FuncOp>(name);
        if (!cache) {
            OpBuilder builder(module.getBodyRegion());
            cache = func::FuncOp::create(builder, module.getLoc(), name,
                                         builder.getFunctionType(inputs, results));
            cache.setPrivate();
        }
        return cache;
    }

    func::FuncOp getMeasurementTokenFn() {
        Type i64 = IntegerType::get(module.getContext(), 64);
        return getOrCreateRuntimeFn(measurementTokenFn, MeasurementTokenFnName, {}, {i64});
    }

    func::FuncOp getBeginClassicalControlFn() {
        Type i64 = IntegerType::get(module.getContext(), 64);
        return getOrCreateRuntimeFn(beginClassicalControlFn, BeginClassicalControlFnName, {i64},
                                    {});
    }

    func::FuncOp getEndClassicalControlFn() {
        return getOrCreateRuntimeFn(endClassicalControlFn, EndClassicalControlFnName, {}, {});
    }

    /// Return the token of `measurement`, taken right after the measurement is performed.
    Value getMeasurementToken(Operation *measurement) {
        auto it = measurementTokens.find(measurement);
        if (it != measurementTokens.end()) {
            return it->second;
        }

        OpBuilder builder(measurement->getContext());
        builder.setInsertionPointAfter(measurement);
        Value token = func::CallOp::create(builder, measurement->getLoc(), getMeasurementTokenFn(),
                                           ValueRange{})
                          .getResult(0);
        measurementTokens[measurement] = token;
        return token;
    }

    /// Build the token of the measurements controlling `controlledOp` at the insertion point.
    ///
    /// Measurements that do not dominate `controlledOp` (e.g. those carried by a loop from a
    /// previous iteration) are represented by the token of the most recent measurement.
    Value buildControlToken(OpBuilder &builder, Operation *controlledOp,
                            const SetVector<Operation *> &measurements) {
        SmallVector<Value> tokens;
        bool needsMostRecentToken = false;
        for (Operation *measurement : measurements) {
            if (domInfo.properlyDominates(measurement, controlledOp)) {
                tokens.push_back(getMeasurementToken(measurement));
            } else {
                needsMostRecentToken = true;
            }
        }
        if (needsMostRecentToken) {
            tokens.push_back(func::CallOp::create(builder, controlledOp->getLoc(),
                                                  getMeasurementTokenFn(), ValueRange{})
                                 .getResult(0));
        }

        Value token = tokens.front();
        for (Value other : ArrayRef<Value>(tokens).drop_front()) {
            token = arith::MaxSIOp::create(builder, controlledOp->getLoc(), token, other);
        }
        return token;
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_INSTRUMENTCLASSICALCONTROLPASS
#include "Quantum/Transforms/Passes.h.inc"

struct InstrumentClassicalControlPass
    : public impl::InstrumentClassicalControlPassBase<InstrumentClassicalControlPass> {
    using impl::InstrumentClassicalControlPassBase<
        InstrumentClassicalControlPass>::InstrumentClassicalControlPassBase;

    void runOnOperation() final {
        ModuleOp module = getOperation();

        // Collect the classically controlled operations before modifying the IR.
        SmallVector<Operation *> controlFlowOps;
        module.walk([&](Operation *op) {
            if (isa<scf::IfOp, scf::IndexSwitchOp, scf::WhileOp, scf::ForOp,
                    pbc::SelectPPMeasurementOp>(op)) {
                controlFlowOps.push_back(op);
            }
        });

        MeasurementCollector collector;
        ClassicalControlInstrumenter instrumenter(module);
        for (Operation *op : controlFlowOps) {
            if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                auto measurements = collector.collect(ifOp.getCondition());
                if (!measurements.empty()) {
                    instrumenter.instrumentRegion(ifOp.getThenRegion(), op, measurements);
                    instrumenter.instrumentRegion(ifOp.getElseRegion(), op, measurements);
                }
            } else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
                auto measurements = collector.collect(switchOp.getArg());
                if (!measurements.empty()) {
                    for (Region &region : switchOp->getRegions()) {
                        instrumenter.instrumentRegion(region, op, measurements);
                    }
                }
            } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                auto measurements = collector.collect(whileOp.getConditionOp().getCondition());
                if (!measurements.empty()) {
                    instrumenter.instrumentRegion(whileOp.getAfter(), op, measurements);
                }
            } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                SmallVector<Value> bounds{forOp.getLowerBound(), forOp.getUpperBound(),
                                          forOp.getStep()};
                auto measurements = collector.collect(bounds);
                if (!measurements.empty()) {
                    instrumenter.instrumentRegion(forOp.getRegion(), op, measurements);
                }
            } else if (auto selectPPMOp = dyn_cast<pbc::SelectPPMeasurementOp>(op)) {
                auto measurements = collector.collect(selectPPMOp.getSelectSwitch());
                if (!measurements.empty()) {
                    instrumenter.instrumentOp(op, measurements);
                }
            }
        }
    }
};

} // namespace quantum
} // namespace catalyst
