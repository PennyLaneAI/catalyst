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

#define DEBUG_TYPE "delegate-feedforward"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_DELEGATEFEEDFORWARDPASS
#define GEN_PASS_DEF_DELEGATEFEEDFORWARDPASS
#include "PBC/Transforms/Passes.h.inc"

} // namespace pbc
} // namespace catalyst

namespace {

using quantum::PredicatedOp;
using quantum::ResultType;

constexpr StringLiteral errorPrefix = "cannot delegate feed-forward to the device: ";

bool isHandle(Value value) { return isa<ResultType>(value.getType()); }

bool isQubit(Value value) { return isa<quantum::QubitType>(value.getType()); }

/// `value` as a result handle: itself, or a constant handle for an i1 constant. Null otherwise.
Value asHandle(Value value, OpBuilder &builder) {
    if (isHandle(value)) {
        return value;
    }
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
        if (auto bit = dyn_cast<IntegerAttr>(constant.getValue());
            bit && bit.getType().isInteger(1)) {
            return quantum::ResultConstantOp::create(
                builder, value.getLoc(), ResultType::get(builder.getContext()),
                builder.getBoolAttr(bit.getValue().getBoolValue()));
        }
    }
    return nullptr;
}

/// The constant value of an i1 constant, if `value` is one.
std::optional<bool> constantBit(Value value) {
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
        if (auto bit = dyn_cast<IntegerAttr>(constant.getValue());
            bit && bit.getType().isInteger(1)) {
            return bit.getValue().getBoolValue();
        }
    }
    return std::nullopt;
}

/// The result of `op` that carries the qubit `operand` on, if `op` is a quantum op that does.
FailureOr<Value> successorQubit(Operation *op, OpOperand &operand) {
    const unsigned index = operand.getOperandNumber();
    return TypeSwitch<Operation *, FailureOr<Value>>(op)
        .Case<quantum::MeasureOp>([](auto measure) { return Value(measure.getOutQubit()); })
        .Case<PredicatedOp>([&](auto predicated) -> FailureOr<Value> {
            // Operand 0 is the condition.
            return Value(predicated.getOutQubits()[index - 1]);
        })
        .Case<pbc::GadgetOp>(
            [&](auto gadget) -> FailureOr<Value> { return Value(gadget.getOutQubits()[index]); })
        .Case<pbc::PBCOpInterface>([&](auto pbcOp) -> FailureOr<Value> {
            for (auto [in, out] : llvm::zip(pbcOp.getInQubits(), pbcOp.getOutQubits())) {
                if (in == operand.get()) {
                    return Value(out);
                }
            }
            return failure();
        })
        .Case<quantum::QuantumOperation>([&](auto gate) -> FailureOr<Value> {
            for (auto [in, out] : llvm::zip(gate.getQubitOperands(), gate.getQubitResults())) {
                if (in == operand.get()) {
                    return Value(out);
                }
            }
            return failure();
        })
        .Default([](Operation *) { return failure(); });
}

/// What happens, inside one region of an `scf.if`, to a qubit the region uses from above.
struct QubitFate {
    enum Kind { Unused, Yielded, Released } kind{Unused};
    /// Yielded: the result index.
    unsigned index{0};
    /// Released: the op that releases it.
    quantum::DeallocQubitOp release{};
};

FailureOr<QubitFate> traceQubit(Value qubit, Region &region) {
    Value current = qubit;
    while (true) {
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : current.getUses()) {
            if (region.isAncestor(use.getOwner()->getParentRegion())) {
                uses.push_back(&use);
            }
        }
        if (uses.empty()) {
            if (current == qubit) {
                return QubitFate{};
            }
            return failure();
        }
        if (uses.size() > 1 || uses.front()->getOwner()->getParentRegion() != &region) {
            return failure();
        }
        OpOperand &use = *uses.front();
        Operation *user = use.getOwner();
        if (isa<scf::YieldOp>(user)) {
            return QubitFate{QubitFate::Yielded, use.getOperandNumber(), {}};
        }
        if (auto release = dyn_cast<quantum::DeallocQubitOp>(user)) {
            return QubitFate{QubitFate::Released, 0, release};
        }
        FailureOr<Value> next = successorQubit(user, use);
        if (failed(next)) {
            return failure();
        }
        current = *next;
    }
}

/// Whether `op` may run under a predicate: a quantum op on qubits, or a pure classical op.
bool isPredicable(Operation *op) {
    if (isa<quantum::ExtractOp, quantum::InsertOp, quantum::AllocOp, quantum::DeallocOp,
            quantum::DeviceInitOp, quantum::DeviceReleaseOp, quantum::MeasurementProcess>(op)) {
        return false;
    }
    if (isa<quantum::QuantumDialect, pbc::PBCDialect>(op->getDialect())) {
        return true;
    }
    return isPure(op) && op->getNumRegions() == 0;
}

class Delegator {
  public:
    Delegator(func::FuncOp func, quantum::PostselectMode mode)
        : func(func), mode(mode), rewriter(func.getContext()) {}

    LogicalResult run() {
        quantum::DeviceInitOp device;
        func.walk([&](quantum::DeviceInitOp op) { device = op; });
        if (!device) {
            return success();
        }

        // Measurement outcomes become handles. Their users are rewritten below; until then the
        // IR is not valid.
        func.walk([&](Operation *op) {
            if (isa<quantum::MeasureOp, pbc::PPMeasurementOp>(op)) {
                op->getResult(0).setType(ResultType::get(op->getContext()));
            }
        });

        // Children before parents, so that a region is rewritten before the op that holds it.
        SmallVector<Operation *> worklist;
        func.walk<WalkOrder::PostOrder>([&](Operation *op) {
            if (isa<arith::XOrIOp, arith::AndIOp, arith::OrIOp, arith::SelectOp, scf::IfOp,
                    pbc::PPRotationOp, pbc::PPRotationArbitraryOp, pbc::SelectPPMeasurementOp>(
                    op)) {
                worklist.push_back(op);
            }
        });
        for (Operation *op : worklist) {
            if (failed(rewrite(op))) {
                return failure();
            }
        }

        if (failed(checkUses())) {
            return failure();
        }

        rewriter.setInsertionPointAfter(device);
        quantum::DelegateOp::create(rewriter, device.getLoc(), mode);
        insertReleases();
        return success();
    }

  private:
    func::FuncOp func;
    quantum::PostselectMode mode;
    IRRewriter rewriter;

    LogicalResult rewrite(Operation *op) {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case<arith::XOrIOp>([&](auto xorOp) { return rewriteXor(xorOp); })
            .Case<arith::AndIOp, arith::OrIOp>([&](auto logic) { return rewriteLogic(logic); })
            .Case<arith::SelectOp>([&](auto select) { return rewriteSelect(select); })
            .Case<scf::IfOp>([&](auto ifOp) { return rewriteIf(ifOp); })
            .Case<pbc::PPRotationOp, pbc::PPRotationArbitraryOp>(
                [&](auto ppr) { return rewriteConditionalRotation(ppr); })
            .Case<pbc::SelectPPMeasurementOp>([&](auto select) { return rewriteSelectPPM(select); })
            .Default([](Operation *) { return success(); });
    }

    LogicalResult rewriteXor(arith::XOrIOp op) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        if (!isHandle(lhs) && !isHandle(rhs)) {
            return success();
        }
        rewriter.setInsertionPoint(op);
        // XOR with a constant is a negation or nothing.
        for (auto [handle, other] : {std::pair{lhs, rhs}, std::pair{rhs, lhs}}) {
            if (std::optional<bool> bit = constantBit(other); bit && isHandle(handle)) {
                Value result = *bit ? quantum::ResultNotOp::create(rewriter, op.getLoc(),
                                                                   handle.getType(), handle)
                                    : handle;
                rewriter.replaceOp(op, result);
                return success();
            }
        }
        Value a = asHandle(lhs, rewriter);
        Value b = asHandle(rhs, rewriter);
        if (!a || !b) {
            return op.emitError(errorPrefix) << "an outcome is combined with a classical value";
        }
        rewriter.replaceOpWithNewOp<quantum::ResultXorOp>(op, a.getType(), a, b);
        return success();
    }

    /// a AND b is SELECT(a, b, false); a OR b is SELECT(a, true, b).
    template <typename LogicOp> LogicalResult rewriteLogic(LogicOp op) {
        if (!isHandle(op->getOperand(0)) && !isHandle(op->getOperand(1))) {
            return success();
        }
        rewriter.setInsertionPoint(op);
        Value a = asHandle(op->getOperand(0), rewriter);
        Value b = asHandle(op->getOperand(1), rewriter);
        if (!a || !b) {
            return op.emitError(errorPrefix) << "an outcome is combined with a classical value";
        }
        const bool isAnd = std::is_same_v<LogicOp, arith::AndIOp>;
        Value constant = quantum::ResultConstantOp::create(rewriter, op.getLoc(), a.getType(),
                                                           rewriter.getBoolAttr(!isAnd));
        if (isAnd) {
            rewriter.replaceOpWithNewOp<quantum::ResultSelectOp>(op, a.getType(), a, b, constant);
        } else {
            rewriter.replaceOpWithNewOp<quantum::ResultSelectOp>(op, a.getType(), a, constant, b);
        }
        return success();
    }

    LogicalResult rewriteSelect(arith::SelectOp op) {
        Value condition = op->getOperand(0);
        if (!isHandle(condition)) {
            if (isHandle(op.getTrueValue()) || isHandle(op.getFalseValue())) {
                return op.emitError(errorPrefix) << "a classical value selects an outcome";
            }
            return success();
        }
        rewriter.setInsertionPoint(op);
        Value a = asHandle(op.getTrueValue(), rewriter);
        Value b = asHandle(op.getFalseValue(), rewriter);
        if (!a || !b) {
            return op.emitError(errorPrefix) << "an outcome selects a classical value";
        }
        rewriter.replaceOpWithNewOp<quantum::ResultSelectOp>(op, a.getType(), condition, a, b);
        return success();
    }

    /// A predicated op on `qubits`, whose regions are filled by `buildThen` and `buildElse`.
    /// Each builder gets the region's qubit arguments and returns the values to yield.
    PredicatedOp createPredicated(Location loc, Value condition, ValueRange qubits,
                                  TypeRange handleTypes,
                                  function_ref<SmallVector<Value>(ValueRange)> buildThen,
                                  function_ref<SmallVector<Value>(ValueRange)> buildElse) {
        SmallVector<Type> resultTypes(qubits.getTypes());
        resultTypes.append(handleTypes.begin(), handleTypes.end());
        auto predicated = PredicatedOp::create(rewriter, loc, resultTypes, condition, qubits);
        OpBuilder::InsertionGuard guard(rewriter);
        SmallVector<Location> locs(qubits.size(), loc);
        for (auto [region, build] : {std::pair{&predicated.getThenRegion(), buildThen},
                                     std::pair{&predicated.getElseRegion(), buildElse}}) {
            Block *block = rewriter.createBlock(region, {}, qubits.getTypes(), locs);
            SmallVector<Value> yielded = build(block->getArguments());
            quantum::YieldOp::create(rewriter, loc, yielded);
        }
        return predicated;
    }

    template <typename RotationOp> LogicalResult rewriteConditionalRotation(RotationOp op) {
        // Untyped access: the typed accessor expects i1.
        MutableOperandRange conditionOperand = op.getConditionMutable();
        if (conditionOperand.empty() || !isHandle(conditionOperand[0].get())) {
            return success();
        }
        Value condition = conditionOperand[0].get();
        rewriter.setInsertionPoint(op);
        auto predicated = createPredicated(
            op.getLoc(), condition, op.getInQubits(), {},
            [&](ValueRange args) {
                IRMapping mapping;
                mapping.map(op.getInQubits(), args);
                auto inner = cast<RotationOp>(rewriter.clone(*op, mapping));
                inner.getConditionMutable().clear();
                return SmallVector<Value>(inner.getOutQubits());
            },
            [](ValueRange args) { return SmallVector<Value>(args); });
        rewriter.replaceOp(op, predicated.getResults());
        return success();
    }

    /// select.ppm (c ? P0 : P1) is a PPM of P0 on the shots where c holds, and of P1 elsewhere.
    LogicalResult rewriteSelectPPM(pbc::SelectPPMeasurementOp op) {
        Value selectSwitch = op.getSelectSwitchMutable().get();
        if (!isHandle(selectSwitch)) {
            return success();
        }
        rewriter.setInsertionPoint(op);
        Type handleType = ResultType::get(op.getContext());
        auto measure = [&](ArrayAttr word, bool negated) {
            return [&, word, negated](ValueRange args) {
                auto ppm = pbc::PPMeasurementOp::create(
                    rewriter, op.getLoc(), handleType, args.getTypes(), word,
                    negated ? rewriter.getUnitAttr() : nullptr, args, /*postselect=*/nullptr);
                SmallVector<Value> yielded(ppm.getOutQubits());
                yielded.push_back(ppm.getMres());
                return yielded;
            };
        };
        auto predicated =
            createPredicated(op.getLoc(), selectSwitch, op.getInQubits(), {handleType},
                             measure(op.getPauliProduct_0(), op.getNegated_0()),
                             measure(op.getPauliProduct_1(), op.getNegated_1()));
        SmallVector<Value> results{predicated.getResultHandles().front()};
        results.append(predicated.getOutQubits().begin(), predicated.getOutQubits().end());
        rewriter.replaceOp(op, results);
        return success();
    }

    LogicalResult rewriteIf(scf::IfOp ifOp) {
        Value condition = ifOp->getOperand(0);
        if (!isHandle(condition)) {
            return success();
        }
        auto fail = [&](const Twine &reason) { return ifOp.emitError(errorPrefix) << reason; };

        // The qubits the branches use from above.
        SetVector<Value> captured;
        WalkResult walk = ifOp->walk([&](Operation *op) {
            if (op == ifOp.getOperation()) {
                return WalkResult::advance();
            }
            if (op->getParentOp() == ifOp.getOperation() && !isa<scf::YieldOp>(op) &&
                !isPredicable(op)) {
                op->emitError(errorPrefix) << "this op cannot run on only some of the shots";
                return WalkResult::interrupt();
            }
            for (Value operand : op->getOperands()) {
                if (isa<quantum::QuregType>(operand.getType()) &&
                    !ifOp->isProperAncestor(operand.getParentRegion()->getParentOp())) {
                    op->emitError(errorPrefix) << "a conditional branch uses a qubit register";
                    return WalkResult::interrupt();
                }
                if (isQubit(operand) &&
                    !ifOp->isAncestor(operand.getParentRegion()->getParentOp())) {
                    captured.insert(operand);
                }
            }
            return WalkResult::advance();
        });
        if (walk.wasInterrupted()) {
            return failure();
        }

        // Each qubit must leave both branches the same way: returned at the same position, or
        // released.
        Region &thenRegion = ifOp.getThenRegion();
        Region &elseRegion = ifOp.getElseRegion();
        if (elseRegion.empty()) {
            return fail("a conditional branch without an else region returns values");
        }
        SmallVector<Value> yieldedQubits(ifOp.getNumResults());
        SmallVector<Value> releasedQubits;
        SmallVector<quantum::DeallocQubitOp> releases;
        for (Value qubit : captured) {
            FailureOr<QubitFate> thenFate = traceQubit(qubit, thenRegion);
            FailureOr<QubitFate> elseFate = traceQubit(qubit, elseRegion);
            if (failed(thenFate) || failed(elseFate)) {
                return fail("a qubit cannot be followed through a conditional branch");
            }
            if (thenFate->kind == QubitFate::Yielded && elseFate->kind == QubitFate::Yielded &&
                thenFate->index == elseFate->index) {
                yieldedQubits[thenFate->index] = qubit;
            } else if (thenFate->kind == QubitFate::Released &&
                       elseFate->kind == QubitFate::Released) {
                releasedQubits.push_back(qubit);
                releases.push_back(thenFate->release);
                releases.push_back(elseFate->release);
            } else {
                return fail("a qubit leaves the two branches of a conditional differently");
            }
        }

        // Operands: the returned qubits in result order, then the released ones.
        SmallVector<Value> operands;
        SmallVector<unsigned> qubitResults;
        SmallVector<unsigned> handleResults;
        for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
            if (isQubit(result)) {
                if (!yieldedQubits[index]) {
                    return fail("a conditional branch returns a qubit it allocated");
                }
                operands.push_back(yieldedQubits[index]);
                qubitResults.push_back(index);
            } else if (result.getType().isInteger(1)) {
                handleResults.push_back(index);
            } else {
                return fail("a conditional branch returns a classical value");
            }
        }
        const size_t numReturned = operands.size();
        operands.append(releasedQubits.begin(), releasedQubits.end());

        rewriter.setInsertionPoint(ifOp);
        SmallVector<Type> handleTypes(handleResults.size(), ResultType::get(ifOp.getContext()));
        SmallVector<Type> resultTypes(ValueRange(operands).getTypes());
        resultTypes.append(handleTypes);
        auto predicated =
            PredicatedOp::create(rewriter, ifOp.getLoc(), resultTypes, condition, operands);

        for (auto [oldRegion, newRegion] : {std::pair{&thenRegion, &predicated.getThenRegion()},
                                            std::pair{&elseRegion, &predicated.getElseRegion()}}) {
            Block *oldBlock = &oldRegion->front();
            Block *block =
                rewriter.createBlock(newRegion, {}, ValueRange(operands).getTypes(),
                                     SmallVector<Location>(operands.size(), ifOp.getLoc()));
            rewriter.mergeBlocks(oldBlock, block, {});
            for (auto [operand, arg] : llvm::zip(operands, block->getArguments())) {
                rewriter.replaceUsesWithIf(operand, arg, [&](OpOperand &use) {
                    return newRegion->isAncestor(use.getOwner()->getParentRegion());
                });
            }

            auto yield = cast<scf::YieldOp>(block->getTerminator());
            rewriter.setInsertionPoint(yield);
            SmallVector<Value> yielded;
            for (unsigned index : qubitResults) {
                yielded.push_back(yield.getOperand(index));
            }
            // A released qubit is returned instead, and released once after the op.
            for (quantum::DeallocQubitOp release : releases) {
                if (newRegion->isAncestor(release->getParentRegion())) {
                    yielded.push_back(release.getQubit());
                }
            }
            for (unsigned index : handleResults) {
                Value handle = asHandle(yield.getOperand(index), rewriter);
                if (!handle) {
                    return fail("a conditional branch returns a classical bit");
                }
                yielded.push_back(handle);
            }
            if (yielded.size() != resultTypes.size()) {
                return fail("a qubit leaves the two branches of a conditional differently");
            }
            quantum::YieldOp::create(rewriter, yield.getLoc(), yielded);
            rewriter.eraseOp(yield);
        }
        for (quantum::DeallocQubitOp release : releases) {
            rewriter.eraseOp(release);
        }

        rewriter.setInsertionPointAfter(predicated);
        for (unsigned i = numReturned; i < operands.size(); i++) {
            quantum::DeallocQubitOp::create(rewriter, ifOp.getLoc(), predicated.getResult(i));
        }
        SmallVector<Value> replacements(ifOp.getNumResults());
        for (auto [position, index] : llvm::enumerate(qubitResults)) {
            replacements[index] = predicated.getResult(position);
        }
        for (auto [position, index] : llvm::enumerate(handleResults)) {
            replacements[index] = predicated.getResultHandles()[position];
        }
        rewriter.replaceOp(ifOp, replacements);
        return success();
    }

    /// Every handle must end in an op that takes handles.
    LogicalResult checkUses() {
        WalkResult walk = func.walk([&](Operation *op) {
            if (isa<quantum::ResultConstantOp, quantum::ResultXorOp, quantum::ResultNotOp,
                    quantum::ResultSelectOp, quantum::ResultReleaseOp, PredicatedOp,
                    quantum::YieldOp>(op)) {
                return WalkResult::advance();
            }
            for (Value operand : op->getOperands()) {
                if (isHandle(operand)) {
                    op->emitError(errorPrefix)
                        << "a measurement outcome is used as a classical value; on a delegated "
                           "program, outcomes may only condition quantum operations";
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
        return failure(walk.wasInterrupted());
    }

    /// Release each handle after its last use in the block that defines it. A handle that a
    /// region yields is released by the lowering of the op that returns it.
    void insertReleases() {
        SmallVector<Value> handles;
        func.walk([&](Operation *op) {
            for (Value result : op->getResults()) {
                if (isHandle(result)) {
                    handles.push_back(result);
                }
            }
        });
        for (Value handle : handles) {
            Operation *definition = handle.getDefiningOp();
            Block *block = definition->getBlock();
            Operation *last = definition;
            for (Operation *user : handle.getUsers()) {
                Operation *ancestor = block->findAncestorOpInBlock(*user);
                if (ancestor && last->isBeforeInBlock(ancestor)) {
                    last = ancestor;
                }
            }
            if (last->hasTrait<OpTrait::IsTerminator>()) {
                continue;
            }
            rewriter.setInsertionPointAfter(last);
            quantum::ResultReleaseOp::create(rewriter, definition->getLoc(), handle);
        }
    }
};

} // namespace

namespace catalyst {
namespace pbc {

struct DelegateFeedforwardPass : public impl::DelegateFeedforwardPassBase<DelegateFeedforwardPass> {
    using DelegateFeedforwardPassBase::DelegateFeedforwardPassBase;

    void runOnOperation() final {
        quantum::PostselectMode mode;
        if (postselectMode == "fill-shots") {
            mode = quantum::PostselectMode::fill_shots;
        } else if (postselectMode == "hw-like") {
            mode = quantum::PostselectMode::hw_like;
        } else {
            getOperation().emitError("unknown postselect-mode '")
                << postselectMode << "': expected fill-shots or hw-like";
            return signalPassFailure();
        }

        WalkResult walk = getOperation().walk([&](func::FuncOp func) {
            return failed(Delegator(func, mode).run()) ? WalkResult::interrupt()
                                                       : WalkResult::advance();
        });
        if (walk.wasInterrupted()) {
            signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
