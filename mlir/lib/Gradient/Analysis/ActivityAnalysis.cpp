// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"

#include "Gradient/Analysis/ActivityAnalysis.h"

using namespace mlir;
using namespace dataflow;

using llvm::errs;

class ValueActivity {
    enum class Activity { Constant, Active };

  public:
    ValueActivity(std::optional<ValueActivity::Activity> activity = std::nullopt)
        : activity(activity)
    {
    }

    bool operator==(const ValueActivity &rhs) const { return activity == rhs.activity; }

    bool isUninitialized() const { return !activity.has_value(); }

    bool isConstant() const { return activity == Activity::Constant; }

    bool isActive() const { return activity == Activity::Active; }

    static ValueActivity getConstant() { return ValueActivity(Activity::Constant); }

    static ValueActivity getActive() { return ValueActivity(Activity::Active); }

    void print(raw_ostream &os) const
    {
        if (isUninitialized()) {
            os << "<uninitialized>";
            return;
        }

        switch (*activity) {
        case Activity::Active:
            os << "Active";
            break;
        case Activity::Constant:
            os << "Constant";
            break;
        }
    }

    /// If either the left-hand side or right-hand side is active, the merged
    /// result is active.
    static ValueActivity merge(const ValueActivity &lhs, const ValueActivity &rhs)
    {
        if (lhs.isUninitialized()) {
            return rhs;
        }
        if (lhs.isUninitialized()) {
            return lhs;
        }

        if (lhs.isConstant() && rhs.isConstant()) {
            return ValueActivity::getConstant();
        }
        return ValueActivity::getActive();
    }

  private:
    std::optional<ValueActivity::Activity> activity;
};

raw_ostream &operator<<(raw_ostream &os, const ValueActivity &valueActivity)
{
    valueActivity.print(os);
    return os;
}

//===----------------------------------------------------------------------===//
// Lattices
//===----------------------------------------------------------------------===//

class ForwardActivity : public AbstractSparseLattice {
  public:
    using AbstractSparseLattice::AbstractSparseLattice;

    ChangeResult join(const AbstractSparseLattice &rhs) override
    {
        return join(static_cast<const ForwardActivity &>(rhs).value);
    }

    ChangeResult join(const ValueActivity &rhs)
    {
        ValueActivity newValue = ValueActivity::merge(value, rhs);

        if (newValue == value) {
            return ChangeResult::NoChange;
        }

        value = newValue;
        return ChangeResult::Change;
    }

    void print(raw_ostream &os) const override { value.print(os); }

    ValueActivity getValue() const { return value; }

  private:
    ValueActivity value;
};

class BackwardActivity : public AbstractSparseLattice {
  public:
    using AbstractSparseLattice::AbstractSparseLattice;

    ChangeResult meet(const AbstractSparseLattice &rhs) override
    {
        return meet(static_cast<const BackwardActivity &>(rhs).value);
    }

    ChangeResult meet(const ValueActivity &rhs)
    {
        ValueActivity newValue = ValueActivity::merge(value, rhs);

        if (newValue == value) {
            return ChangeResult::NoChange;
        }

        value = newValue;
        return ChangeResult::Change;
    }

    void print(raw_ostream &os) const override { value.print(os); }

    ValueActivity getValue() const { return value; }

  private:
    ValueActivity value;
};

//===----------------------------------------------------------------------===//
// Sub-analyses
//===----------------------------------------------------------------------===//

class ForwardActivityAnalysis : public SparseDataFlowAnalysis<ForwardActivity> {
  public:
    using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

    void visitOperation(Operation *op, ArrayRef<const ForwardActivity *> operands,
                        ArrayRef<ForwardActivity *> results) override
    {
        if (op->hasTrait<OpTrait::ConstantLike>()) {
            for (ForwardActivity *result : results) {
                propagateIfChanged(result, result->join(ValueActivity::getConstant()));
            }
            return;
        }

        // A result is (forward) active iff it has any active operands.
        for (ForwardActivity *result : results) {
            for (const ForwardActivity *operand : operands) {
                join(result, *operand);
            }
        }
    }

    /// In general we can't reason about activity at arbitrary entry states.
    void setToEntryState(ForwardActivity *lattice) override { lattice->join(ValueActivity()); }
};

class BackwardActivityAnalysis : public SparseBackwardDataFlowAnalysis<BackwardActivity> {
  public:
    using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

    void visitOperation(Operation *op, ArrayRef<BackwardActivity *> operands,
                        ArrayRef<const BackwardActivity *> results) override
    {
        // An operand is (backward) active iff any of its results are active
        for (BackwardActivity *operand : operands) {
            for (const BackwardActivity *result : results) {
                meet(operand, *result);
            }
        }
    }

    /// We don't need any special handling of branch operands.
    void visitBranchOperand(OpOperand &operand) override{};

    /// In general we can't reason about activity at arbitrary exit states.
    void setToExitState(BackwardActivity *lattice) override { lattice->meet(ValueActivity()); }
};

//===----------------------------------------------------------------------===//
// ActivityAnalyzer
//===----------------------------------------------------------------------===//

catalyst::gradient::ActivityAnalyzer::ActivityAnalyzer(FunctionOpInterface callee,
                                                       ArrayRef<size_t> diffArgIndices)
{
    SymbolTableCollection symbolTable;
    solver.load<ForwardActivityAnalysis>();
    solver.load<BackwardActivityAnalysis>(symbolTable);

    // These are required by the dataflow framework to traverse region control flow.
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();

    // Initialize the activity states of the arguments and function returns
    DenseSet<size_t> activeArgs{diffArgIndices.begin(), diffArgIndices.end()};
    for (BlockArgument arg : callee.getArguments()) {
        ForwardActivity *argState = solver.getOrCreateState<ForwardActivity>(arg);
        argState->join(activeArgs.contains(arg.getArgNumber()) ? ValueActivity::getActive()
                                                               : ValueActivity::getConstant());
    }

    for (Operation &op : callee.getFunctionBody().getOps()) {
        if (op.hasTrait<OpTrait::ReturnLike>()) {
            // Assume that all function returns are active.
            for (auto operand : op.getOperands()) {
                BackwardActivity *returnState = solver.getOrCreateState<BackwardActivity>(operand);
                returnState->meet(ValueActivity::getActive());
            }
        }
    }

    if (failed(solver.initializeAndRun(callee->getParentOfType<ModuleOp>()))) {
        assert(false && "dataflow failed");
    }

    // Print the results
    for (BlockArgument arg : callee.getArguments()) {
        Attribute label = callee.getArgAttr(arg.getArgNumber(), "activity.id");
        if (label) {
            ForwardActivity *fwdState = solver.getOrCreateState<ForwardActivity>(arg);
            BackwardActivity *bwdState = solver.getOrCreateState<BackwardActivity>(arg);

            errs() << label << ": " << (isActive(arg) ? "Active" : "Constant") << " (fwd "
                   << fwdState->getValue() << " bwd " << bwdState->getValue() << ")\n";
        }
    }

    callee.walk([this](Operation *op) {
        if (op->hasAttr("activity.id")) {
            errs() << op->getAttr("activity.id") << ": ";
            for (OpResult result : op->getResults()) {
                ForwardActivity *fwdState = solver.getOrCreateState<ForwardActivity>(result);
                BackwardActivity *bwdState = solver.getOrCreateState<BackwardActivity>(result);

                errs() << (isActive(result) ? "Active" : "Constant") << " (fwd "
                       << fwdState->getValue() << " bwd " << bwdState->getValue() << ") ";
            }
            errs() << "\n";
        }
    });
}

bool catalyst::gradient::ActivityAnalyzer::isActive(Value value) const
{
    auto *forwardState = solver.lookupState<ForwardActivity>(value);
    auto *backwardState = solver.lookupState<BackwardActivity>(value);

    if (!(forwardState && backwardState)) {
        return false;
    }

    // A value is overall active iff it is both forward and backward active.
    return forwardState->getValue().isActive() && backwardState->getValue().isActive();
}
