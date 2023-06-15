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

/**
 * A dataflow analysis that determines the set of values that depend on measurements and mid-circuit
 * measurements. This traverses def-use chains to answer questions such as:
 *
 * - Does this value depend on a quantum measurement?
 * - Does this value depend on a mid-circuit measurement?
 *
 * Here, a mid-circuit measurement is a measurement that is later used by a quantum operation.
 * Measurements that undergo purely classical post-processing are just measurements, not mid-circuit
 * measurements.
 */
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "Quantum/Analysis/QuantumDependenceAnalysis.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

#define DEBUG_TYPE "qdep-analysis"

using namespace mlir;
using namespace mlir::dataflow;
using namespace catalyst;

class QuantumDependence {
  public:
    enum class DependenceType {
        NoDependence,
        DependsOnMeasurement,
        DependsOnMidCircuitMeasurement,
    };

    explicit QuantumDependence() = default;

    QuantumDependence(DependenceType dependenceType) : dependenceType(dependenceType) {}

    bool isUninitialized() const { return !dependenceType.has_value(); }

    bool dependsOnMCM() const
    {
        return dependenceType == DependenceType::DependsOnMidCircuitMeasurement;
    }

    bool dependsOnMeasurement() const
    {
        return dependenceType == DependenceType::DependsOnMeasurement || dependsOnMCM();
    }

    bool operator==(const QuantumDependence &rhs) const
    {
        return dependenceType == rhs.dependenceType;
    }

    void print(raw_ostream &os) const
    {
        if (isUninitialized()) {
            os << "<UNINITIALIZED>";
            return;
        }

        switch (dependenceType.value()) {
        case DependenceType::NoDependence:
            os << "<does not depend on measurement>";
            break;
        case DependenceType::DependsOnMeasurement:
            os << "<depends on measurement>";
            break;
        case DependenceType::DependsOnMidCircuitMeasurement:
            os << "<depends on mid-circuit measurement>";
            break;
        }
    }

    friend raw_ostream &operator<<(raw_ostream &os, const QuantumDependence &mcm)
    {
        mcm.print(os);
        return os;
    }

    /// The state where the mid-circuit measurement value is uninitialized. This happens when the
    /// state hasn't been set during the analysis.
    static QuantumDependence getUninitialized() { return QuantumDependence{}; }

    static QuantumDependence getNoDependence()
    {
        return QuantumDependence{DependenceType::NoDependence};
    }
    static QuantumDependence getDependsOnMeasurement()
    {
        return QuantumDependence{DependenceType::DependsOnMeasurement};
    }
    static QuantumDependence getDependsOnMidCircuitMeasurement()
    {
        return QuantumDependence{DependenceType::DependsOnMidCircuitMeasurement};
    }

    static QuantumDependence join(const QuantumDependence &lhs, const QuantumDependence &rhs)
    {
        if (lhs.isUninitialized()) {
            return rhs;
        }
        if (rhs.isUninitialized()) {
            return lhs;
        }

        if (lhs.dependsOnMCM() || rhs.dependsOnMCM()) {
            return QuantumDependence::getDependsOnMidCircuitMeasurement();
        }
        if (lhs.dependsOnMeasurement() || rhs.dependsOnMeasurement()) {
            return QuantumDependence::getDependsOnMeasurement();
        }

        return QuantumDependence::getNoDependence();
    }

  private:
    llvm::Optional<DependenceType> dependenceType;
};

struct QuantumDependenceDataFlowAnalysis
    : public SparseDataFlowAnalysis<Lattice<QuantumDependence>> {
    using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

    void visitOperation(Operation *op, ArrayRef<const Lattice<QuantumDependence> *> operands,
                        ArrayRef<Lattice<QuantumDependence> *> results) override
    {
        LLVM_DEBUG(llvm::dbgs() << "QDEP: Visiting operation " << *op << "\n");
        if (auto measurement = dyn_cast<quantum::MeasurementProcess>(op)) {
            LLVM_DEBUG(llvm::dbgs() << "QDEP: Found a measurement " << measurement << "\n");
            for (Lattice<QuantumDependence> *result : results) {
                propagateIfChanged(result,
                                   result->join(QuantumDependence::getDependsOnMeasurement()));
            }
        }
        else if (auto quantumOp = dyn_cast<quantum::QuantumGate>(op)) {
            if (llvm::any_of(operands, [&](const Lattice<QuantumDependence> *operand) {
                    return operand->getValue().dependsOnMeasurement();
                })) {
                LLVM_DEBUG(llvm::dbgs() << "QDEP: Found a gate that depends on a measurement "
                                        << quantumOp << "\n");
                for (auto *result : results) {
                    propagateIfChanged(
                        result,
                        result->join(QuantumDependence::getDependsOnMidCircuitMeasurement()));
                }
            }
        }
        else {
            // Set the result lattice to the join of all of the operand lattices.
            QuantumDependence operandAgg;
            for (auto *operand : operands) {
                operandAgg = QuantumDependence::join(operandAgg, operand->getValue());
            }
            for (auto *result : results) {
                propagateIfChanged(result, result->join(operandAgg));
            }
        }
    }

    void setToEntryState(Lattice<QuantumDependence> *lattice) override
    {
        propagateIfChanged(lattice, lattice->join(QuantumDependence::getNoDependence()));
    }
};

quantum::QuantumDependenceAnalysis::QuantumDependenceAnalysis(Operation *op)
{
    solver.load<QuantumDependenceDataFlowAnalysis>();
    // Dead code analysis and sparse constant propagation are required to be loaded for the region
    // traversal of MLIR's dataflow framework to work.
    solver.load<SparseConstantPropagation>();
    solver.load<DeadCodeAnalysis>();
    if (failed(solver.initializeAndRun(op))) {
        // TODO(jacob): error handling
        assert(0 && "unexpected dataflow failure\n");
    }
}

bool quantum::QuantumDependenceAnalysis::dependsOnMeasurement(Value value)
{
    auto *state = solver.lookupState<Lattice<QuantumDependence>>(value);
    if (state) {
        // A mid-circuit measurement implies a measurement.
        return state->getValue().dependsOnMeasurement() || state->getValue().dependsOnMCM();
    }

    // If not visited, we assume it does not depend on a measurement.
    return false;
}

bool quantum::QuantumDependenceAnalysis::dependsOnMeasurement(Operation *op)
{
    return llvm::any_of(op->getOperands(),
                        [this](Value operand) { return dependsOnMeasurement(operand); });
}

bool quantum::QuantumDependenceAnalysis::dependsOnMidCircuitMeasurement(Value value)
{
    auto *state = solver.lookupState<Lattice<QuantumDependence>>(value);
    if (state) {
        return state->getValue().dependsOnMCM();
    }

    // If not visited, we assume it does not depend on a mid-circuit measurement.
    return false;
}

bool quantum::QuantumDependenceAnalysis::isFunctionLive(FunctionOpInterface funcOp)
{
    auto *state = solver.getOrCreateState<Executable>(&funcOp.getFunctionBody().front());
    if (state) {
        return state->isLive();
    }

    return false;
}
