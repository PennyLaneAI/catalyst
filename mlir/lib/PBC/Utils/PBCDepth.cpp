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

#include "PBC/Utils/PBCDepth.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Catalyst/Analysis/ResourceInterfaces.h"
#include "Catalyst/Utils/SCFUtils.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst::pbc {
namespace {

bool isQuantum(Type type) { return isa<quantum::QubitType, quantum::QuregType>(type); }

std::optional<int64_t> constantIndex(std::optional<int64_t> attr, Value value) {
    if (attr) {
        return attr;
    }
    llvm::APInt index;
    if (value && matchPattern(value, m_ConstantInt(&index)) && index.isSignedIntN(64)) {
        return index.getSExtValue();
    }
    return std::nullopt;
}

struct Register {
    Value root;
    llvm::DenseMap<int64_t, unsigned> slots;
};

struct Context {
    // Call DAGs can expand exponentially. This optional estimate must not make
    // resource analysis unbounded; exceeding either limit omits the total.
    static constexpr unsigned maxOperations = 100000;
    static constexpr unsigned maxCallDepth = 256;
    unsigned operations = 0;
    unsigned nextQubit = 0;
    llvm::DenseMap<std::pair<Value, int64_t>, unsigned> registerQubits;
    llvm::DenseSet<Operation *> activeCalls;
};

// Qubit identity is distinct from SSA identity: insert/extract and region boundaries
// must not turn successive versions of the same physical qubit into parallel wires.
class Schedule {
    Context &context;
    llvm::DenseMap<Value, unsigned> qubits;
    llvm::DenseMap<Value, Register> registers;
    llvm::DenseMap<Value, double> classical;
    llvm::DenseMap<unsigned, double> clocks;
    llvm::DenseSet<unsigned> touched;
    double memoryClock = 0;
    bool accessesMemory = false;
    double depth = 0;
    bool hasPBC = false;

    unsigned qubit(Value value) {
        auto [it, inserted] = qubits.try_emplace(value, context.nextQubit);
        if (inserted) {
            ++context.nextQubit;
        }
        return it->second;
    }

    Register &reg(Value value) {
        return registers.try_emplace(value, Register{value, llvm::DenseMap<int64_t, unsigned>{}})
            .first->second;
    }

    unsigned slot(Register &value, int64_t index) {
        if (auto it = value.slots.find(index); it != value.slots.end()) {
            return it->second;
        }
        auto [it, inserted] =
            context.registerQubits.try_emplace({value.root, index}, context.nextQubit);
        if (inserted) {
            ++context.nextQubit;
        }
        return it->second;
    }

    double time(Value value) {
        if (isa<quantum::QubitType>(value.getType())) {
            unsigned id = qubit(value);
            touched.insert(id);
            return clocks.lookup(id);
        }
        if (isa<quantum::QuregType>(value.getType())) {
            Register &valueReg = reg(value);
            double result = 0;
            for (auto [key, id] : context.registerQubits) {
                if (key.first == valueReg.root) {
                    unsigned current = slot(valueReg, key.second);
                    touched.insert(current);
                    result = std::max(result, clocks.lookup(current));
                }
            }
            for (auto [index, id] : valueReg.slots) {
                touched.insert(id);
                result = std::max(result, clocks.lookup(id));
            }
            return result;
        }
        return classical.lookup(value);
    }

    double operandTime(ValueRange values) {
        double result = 0;
        for (Value value : values) {
            result = std::max(result, time(value));
        }
        return result;
    }

    // A child starts at time zero but retains physical identities. Its footprint
    // and duration can then be placed on the parent's clocks without expanding loops.
    Schedule child() {
        Schedule result(context);
        result.qubits = qubits;
        result.registers = registers;
        return result;
    }

    void bind(Value target, Value source, Schedule &from) {
        if (isa<quantum::QubitType>(target.getType())) {
            qubits[target] = from.qubit(source);
        } else if (isa<quantum::QuregType>(target.getType())) {
            // Source and destination can share this map (e.g. a zero-trip loop).
            // Copy before insertion, which may invalidate references on rehash.
            Register value = from.reg(source);
            registers[target] = std::move(value);
        }
    }

    bool sameIdentity(Value left, Schedule &other, Value right) {
        if (isa<quantum::QubitType>(left.getType())) {
            return qubit(left) == other.qubit(right);
        }
        if (isa<quantum::QuregType>(left.getType())) {
            Register &a = reg(left);
            Register &b = other.reg(right);
            if (a.root != b.root) {
                return false;
            }
            for (auto [index, id] : a.slots) {
                if (id != other.slot(b, index)) {
                    return false;
                }
            }
            for (auto [index, id] : b.slots) {
                if (id != slot(a, index)) {
                    return false;
                }
            }
        }
        return true;
    }

    double dependencies(Operation &op) {
        double result = 0;
        auto read = [&](Operation *nested) {
            for (Value value : nested->getOperands()) {
                // Quantum captures are accounted for by the child's actual footprint.
                if (!isQuantum(value.getType())) {
                    result = std::max(result, classical.lookup(value));
                }
            }
        };
        op.walk(read);
        return result;
    }

    double place(Schedule &body, double dependency, double count = 1) {
        double start = dependency;
        for (unsigned id : body.touched) {
            start = std::max(start, clocks.lookup(id));
        }
        if (body.accessesMemory) {
            start = std::max(start, memoryClock);
            accessesMemory = true;
        }
        double end = start + count * body.depth;
        for (unsigned id : body.touched) {
            clocks[id] = end;
            touched.insert(id);
        }
        if (body.accessesMemory) {
            // A memory dependence created within a repeated region may occur in
            // its final iteration. Reserving until the region ends is conservative.
            memoryClock = std::max(memoryClock, body.memoryClock > 0 ? end : start);
        }
        hasPBC |= body.hasPBC;
        if (body.hasPBC) {
            depth = std::max(depth, end);
        }
        return end;
    }

    bool loop(scf::ForOp op) {
        auto count = resolveForLoopTripCount(op);
        if (!count || !std::isfinite(*count) || *count < 0) {
            return false;
        }
        if (*count == 0) {
            for (auto [result, input] : llvm::zip(op.getResults(), op.getInitArgs())) {
                bind(result, input, *this);
                classical[result] = time(input);
            }
            return true;
        }
        Schedule body = child();
        for (auto [arg, input] : llvm::zip(op.getRegionIterArgs(), op.getInitArgs())) {
            body.bind(arg, input, *this);
        }
        if (!body.run(op.getBody())) {
            return false;
        }
        auto outputs = op.getBody()->getTerminator()->getOperands();
        for (auto [input, output] : llvm::zip(op.getInitArgs(), outputs)) {
            if (!sameIdentity(input, body, output)) {
                return false;
            }
        }
        double end = place(body, dependencies(*op), *count);
        for (auto [result, output] : llvm::zip(op.getResults(), outputs)) {
            bind(result, output, body);
            classical[result] = end;
        }
        return std::isfinite(end);
    }

    bool branches(Operation &op) {
        Schedule combined = child();
        bool first = true;
        for (Region &region : op.getRegions()) {
            if (region.empty()) {
                continue;
            }
            if (!region.hasOneBlock()) {
                return false;
            }
            Schedule branch = child();
            if (!branch.run(&region.front())) {
                return false;
            }
            auto outputs = region.front().getTerminator()->getOperands();
            for (auto [result, output] : llvm::zip(op.getResults(), outputs)) {
                if (!first && !sameIdentity(result, branch, output)) {
                    return false;
                }
                bind(result, output, branch);
            }
            combined.depth = std::max(combined.depth, branch.depth);
            combined.memoryClock = std::max(combined.memoryClock, branch.memoryClock);
            combined.accessesMemory |= branch.accessesMemory;
            combined.hasPBC |= branch.hasPBC;
            combined.touched.insert(branch.touched.begin(), branch.touched.end());
            first = false;
        }
        double end = place(combined, dependencies(op));
        for (Value result : op.getResults()) {
            classical[result] = end;
        }
        return true;
    }

    bool call(func::CallOp op) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        if (!callee) {
            return false;
        }
        if (callee.isDeclaration()) {
            if (llvm::any_of(op.getOperandTypes(), isQuantum) ||
                llvm::any_of(op.getResultTypes(), isQuantum)) {
                return false;
            }
            return plain(*op);
        }
        if (!callee.getBody().hasOneBlock() ||
            context.activeCalls.size() >= Context::maxCallDepth ||
            !context.activeCalls.insert(callee).second) {
            return false;
        }
        Schedule body = child();
        for (auto [arg, input] : llvm::zip(callee.getArguments(), op.getOperands())) {
            body.bind(arg, input, *this);
        }
        bool supported = body.run(&callee.getBody().front());
        context.activeCalls.erase(callee);
        if (!supported) {
            return false;
        }
        double end = place(body, dependencies(*op));
        auto outputs = callee.getBody().front().getTerminator()->getOperands();
        for (auto [result, output] : llvm::zip(op.getResults(), outputs)) {
            bind(result, output, body);
            classical[result] = end;
        }
        return true;
    }

    bool plain(Operation &op) {
        double ready = operandTime(op.getOperands());
        if (!isMemoryEffectFree(&op)) {
            accessesMemory = true;
            memoryClock = std::max(memoryClock, ready);
            ready = memoryClock;
        }
        for (Value result : op.getResults()) {
            classical[result] = ready;
        }
        return true;
    }

    bool operation(Operation &op) {
        if (++context.operations > Context::maxOperations) {
            return false;
        }
        if (op.hasTrait<OpTrait::IsTerminator>()) {
            return true;
        }
        if (auto loopOp = dyn_cast<scf::ForOp>(op)) {
            return loop(loopOp);
        }
        if (isa<scf::IfOp, scf::IndexSwitchOp>(op)) {
            return branches(op);
        }
        if (auto callOp = dyn_cast<func::CallOp>(op)) {
            return call(callOp);
        }
        if (auto extract = dyn_cast<quantum::ExtractOp>(op)) {
            auto index = constantIndex(extract.getIdxAttr(), extract.getIdx());
            if (!index) {
                return false;
            }
            qubits[extract.getQubit()] = slot(reg(extract.getQreg()), *index);
            return true;
        }
        if (auto insert = dyn_cast<quantum::InsertOp>(op)) {
            auto index = constantIndex(insert.getIdxAttr(), insert.getIdx());
            if (!index) {
                return false;
            }
            Register value = reg(insert.getInQreg());
            value.slots[*index] = qubit(insert.getQubit());
            registers[insert.getOutQreg()] = std::move(value);
            return true;
        }
        if (isa<quantum::AllocOp, quantum::AllocQubitOp, PrepareStateOp, FabricateOp>(op)) {
            // A measurement-dependent register size needs allocation readiness
            // for not-yet-materialized slots, which this fixed-wire analysis omits.
            if (operandTime(op.getOperands()) != 0) {
                return false;
            }
            for (Value result : op.getResults()) {
                if (isa<quantum::QubitType>(result.getType())) {
                    qubit(result);
                } else {
                    reg(result);
                }
            }
            return true;
        }
        ValueRange inputs, outputs;
        if (auto pbcOp = dyn_cast<PBCOpInterface>(op)) {
            inputs = pbcOp.getInQubits();
            outputs = pbcOp.getOutQubits();
        } else if (auto select = dyn_cast<SelectPPMeasurementOp>(op)) {
            inputs = select.getInQubits();
            outputs = select.getOutQubits();
        }
        if (!inputs.empty()) {
            double end = operandTime(op.getOperands()) + 1;
            for (auto [input, output] : llvm::zip(inputs, outputs)) {
                unsigned id = qubit(input);
                qubits[output] = id;
                clocks[id] = end;
            }
            for (Value result : op.getResults()) {
                classical[result] = end;
            }
            depth = std::max(depth, end);
            hasPBC = true;
            return true;
        }
        if (op.getNumRegions() != 0) {
            return false;
        }
        // These operations contribute no PBC layers. Terminal measurement results
        // nevertheless carry the readiness of the observed quantum state.
        if (isa<quantum::DeallocOp, quantum::DeallocQubitOp, quantum::DeviceInitOp,
                quantum::DeviceReleaseOp, quantum::InitializeOp, quantum::FinalizeOp,
                quantum::NumQubitsOp, quantum::ComputationalBasisOp, quantum::NamedObsOp,
                quantum::HermitianOp, quantum::TensorOp, quantum::HamiltonianOp,
                ResourceMeasurementOpInterface>(op)) {
            double ready = operandTime(op.getOperands());
            for (Value result : op.getResults()) {
                classical[result] = ready;
            }
            return true;
        }
        if (auto phase = dyn_cast<quantum::GlobalPhaseOp>(op)) {
            return phase.getInCtrlQubits().empty();
        }
        if (op.getDialect() && (op.getDialect()->getNamespace() == "quantum" ||
                                op.getDialect()->getNamespace() == "pbc" ||
                                op.getDialect()->getNamespace() == "qref" ||
                                op.getDialect()->getNamespace() == "mbqc")) {
            return false;
        }
        if (llvm::any_of(op.getOperandTypes(), isQuantum) ||
            llvm::any_of(op.getResultTypes(), isQuantum)) {
            return false;
        }
        return plain(op);
    }

  public:
    explicit Schedule(Context &context) : context(context) {}

    bool run(Block *block) {
        for (BlockArgument arg : block->getArguments()) {
            if (isa<quantum::QubitType>(arg.getType())) {
                qubit(arg);
            } else if (isa<quantum::QuregType>(arg.getType())) {
                reg(arg);
            }
        }
        for (Operation &op : *block) {
            if (!operation(op)) {
                return false;
            }
        }
        return true;
    }

    std::optional<double> result() const {
        return hasPBC && std::isfinite(depth) ? std::optional<double>(depth) : std::nullopt;
    }
};

} // namespace

std::optional<double> computeQubitDisjointDepthTotal(Block *block) {
    if (!block->getParent()->hasOneBlock()) {
        return std::nullopt;
    }
    Context context;
    Schedule schedule(context);
    return schedule.run(block) ? schedule.result() : std::nullopt;
}

} // namespace catalyst::pbc
