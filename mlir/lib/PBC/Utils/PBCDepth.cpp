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
#include <map>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "Catalyst/Analysis/ResourceInterfaces.h"
#include "Catalyst/Utils/SCFUtils.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Utils/PBCWireIndex.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst::pbc {
namespace {

bool isQuantum(Type type) { return isa<quantum::QubitType, quantum::QuregType>(type); }

bool isQuantumOperation(Operation *op) {
    return llvm::any_of(op->getOperandTypes(), isQuantum) ||
           llvm::any_of(op->getResultTypes(), isQuantum) ||
           (op->getDialect() && (op->getDialect()->getNamespace() == "quantum" ||
                                 op->getDialect()->getNamespace() == "pbc" ||
                                 op->getDialect()->getNamespace() == "qref" ||
                                 op->getDialect()->getNamespace() == "mbqc"));
}

// LowerPBCInitOps represents prepare/fabricate as fresh allocations followed by
// these gates. They retain the zero PBC-layer cost of the original initialization.
// Do not extend this exemption to gates acting on computational qubits.
bool isPBCInitializationGate(quantum::CustomOp gate) {
    auto isSingleQubit = [](quantum::CustomOp op) {
        return op.getInQubits().size() == 1 && op.getOutQubits().size() == 1 &&
               op.getInCtrlQubits().empty() && op.getParams().empty() &&
               op.getInQubits().front().hasOneUse();
    };
    if (!isSingleQubit(gate)) {
        return false;
    }
    Value input = gate.getInQubits().front();
    StringRef name = gate.getGateName();
    if (input.getDefiningOp<quantum::AllocQubitOp>()) {
        return !gate.getAdjoint() && (name == "Hadamard" || name == "PauliX");
    }
    auto hadamard = input.getDefiningOp<quantum::CustomOp>();
    if (!hadamard || !isSingleQubit(hadamard) || hadamard.getGateName() != "Hadamard" ||
        hadamard.getAdjoint() ||
        !hadamard.getInQubits().front().getDefiningOp<quantum::AllocQubitOp>()) {
        return false;
    }
    return name == "S" || name == "T" || (name == "PauliZ" && !gate.getAdjoint());
}

struct Register {
    Value root;
    llvm::DenseMap<int64_t, unsigned> slots;
    std::optional<int64_t> size;
};

struct Qubit {
    // A dynamic extraction may refer to any slot in a fixed-size register.
    // Keep the slot order so that insertion can detect register permutations.
    llvm::SmallVector<unsigned, 1> aliases;
    Value registerRoot;
    Value index;
};

struct Footprint {
    llvm::DenseSet<unsigned> touched;
    double depth = 0;
    bool hasPBC = false;
};

struct CallOutput {
    std::variant<std::monostate, Qubit, Register> identity;
    Attribute constant;
};

struct CallSummary : Footprint {
    llvm::SmallVector<CallOutput> outputs;
};

using CallKey = std::vector<uintptr_t>;

struct Context {
    // Call DAGs can expand exponentially. This optional estimate must not make
    // resource analysis unbounded; exceeding these limits omits the total.
    // Application workflows can exceed 100000 distinct visits even with caching.
    static constexpr unsigned maxOperations = 1000000;
    static constexpr unsigned maxCallDepth = 256;
    static constexpr unsigned maxRegisterSize = 100000;
    unsigned operations = 0;
    unsigned nextQubit = 0;
    llvm::DenseMap<std::pair<Value, int64_t>, unsigned> registerQubits;
    llvm::DenseSet<Operation *> activeCalls;
    std::map<CallKey, CallSummary> calls;
    llvm::DenseMap<Operation *, bool> classicalFunctions;
    llvm::DenseSet<Operation *> activeClassicalCalls;
};

bool isClassicalRegion(Operation &op, Context &context) {
    return !op.walk([&](Operation *nested) {
                  if (++context.operations > Context::maxOperations || isQuantumOperation(nested)) {
                      return WalkResult::interrupt();
                  }
                  if (!isa<CallOpInterface>(nested)) {
                      return WalkResult::advance();
                  }
                  auto call = dyn_cast<func::CallOp>(nested);
                  auto callee = call ? SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                                           call, call.getCalleeAttr())
                                     : func::FuncOp{};
                  if (!callee || callee.isDeclaration()) {
                      return WalkResult::interrupt();
                  }
                  if (auto it = context.classicalFunctions.find(callee);
                      it != context.classicalFunctions.end()) {
                      return it->second ? WalkResult::advance() : WalkResult::interrupt();
                  }
                  if (context.activeClassicalCalls.size() >= Context::maxCallDepth ||
                      !context.activeClassicalCalls.insert(callee).second) {
                      return WalkResult::interrupt();
                  }
                  bool classical = isClassicalRegion(*callee, context);
                  context.activeClassicalCalls.erase(callee);
                  context.classicalFunctions[callee] = classical;
                  return classical ? WalkResult::advance() : WalkResult::interrupt();
              }).wasInterrupted();
}

// Qubit identity is distinct from SSA identity: insert/extract and region boundaries
// must not turn successive versions of the same physical qubit into parallel wires.
class Schedule : Footprint {
    Context &context;
    llvm::DenseMap<Value, Qubit> qubits;
    llvm::DenseMap<Value, Register> registers;
    llvm::DenseMap<unsigned, double> clocks;
    WireIndexResolver wireIndices;

    std::optional<int64_t> index(std::optional<int64_t> attr, Value value) const {
        return attr ? attr : wireIndices.resolveIndex(value);
    }

    const Qubit &qubit(Value value) {
        auto [it, inserted] =
            qubits.try_emplace(value, Qubit{{context.nextQubit}, Value{}, Value{}});
        if (inserted) {
            ++context.nextQubit;
        }
        return it->second;
    }

    Register &reg(Value value) {
        return registers
            .try_emplace(value, Register{value, llvm::DenseMap<int64_t, unsigned>{}, std::nullopt})
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

    std::optional<llvm::SmallVector<unsigned, 1>> allSlots(Register &value) {
        if (!value.size || *value.size <= 0 || *value.size > Context::maxRegisterSize) {
            return std::nullopt;
        }
        llvm::SmallVector<unsigned, 1> result;
        result.reserve(*value.size);
        for (int64_t index = 0; index < *value.size; ++index) {
            result.push_back(slot(value, index));
        }
        return result;
    }

    double qubitTime(ValueRange values) {
        double result = 0;
        for (Value value : values) {
            for (unsigned id : qubit(value).aliases) {
                touched.insert(id);
                result = std::max(result, clocks.lookup(id));
            }
        }
        return result;
    }

    // A child starts at time zero but retains physical identities. Its footprint
    // and duration can then be placed on the parent's clocks without expanding loops.
    Schedule child() {
        Schedule result(context);
        result.qubits = qubits;
        result.registers = registers;
        result.wireIndices = wireIndices;
        return result;
    }

    void bind(Value target, Value source, Schedule &from, bool bindConstants = true) {
        if (isa<quantum::QubitType>(target.getType())) {
            Qubit value = from.qubit(source);
            qubits[target] = std::move(value);
        } else if (isa<quantum::QuregType>(target.getType())) {
            // Source and destination can share this map (e.g. a zero-trip loop).
            // Copy before insertion, which may invalidate references on rehash.
            Register value = from.reg(source);
            registers[target] = std::move(value);
        } else if (bindConstants) {
            wireIndices.bind(target, source, from.wireIndices);
        } else {
            wireIndices.bindConstant(target, {});
        }
    }

    bool sameIdentity(Value left, Schedule &other, Value right) {
        if (isa<quantum::QubitType>(left.getType())) {
            Qubit a = qubit(left);
            const Qubit &b = other.qubit(right);
            return a.aliases == b.aliases && a.registerRoot == b.registerRoot && a.index == b.index;
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

    double place(const Footprint &body, double count = 1) {
        double start = 0;
        for (unsigned id : body.touched) {
            start = std::max(start, clocks.lookup(id));
        }
        double end = start + count * body.depth;
        for (unsigned id : body.touched) {
            clocks[id] = end;
            touched.insert(id);
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
            }
            return true;
        }
        Schedule body = child();
        for (auto [arg, input] : llvm::zip(op.getRegionIterArgs(), op.getInitArgs())) {
            // A loop-carried classical value may change after the first iteration.
            body.bind(arg, input, *this, false);
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
        double end = place(body, *count);
        for (auto [result, output] : llvm::zip(op.getResults(), outputs)) {
            bind(result, output, body);
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
                // A result need not have the same constant value in both branches.
                bind(result, output, branch, false);
            }
            combined.depth = std::max(combined.depth, branch.depth);
            combined.hasPBC |= branch.hasPBC;
            combined.touched.insert(branch.touched.begin(), branch.touched.end());
            first = false;
        }
        place(combined);
        return true;
    }

    Attribute callConstant(Value value) const { return wireIndices.resolveConstant(value); }

    CallKey callKey(func::CallOp op, func::FuncOp callee) {
        CallKey key{reinterpret_cast<uintptr_t>(callee.getOperation())};
        for (Value input : op.getOperands()) {
            if (isa<quantum::QubitType>(input.getType())) {
                const Qubit &value = qubit(input);
                key.push_back(value.aliases.size());
                key.insert(key.end(), value.aliases.begin(), value.aliases.end());
                key.push_back(reinterpret_cast<uintptr_t>(value.registerRoot.getAsOpaquePointer()));
                key.push_back(reinterpret_cast<uintptr_t>(value.index.getAsOpaquePointer()));
            } else if (isa<quantum::QuregType>(input.getType())) {
                const Register &value = reg(input);
                key.push_back(reinterpret_cast<uintptr_t>(value.root.getAsOpaquePointer()));
                key.push_back(value.size.has_value());
                key.push_back(value.size.value_or(0));
                llvm::SmallVector<std::pair<int64_t, unsigned>> slots(value.slots.begin(),
                                                                      value.slots.end());
                llvm::sort(slots);
                key.push_back(slots.size());
                for (auto [index, id] : slots) {
                    key.push_back(index);
                    key.push_back(id);
                }
            } else {
                // Attributes are uniqued in the MLIR context. Unknown arguments
                // retain the conservative branch and alias behavior of the analysis.
                key.push_back(
                    reinterpret_cast<uintptr_t>(callConstant(input).getAsOpaquePointer()));
            }
        }
        return key;
    }

    CallSummary summarize(ValueRange outputs) {
        CallSummary summary;
        static_cast<Footprint &>(summary) = *this;
        for (Value output : outputs) {
            CallOutput result;
            if (isa<quantum::QubitType>(output.getType())) {
                result.identity = qubit(output);
            } else if (isa<quantum::QuregType>(output.getType())) {
                result.identity = reg(output);
            } else {
                result.constant = callConstant(output);
            }
            summary.outputs.push_back(std::move(result));
        }
        return summary;
    }

    void bindCallOutputs(ValueRange results, const CallSummary &summary) {
        for (auto [result, output] : llvm::zip(results, summary.outputs)) {
            if (const auto *value = std::get_if<Qubit>(&output.identity)) {
                qubits[result] = *value;
            } else if (const auto *value = std::get_if<Register>(&output.identity)) {
                registers[result] = *value;
            } else {
                wireIndices.bindConstant(result, output.constant);
            }
        }
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
            return true;
        }
        if (!callee.getBody().hasOneBlock() ||
            context.activeCalls.size() >= Context::maxCallDepth ||
            !context.activeCalls.insert(callee).second) {
            return false;
        }
        CallKey key = callKey(op, callee);
        if (auto it = context.calls.find(key); it != context.calls.end()) {
            context.activeCalls.erase(callee);
            place(it->second);
            bindCallOutputs(op.getResults(), it->second);
            return true;
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
        auto outputs = callee.getBody().front().getTerminator()->getOperands();
        // Reusing internal ancilla identities reserves them across calls. This
        // may overestimate depth but cannot introduce unsupported parallelism.
        auto [it, inserted] = context.calls.emplace(std::move(key), body.summarize(outputs));
        place(it->second);
        bindCallOutputs(op.getResults(), it->second);
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
            auto index = this->index(extract.getIdxAttr(), extract.getIdx());
            Register &value = reg(extract.getQreg());
            if (index) {
                if (*index < 0 || (value.size && *index >= *value.size)) {
                    return false;
                }
                qubits[extract.getQubit()] = Qubit{{slot(value, *index)}, Value{}, Value{}};
                return true;
            }
            auto aliases = allSlots(value);
            if (!aliases) {
                return false;
            }
            qubits[extract.getQubit()] = Qubit{std::move(*aliases), value.root, extract.getIdx()};
            return true;
        }
        if (auto insert = dyn_cast<quantum::InsertOp>(op)) {
            auto index = this->index(insert.getIdxAttr(), insert.getIdx());
            Register value = reg(insert.getInQreg());
            const Qubit &input = qubit(insert.getQubit());
            if (index) {
                if (*index < 0 || (value.size && *index >= *value.size) ||
                    input.aliases.size() != 1) {
                    return false;
                }
                value.slots[*index] = input.aliases.front();
            } else {
                // Only a matching extract/insert pair preserves every slot's
                // physical identity. A different index or replacement is unresolved.
                auto aliases = allSlots(value);
                if (!aliases || input.registerRoot != value.root ||
                    input.index != insert.getIdx() || input.aliases != *aliases) {
                    return false;
                }
            }
            registers[insert.getOutQreg()] = std::move(value);
            return true;
        }
        if (isa<quantum::AllocOp, quantum::AllocQubitOp, PrepareStateOp, FabricateOp>(op)) {
            for (Value result : op.getResults()) {
                if (isa<quantum::QubitType>(result.getType())) {
                    qubit(result);
                } else {
                    Register &value = reg(result);
                    if (auto alloc = dyn_cast<quantum::AllocOp>(op)) {
                        value.size = index(alloc.getNqubitsAttr(), alloc.getNqubits());
                    }
                }
            }
            return true;
        }
        if (auto gate = dyn_cast<quantum::CustomOp>(op)) {
            if (!isPBCInitializationGate(gate)) {
                return false;
            }
            bind(gate.getOutQubits().front(), gate.getInQubits().front(), *this);
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
            // Match PBCLayer's qubit-disjoint metric: classical conditions and
            // memory operations do not add quantum layers or synchronization.
            double end = qubitTime(inputs) + 1;
            for (auto [input, output] : llvm::zip(inputs, outputs)) {
                Qubit value = qubit(input);
                for (unsigned id : value.aliases) {
                    clocks[id] = end;
                }
                qubits[output] = std::move(value);
            }
            depth = std::max(depth, end);
            hasPBC = true;
            return true;
        }
        if (op.getNumRegions() != 0) {
            // Classical reductions/scatters can contain regions. They add no
            // quantum layers, but nested quantum work and unresolved calls must
            // not be discarded.
            return isClassicalRegion(op, context);
        }
        if (auto device = dyn_cast<quantum::DeviceInitOp>(op);
            device && device.getAutoQubitManagement()) {
            // Automatic management can grow a register after allocation. Its
            // initial size does not bound the aliases of a dynamic extraction.
            return false;
        }
        // These operations contribute no PBC layers.
        if (isa<quantum::DeallocOp, quantum::DeallocQubitOp, quantum::DeviceInitOp,
                quantum::DeviceReleaseOp, quantum::InitializeOp, quantum::FinalizeOp,
                quantum::NumQubitsOp, quantum::ComputationalBasisOp, quantum::NamedObsOp,
                quantum::HermitianOp, quantum::TensorOp, quantum::HamiltonianOp,
                ResourceMeasurementOpInterface>(op)) {
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
        return true;
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
