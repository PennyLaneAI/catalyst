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
#include <cstdint>
#include <limits>
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
    unsigned root;
    llvm::DenseMap<int64_t, unsigned> slots;
    std::optional<int64_t> size;
};

// A clock represents max(constant, input[i] + offsets[i]). Retaining every input
// dependency avoids reserving all of a callee's qubits for its longest path.
struct Clock {
    int64_t constant = 0;
    llvm::DenseMap<unsigned, int64_t> offsets;
    Clock() = default;
};

struct Transfer {
    llvm::DenseMap<unsigned, Clock> clocks;
    Clock depth;
    bool hasPBC = false;
};

struct Binding {
    std::variant<std::monostate, unsigned, Register> identity;
    Attribute constant;
};

struct CallSummary : Transfer {
    llvm::SmallVector<Binding> outputs;
};

using CallKey = std::vector<uintptr_t>;

struct Context {
    static constexpr unsigned maxOperations = 1000000;
    static constexpr unsigned maxCallDepth = 256;
    static constexpr uint64_t maxUnrolledIterations = 1024;
    unsigned operations = 0;
    unsigned nextQubit = 0;
    unsigned nextRegister = 0;
    bool overflow = false;
    llvm::DenseMap<std::pair<unsigned, int64_t>, unsigned> registerQubits;
    llvm::DenseMap<unsigned, unsigned> qubitRegisters;
    llvm::DenseSet<Operation *> activeCalls;
    std::map<CallKey, CallSummary> calls;
    llvm::DenseMap<Operation *, bool> classicalFunctions;
    llvm::DenseSet<Operation *> activeClassicalCalls;

    bool work() { return !overflow && ++operations <= maxOperations; }

    int64_t add(int64_t left, int64_t right) {
        if (right > std::numeric_limits<int64_t>::max() - left) {
            overflow = true;
            return 0;
        }
        return left + right;
    }
};

void mergeClock(Clock &target, const Clock &other) {
    target.constant = std::max(target.constant, other.constant);
    for (auto [input, offset] : other.offsets) {
        auto [it, inserted] = target.offsets.try_emplace(input, offset);
        if (!inserted) {
            it->second = std::max(it->second, offset);
        }
    }
}

bool equalClock(const Clock &left, const Clock &right) {
    if (left.constant != right.constant || left.offsets.size() != right.offsets.size()) {
        return false;
    }
    return llvm::all_of(left.offsets, [&](const auto &entry) {
        auto it = right.offsets.find(entry.first);
        return it != right.offsets.end() && it->second == entry.second;
    });
}

bool equalTransfer(const Transfer &left, const Transfer &right) {
    if (!equalClock(left.depth, right.depth) || left.clocks.size() != right.clocks.size()) {
        return false;
    }
    return llvm::all_of(left.clocks, [&](const auto &entry) {
        auto it = right.clocks.find(entry.first);
        return it != right.clocks.end() && equalClock(entry.second, it->second);
    });
}

Clock shifted(Clock value, int64_t offset, Context &context) {
    value.constant = context.add(value.constant, offset);
    for (auto &entry : value.offsets) {
        entry.second = context.add(entry.second, offset);
    }
    return value;
}

Clock substitute(const Clock &value, const llvm::DenseMap<unsigned, Clock> &inputs,
                 Context &context) {
    Clock result;
    result.constant = value.constant;
    for (auto [input, offset] : value.offsets) {
        if (!context.work()) {
            return {};
        }
        if (auto it = inputs.find(input); it != inputs.end()) {
            mergeClock(result, shifted(it->second, offset, context));
        } else {
            Clock identity;
            identity.offsets[input] = 0;
            mergeClock(result, shifted(std::move(identity), offset, context));
        }
    }
    return result;
}

// Compose two circuits in program order, including completed internal-ancilla
// paths. Qubit clocks absent from a transfer have the identity transformation.
Transfer compose(const Transfer &first, const Transfer &second, Context &context) {
    Transfer result = first;
    for (const auto &entry : second.clocks) {
        result.clocks[entry.first] = substitute(entry.second, first.clocks, context);
    }
    mergeClock(result.depth, substitute(second.depth, first.clocks, context));
    result.hasPBC |= second.hasPBC;
    return result;
}

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

bool containsPBC(Operation &op, Context &context, llvm::DenseSet<Operation *> &visited) {
    bool found = false;
    op.walk([&](Operation *nested) {
        if (!context.work()) {
            return WalkResult::interrupt();
        }
        if (isa<PBCOpInterface, SelectPPMeasurementOp>(nested)) {
            found = true;
            return WalkResult::interrupt();
        }
        if (auto call = dyn_cast<func::CallOp>(nested)) {
            auto callee =
                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
            if (callee && !callee.isDeclaration() && visited.size() < Context::maxCallDepth &&
                visited.insert(callee).second && containsPBC(*callee, context, visited)) {
                found = true;
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return found;
}

class Schedule : Transfer {
    Context &context;
    llvm::DenseMap<Value, unsigned> qubits;
    llvm::DenseMap<Value, Register> registers;
    llvm::DenseSet<unsigned> inputQubits;
    llvm::DenseSet<unsigned> inputRegisters;
    WireIndexResolver wireIndices;

    std::optional<int64_t> index(std::optional<int64_t> attr, Value value) const {
        return attr ? attr : wireIndices.resolveIndex(value);
    }

    unsigned qubit(Value value) {
        auto [it, inserted] = qubits.try_emplace(value, context.nextQubit);
        if (inserted) {
            ++context.nextQubit;
        }
        return it->second;
    }

    Register &reg(Value value) {
        auto [it, inserted] = registers.try_emplace(
            value,
            Register{context.nextRegister, llvm::DenseMap<int64_t, unsigned>{}, std::nullopt});
        if (inserted) {
            ++context.nextRegister;
        }
        return it->second;
    }

    unsigned slot(Register &value, int64_t index) {
        if (auto it = value.slots.find(index); it != value.slots.end()) {
            return it->second;
        }
        auto [it, inserted] = context.registerQubits.try_emplace(std::make_pair(value.root, index),
                                                                 context.nextQubit);
        if (inserted) {
            context.qubitRegisters[context.nextQubit++] = value.root;
        }
        return it->second;
    }

    bool isInput(unsigned id) const {
        if (inputQubits.contains(id)) {
            return true;
        }
        auto it = context.qubitRegisters.find(id);
        return it != context.qubitRegisters.end() && inputRegisters.contains(it->second);
    }

    Clock &clock(unsigned id) {
        auto [it, inserted] = clocks.try_emplace(id);
        if (inserted && isInput(id)) {
            it->second.offsets[id] = 0;
        }
        return it->second;
    }

    Schedule child() {
        Schedule result(context);
        result.qubits = qubits;
        result.registers = registers;
        result.wireIndices = wireIndices;
        for (auto [value, id] : qubits) {
            result.inputQubits.insert(id);
        }
        for (const auto &entry : registers) {
            result.inputRegisters.insert(entry.second.root);
            for (auto [index, id] : entry.second.slots) {
                result.inputQubits.insert(id);
            }
        }
        return result;
    }

    Binding binding(Value value) {
        Binding result;
        if (isa<quantum::QubitType>(value.getType())) {
            result.identity = qubit(value);
        } else if (isa<quantum::QuregType>(value.getType())) {
            result.identity = reg(value);
        } else {
            result.constant = wireIndices.resolveConstant(value);
        }
        return result;
    }

    void bind(Value target, const Binding &value) {
        if (auto *id = std::get_if<unsigned>(&value.identity)) {
            qubits[target] = *id;
        } else if (auto *reg = std::get_if<Register>(&value.identity)) {
            registers[target] = *reg;
        } else {
            wireIndices.bindConstant(target, value.constant);
        }
    }

    // Snapshot first: loop yields can permute arguments or refer to other
    // arguments whose bindings are overwritten by the same operation.
    void bindAll(ValueRange targets, ValueRange sources, Schedule &from) {
        llvm::SmallVector<Binding> values;
        for (Value value : sources) {
            values.push_back(from.binding(value));
        }
        for (auto [target, value] : llvm::zip(targets, values)) {
            bind(target, value);
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

    Transfer inputTransfer() const {
        Transfer result;
        result.depth = depth;
        result.hasPBC = hasPBC;
        for (const auto &entry : clocks) {
            if (isInput(entry.first)) {
                result.clocks.insert(entry);
            }
        }
        return result;
    }

    void markSkippedPBC(Operation &op) {
        llvm::DenseSet<Operation *> visited;
        hasPBC |= containsPBC(op, context, visited);
    }

    void apply(const Transfer &body) {
        llvm::DenseMap<unsigned, Clock> inputs;
        auto gather = [&](const Clock &value) {
            for (auto [id, offset] : value.offsets) {
                inputs.try_emplace(id, clock(id));
            }
        };
        gather(body.depth);
        for (const auto &entry : body.clocks) {
            gather(entry.second);
        }
        mergeClock(depth, substitute(body.depth, inputs, context));
        for (const auto &entry : body.clocks) {
            clocks[entry.first] = substitute(entry.second, inputs, context);
        }
        hasPBC |= body.hasPBC;
    }

    bool loop(scf::ForOp op) {
        // Iteration hints describe estimates, not the executed static graph.
        auto lower = wireIndices.resolveIndex(op.getLowerBound());
        auto upper = wireIndices.resolveIndex(op.getUpperBound());
        auto step = wireIndices.resolveIndex(op.getStep());
        if (!lower || !upper || !step || *step <= 0) {
            return false;
        }
        uint64_t count = 0;
        if (*upper > *lower) {
            uint64_t distance = static_cast<uint64_t>(*upper) - static_cast<uint64_t>(*lower);
            count = (distance - 1) / static_cast<uint64_t>(*step) + 1;
        }
        if (count == 0) {
            markSkippedPBC(*op);
            bindAll(op.getResults(), op.getInitArgs(), *this);
            return true;
        }
        auto outputs = op.getBody()->getTerminator()->getOperands();
        if (count <= Context::maxUnrolledIterations) {
            bindAll(op.getRegionIterArgs(), op.getInitArgs(), *this);
            int64_t induction = *lower;
            for (uint64_t iteration = 0; iteration < count; ++iteration) {
                wireIndices.bindConstant(
                    op.getInductionVar(),
                    IntegerAttr::get(op.getInductionVar().getType(), induction));
                if (!run(op.getBody())) {
                    return false;
                }
                if (iteration + 1 < count) {
                    bindAll(op.getRegionIterArgs(), outputs, *this);
                    // The next value remains below the representable upper bound.
                    induction += *step;
                }
            }
            bindAll(op.getResults(), outputs, *this);
            return true;
        }

        Schedule body = child();
        body.bindAll(op.getRegionIterArgs(), op.getInitArgs(), *this);
        body.wireIndices.bindConstant(op.getInductionVar(), {});
        for (auto [arg, output] : llvm::zip(op.getRegionIterArgs(), outputs)) {
            if (!isQuantum(arg.getType()) && arg != output) {
                // A changing classical value must not be assumed constant while
                // proving the repeated quantum graph independent of iteration.
                body.wireIndices.bindConstant(arg, {});
            }
        }
        if (!body.run(op.getBody())) {
            return false;
        }
        for (auto [input, output] : llvm::zip(op.getInitArgs(), outputs)) {
            if (!sameIdentity(input, body, output)) {
                return false;
            }
        }
        Transfer power = body.inputTransfer();
        Transfer repeated;
        while (count) {
            if (count & 1) {
                repeated = compose(repeated, power, context);
            }
            count >>= 1;
            if (count) {
                power = compose(power, power, context);
            }
            if (!context.work()) {
                return false;
            }
        }
        apply(repeated);
        bindAll(op.getResults(), outputs, body);
        return !context.overflow;
    }

    bool branches(Operation &op) {
        Region *selected = nullptr;
        if (auto conditional = dyn_cast<scf::IfOp>(op)) {
            if (auto condition = wireIndices.resolveIndex(conditional.getCondition())) {
                selected = *condition ? &conditional.getThenRegion() : &conditional.getElseRegion();
            }
        } else if (auto branch = dyn_cast<scf::IndexSwitchOp>(op)) {
            if (auto value = wireIndices.resolveIndex(branch.getArg())) {
                selected = &branch.getDefaultRegion();
                for (auto [index, item] : llvm::enumerate(branch.getCases())) {
                    if (item == *value) {
                        selected = &branch.getCaseRegions()[index];
                        break;
                    }
                }
            }
        }
        if (!selected) {
            std::optional<Schedule> first;
            Transfer common;
            llvm::SmallVector<Value> firstOutputs;
            llvm::SmallVector<Binding> commonOutputs;
            for (Region &region : op.getRegions()) {
                Schedule branch = child();
                llvm::SmallVector<Value> outputs;
                if (!region.empty()) {
                    if (!region.hasOneBlock() || !branch.run(&region.front())) {
                        return false;
                    }
                    outputs.append(region.front().getTerminator()->operand_begin(),
                                   region.front().getTerminator()->operand_end());
                }
                Transfer transfer = branch.inputTransfer();
                if (!first) {
                    common = std::move(transfer);
                    firstOutputs = outputs;
                    for (Value output : outputs) {
                        commonOutputs.push_back(branch.binding(output));
                    }
                    first.emplace(std::move(branch));
                    continue;
                }
                if (!equalTransfer(common, transfer) || firstOutputs.size() != outputs.size()) {
                    return false;
                }
                common.hasPBC |= transfer.hasPBC;
                for (auto [i, output] : llvm::enumerate(outputs)) {
                    if (!first->sameIdentity(firstOutputs[i], branch, output)) {
                        return false;
                    }
                    if (commonOutputs[i].constant != branch.binding(output).constant) {
                        commonOutputs[i].constant = {};
                    }
                }
            }
            apply(common);
            for (auto [result, value] : llvm::zip(op.getResults(), commonOutputs)) {
                bind(result, value);
            }
            return true;
        }
        markSkippedPBC(op);
        if (selected->empty()) {
            return op.getNumResults() == 0;
        }
        if (!selected->hasOneBlock() || !run(&selected->front())) {
            return false;
        }
        bindAll(op.getResults(), selected->front().getTerminator()->getOperands(), *this);
        return true;
    }

    CallKey callKey(func::CallOp op, func::FuncOp callee) {
        CallKey key{reinterpret_cast<uintptr_t>(callee.getOperation())};
        for (Value input : op.getOperands()) {
            if (isa<quantum::QubitType>(input.getType())) {
                key.push_back(qubit(input));
            } else if (isa<quantum::QuregType>(input.getType())) {
                const Register &value = reg(input);
                key.push_back(value.root);
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
                key.push_back(reinterpret_cast<uintptr_t>(
                    wireIndices.resolveConstant(input).getAsOpaquePointer()));
            }
        }
        return key;
    }

    void applyCall(ValueRange results, const CallSummary &summary) {
        apply(summary);
        for (auto [result, output] : llvm::zip(results, summary.outputs)) {
            bind(result, output);
        }
    }

    bool call(func::CallOp op) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        if (!callee || callee.isDeclaration() || !callee.getBody().hasOneBlock() ||
            context.activeCalls.size() >= Context::maxCallDepth ||
            !context.activeCalls.insert(callee).second) {
            return false;
        }
        CallKey key = callKey(op, callee);
        if (auto it = context.calls.find(key); it != context.calls.end()) {
            context.activeCalls.erase(callee);
            applyCall(op.getResults(), it->second);
            return true;
        }
        Schedule body = child();
        body.bindAll(callee.getArguments(), op.getOperands(), *this);
        bool supported = body.run(&callee.getBody().front());
        context.activeCalls.erase(callee);
        if (!supported) {
            return false;
        }
        CallSummary summary;
        static_cast<Transfer &>(summary) = body;
        bool cacheable = true;
        for (Value output : callee.getBody().front().getTerminator()->getOperands()) {
            Binding value = body.binding(output);
            if (auto *id = std::get_if<unsigned>(&value.identity)) {
                cacheable &= body.isInput(*id);
            } else if (auto *valueReg = std::get_if<Register>(&value.identity)) {
                cacheable &= body.inputRegisters.contains(valueReg->root);
                for (auto [index, id] : valueReg->slots) {
                    cacheable &= body.isInput(id);
                }
            }
            summary.outputs.push_back(std::move(value));
        }
        // Internal ancillas begin at zero in every invocation. Escaping fresh
        // identities must be allocated afresh, so those calls are not cached.
        applyCall(op.getResults(), summary);
        if (cacheable) {
            context.calls.emplace(std::move(key), std::move(summary));
        }
        return true;
    }

    bool operation(Operation &op) {
        if (!context.work()) {
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
            auto resolved = index(extract.getIdxAttr(), extract.getIdx());
            Register &value = reg(extract.getQreg());
            if (!resolved || *resolved < 0 || (value.size && *resolved >= *value.size)) {
                return false;
            }
            qubits[extract.getQubit()] = slot(value, *resolved);
            return true;
        }
        if (auto insert = dyn_cast<quantum::InsertOp>(op)) {
            auto resolved = index(insert.getIdxAttr(), insert.getIdx());
            Register value = reg(insert.getInQreg());
            if (!resolved || *resolved < 0 || (value.size && *resolved >= *value.size)) {
                return false;
            }
            value.slots[*resolved] = qubit(insert.getQubit());
            registers[insert.getOutQreg()] = std::move(value);
            return true;
        }
        if (auto alloc = dyn_cast<quantum::AllocOp>(op)) {
            auto size = index(alloc.getNqubitsAttr(), alloc.getNqubits());
            if (!size || *size < 0) {
                return false;
            }
            registers[alloc.getQreg()] =
                Register{context.nextRegister++, llvm::DenseMap<int64_t, unsigned>{}, size};
            return true;
        }
        if (isa<quantum::AllocQubitOp, PrepareStateOp, FabricateOp>(op)) {
            for (Value result : op.getResults()) {
                qubits[result] = context.nextQubit++;
            }
            return true;
        }
        if (auto gate = dyn_cast<quantum::CustomOp>(op)) {
            if (!isPBCInitializationGate(gate)) {
                return false;
            }
            qubits[gate.getOutQubits().front()] = qubit(gate.getInQubits().front());
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
            Clock end;
            llvm::SmallVector<unsigned> ids;
            for (Value input : inputs) {
                unsigned id = qubit(input);
                ids.push_back(id);
                mergeClock(end, clock(id));
            }
            end = shifted(std::move(end), 1, context);
            for (auto [id, output] : llvm::zip(ids, outputs)) {
                clocks[id] = end;
                qubits[output] = id;
            }
            mergeClock(depth, end);
            hasPBC = true;
            return !context.overflow;
        }
        if (op.getNumRegions() != 0) {
            return isClassicalRegion(op, context);
        }
        if (auto device = dyn_cast<quantum::DeviceInitOp>(op);
            device && device.getAutoQubitManagement()) {
            return false;
        }
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
        return !isQuantumOperation(&op);
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
        return context.work();
    }

    std::optional<int64_t> result() const {
        if (!hasPBC || context.overflow || context.operations > Context::maxOperations) {
            return std::nullopt;
        }
        int64_t result = depth.constant;
        for (auto [input, offset] : depth.offsets) {
            result = std::max(result, offset);
        }
        return result;
    }
};

} // namespace

std::optional<int64_t> computeQubitDisjointDepthTotal(Block *block) {
    if (!block->getParent()->hasOneBlock()) {
        return std::nullopt;
    }
    Context context;
    Schedule schedule(context);
    return schedule.run(block) ? schedule.result() : std::nullopt;
}

} // namespace catalyst::pbc
