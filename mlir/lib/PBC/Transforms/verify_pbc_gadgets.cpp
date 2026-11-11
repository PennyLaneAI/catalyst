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

#define DEBUG_TYPE "verify-pbc-gadgets"

#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_VERIFYPBCGADGETSPASS
#define GEN_PASS_DEF_VERIFYPBCGADGETSPASS
#include "PBC/Transforms/Passes.h.inc"

} // namespace pbc
} // namespace catalyst

namespace {

using Complex = std::complex<double>;

/// The tolerance on the distance between a branch's unitary and the source rotation.
constexpr double tolerance = 1e-12;

/// The largest number of qubits the check simulates: twice the data qubits, plus auxiliaries.
constexpr unsigned maxQubits = 16;

/// Why a gadget could not be checked, or does not implement its source.
struct Finding {
    GadgetVerdict verdict;
    std::string reason{};
};

/// A state vector on a few qubits, qubit i being bit i of the index.
class State {
  public:
    explicit State(unsigned numQubits) : numQubits(numQubits), amplitudes(size_t{1} << numQubits) {}

    unsigned numQubits;
    std::vector<Complex> amplitudes;

    /// P |psi> for the Pauli word on the given qubits.
    std::vector<Complex> applyPauli(ArrayRef<char> word, ArrayRef<unsigned> qubits) const {
        size_t flip = 0;
        for (auto [pauli, qubit] : llvm::zip(word, qubits)) {
            if (pauli == 'X' || pauli == 'Y') {
                flip |= size_t{1} << qubit;
            }
        }
        std::vector<Complex> result(amplitudes.size());
        for (size_t k = 0; k < amplitudes.size(); k++) {
            Complex phase{1.0, 0.0};
            for (auto [pauli, qubit] : llvm::zip(word, qubits)) {
                const bool bit = (k >> qubit) & 1;
                if (pauli == 'Z') {
                    phase *= bit ? -1.0 : 1.0;
                } else if (pauli == 'Y') {
                    // Y|0> = i|1>, Y|1> = -i|0>
                    phase *= bit ? Complex{0.0, -1.0} : Complex{0.0, 1.0};
                }
            }
            result[k ^ flip] += phase * amplitudes[k];
        }
        return result;
    }

    /// psi <- a psi + b P psi.
    void applyLinear(ArrayRef<char> word, ArrayRef<unsigned> qubits, Complex a, Complex b) {
        std::vector<Complex> pauli = applyPauli(word, qubits);
        for (size_t k = 0; k < amplitudes.size(); k++) {
            amplitudes[k] = a * amplitudes[k] + b * pauli[k];
        }
    }

    double norm() const {
        double total = 0.0;
        for (Complex amplitude : amplitudes) {
            total += std::norm(amplitude);
        }
        return total;
    }

    /// Append a qubit in the state `single` as the new highest qubit.
    unsigned addQubit(std::array<Complex, 2> single) {
        std::vector<Complex> next(2 * amplitudes.size());
        for (size_t k = 0; k < amplitudes.size(); k++) {
            next[k] = amplitudes[k] * single[0];
            next[k + amplitudes.size()] = amplitudes[k] * single[1];
        }
        amplitudes = std::move(next);
        return numQubits++;
    }

    /// Remove `qubit`, which must be unentangled with the rest. Higher qubits move down by one.
    bool removeQubit(unsigned qubit) {
        const size_t bit = size_t{1} << qubit;
        // Reduced density matrix of the qubit.
        Complex rho[2][2] = {};
        for (size_t k = 0; k < amplitudes.size(); k++) {
            if (k & bit) {
                continue;
            }
            const Complex v[2] = {amplitudes[k], amplitudes[k | bit]};
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    rho[a][b] += v[a] * std::conj(v[b]);
                }
            }
        }
        const double trace = (rho[0][0] + rho[1][1]).real();
        const double det = std::abs(rho[0][0] * rho[1][1] - rho[0][1] * rho[1][0]);
        if (det > tolerance * trace * trace) {
            return false;
        }
        // The qubit's state is the dominant column of rho.
        const int column = rho[0][0].real() >= rho[1][1].real() ? 0 : 1;
        Complex phi[2] = {rho[0][column], rho[1][column]};
        const double scale = std::sqrt(std::norm(phi[0]) + std::norm(phi[1]));
        phi[0] /= scale;
        phi[1] /= scale;

        std::vector<Complex> next(amplitudes.size() / 2);
        for (size_t t = 0; t < next.size(); t++) {
            const size_t low = t & (bit - 1);
            const size_t k = ((t ^ low) << 1) | low;
            next[t] = std::conj(phi[0]) * amplitudes[k] + std::conj(phi[1]) * amplitudes[k | bit];
        }
        amplitudes = std::move(next);
        numQubits--;
        return true;
    }
};

std::array<Complex, 2> initialState(LogicalInitKind kind) {
    const double h = M_SQRT1_2;
    const Complex magic = std::polar(1.0, llvm::numbers::pi / 4);
    switch (kind) {
    case LogicalInitKind::zero:
        return {Complex{1.0}, Complex{0.0}};
    case LogicalInitKind::one:
        return {Complex{0.0}, Complex{1.0}};
    case LogicalInitKind::plus:
        return {Complex{h}, Complex{h}};
    case LogicalInitKind::minus:
        return {Complex{h}, Complex{-h}};
    case LogicalInitKind::plus_i:
        return {Complex{h}, Complex{0.0, h}};
    case LogicalInitKind::minus_i:
        return {Complex{h}, Complex{0.0, -h}};
    case LogicalInitKind::magic:
        return {Complex{h}, h * magic};
    case LogicalInitKind::magic_conj:
        return {Complex{h}, h * std::conj(magic)};
    }
    llvm_unreachable("unknown initial state");
}

SmallVector<char> pauliChars(ArrayAttr word) {
    SmallVector<char> chars;
    for (Attribute pauli : word) {
        chars.push_back(cast<StringAttr>(pauli).getValue().front());
    }
    return chars;
}

/// One run of a gadget's region along a fixed sequence of measurement outcomes.
class Branch {
  public:
    /// With `reduced`, the data qubits are one check qubit on which the source word is Z; every
    /// op must then act on the data as the identity or as the source word.
    Branch(GadgetOp gadget, ArrayRef<bool> outcomes, bool reduced)
        : outcomes(outcomes), state(0), reduced(reduced),
          source(pauliChars(gadget.getPauliProduct())) {
        // A Choi state: data qubit i is maximally entangled with reference qubit k + i.
        const unsigned k = reduced ? 1 : gadget.getInQubits().size();
        state = State(2 * k);
        const size_t d = size_t{1} << k;
        for (size_t i = 0; i < d; i++) {
            state.amplitudes[i | (i << k)] = 1.0 / std::sqrt(static_cast<double>(d));
        }
        for (auto [i, arg] : llvm::enumerate(gadget.getRegion().getArguments())) {
            if (reduced) {
                dataIndex[arg] = static_cast<unsigned>(i);
            } else {
                slots[arg] = static_cast<unsigned>(i);
            }
        }
    }

    /// Run the block; returns whether the run ended (false: it needs one more outcome).
    FailureOr<bool> run(Block &block, SmallVectorImpl<Value> &yielded) {
        for (Operation &op : block) {
            if (op.hasTrait<OpTrait::IsTerminator>()) {
                yielded.assign(op.getOperands().begin(), op.getOperands().end());
                return true;
            }
            FailureOr<bool> done = step(&op);
            if (failed(done) || !*done) {
                return done;
            }
        }
        return finding(GadgetVerdict::unchecked, "the region has no terminator");
    }

    ArrayRef<bool> outcomes;
    size_t measured{0};
    State state;
    DenseMap<Value, unsigned> slots;
    DenseMap<Value, bool> bits;
    std::optional<Finding> result;
    bool reduced;
    SmallVector<char> source;
    /// Reduced runs only: the data qubit each data value is.
    DenseMap<Value, unsigned> dataIndex;
    /// The run failed because an op does not fit the reduction.
    bool irreducible{false};

  private:
    FailureOr<bool> finding(GadgetVerdict verdict, const Twine &reason) {
        result = Finding{verdict, reason.str()};
        return failure();
    }

    FailureOr<bool> bit(Value value, bool &out) {
        if (auto it = bits.find(value); it != bits.end()) {
            out = it->second;
            return true;
        }
        if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
            if (auto attr = dyn_cast<IntegerAttr>(constant.getValue())) {
                out = attr.getValue().getBoolValue();
                return true;
            }
        }
        return finding(GadgetVerdict::unchecked, "a condition comes from outside the gadget");
    }

    FailureOr<SmallVector<unsigned>> qubitSlots(ValueRange qubits) {
        SmallVector<unsigned> result;
        for (Value qubit : qubits) {
            auto it = slots.find(qubit);
            if (it == slots.end()) {
                result.clear();
                (void)finding(GadgetVerdict::unchecked, "a qubit comes from outside the gadget");
                return failure();
            }
            result.push_back(it->second);
        }
        return result;
    }

    void forward(ValueRange from, ValueRange to) {
        for (auto [in, out] : llvm::zip(from, to)) {
            if (auto it = dataIndex.find(in); it != dataIndex.end()) {
                dataIndex[out] = it->second;
            } else {
                slots[out] = slots[in];
            }
        }
    }

    /// The word's Paulis on the check's qubits.
    FailureOr<std::pair<SmallVector<char>, SmallVector<unsigned>>> paulisOn(ArrayAttr word,
                                                                            ValueRange qubits) {
        SmallVector<char> chars = pauliChars(word);
        std::pair<SmallVector<char>, SmallVector<unsigned>> result;
        SmallVector<char> onData(source.size(), 'I');
        for (auto [pauli, qubit] : llvm::zip(chars, qubits)) {
            if (auto it = dataIndex.find(qubit); it != dataIndex.end()) {
                onData[it->second] = pauli;
                continue;
            }
            auto slot = slots.find(qubit);
            if (slot == slots.end()) {
                (void)finding(GadgetVerdict::unchecked, "a qubit comes from outside the gadget");
                return failure();
            }
            result.first.push_back(pauli);
            result.second.push_back(slot->second);
        }
        if (reduced && llvm::any_of(onData, [](char p) { return p != 'I'; })) {
            if (onData != source) {
                irreducible = true;
                (void)finding(GadgetVerdict::unchecked, "an op acts on the data qubits as "
                                                        "another word than the source");
                return failure();
            }
            result.first.push_back('Z');
            result.second.push_back(0);
        }
        return result;
    }

    unsigned addQubit(Value qubit, std::array<Complex, 2> single) {
        const unsigned slot = state.addQubit(single);
        slots[qubit] = slot;
        return slot;
    }

    FailureOr<bool> rotate(ArrayAttr word, ValueRange in, ValueRange out, double angle) {
        auto paulis = paulisOn(word, in);
        if (failed(paulis)) {
            return failure();
        }
        // exp(-i angle P)
        state.applyLinear(paulis->first, paulis->second, std::cos(angle),
                          Complex{0.0, -std::sin(angle)});
        forward(in, out);
        return true;
    }

    FailureOr<bool> measure(ArrayAttr word, bool negated, ValueRange in, ValueRange out,
                            Value result) {
        auto paulis = paulisOn(word, in);
        if (failed(paulis)) {
            return failure();
        }
        if (measured == outcomes.size()) {
            return false;
        }
        // Outcome 0 is the +1 eigenvalue of the signed word.
        const bool outcome = outcomes[measured++];
        const double sign = (negated != outcome) ? -0.5 : 0.5;
        state.applyLinear(paulis->first, paulis->second, 0.5, sign);
        bits[result] = outcome;
        forward(in, out);
        return true;
    }

    FailureOr<bool> release(Value qubit) {
        if (dataIndex.contains(qubit)) {
            return finding(GadgetVerdict::unchecked, "a data qubit is released in the gadget");
        }
        FailureOr<SmallVector<unsigned>> qubits = qubitSlots(qubit);
        if (failed(qubits)) {
            return failure();
        }
        const unsigned slot = qubits->front();
        if (!state.removeQubit(slot)) {
            return finding(GadgetVerdict::mismatch,
                           "entangled-release: an auxiliary qubit is released while entangled");
        }
        slots.erase(qubit);
        for (auto &entry : slots) {
            if (entry.second > slot) {
                entry.second--;
            }
        }
        return true;
    }

    FailureOr<bool> step(Operation *op) {
        if (state.numQubits > maxQubits) {
            return finding(GadgetVerdict::unchecked, "the gadget is too large to check");
        }
        return TypeSwitch<Operation *, FailureOr<bool>>(op)
            .Case<PPRotationOp>([&](auto ppr) -> FailureOr<bool> {
                if (Value condition = ppr.getCondition()) {
                    bool apply;
                    if (failed(bit(condition, apply))) {
                        return failure();
                    }
                    if (!apply) {
                        forward(ppr.getInQubits(), ppr.getOutQubits());
                        return true;
                    }
                }
                const double angle = llvm::numbers::pi / static_cast<double>(ppr.getRotationKind());
                return rotate(ppr.getPauliProduct(), ppr.getInQubits(), ppr.getOutQubits(), angle);
            })
            .Case<PPMeasurementOp>([&](auto ppm) {
                return measure(ppm.getPauliProduct(), ppm.getNegated(), ppm.getInQubits(),
                               ppm.getOutQubits(), ppm.getMres());
            })
            .Case<SelectPPMeasurementOp>([&](auto select) -> FailureOr<bool> {
                bool first;
                if (failed(bit(select.getSelectSwitch(), first))) {
                    return failure();
                }
                return first
                           ? measure(select.getPauliProduct_0(), select.getNegated_0(),
                                     select.getInQubits(), select.getOutQubits(), select.getMres())
                           : measure(select.getPauliProduct_1(), select.getNegated_1(),
                                     select.getInQubits(), select.getOutQubits(), select.getMres());
            })
            .Case<FabricateOp, PrepareStateOp>([&](auto init) {
                for (Value qubit : init.getOutQubits()) {
                    addQubit(qubit, initialState(init.getInitState()));
                }
                return FailureOr<bool>(true);
            })
            .Case<quantum::AllocQubitOp>([&](auto alloc) {
                addQubit(alloc.getQubit(), initialState(LogicalInitKind::zero));
                return FailureOr<bool>(true);
            })
            .Case<quantum::DeallocQubitOp>(
                [&](auto dealloc) { return release(dealloc.getQubit()); })
            .Case<arith::ConstantOp>([](auto) { return FailureOr<bool>(true); })
            .Case<arith::XOrIOp, arith::AndIOp, arith::OrIOp>([&](auto logic) -> FailureOr<bool> {
                bool a, b;
                if (failed(bit(logic.getLhs(), a)) || failed(bit(logic.getRhs(), b))) {
                    return failure();
                }
                using LogicOp = std::decay_t<decltype(logic)>;
                bits[logic.getResult()] = std::is_same_v<LogicOp, arith::XOrIOp>   ? (a != b)
                                          : std::is_same_v<LogicOp, arith::AndIOp> ? (a && b)
                                                                                   : (a || b);
                return true;
            })
            .Case<scf::IfOp>([&](auto ifOp) -> FailureOr<bool> {
                bool taken;
                if (failed(bit(ifOp.getCondition(), taken))) {
                    return failure();
                }
                SmallVector<Value> yielded;
                Region &region = taken ? ifOp.getThenRegion() : ifOp.getElseRegion();
                if (region.empty()) {
                    return true;
                }
                FailureOr<bool> done = run(region.front(), yielded);
                if (failed(done) || !*done) {
                    return done;
                }
                for (auto [value, result] : llvm::zip(yielded, ifOp.getResults())) {
                    if (auto data = dataIndex.find(value); data != dataIndex.end()) {
                        dataIndex[result] = data->second;
                    } else if (auto it = slots.find(value); it != slots.end()) {
                        slots[result] = it->second;
                    } else if (auto bitIt = bits.find(value); bitIt != bits.end()) {
                        bits[result] = bitIt->second;
                    }
                }
                return true;
            })
            .Default([&](Operation *other) {
                return finding(GadgetVerdict::unchecked,
                               "the check cannot follow '" + other->getName().getStringRef() + "'");
            });
    }
};

/// Check `gadget` on every branch of its measurement outcomes. With `reduced`, the data qubits
/// are checked as one qubit on which the source word is Z.
Finding checkGadget(GadgetOp gadget, bool reduced) {
    const unsigned k = reduced ? 1 : gadget.getInQubits().size();
    if (2 * k > maxQubits) {
        return {GadgetVerdict::unchecked, "the gadget is too large to check"};
    }
    const size_t d = size_t{1} << k;
    SmallVector<char> word =
        reduced ? SmallVector<char>{'Z'} : pauliChars(gadget.getPauliProduct());
    const double theta = gadget.getTheta().convertToDouble();

    // The source rotation exp(-i theta/2 P), as a d x d matrix.
    std::vector<Complex> source(d * d);
    {
        State basis(k);
        SmallVector<unsigned> qubits;
        for (unsigned i = 0; i < k; i++) {
            qubits.push_back(i);
        }
        for (size_t j = 0; j < d; j++) {
            std::fill(basis.amplitudes.begin(), basis.amplitudes.end(), Complex{});
            basis.amplitudes[j] = 1.0;
            basis.applyLinear(word, qubits, std::cos(theta / 2),
                              Complex{0.0, -std::sin(theta / 2)});
            for (size_t i = 0; i < d; i++) {
                source[i * d + j] = basis.amplitudes[i];
            }
        }
    }

    SmallVector<SmallVector<bool>> pending{{}};
    double totalWeight = 0.0;
    size_t branches = 0;
    while (!pending.empty()) {
        SmallVector<bool> outcomes = pending.pop_back_val();
        Branch branch(gadget, outcomes, reduced);
        SmallVector<Value> yielded;
        FailureOr<bool> done = branch.run(gadget.getRegion().front(), yielded);
        if (failed(done)) {
            if (branch.irreducible) {
                return checkGadget(gadget, /*reduced=*/false);
            }
            return *branch.result;
        }
        if (!*done) {
            for (bool outcome : {false, true}) {
                SmallVector<bool> longer(outcomes);
                longer.push_back(outcome);
                pending.push_back(longer);
            }
            continue;
        }
        const double weight = branch.state.norm();
        if (weight < tolerance * tolerance) {
            continue;
        }
        branches++;
        totalWeight += weight;
        if (branch.state.numQubits != 2 * k) {
            return {GadgetVerdict::unchecked, "an auxiliary qubit is not released in the gadget"};
        }

        // The branch's unitary U[i][j] = sqrt(d / weight) <i|_data <j|_ref |psi>.
        SmallVector<unsigned> dataSlots;
        for (auto [position, qubit] : llvm::enumerate(yielded)) {
            if (reduced) {
                auto data = branch.dataIndex.find(qubit);
                if (data == branch.dataIndex.end() || data->second != position) {
                    return {GadgetVerdict::unchecked, "the gadget permutes its data qubits"};
                }
                continue;
            }
            dataSlots.push_back(branch.slots.lookup(qubit));
        }
        if (reduced) {
            dataSlots.push_back(0);
        }
        std::vector<Complex> unitary(d * d);
        Complex overlap{};
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                size_t index = j << k;
                for (unsigned q = 0; q < k; q++) {
                    index |= ((i >> q) & 1) << dataSlots[q];
                }
                unitary[i * d + j] =
                    branch.state.amplitudes[index] * std::sqrt(static_cast<double>(d) / weight);
                overlap += std::conj(source[i * d + j]) * unitary[i * d + j];
            }
        }
        // || U - e^{i phi} V ||_F / sqrt(d), with the best global phase.
        const Complex phase = std::abs(overlap) > 0 ? overlap / std::abs(overlap) : Complex{1.0};
        double distance = 0.0;
        for (size_t e = 0; e < d * d; e++) {
            distance += std::norm(unitary[e] - phase * source[e]);
        }
        distance = std::sqrt(distance / static_cast<double>(d));
        if (distance > tolerance) {
            std::string outcomeText;
            for (bool outcome : outcomes) {
                outcomeText += outcome ? '1' : '0';
            }
            return {GadgetVerdict::mismatch,
                    "the branch with outcomes '" + outcomeText +
                        "' does not implement the source rotation (distance " +
                        std::to_string(distance) + ")"};
        }
    }
    if (branches == 0 || std::abs(totalWeight - 1.0) > 1e-9) {
        return {GadgetVerdict::unchecked, "the branch weights do not sum to one"};
    }
    return {GadgetVerdict::verified};
}

} // namespace

namespace catalyst {
namespace pbc {

struct VerifyPBCGadgetsPass : public impl::VerifyPBCGadgetsPassBase<VerifyPBCGadgetsPass> {
    using VerifyPBCGadgetsPassBase::VerifyPBCGadgetsPassBase;

    void runOnOperation() final {
        bool failedStrict = false;
        getOperation()->walk([&](GadgetOp gadget) {
            Finding finding = checkGadget(gadget, /*reduced=*/true);
            gadget.setVerdictAttr(GadgetVerdictAttr::get(gadget.getContext(), finding.verdict));
            if (finding.verdict == GadgetVerdict::mismatch) {
                if (strict) {
                    gadget.emitError("gadget does not implement its source rotation: ")
                        << finding.reason;
                    failedStrict = true;
                } else {
                    gadget.emitWarning("gadget does not implement its source rotation: ")
                        << finding.reason;
                }
            } else if (finding.verdict == GadgetVerdict::unchecked) {
                gadget.emitRemark("gadget not checked: ") << finding.reason;
            }
        });
        if (failedStrict) {
            signalPassFailure();
        }
    }
};

} // namespace pbc
} // namespace catalyst
