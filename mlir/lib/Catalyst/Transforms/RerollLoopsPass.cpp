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

#define DEBUG_TYPE "reroll-loops"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace llvm;
using namespace mlir;

// Tracing unrolls Python loops: N identical circuit segments (Trotter steps,
// layers, folds) arrive as N copies of the same op sequence, and every later
// stage amplifies each copy. This pass reconstructs the loops in four steps:
// structural hashing of each op, tandem-repeat detection on the hash sequence
// (maximal runs with H[i] == H[i+p]), semantic verification that consecutive
// windows are isomorphic with cross-window dataflow limited to threaded values
// (e.g. qubits) plus loop invariants, and replacement of the repeat with an
// scf.for whose iter_args are the threaded values.

namespace {

/// Ops that may participate in a rerolled window: single-block-region-free,
/// no successors, and not a terminator.
bool isRerollableOp(Operation *op)
{
    return op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
           !op->hasTrait<OpTrait::IsTerminator>();
}

/// Compute structural hashes for all ops of a block (excluding the
/// terminator). Non-rerollable ops receive unique sentinel hashes. In-block
/// operand defs are hashed by result number only (not identity, which would
/// make every window distinct); the weak discrimination this leaves is
/// compensated by semantic verification of every candidate.
void computeHashes(ArrayRef<Operation *> ops, const llvm::DenseMap<Operation *, size_t> &indexOf,
                   SmallVectorImpl<uint64_t> &hashes)
{
    uint64_t sentinel = 0;
    for (Operation *op : ops) {
        if (!isRerollableOp(op)) {
            hashes.push_back(hash_combine(0xdeadbeefULL, ++sentinel));
            continue;
        }
        hash_code h = hash_combine(op->getName().getTypeID(),
                                   op->getAttrDictionary().getAsOpaquePointer());
        for (Type t : op->getResultTypes()) {
            h = hash_combine(h, t.getAsOpaquePointer());
        }
        for (Value v : op->getOperands()) {
            Operation *def = v.getDefiningOp();
            auto it = def ? indexOf.find(def) : indexOf.end();
            if (it != indexOf.end()) {
                h = hash_combine(h, 1, cast<OpResult>(v).getResultNumber());
            }
            else {
                // Block argument or out-of-block def: hash the value identity.
                h = hash_combine(h, 2, v.getAsOpaquePointer());
            }
        }
        hashes.push_back(h);
    }
}

struct Candidate {
    size_t start;  // index of the first op of the first window
    size_t period; // ops per window
    size_t count;  // number of windows
};

/// Harvest candidate periods from the gap distribution between successive
/// occurrences of equal hashes, then find maximal H[i] == H[i+p] runs.
void findCandidates(ArrayRef<uint64_t> hashes, unsigned minPeriod, unsigned maxPeriods,
                    SmallVectorImpl<Candidate> &out)
{
    size_t n = hashes.size();
    llvm::DenseMap<uint64_t, size_t> lastSeen;
    llvm::DenseMap<size_t, size_t> gapWeight;
    for (size_t i = 0; i < n; ++i) {
        auto [it, inserted] = lastSeen.try_emplace(hashes[i], i);
        if (!inserted) {
            size_t gap = i - it->second;
            if (gap >= minPeriod) {
                gapWeight[gap]++;
            }
            it->second = i;
        }
    }

    SmallVector<std::pair<size_t, size_t>> periods(gapWeight.begin(), gapWeight.end());
    // Favor periods that explain the most ops.
    llvm::sort(periods, [](auto &a, auto &b) {
        return a.first * a.second > b.first * b.second;
    });
    if (periods.size() > maxPeriods) {
        periods.resize(maxPeriods);
    }

    for (auto &[p, weight] : periods) {
        // Require the period to repeat enough times to be worth a loop.
        if (weight < 2 * p) {
            continue;
        }
        size_t runStart = SIZE_MAX;
        for (size_t i = 0; i + p <= n; ++i) {
            bool match = (i + p < n) && hashes[i] == hashes[i + p];
            if (match && runStart == SIZE_MAX) {
                runStart = i;
            }
            if (!match && runStart != SIZE_MAX) {
                size_t runLen = i - runStart;
                size_t count = runLen / p + 1;
                if (count >= 3) {
                    out.push_back({runStart, p, count});
                }
                runStart = SIZE_MAX;
            }
        }
    }
}

struct RerollPlan {
    Candidate cand;
    // Threaded slots: values flowing window -> next window, identified by
    // (defining op position within the window, result number). Order defines
    // the iter_args order.
    SmallVector<std::pair<unsigned, unsigned>> slots;
    // Initial value of each slot (operand of the first window).
    SmallVector<Value> inits;
    // Classification of each cross-window operand use:
    // (window-relative op position, operand number) -> slot index.
    llvm::DenseMap<std::pair<unsigned, unsigned>, unsigned> threadedUse;
};

/// Verify that the candidate's windows are isomorphic with dataflow limited to
/// threaded slots + loop-invariant values, and build the reroll plan.
std::optional<RerollPlan> verifyCandidate(ArrayRef<Operation *> ops,
                                          const llvm::DenseMap<Operation *, size_t> &indexOf,
                                          Candidate cand)
{
    size_t start = cand.start, p = cand.period, count = cand.count;
    size_t end = start + count * p;
    if (end > ops.size()) {
        return std::nullopt;
    }

    RerollPlan plan;
    plan.cand = cand;
    llvm::DenseMap<std::pair<unsigned, unsigned>, unsigned> slotIndex; // (defPos,resNo) -> idx

    auto getSlot = [&](unsigned defPos, unsigned resNo) -> unsigned {
        auto [it, inserted] = slotIndex.try_emplace({defPos, resNo}, plan.slots.size());
        if (inserted) {
            plan.slots.push_back({defPos, resNo});
            plan.inits.push_back(Value());
        }
        return it->second;
    };

    for (size_t w = 1; w < count; ++w) {
        for (size_t j = 0; j < p; ++j) {
            Operation *a = ops[start + (w - 1) * p + j];
            Operation *b = ops[start + w * p + j];
            if (!isRerollableOp(a) || !isRerollableOp(b)) {
                return std::nullopt;
            }
            if (a->getName() != b->getName() ||
                a->getAttrDictionary() != b->getAttrDictionary() ||
                a->getResultTypes() != b->getResultTypes() ||
                a->getNumOperands() != b->getNumOperands()) {
                return std::nullopt;
            }
            for (unsigned t = 0; t < b->getNumOperands(); ++t) {
                Value vb = b->getOperand(t);
                Value va = a->getOperand(t);
                Operation *defB = vb.getDefiningOp();
                auto itB = defB ? indexOf.find(defB) : indexOf.end();
                size_t ib = (itB != indexOf.end()) ? itB->second : SIZE_MAX;

                if (ib != SIZE_MAX && ib >= start + w * p) {
                    // Within current window: the counterpart must reference the
                    // same relative position.
                    Operation *defA = va.getDefiningOp();
                    auto itA = defA ? indexOf.find(defA) : indexOf.end();
                    if (itA == indexOf.end() || itA->second + p != ib ||
                        cast<OpResult>(va).getResultNumber() !=
                            cast<OpResult>(vb).getResultNumber()) {
                        return std::nullopt;
                    }
                }
                else if (ib != SIZE_MAX && ib >= start + (w - 1) * p) {
                    // Threaded from the previous window.
                    unsigned defPos = ib - (start + (w - 1) * p);
                    unsigned resNo = cast<OpResult>(vb).getResultNumber();
                    unsigned slot = getSlot(defPos, resNo);
                    auto [uit, uinserted] = plan.threadedUse.try_emplace({(unsigned)j, t}, slot);
                    if (!uinserted && uit->second != slot) {
                        return std::nullopt;
                    }
                    // The counterpart operand must thread identically.
                    if (w == 1) {
                        // va is the init value; it must be loop-invariant w.r.t.
                        // the region (defined before it).
                        Operation *defA = va.getDefiningOp();
                        auto itA = defA ? indexOf.find(defA) : indexOf.end();
                        if (itA != indexOf.end() && itA->second >= start) {
                            return std::nullopt;
                        }
                        if (plan.inits[slot] && plan.inits[slot] != va) {
                            return std::nullopt;
                        }
                        plan.inits[slot] = va;
                    }
                    else {
                        Operation *defA = va.getDefiningOp();
                        auto itA = defA ? indexOf.find(defA) : indexOf.end();
                        if (itA == indexOf.end() || itA->second + p != ib ||
                            cast<OpResult>(va).getResultNumber() != resNo) {
                            return std::nullopt;
                        }
                    }
                }
                else {
                    // Loop-invariant: must be the exact same value, defined
                    // before the region.
                    if (va != vb) {
                        return std::nullopt;
                    }
                    if (ib != SIZE_MAX && ib >= start) {
                        return std::nullopt;
                    }
                }
            }
        }
    }

    // Every slot must have an init.
    for (Value init : plan.inits) {
        if (!init) {
            return std::nullopt;
        }
    }

    // Uses of window results outside the allowed range:
    //  - windows 0..count-2: results may only be used inside their own window or
    //    the next one (threaded uses were verified above; any other use pattern
    //    is unsupported).
    //  - last window: external uses allowed only for threaded slots (they become
    //    loop results).
    for (size_t w = 0; w < count; ++w) {
        bool isLast = (w == count - 1);
        for (size_t j = 0; j < p; ++j) {
            Operation *op = ops[start + w * p + j];
            for (OpResult res : op->getResults()) {
                for (OpOperand &use : res.getUses()) {
                    Operation *user = use.getOwner();
                    auto uit = indexOf.find(user);
                    size_t ui = (uit != indexOf.end()) ? uit->second : SIZE_MAX;
                    bool inOwnWindow =
                        ui != SIZE_MAX && ui >= start + w * p && ui < start + (w + 1) * p;
                    bool inNextWindow = !isLast && ui != SIZE_MAX &&
                                        ui >= start + (w + 1) * p &&
                                        ui < start + (w + 2) * p;
                    if (inOwnWindow || inNextWindow) {
                        continue;
                    }
                    // External use.
                    if (!isLast) {
                        return std::nullopt;
                    }
                    if (!slotIndex.count({(unsigned)j, res.getResultNumber()})) {
                        return std::nullopt;
                    }
                }
            }
        }
    }

    return plan;
}

/// Try to extend a verified candidate by whole windows to the left/right; the
/// hash sequence misses the first window (its cross-window references point at
/// the prologue, at different distances), so this recovers it.
RerollPlan extendCandidate(ArrayRef<Operation *> ops,
                           const llvm::DenseMap<Operation *, size_t> &indexOf, RerollPlan plan)
{
    while (plan.cand.start >= plan.cand.period) {
        Candidate c = plan.cand;
        c.start -= c.period;
        c.count += 1;
        auto extended = verifyCandidate(ops, indexOf, c);
        if (!extended) {
            break;
        }
        plan = *extended;
    }
    while (true) {
        Candidate c = plan.cand;
        c.count += 1;
        auto extended = verifyCandidate(ops, indexOf, c);
        if (!extended) {
            break;
        }
        plan = *extended;
    }
    return plan;
}

/// Replace the repeat with an scf.for and erase the original ops.
void materialize(ArrayRef<Operation *> ops, const RerollPlan &plan)
{
    size_t start = plan.cand.start, p = plan.cand.period, count = plan.cand.count;
    Operation *first = ops[start];
    Location loc = first->getLoc();
    OpBuilder builder(first);

    Value lb = arith::ConstantIndexOp::create(builder, loc, 0);
    Value ub = arith::ConstantIndexOp::create(builder, loc, count);
    Value step = arith::ConstantIndexOp::create(builder, loc, 1);

    auto forOp = scf::ForOp::create(builder, loc, lb, ub, step, plan.inits);
    Block *body = forOp.getBody();
    builder.setInsertionPointToStart(body);

    // Clone window 0 as the loop body, remapping operands per classification.
    SmallVector<Operation *> cloned(p);
    IRMapping mapping; // within-window result mapping
    for (size_t j = 0; j < p; ++j) {
        Operation *proto = ops[start + j];
        Operation *clone = proto->cloneWithoutRegions(mapping);
        // Fix up operands that are threaded from the previous iteration: the
        // prototype (window 0) uses the init values there.
        for (unsigned t = 0; t < clone->getNumOperands(); ++t) {
            auto it = plan.threadedUse.find({(unsigned)j, t});
            if (it != plan.threadedUse.end()) {
                clone->setOperand(t, forOp.getRegionIterArg(it->second));
            }
        }
        builder.insert(clone);
        cloned[j] = clone;
    }
    SmallVector<Value> yields;
    for (auto [defPos, resNo] : plan.slots) {
        yields.push_back(cloned[defPos]->getResult(resNo));
    }
    scf::YieldOp::create(builder, loc, yields);

    // Rewire external uses of the last window's results to the loop results.
    for (const auto &[slotIdx, slot] : llvm::enumerate(plan.slots)) {
        auto [defPos, resNo] = slot;
        Operation *lastOp = ops[start + (count - 1) * p + defPos];
        lastOp->getResult(resNo).replaceAllUsesWith(forOp.getResult(slotIdx));
    }

    // Erase the original ops, last first (uses before defs).
    for (size_t i = start + count * p; i-- > start;) {
        ops[i]->dropAllUses();
        ops[i]->erase();
    }
}

bool processBlock(Block &block, unsigned minPeriod, unsigned minSavings)
{
    SmallVector<Operation *> ops;
    for (Operation &op : block.without_terminator()) {
        ops.push_back(&op);
    }
    if (ops.size() < 2 * minPeriod) {
        return false;
    }

    llvm::DenseMap<Operation *, size_t> indexOf;
    for (const auto &[i, op] : llvm::enumerate(ops)) {
        indexOf[op] = i;
    }

    SmallVector<uint64_t> hashes;
    computeHashes(ops, indexOf, hashes);

    SmallVector<Candidate> candidates;
    findCandidates(hashes, minPeriod, /*maxPeriods=*/16, candidates);

    // Verify, extend, and pick non-overlapping plans greedily by savings.
    SmallVector<RerollPlan> plans;
    for (Candidate cand : candidates) {
        auto plan = verifyCandidate(ops, indexOf, cand);
        if (!plan) {
            continue;
        }
        *plan = extendCandidate(ops, indexOf, *plan);
        if ((plan->cand.count - 1) * plan->cand.period >= minSavings) {
            plans.push_back(std::move(*plan));
        }
    }
    llvm::sort(plans, [](const RerollPlan &a, const RerollPlan &b) {
        return (a.cand.count - 1) * a.cand.period > (b.cand.count - 1) * b.cand.period;
    });

    SmallVector<std::pair<size_t, size_t>> used;
    SmallVector<const RerollPlan *> accepted;
    for (const RerollPlan &plan : plans) {
        size_t s = plan.cand.start, e = s + plan.cand.count * plan.cand.period;
        bool overlaps = llvm::any_of(
            used, [&](auto range) { return s < range.second && range.first < e; });
        if (!overlaps) {
            used.push_back({s, e});
            accepted.push_back(&plan);
        }
    }

    // Materialize from the highest start index down so op indices of pending
    // plans remain valid.
    llvm::sort(accepted, [](const RerollPlan *a, const RerollPlan *b) {
        return a->cand.start > b->cand.start;
    });
    for (const RerollPlan *plan : accepted) {
        LLVM_DEBUG(dbgs() << "rerolling: start=" << plan->cand.start
                          << " period=" << plan->cand.period
                          << " count=" << plan->cand.count
                          << " slots=" << plan->slots.size() << "\n");
        materialize(ops, *plan);
    }
    return !accepted.empty();
}

} // namespace

namespace catalyst {

#define GEN_PASS_DECL_REROLLLOOPSPASS
#define GEN_PASS_DEF_REROLLLOOPSPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct RerollLoopsPass : public impl::RerollLoopsPassBase<RerollLoopsPass> {
    using impl::RerollLoopsPassBase<RerollLoopsPass>::RerollLoopsPassBase;

    void runOnOperation() override
    {
        // Iterate to a fixpoint (bounded): rerolling creates new blocks (loop
        // bodies) that may contain further repeats, e.g. nested loops.
        bool changed = true;
        unsigned rounds = 0;
        while (changed && rounds++ < 4) {
            changed = false;
            SmallVector<Block *> blocks;
            getOperation()->walk([&](Block *block) { blocks.push_back(block); });
            for (Block *block : blocks) {
                changed |= processBlock(*block, minPeriod, minSavings);
            }
        }
    }
};

} // namespace catalyst
