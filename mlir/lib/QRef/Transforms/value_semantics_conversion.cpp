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

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>
#define DEBUG_TYPE "value-semantics-conversion"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"  // for PassManager
#include "mlir/Transforms/Passes.h" // for createCSEPass

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"

using namespace mlir;
using namespace catalyst;

// In this file, variable names like "vQubit" stand for "qubits in value semantics",
// and variable names like "rQubit" stand for "qubits in reference semantics".

namespace {

/**
 * @brief Given a rQubit Value, return whether it is a root rQubit Value or not.
 * Root rQubit Values are either scope arguments, or results of single qubit allocations.
 * All non-root rQubit Values are results of qref.get operations.
 *
 * @param rQubit
 * @return bool
 */
bool isRootRQubit(Value rQubit)
{
    assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");

    if (isa<BlockArgument>(rQubit) || isa<qref::AllocQubitOp>(rQubit.getDefiningOp())) {
        return true;
    }

    assert(isa<qref::GetOp>(rQubit.getDefiningOp()) &&
           "Only qref.get ops can produce qref.bit SSA values");
    return false;
}

/**
 * @brief Given a non-root rQubit Value, return the rQreg Value that it belongs to.
 *
 * @param rQubit
 * @return Value
 */
Value getRSourceRegisterValue(Value rQubit)
{
    assert(isa<qref::QubitType>(rQubit.getType()) && "Can only query qref.bit types");
    assert(!isRootRQubit(rQubit) && "Only non-root rQubits can come from a rQreg");
    return cast<qref::GetOp>(rQubit.getDefiningOp()).getQreg();
}

/**
Collect the rQreg and rQubit Values that are captured into a region from above via closure.
This includes any rQregs that produce rQubits that the region captures from above via closure.

Reference semantics dialect operations do not take in or produce qreg Values, which means all
qreg Values are taken in via closure from above.

When converting to value semantics, the vQregs and vQubits need to be taken in by the region-ed
operations explicitly.
 */
void getNecessaryRegionRValues(Region &r, SetVector<Value> &necessaryRegionRValues)
{
    auto *qrefDialect = r.getContext()->getLoadedDialect<qref::QRefDialect>();

    r.walk([&](Operation *op) {
        if (op->getDialect() != qrefDialect) {
            return;
        }
        if (isa<qref::GetOp>(op)) {
            // qref.get is not a gate, do not count it as a user
            // For example, if the rQubit result from a qref.get has no users, the get op is not
            // actually needed by the region.
            return;
        }
        for (Value v : op->getOperands()) {
            if (isa<qref::QuregType>(v.getType())) {
                // Ignore allocations from inside the region itself
                if (v.getParentRegion()->isProperAncestor(&r)) {
                    necessaryRegionRValues.insert(v);
                }
            }
            else if (isa<qref::QubitType>(v.getType())) {
                if (isRootRQubit(v)) {
                    // Ignore allocations from inside the region itself
                    if (v.getParentRegion()->isProperAncestor(&r)) {
                        necessaryRegionRValues.insert(v);
                    }
                }
                else {
                    Value rQreg = getRSourceRegisterValue(v);
                    if (rQreg.getParentRegion()->isProperAncestor(&r)) {
                        necessaryRegionRValues.insert(v);
                    }
                }
            }
        }
    });
}

/**
 * @brief This struct tracks the current vQreg and vQubit Values for the root rQreg and rQubit
 * Values in a given region. There should be exactly one tracker instance per region.
 *
 * In a region, the root rQreg and rQubit values can come from one of two places:
 * - An allocation, i.e. a qref.alloc or qref.alloc_qb operation
 * - An argument, e.g. arguments to a subroutine or control flow region
 *
 * These Values are considered "root" because they are guaranteed to be uniquely defined: different
 * allocations and arguments are semantically distinct.
 *
 * By contrast, rQubit Values returned from qref.get operations are not considered root values,
 * because multiple qref.get operations can alias each other, making two rQubit values effectively
 * the same. This creates a problem of aliasing: if we were to track the current vQubit Values of
 * all rQubit Values in the program, a new vQubit Value would need to be updated across all aliasing
 * rQubit Values. This is not so simple to do.
 *
 * Importantly, vQubit Values extracted from a vQreg are not tracked. Instead, each usage of the
 * corresponding rQubit Value (from a qref.get operation) will extract the vQubit Values before the
 * use, and insert the vQubit Values after the use.
 *
 */
struct QubitValueTracker {
  public:
    QubitValueTracker() = default;

    /**
     * @brief Return the current vQreg Value corresponding to a root rQreg Value.
     *
     * @param rQreg
     * @return const Value
     */
    const Value getCurrentVQreg(Value rQreg)
    {
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");

        Value vQreg = this->qreg_map.at(rQreg);
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");
        return vQreg;
    }

    /**
     * @brief Set the current vQreg Value corresponding to a root rQreg Value.
     *
     * @param rQreg
     * @param vQreg
     */
    void setCurrentVQreg(Value rQreg, Value vQreg)
    {
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");
        if (this->qreg_map.contains(rQreg)) {
            this->qreg_map[rQreg] = vQreg;
        }
        else {
            this->qreg_map.insert({rQreg, vQreg});
        }
    }

    /**
     * @brief Return the current vQubit Value corresponding to a root rQubit Value.
     *
     * @param rQubit
     * @return const Value
     */
    const Value getCurrentVQubit(Value rQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        assert(isRootRQubit(rQubit) && "rQubit Values indexed from a rQreg are not tracked");

        Value vQubit = this->qubit_map.at(rQubit);
        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");
        return vQubit;
    }

    /**
     * @brief Set the current vQubit Value corresponding to a root rQubit Value.
     *
     * @param rQubit
     * @param vQubit
     */
    void setCurrentVQubit(Value rQubit, Value vQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");
        assert(isRootRQubit(rQubit) && "rQubit Values indexed from a rQreg are not tracked");

        if (this->qubit_map.contains(rQubit)) {
            this->qubit_map[rQubit] = vQubit;
        }
        else {
            this->qubit_map.insert({rQubit, vQubit});
        }
    }

  private:
    // The map is from root qref.qreg values of a region to the current quantum.qreg values.
    llvm::DenseMap<Value, Value> qreg_map;

    // The map is from root qref.bit values of a region to the current quantum.bit values.
    llvm::DenseMap<Value, Value> qubit_map;
};

/**
 * @brief This struct is responsible for extracting and inserting vQubits before and after uses.
 *
 * @details This struct is responsible for extracting and inserting vQubits before and after uses.
 *
 * This class must be the sole source of the creation of quantum.extract and insert operations.
 *
 * We follow a very simple strategy in this semantics conversion, which we now describe.
 *
 * Whenever a qref operation (gate, observables, or any region containing those) uses some rQubit
 * and rQreg Values, these Values are either root or non-root.
 *
 * Root Values are tracked in the tracker maps directly and do not need to be extracted, as they
 * have no source registers. Note that all rQreg Values are root.
 *
 * Non-root rQubit Values are all considered transient. That is, their corresponding vQubit Values
 * are extracted immediately before the uses (from their source rQreg Value's current vQreg Value),
 * and inserted back immediately after the use. They are not tracked globally across uses, unlike
 * the root Values.
 *
 * If the user op consumes the extracted vQubits but does not return vQubit results (e.g. observable
 * ops), the extracted vQubits are inserted.
 *
 * The order of the extract and insert operations created are in the order that the non-root
 * rQubit Values appear in the given operation.
 *
 * When an operation (rOp) using non-root rQubit Values is interfacing with this struct, it must
 * guarantee the following are true on the new operation (vOp) it creates in the value semantics
 * dialect:
 * 1. If the rOp returns any results, the vOp must return results of the same types at the same
 * result indices. This is because even though a rOp cannot return any qubit or qreg Values, it
 * can still return classical Values (e.g. from regions) or observables (e.g. from namedobs op).
 * 2. Any vQubit and vQreg Values returned from vOp must appear at the end of the vOp's list of
 * results, after any existing result Values from the previous point, without being interleaved
 * with any other results.
 * 3. The order in which these new vQubit and vQreg results appear must match the order in which
 * they appear in the rOp's operands.
 *
 * For example, denoting the 3 kinds (qreg, root qubit, non-root qubit) by A, B, C, and classical
 * Values by regular names, the valid vOp for the following rOp
 *   %0, %1 = qref.some_op %arg0, %A1, %arg1, %B1, %B2, %C1, %A2, %arg2, %C2
 * is
 *   %0, %1, %A1_out, %B1_out, %B2_out, %C1_out, %A2_out, %C2_out = quantum.some_op ....
 *
 * This struct offers methods to query the expected indices that follow the above convention.
 * The "getOperandIndices" methods return the operand indices of requested kind on the rOp.
 * The "getResultIndices" methods return the expected result indices of requested kind on the vOp.
 * In the above example, the methods would return:
 * getQRegOperandIndices(): [1, 6]
 * getRootQubitOperandIndices(): [3, 4]
 * getNonRootQubitOperandIndices(): [5, 8]
 * getQRegResultIndices(): [2, 6]
 * getRootQubitResultIndices(): [3, 4]
 * getNonRootQubitResultIndices(): [5, 7]

 * @param rOp
 * @param tracker
 * @param builder
 */
struct TransientQubitExtractor {
  public:
    TransientQubitExtractor(Operation *_rOp, QubitValueTracker &_tracker, IRRewriter &_builder)
        : rOp(_rOp), tracker(_tracker), builder(_builder)
    {
        OpBuilder::InsertionGuard guard(this->builder);
        this->builder.setInsertionPoint(this->rOp);

        this->analyzeROpQuantumOperandPatterns();

        for (auto [i, operand] : llvm::enumerate(this->rOp->getOperands())) {
            if (!isa<qref::QubitType>(operand.getType()) || isRootRQubit(operand)) {
                continue;
            }

            this->sourceRQregs.push_back(getRSourceRegisterValue(operand));
            this->extractOps.push_back(this->createExtractOp(operand));
        }
    }

    TransientQubitExtractor(Region &r, QubitValueTracker &_tracker, IRRewriter &_builder)
        : rOp(r.getParentOp()), tracker(_tracker), builder(_builder)
    {
        assert(r.hasOneBlock() && "Expected single block region");
        OpBuilder::InsertionGuard guard(this->builder);
        this->builder.setInsertionPoint(this->rOp);

        SmallVector<Value> regionOperands(r.front().getArguments());
        SetVector<Value> rValuesUsedByRegion;
        getNecessaryRegionRValues(r, rValuesUsedByRegion);
        regionOperands.append(rValuesUsedByRegion.begin(), rValuesUsedByRegion.end());
        this->analyzeROpQuantumOperandPatterns(&regionOperands);

        for (auto [i, operand] : llvm::enumerate(rValuesUsedByRegion)) {
            if (!isa<qref::QubitType>(operand.getType()) || isRootRQubit(operand)) {
                continue;
            }

            this->sourceRQregs.push_back(getRSourceRegisterValue(operand));
            this->extractOps.push_back(this->createExtractOp(operand));
        }
    }

    ~TransientQubitExtractor()
    {
        if (this->getNonRootQubitResultIndices().size() == 0) {
            // Nothing to insert
            return;
        }

        OpBuilder::InsertionGuard guard(this->builder);
        this->builder.setInsertionPointAfter(this->rOp);

        // 0. Get the new vOp. It is the user of the extracted vQubits.
        SetVector<Operation *> vOps;
        for (auto extractOp : this->extractOps) {
            assert(extractOp->hasOneUse() && "Expected the extracted vQubit to have only one use");
            vOps.insert(*extractOp->getUsers().begin());
        }
        assert(vOps.size() == 1 && "Expected the extracted vQubits to be used by the same vOp");
        Operation *vOp = vOps[0];

        // 1. Fetch the vQubit results from the newly created value semantics user op
        // Only need to insert non-root rQubits.
        //
        // e.g. if the new vOp is
        // %classical, %vQreg, %vQubit_non_root0, %vQubit_root, %vQubit_non_root1
        //  = quantum.custom .... %inQreg, %inQubit_non_root0, %inQubit_root, %inQubit_non_root1
        //
        // we need to insert result Values at indices 2 and 4
        std::vector<unsigned> nonRootQubitResultIndices = this->getNonRootQubitResultIndices();

        // 2. Create the insert ops
        for (auto triplet :
             llvm::zip_equal(this->extractOps, this->sourceRQregs, nonRootQubitResultIndices)) {
            quantum::ExtractOp extractOp = std::get<0>(triplet);
            Value sourceRQreg = std::get<1>(triplet);
            unsigned resultIdx = std::get<2>(triplet);

            Value qubitToInsert =
                resultIdx >= (vOp->getNumResults())
                    ? extractOp.getQubit()
                    : dyn_cast<TypedValue<quantum::QubitType>>(vOp->getResult(resultIdx));

            auto insertOp = quantum::InsertOp::create(
                this->builder, extractOp->getLoc(), quantum::QuregType::get(extractOp.getContext()),
                this->tracker.getCurrentVQreg(sourceRQreg), extractOp.getIdx(),
                extractOp.getIdxAttrAttr(), qubitToInsert);
            this->tracker.setCurrentVQreg(sourceRQreg, insertOp.getOutQreg());
            this->builder.setInsertionPointAfter(insertOp);
        }
    }

    /**
     * @brief Get the extracted vQubit Values.
     *
     * @return SmallVector<Value>
     */
    SmallVector<Value> getExtractedVQubits()
    {
        SmallVector<Value> result;
        for (quantum::ExtractOp &extractOp : this->extractOps) {
            result.push_back(extractOp.getQubit());
        }
        return result;
    }

    const std::vector<unsigned> &getQRegOperandIndices() { return this->rQregOperandIndices; }

    const std::vector<unsigned> &getRootQubitOperandIndices()
    {
        return this->rootRQubitOperandIndices;
    }

    const std::vector<unsigned> &getNonRootQubitOperandIndices()
    {
        return this->nonRootRQubitOperandIndices;
    }

    const std::vector<unsigned> &getQRegResultIndices() { return this->vQregResultIndices; }

    const std::vector<unsigned> &getRootQubitResultIndices()
    {
        return this->rootVQubitResultIndices;
    }

    const std::vector<unsigned> &getNonRootQubitResultIndices()
    {
        return this->nonRootVQubitResultIndices;
    }

  private:
    Operation *rOp;
    QubitValueTracker &tracker;
    IRRewriter &builder;

    std::vector<unsigned> nonRootRQubitOperandIndices;
    std::vector<unsigned> rootRQubitOperandIndices;
    std::vector<unsigned> rQregOperandIndices;
    std::vector<unsigned> nonRootVQubitResultIndices;
    std::vector<unsigned> rootVQubitResultIndices;
    std::vector<unsigned> vQregResultIndices;

    SmallVector<Value> sourceRQregs;
    SmallVector<quantum::ExtractOp> extractOps;

    void analyzeROpQuantumOperandPatterns(SmallVector<Value> *regionOperands = nullptr)
    {
        unsigned resIdx = 0;
        unsigned existingNumResults = this->rOp->getNumResults();

        SmallVector<Value> operands;
        if (regionOperands == nullptr) {
            operands.append(this->rOp->getOperands().begin(), this->rOp->getOperands().end());
        }
        else {
            operands.append(regionOperands->begin(), regionOperands->end());
        }

        for (auto [i, operand] : llvm::enumerate(operands)) {
            if (isa<qref::QuregType>(operand.getType())) {
                this->vQregResultIndices.push_back(existingNumResults + resIdx);
                this->rQregOperandIndices.push_back(i);
                resIdx++;
            }
            else if (isa<qref::QubitType>(operand.getType())) {
                if (isRootRQubit(operand)) {
                    this->rootVQubitResultIndices.push_back(existingNumResults + resIdx);
                    this->rootRQubitOperandIndices.push_back(i);
                }
                else {
                    this->nonRootVQubitResultIndices.push_back(existingNumResults + resIdx);
                    this->nonRootRQubitOperandIndices.push_back(i);
                }
                resIdx++;
            }
        }
    }

    /**
     * @brief Given an non-root rQubit Value, create a quantum.extract operation from the current
     * vQreg of the rQreg it belongs to. The extract op is created from the same index as the
     * rQubit's defining qref.get operation.
     */
    quantum::ExtractOp createExtractOp(Value rQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
        assert(getOp && "Only qref.get ops can produce qref.bit SSA values");

        OpBuilder::InsertionGuard guard(this->builder);

        std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
        Value idxValue = getOp.getIdx();
        assert((idxAttr.has_value() ^ (idxValue != nullptr)) &&
               "expected exactly one index for extract op");

        Value vQreg = this->tracker.getCurrentVQreg(getOp.getQreg());

        Type qubitType = quantum::QubitType::get(rQubit.getContext());
        Type i64Type = this->builder.getI64Type();
        quantum::ExtractOp extractOp;
        if (idxAttr.has_value()) {
            extractOp = quantum::ExtractOp::create(this->builder, rQubit.getLoc(), qubitType, vQreg,
                                                   {}, IntegerAttr::get(i64Type, idxAttr.value()));
        }
        else {
            extractOp = quantum::ExtractOp::create(this->builder, rQubit.getLoc(), qubitType, vQreg,
                                                   idxValue, nullptr);
        }

        return extractOp;
    }
};

void handleRegion(IRRewriter &builder, Region &r);

//
// Misc helpers
//

/**
 * @brief Given a reference semantics operation instance, migrate it to value semantics.
 *
 * We create the corresponding value semantics operation, with exactly the same operands and
 * attributes, except we replace the rQubit (and rQreg) Values with the corresponding current
 * vQubit (and vQreg) Values.
 *
 * If the vQubit Values do not exist in the IR yet, a quantum.extract op from the corresponding
 * quantum.reg is created, and the newly extracted quantum.bit Value is used.
 *
 * The result types of the value semantics operation will be the same as the old one, unless
 * explicitly overriden in the `newResultTypes` argument. When overridden, all new results types
 * that is a quantum.bit or quantum.reg will appear at the end of the results of the new operation.
 *
 * Any vQreg and root vQubit result Values from the newly created value semantics operation is
 * updated into the tracker. If no `newResultTypes` are provided, this update does not happen,
 * as none of the results from the old qref op could be a vQreg or vQubit.
 *
 * @tparam OpTy
 * @param builder
 * @param qrefOp
 * @param tracker
 * @param newResultTypes
 * @return OpTy
 */
template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp, QubitValueTracker &tracker,
                               std::optional<TypeRange> newResultTypes = std::nullopt)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(qrefOp);
    Location loc = qrefOp->getLoc();

    TransientQubitExtractor extractor(qrefOp, tracker, builder);

    // Create the new op using the generic state-based approach
    // We cannot just clone, since we are changing the op type
    OperationState state(loc, OpTy::getOperationName());

    SmallVector<Value> vOperands;
    for (Value v : qrefOp->getOperands()) {
        if (isa<qref::QubitType>(v.getType()) && isRootRQubit(v)) {
            vOperands.push_back(tracker.getCurrentVQubit(v));
        }
        else if (isa<qref::QuregType>(v.getType())) {
            vOperands.push_back(tracker.getCurrentVQreg(v));
        }
        else {
            vOperands.push_back(v);
        }
    }
    for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                              extractor.getNonRootQubitOperandIndices())) {
        vOperands[idx] = vQubit;
    }

    state.addOperands(vOperands);
    state.addAttributes(qrefOp->getAttrs());

    SmallVector<Type> outTypes(qrefOp->getResultTypes());
    if (newResultTypes.has_value()) {
        outTypes.append(newResultTypes->begin(), newResultTypes->end());
    }
    state.addTypes(outTypes);

    Operation *newOp = builder.create(state);

    // Update tracker with results
    if (newResultTypes.has_value()) {
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(qrefOp->getOperand(i), newOp->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(qrefOp->getOperand(i), newOp->getResult(j));
        }
    }

    return cast<OpTy>(newOp);
}

/**
Given a qref gate operation, compute the result segment sizes for the corresponding value semantics
gate operation.

The reference semantics gates do not produce results.
Therefore, we need to manually set the result segment sizes for the corresponding
value semantics gate.
 */
DenseI32ArrayAttr getResultSegmentSizes(IRRewriter &builder, qref::QuantumGate rGateOp)
{
    int32_t non_ctrl_len = rGateOp.getNonCtrlQubitOperands().size();
    int32_t ctrl_len = rGateOp.getCtrlQubitOperands().size();
    return builder.getDenseI32ArrayAttr({non_ctrl_len, ctrl_len});
}

//
// Handlers for each op
//

// Memory ops

void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rAllocOp);

    Location loc = rAllocOp.getLoc();
    MLIRContext *ctx = rAllocOp.getContext();
    Type qregType = quantum::QuregType::get(ctx);
    Type i64Type = builder.getI64Type();

    quantum::AllocOp vAllocOp;
    std::optional<uint64_t> nqubitsAttr = rAllocOp.getNqubitsAttr();
    if (nqubitsAttr.has_value()) {
        vAllocOp = quantum::AllocOp::create(builder, loc, qregType, {},
                                            IntegerAttr::get(i64Type, *nqubitsAttr));
    }
    else {
        vAllocOp = quantum::AllocOp::create(builder, loc, qregType, rAllocOp.getNqubits(), nullptr);
    }
    tracker.setCurrentVQreg(rAllocOp.getQreg(), vAllocOp.getQreg());
}

void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rDeallocOp);
    Location loc = rDeallocOp.getLoc();

    quantum::DeallocOp::create(builder, loc, tracker.getCurrentVQreg(rDeallocOp.getQreg()));

    builder.eraseOp(rDeallocOp);
}

// Gate Ops

void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rGateOp.getContext();

    SmallVector<Type> qubitResultsType;
    for (size_t i = 0; i < rGateOp.getQubitOperands().size(); i++) {
        qubitResultsType.push_back(quantum::QubitType::get(ctx));
    }

    quantum::QuantumOperation vGateOp;
    Operation *_rGateOp = rGateOp.getOperation();
    if (auto rCustomOp = dyn_cast<qref::CustomOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::CustomOp>(builder, rCustomOp, tracker,
                                                               qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rCustomOp));
    }
    else if (auto rPauliRotOP = dyn_cast<qref::PauliRotOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PauliRotOp>(builder, rPauliRotOP, tracker,
                                                                 qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rPauliRotOP));
    }
    else if (auto rGPhaseOp = dyn_cast<qref::GlobalPhaseOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::GlobalPhaseOp>(builder, rGPhaseOp, tracker,
                                                                    qubitResultsType);
    }
    else if (auto rMultiRZOp = dyn_cast<qref::MultiRZOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::MultiRZOp>(builder, rMultiRZOp, tracker,
                                                                qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rMultiRZOp));
    }
    else if (auto rPCPhaseOp = dyn_cast<qref::PCPhaseOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PCPhaseOp>(builder, rPCPhaseOp, tracker,
                                                                qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rPCPhaseOp));
    }
    else if (auto qQubitUnitaryOp = dyn_cast<qref::QubitUnitaryOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::QubitUnitaryOp>(builder, qQubitUnitaryOp,
                                                                     tracker, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, qQubitUnitaryOp));
    }
    else if (auto rSetStateOp = dyn_cast<qref::SetStateOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::SetStateOp>(builder, rSetStateOp, tracker,
                                                                 qubitResultsType);
    }
    else if (auto rSetBasisStateOp = dyn_cast<qref::SetBasisStateOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::SetBasisStateOp>(builder, rSetBasisStateOp,
                                                                      tracker, qubitResultsType);
    }

    builder.eraseOp(rGateOp);
}

// Observable Ops

void handleCompbasis(IRRewriter &builder, qref::ComputationalBasisOp rCompbasisOp,
                     QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vCompbasisOp =
        migrateOpToValueSemantics<quantum::ComputationalBasisOp>(builder, rCompbasisOp, tracker);
    builder.replaceOp(rCompbasisOp, vCompbasisOp);
}

void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vNamedObsOp =
        migrateOpToValueSemantics<quantum::NamedObsOp>(builder, rNamedObsOp, tracker);
    builder.replaceOp(rNamedObsOp, vNamedObsOp);
}

void handleHermitian(IRRewriter &builder, qref::HermitianOp rHermitianOp,
                     QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vHermitianOp =
        migrateOpToValueSemantics<quantum::HermitianOp>(builder, rHermitianOp, tracker);
    builder.replaceOp(rHermitianOp, vHermitianOp);
}

// Control flow

void handleFor(IRRewriter &builder, scf::ForOp forOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    Location loc = forOp->getLoc();
    // MLIRContext *ctx = forOp.getContext();

    TransientQubitExtractor extractor(forOp->getRegion(0), tracker, builder);

    // 1. Append the rQuantum values needed by the loop to the loop arguments
    SetVector<Value> rValuesUsedByRegion;
    getNecessaryRegionRValues(forOp->getRegion(0), rValuesUsedByRegion);

    SmallVector<Value> newIterArgs(forOp.getInitArgs());
    builder.setInsertionPoint(forOp);
    for (Value v : rValuesUsedByRegion) {
        newIterArgs.push_back(v);
    }

    for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                              extractor.getNonRootQubitOperandIndices())) {
        // The body block of the for loop region always has the loop index as the first argument
        newIterArgs[idx - 1] = vQubit;
    }
    auto newLoop = scf::ForOp::create(builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                                      forOp.getStep(), newIterArgs);

    // 2. Move operations from old body to new body
    // Map any root values to the new loop arguments
    // builder.inlineRegionBefore(forOp.getRegion(), newLoop.getRegion(),
    // newLoop.getRegion().end()); for (auto rValue : rValuesUsedByRegion) {
    //     // assert(isa<qref::QuregType, qref:QubitType>(rValue.getType()) &&
    //     //     "Expected the only extra loop arguments in value semantics to be quantum registers
    //     and qubits");
    //     // Value vNewArg =
    //     //     newLoop.getRegion().getBlocks().front().addArgument(quantum::QuregType::get(ctx),
    //     //     loc);
    //     //qubitValueTrackers.at(rQreg)->setCurrentVQreg(vQregNewArg);
    //     if (isa<qref::QuregType>(rValue.getType())){
    //         tracker.setCurrentVQreg(, );
    //     }
    // }

    // 3. Replace the

    // llvm::errs() << "dumping: " << newLoop << "\n";
    llvm::errs() << "dumping: " << newLoop->getParentOfType<func::FuncOp>() << "\n";

    // // 2. The new forOp iteration args must take in the registers
    // // Remove the default empty block, it doesn't have the new block arg signature
    // auto newLoop = scf::ForOp::create(builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
    //                                   forOp.getStep(), newInitArgs);
    // builder.eraseBlock(newLoop.getBody());

    // // 3. Move operations from old body to new body
    // // The old loop body still refers to the old block arguments.
    // // We must map them to the new ones.
    // builder.inlineRegionBefore(forOp.getRegion(), newLoop.getRegion(),
    // newLoop.getRegion().end()); for (auto rQreg : rQregsUsedByRegion) {
    //     assert(isa<qref::QuregType>(rQreg.getType()) &&
    //            "Expected the only extra loop arguments in value semantics to be quantum
    //            registers");
    //     Value vQregNewArg =
    //         newLoop.getRegion().getBlocks().front().addArgument(quantum::QuregType::get(ctx),
    //         loc);
    //     qubitValueTrackers.at(rQreg)->setCurrentVQreg(vQregNewArg);
    // }

    // handleRegion(builder, newLoop.getRegion(), qubitValueTrackers);

    // // 4. Insert loop region registers and Yield
    // // builder.setInsertionPoint(newLoop.getRegion().getBlocks().back().getTerminator());
    // auto yieldOp = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    // size_t numOldYields = yieldOp->getNumResults();
    // SmallVector<Value> yieldOperands(yieldOp.getOperands());
    // for (Value rQreg : rQregsUsedByRegion) {
    //     Value insertedQreg = qubitValueTrackers.at(rQreg)->insertAllDanglingQubits(yieldOp);
    //     yieldOperands.push_back(insertedQreg);
    // }
    // builder.setInsertionPoint(yieldOp);
    // scf::YieldOp::create(builder, loc, yieldOperands);
    // builder.eraseOp(yieldOp);

    // // 5. New for loop's returned qreg is the new value semantics qreg value in the outer scope
    // for (auto [i, rQreg] : llvm::enumerate(rQregsUsedByRegion)) {
    //     qubitValueTrackers.at(rQreg)->setCurrentVQreg(newLoop->getResult(numOldYields + i));
    // }

    // builder.eraseOp(forOp);
}

// Driver

void handleRegion(IRRewriter &builder, Region &r)
{
    QubitValueTracker tracker;
    Location loc = r.getLoc();
    MLIRContext *ctx = r.getContext();

    // Set up the root values in the tracker if there are any arguments to the region
    assert(r.hasOneBlock() && "Expected single block region");
    Block &block = r.front();
    std::vector<unsigned> rArgsErasureIndices;
    SmallVector<BlockArgument> regionArgs(block.getArguments());
    unsigned _numCreatedNewArgs = 0;
    for (auto [i, arg] : llvm::enumerate(regionArgs)) {
        if (!isa<qref::QubitType, qref::QuregType>(arg.getType())) {
            continue;
        }
        unsigned newArgIdx = i + _numCreatedNewArgs;
        if (isa<qref::QubitType>(arg.getType())) {
            Value vQubit = block.insertArgument(newArgIdx, quantum::QubitType::get(ctx), loc);
            tracker.setCurrentVQubit(arg, vQubit);
        }
        else if (isa<qref::QuregType>(arg.getType())) {
            Value vQreg = block.insertArgument(newArgIdx, quantum::QuregType::get(ctx), loc);
            tracker.setCurrentVQreg(arg, vQreg);
        }
        _numCreatedNewArgs++;
        rArgsErasureIndices.push_back(newArgIdx + 1);
    }

    r.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto rAllocOp = dyn_cast<qref::AllocOp>(op)) {
            handleAlloc(builder, rAllocOp, tracker);
        }
        else if (auto rGateOp = dyn_cast<qref::QuantumOperation>(op)) {
            handleGate(builder, rGateOp, tracker);
        }
        else if (auto rCompbasisOp = dyn_cast<qref::ComputationalBasisOp>(op)) {
            handleCompbasis(builder, rCompbasisOp, tracker);
        }
        else if (auto rNamedObsOp = dyn_cast<qref::NamedObsOp>(op)) {
            handleNamedObs(builder, rNamedObsOp, tracker);
        }
        else if (auto rHermitianOp = dyn_cast<qref::HermitianOp>(op)) {
            handleHermitian(builder, rHermitianOp, tracker);
        }
        else if (auto rDeallocOp = dyn_cast<qref::DeallocOp>(op)) {
            handleDealloc(builder, rDeallocOp, tracker);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            handleFor(builder, forOp, tracker);
        }
    });

    r.walk([&](qref::GetOp getOp) {
        assert(getOp.use_empty() &&
               "qref.bit Values must have no uses after the semantic conversion");
        builder.eraseOp(getOp);
    });

    // Special: Need to edit function type and return vQubits if the region comes from a funcop
    // Since we don't do a designated handler for func ops, unlike the other control flow ops
    if (auto funcOp = dyn_cast<func::FuncOp>(r.getParentOp())) {
        auto retOp = cast<func::ReturnOp>(funcOp.getBody().back().getTerminator());
        SmallVector<Value> retVals(retOp.getOperands());
        for (auto arg : block.getArguments()) {
            if (isa<qref::QubitType>(arg.getType())) {
                retVals.push_back(tracker.getCurrentVQubit(arg));
            }
            else if (isa<qref::QuregType>(arg.getType())) {
                retVals.push_back(tracker.getCurrentVQreg(arg));
            }
        }
        retOp->setOperands(retVals);

        for (unsigned i : llvm::reverse(rArgsErasureIndices)) {
            // A note on the reverse: if we want to erase args at indices 0 and 1, and the original
            // args are [a, b, c, d], we need to erase 1 (b) first, then 0 (a). Because if we erase
            // arg0 (a) first, then after that erasure iteration, the 3 new args are [b, c, d], and
            // the arg now at index 1 was the original arg2 (c)!
            r.front().eraseArgument(i);
        }
        funcOp.setFunctionType(
            FunctionType::get(ctx, block.getArgumentTypes(), TypeRange(retVals)));
    }
    else {
        // Not a function, just delete qref args, no need to set function type
        for (unsigned i : llvm::reverse(rArgsErasureIndices)) {
            r.front().eraseArgument(i);
        }
    }
}

} // anonymous namespace

namespace catalyst {
namespace qref {

#define GEN_PASS_DECL_VALUESEMANTICSCONVERSIONPASS
#define GEN_PASS_DEF_VALUESEMANTICSCONVERSIONPASS
#include "QRef/Transforms/Passes.h.inc"

struct ValueSemanticsConversionPass
    : impl::ValueSemanticsConversionPassBase<ValueSemanticsConversionPass> {
    using ValueSemanticsConversionPassBase::ValueSemanticsConversionPassBase;

    void runOnOperation() final
    {
        Operation *mod = getOperation();
        MLIRContext *ctx = mod->getContext();
        auto *qrefDialect = ctx->getLoadedDialect<qref::QRefDialect>();
        IRRewriter builder(ctx);

        // // CSE potential duplicated getOps
        // PassManager pm(ctx);
        // pm.addPass(createCSEPass());

        // Collect all qnode functions.
        SetVector<func::FuncOp> qnodeFuncs;
        mod->walk([&](func::FuncOp f) {
            if (f->hasAttrOfType<UnitAttr>("quantum.node")) {
                qnodeFuncs.insert(f);
            }
        });

        for (auto qnodeFunc : qnodeFuncs) {
            handleRegion(builder, qnodeFunc.getBody());

            // Clean up: erase remaining qref dialect ops
            // Due to the nature of the reference semantics dialect, qref ops all have full side
            // effects, and will not delete themselves.
            SmallVector<Operation *> toErase;
            qnodeFunc->walk([&](Operation *op) {
                if (op->getDialect() == qrefDialect) {
                    toErase.push_back(op);
                }
            });
            for (Operation *op : llvm::reverse(toErase)) {
                op->erase();
            }
        }
    }
};

} // namespace qref
} // namespace catalyst
