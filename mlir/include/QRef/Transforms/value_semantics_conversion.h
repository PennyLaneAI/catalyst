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

#pragma once

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseMapInfo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst;

namespace ReferenceToValueSemanticsConversion {

// Structs holding the core conversion logic
struct QubitValueTracker;
struct TransientQubitExtractor;

// A struct to store the register and the index of rQubits from a qref.get operation.
// This struct is intended to be the keys in `llvm::DenseMap`s.
struct rQubitGetOpInfo {
    Value reg;
    int64_t idxAttr;
    Value idx;

    rQubitGetOpInfo(Value _reg, Value _idx) : reg(_reg), idxAttr(-1), idx(_idx) {}

    rQubitGetOpInfo(Value _reg, int64_t _idxAttr) : reg(_reg), idxAttr(_idxAttr), idx(nullptr) {}

    bool operator==(const rQubitGetOpInfo &other) const
    {
        return reg == other.reg && idxAttr == other.idxAttr && idx == other.idx;
    }
};

// Misc helper functions
std::optional<rQubitGetOpInfo> getGetOpInfo(Value rQubit);
Value getRSourceRegisterValue(Value rQubit);
void getNecessaryRegionRValues(Region &r, SetVector<Value> &necessaryRegionRValues);
DenseI32ArrayAttr getResultSegmentSizes(IRRewriter &builder, qref::QuantumGate rGateOp);
void addRootVValuesToRetOp(Operation *retOp, ArrayRef<Value> rValuesToReturn,
                           QubitValueTracker &tracker);

// The main converter function
template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp, QubitValueTracker &tracker,
                               std::optional<TypeRange> newResultTypes = std::nullopt);

// Individual handlers for each op
void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp, QubitValueTracker &tracker);
void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp, QubitValueTracker &tracker);
void handleAllocQubit(IRRewriter &builder, qref::AllocQubitOp rAllocQbOp,
                      QubitValueTracker &tracker);
void handleDeallocQubit(IRRewriter &builder, qref::DeallocQubitOp rDeallocQbOp,
                        QubitValueTracker &tracker);
void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp, QubitValueTracker &tracker);
void handleMeasure(IRRewriter &builder, qref::MeasureOp rMeasureOp, QubitValueTracker &tracker);
void handleCall(IRRewriter &builder, func::CallOp callOp, QubitValueTracker &tracker);
void handleCompbasis(IRRewriter &builder, qref::ComputationalBasisOp rCompbasisOp,
                     QubitValueTracker &tracker);
void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp, QubitValueTracker &tracker);
void handleHermitian(IRRewriter &builder, qref::HermitianOp rHermitianOp,
                     QubitValueTracker &tracker);
void handleAdjoint(IRRewriter &builder, qref::AdjointOp rAdjointOp, QubitValueTracker &tracker);
void handleIf(IRRewriter &builder, scf::IfOp ifOp, QubitValueTracker &tracker);
void handleSwitch(IRRewriter &builder, scf::IndexSwitchOp switchOp, QubitValueTracker &tracker);
void handleFor(IRRewriter &builder, scf::ForOp forOp, QubitValueTracker &tracker);
void handleWhile(IRRewriter &builder, scf::WhileOp whileOp, QubitValueTracker &tracker);
void handleSubroutine(IRRewriter &builder, func::FuncOp f,
                      SetVector<Value> &rValuesUsedBySubroutine);

// Main driver
void handleRegion(IRRewriter &builder, Region &r, QubitValueTracker &tracker);
} // namespace ReferenceToValueSemanticsConversion

namespace llvm {

// Boilerplate to enable using `rQubitGetOpInfo` as DenseMap keys.
template <> struct DenseMapInfo<ReferenceToValueSemanticsConversion::rQubitGetOpInfo> {
    using rQubitGetOpInfo = ReferenceToValueSemanticsConversion::rQubitGetOpInfo;

    static inline rQubitGetOpInfo getEmptyKey()
    {
        return rQubitGetOpInfo(DenseMapInfo<Value>::getEmptyKey(), -1);
    }

    static inline rQubitGetOpInfo getTombstoneKey()
    {
        return rQubitGetOpInfo(DenseMapInfo<Value>::getTombstoneKey(), -2);
    }

    static unsigned getHashValue(const rQubitGetOpInfo &val)
    {
        return hash_combine(hash_value(val.reg.getAsOpaquePointer()), val.idxAttr,
                            val.idx ? static_cast<size_t>(hash_value(val.idx.getAsOpaquePointer()))
                                    : 0);
    }

    static bool isEqual(const rQubitGetOpInfo &lhs, const rQubitGetOpInfo &rhs)
    {
        return lhs == rhs;
    }
};
} // namespace llvm
