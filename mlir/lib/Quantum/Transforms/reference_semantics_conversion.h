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

#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "MBQC/IR/MBQCOps.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace {

struct QubitValueTracker;

// The main converter function
template <typename OpTy>
OpTy migrateOpToReferenceSemantics(IRRewriter &builder, Operation *vOp, QubitValueTracker &tracker);

// Individual handlers for each op
void handleAlloc(IRRewriter &builder, quantum::AllocOp vAllocOp, QubitValueTracker &tracker,
                 SmallVector<Operation *> &erasureWorklist);
void handleDealloc(IRRewriter &builder, quantum::DeallocOp vDeallocOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist);
void handleAllocQubit(IRRewriter &builder, quantum::AllocQubitOp vAllocQbOp,
                      QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist);
void handleDeallocQubit(IRRewriter &builder, quantum::DeallocQubitOp vDeallocQbOp,
                        QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist);
void handleExtract(IRRewriter &builder, quantum::ExtractOp vExtractOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist);
void handleInsert(quantum::InsertOp vInsertOp, QubitValueTracker &tracker,
                  SmallVector<Operation *> &erasureWorklist);
void handleGate(IRRewriter &builder, quantum::QuantumOperation vGateOp, QubitValueTracker &tracker,
                SmallVector<Operation *> &erasureWorklist);
void handleMeasure(IRRewriter &builder, quantum::MeasureOp vMeasureOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist);
void handleMeasureInBasis(IRRewriter &builder, mbqc::MeasureInBasisOp vMeasureInBasisOp,
                          QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist);
void handlePPM(IRRewriter &builder, pbc::PPMeasurementOp vPPMOp, QubitValueTracker &tracker,
               SmallVector<Operation *> &erasureWorklist);
// void handleCall(IRRewriter &builder, func::CallOp callOp, QubitValueTracker &tracker);
void handleCompbasis(IRRewriter &builder, quantum::ComputationalBasisOp vCompbasisOp,
                     QubitValueTracker &tracker);
void handleNamedObs(IRRewriter &builder, quantum::NamedObsOp vNamedObsOp,
                    QubitValueTracker &tracker);
void handleHermitian(IRRewriter &builder, quantum::HermitianOp vHermitianOp,
                     QubitValueTracker &tracker);
void handleGraphStatePrep(IRRewriter &builder, mbqc::GraphStatePrepOp vGraphStatePrepOp,
                          QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist);
void handleAdjoint(IRRewriter &builder, quantum::AdjointOp vAdjointOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist);
void handleIf(IRRewriter &builder, scf::IfOp ifOp, QubitValueTracker &tracker,
              SmallVector<Operation *> &erasureWorklist);
void handleSwitch(IRRewriter &builder, scf::IndexSwitchOp switchOp, QubitValueTracker &tracker,
                  SmallVector<Operation *> &erasureWorklist);
void handleFor(IRRewriter &builder, scf::ForOp forOp, QubitValueTracker &tracker,
               SmallVector<Operation *> &erasureWorklist);
void handleWhile(IRRewriter &builder, scf::WhileOp whileOp, QubitValueTracker &tracker,
                 SmallVector<Operation *> &erasureWorklist);
// void handleSubroutine(IRRewriter &builder, func::FuncOp f,
//                       const SetVector<Value> &rValuesUsedBySubroutine);

// Main driver
std::optional<SmallVector<Operation *>> handleRegion(IRRewriter &builder, Region &r,
                                                     QubitValueTracker &tracker, bool erase = true);
} // anonymous namespace
