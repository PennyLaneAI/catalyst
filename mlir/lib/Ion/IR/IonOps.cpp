// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Ion/IR/IonDialect.h"
#include "Ion/IR/IonOps.h"

using namespace mlir;
using namespace catalyst::ion;

//===----------------------------------------------------------------------===//
// Ion op definitions.
//===----------------------------------------------------------------------===//

#include "Ion/IR/IonEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Ion/IR/IonOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Ion op builders.
//===----------------------------------------------------------------------===//

void ParallelProtocolOp::build(OpBuilder &builder, OperationState &result, ValueRange inQubits,
                               BodyBuilderFn bodyBuilder)
{
    OpBuilder::InsertionGuard guard(builder);
    Location loc = result.location;
    Type ionType = IonType::get(result.getContext());

    // The parallel protocol op can interact with the outside world by accepting
    // either ion types or qubit types.
    // We allow qubit types because during `quantum-to-ion`, during gate-to-pulse
    // decomposition, we still need the core quantum dialect to track SSA def use
    // chains of qubit values.
    // After gate-to-pulse decomposition is finished, we change all parallel protocol
    // ops to return ion types.
    // Note that the body region is shielded from the outside, so its block can
    // have an ion type argument directly
    result.addOperands(inQubits);
    for (Value v : inQubits) {
        result.addTypes(v.getType());
    }

    Region *bodyRegion = result.addRegion();
    Block *bodyBlock = builder.createBlock(bodyRegion);
    for (Value v : inQubits) {
        bodyBlock->addArgument(ionType, v.getLoc());
    }

    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, loc, bodyBlock->getArguments());

    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<ion::YieldOp>(loc, bodyBlock->getArguments());
}

//===----------------------------------------------------------------------===//
// Ion op canonicalizers.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Ion op verifiers.
//===----------------------------------------------------------------------===//
