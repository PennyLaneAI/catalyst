// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "RTIO/IR/RTIOOps.h"

using namespace mlir;
using namespace catalyst::rtio;

//===----------------------------------------------------------------------===//
// RTIO Operations
//===----------------------------------------------------------------------===//

LogicalResult RTIOSyncOp::verify()
{
    // Ensure at least one event is provided
    if (getEvents().empty()) {
        return emitOpError("requires at least one event to synchronize");
    }
    return success();
}

// LogicalResult RTIOQubitToChannelOp::canonicalize(RTIOQubitToChannelOp op,
//                                                  mlir::PatternRewriter &rewriter)
// {
//     Block *currentBlock = op->getBlock();
//     Value qubit = op.getQubit();
//     Type channelType = op.getChannel().getType();



//     // Try to find the same qubit_to_channel operation between [block->begin, op)
//     for (Operation &prevOp : llvm::make_range(currentBlock->begin(), op->getIterator())) {
//         if (auto prevQubitToChannel = dyn_cast<RTIOQubitToChannelOp>(&prevOp)) {
//             if (prevQubitToChannel.getQubit() == qubit &&
//                 prevQubitToChannel.getChannel().getType() == channelType) {
//                 rewriter.replaceOp(op, prevQubitToChannel.getChannel());
//                 return success();
//             }
//         }
//     }

//     return failure();
// }

// LogicalResult RTIOChannelOp::canonicalize(RTIOChannelOp op, mlir::PatternRewriter &rewriter)
// {
//     Block *currentBlock = op->getBlock();
//     Value channel = op.getChannel();
//     Type channelType = channel.getType();

//     for (Operation &prevOp : llvm::make_range(currentBlock->begin(), op->getIterator())) {
//         if (auto prevChannelOp = dyn_cast<RTIOChannelOp>(&prevOp)) {
//             if (prevChannelOp.getChannel().getType() == channelType) {
//                 rewriter.replaceOp(op, prevChannelOp.getChannel());
//                 return success();
//             }
//         }
//     }
//     return failure();
// }

#define GET_OP_CLASSES
#include "RTIO/IR/RTIOOps.cpp.inc"
