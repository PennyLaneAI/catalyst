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

#include "Transport/IR/TransportDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "Transport/IR/TransportOps.h"

using namespace mlir;
using namespace catalyst::transport;

//===----------------------------------------------------------------------===//
// Transport dialect definitions.
//===----------------------------------------------------------------------===//

#include "Transport/IR/TransportEnums.cpp.inc"
#include "Transport/IR/TransportOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Transport type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Transport/IR/TransportOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Transport attribute definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Transport/IR/TransportAttributes.cpp.inc"

StringAttr NodeAttr::keyOr(llvm::StringRef fallback) const {
    if (StringAttr n = getName(); n && !n.getValue().empty()) {
        return n;
    }
    return StringAttr::get(getContext(), fallback);
}

bool NodeAttr::isOutOfProcess() const {
    BoolAttr r = getOutOfProcess();
    return r && r.getValue();
}

static int64_t intOr(IntegerAttr field, int64_t dflt) { return field ? field.getInt() : dflt; }

int64_t NodeAttr::oobPort() const {
    // A port is unsigned, so zero-extend
    IntegerAttr p = getOobPort();
    return p ? static_cast<int64_t>(p.getValue().getZExtValue()) : 0;
}
int64_t NodeAttr::inBytes() const { return getInBytes().getInt(); }
int64_t NodeAttr::outBytes() const { return getOutBytes().getInt(); }
int64_t NodeAttr::workItemIdx() const { return intOr(getWorkItemIdx(), 0); }

LogicalResult BacklineAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   StringAttr transport, NodeAttr controller,
                                   llvm::ArrayRef<NodeAttr> coprocessors) {
    if (!controller) {
        return emitError() << "backline requires a controller";
    }
    if (!transport || (transport.getValue() != "rdma" && transport.getValue() != "memcpy")) {
        return emitError() << "backline transport must be 'rdma' or 'memcpy'";
    }
    for (NodeAttr c : coprocessors) {
        if (!c) {
            return emitError() << "null coprocessor";
        }
        // A 'peer' is the address of the out-of-band handshake, which only rdma performs.
        if (transport.getValue() == "rdma" && (!c.getPeer() || c.getPeer().getValue().empty())) {
            return emitError() << "coprocessor requires a 'peer' under the 'rdma' transport";
        }
        // The attribute is i32 so it can hold the whole unsigned range, but the runtime call
        // takes a uint16_t. Reject an out-of-range port here rather than truncate it silently.
        if (c.getOobPort() && c.oobPort() > 65535) {
            return emitError() << "coprocessor 'oob_port' must be in 0..65535, got " << c.oobPort();
        }
        if (!c.getSymbol() || c.getSymbol().getValue().empty()) {
            return emitError() << "coprocessor requires a 'symbol'";
        }
        if (transport.getValue() == "memcpy") {
            if (c.isOutOfProcess() != controller.isOutOfProcess()) {
                return emitError()
                       << "memcpy transport requires controller and coprocessor on the same node";
            }
            // Memcpy is in-process, so two out-of-process nodes are only on the "same node" if
            // they share a catalyst-executor address.
            if (c.isOutOfProcess()) {
                llvm::StringRef ctrlAddr = controller.getAddress().getValue();
                llvm::StringRef copAddr = c.getAddress().getValue();
                if (ctrlAddr != copAddr) {
                    return emitError()
                           << "memcpy transport with remote nodes requires controller and "
                              "coprocessor on the same executor address (got '"
                           << ctrlAddr << "' vs '" << copAddr << "')";
                }
            }
        }
    }
    if (!controller.getInBytes() || !controller.getOutBytes()) {
        return emitError() << "controller requires 'in_bytes' and 'out_bytes'";
    }
    return success();
}

void TransportDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Transport/IR/TransportOpsTypes.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "Transport/IR/TransportAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Transport/IR/TransportOps.cpp.inc"
        >();
}
