// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef QUANTUM_GRAPH_DECOMPOSITION_INTERFACES
#define QUANTUM_GRAPH_DECOMPOSITION_INTERFACES

include "mlir/IR/OpBase.td"
include "Catalyst/Analysis/ResourceInterfaces.td"

def DecomposableGate : OpInterface<"DecomposableGate"> {
    let description = [{
        This interface provides a generic way to query decomposition data from a quantum operation.
        An operation must implement this interface in order to compatible with the `graph-decomposition` pass.
        Note that implementing this interface does not guarantee that an operation will decomposed successfully, only that it will be registered with the graph.
    }];

    let cppNamespace = "::catalyst::quantum";

    let methods = [
        InterfaceMethod<
            "Return the name of the corresponding frontend operator (i.e. the rule supplier).", "std::string", "getOperatorName"
        >,
        InterfaceMethod<
            "Return a map of frontend parameter name to shape and dtype of dynamic data.",
            "llvm::StringMap<llvm::SmallVector<mlir::Type>>", "getDynamicShape"
        >,
        InterfaceMethod<
            "Return a map of frontend parameter name to number of wires (excluding control wires).",
            "llvm::StringMap<size_t>", "getWireLens"
        >,
        InterfaceMethod<
            "Return a map of frontend parameter name to static data values.",
            "mlir::DictionaryAttr", "getStaticData"
        >,
        InterfaceMethod<
            "Return a string representation of any additional data an operator may need to provide"
            " for uniqueness (ex. UID), or emptystring if not applicable.",
            "std::string", "getExtraData", (ins), [{}], [{ return ""; }]
        >,
        InterfaceMethod<
            "Return the operation's ID for use with the graph. "
            "This should be computed by the following convention:\n"
            "```\n"
            "name + {map params to list of types} + {map wire args to number of wires} + {static data} + [extra data]\n"
            "```\n"
            "Note that extra data is currently specific to `OperatorOp`s, and should only appear when present."
            "Each of these maps should be sorted lexicographically by key."
            "The static data map is spelled by MLIR's attribute printer, which the frontend also "
            "depends on identical id generation."
            "In general, this should be computed by the default here and not overridden.",
            "std::string", "getGraphOpId", (ins), [{}], [{ return ::catalyst::quantum::defaultGetGraphOpId($_op.getOperation()); }]
        >
    ];
}

#endif // QUANTUM_GRAPH_DECOMPOSITION_INTERFACES
