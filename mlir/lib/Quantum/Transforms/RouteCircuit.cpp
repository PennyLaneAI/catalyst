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

// This algorithm is taken from https://arxiv.org/pdf/2012.07711, table 6 (Equivalences for
// basis-states in SWAP gate)

#define DEBUG_TYPE "routecircuit"

#include <iostream>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#include "c++/z3++.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"


using namespace mlir;
using namespace catalyst;

using namespace z3;

namespace catalyst {
#define GEN_PASS_DEF_ROUTINGPASS
#define GEN_PASS_DECL_ROUTINGPASS
#include "Quantum/Transforms/Passes.h.inc"


struct RoutingPass : public impl::RoutingPassBase<RoutingPass> {
    using impl::RoutingPassBase<RoutingPass>::RoutingPassBase;

    std::vector<std::pair<int, int>> parseHardwareGraph(std::string s, std::string delimiter, std::set<int> *physicalQubits) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector< std::pair<int, int> > res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr (pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;

            size_t commaPos = token.find(',');
            int u = std::stoi(token.substr(1, commaPos));
            int v = std::stoi(token.substr(commaPos + 1, token.size() - 2));
            std::pair<int, int> edge = std::make_pair(u,v);
            (*physicalQubits).insert(u);
            (*physicalQubits).insert(v);
            res.push_back (edge);
        }
        return res;
    }

    int countLogicalQubit(Operation *op)
    {
        int numQubits = cast<quantum::AllocOp>(op).getNqubitsAttr().value_or(0);
        assert(numQubits != 0 && "PPM specs with dynamic number of qubits is not implemented");
        auto parentFuncOp = op->getParentOfType<func::FuncOp>();
        return numQubits;
    }

    void runOnOperation() override {

        std::set<int> physicalQubits;
        std::vector<std::pair<int, int>> couplingMap = parseHardwareGraph(hardwareGraph, ";", &physicalQubits);
        int numPhysicalQubits = physicalQubits.size();
        int numLogicalQubits = physicalQubits.size();

        llvm::outs() << "Number of Physical Qubits on the Hardware : " << numPhysicalQubits << "\n";
        llvm::outs() << "Physical Qubits on the Hardware :\n";
        for (auto i : physicalQubits) llvm::outs() << i << "\n";

        llvm::outs() << "Hardware Topology :\n";
        for (auto i : couplingMap) llvm::outs() << "(" << i.first << "," << i.second << ")\n";

        getOperation()->walk([&](Operation *op) {
            // Skip top-level container ops if desired
            if (isa<ModuleOp>(op)) {
                return;
            }

            else if (isa<quantum::AllocOp>(op)) {
                numLogicalQubits = countLogicalQubit(op);
                llvm::outs() << "Number of Logical Qubits in the Circuit : " << numLogicalQubits << "\n";
            }
        });

        context c;

        expr x = c.bool_const("x");
        expr y = c.bool_const("y");
        expr conjecture = (!(x && y)) == (!x || !y);
        
        solver s(c);
        s.add(!conjecture);
        llvm::outs() <<  "Z3 result " << s.to_smt2() << "\n";
        llvm::outs() <<  "Z3 check " << s.check() << "\n";
        llvm::outs() <<  "sat " << z3::sat << "\n";
        llvm::outs() <<  "unsat " << z3::unsat << "\n";
        llvm::outs() <<  "unknown " << z3::unknown << "\n";

    }
};

std::unique_ptr<Pass> createRoutingPass()
{
    return std::make_unique<RoutingPass>();
}

} // namespace catalyst
