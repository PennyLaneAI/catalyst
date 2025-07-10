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
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#include "c++/z3++.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Analysis/SliceAnalysis.h"
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

    llvm::DenseMap<std::pair<int, int>, bool> parseHardwareGraph(std::string s, std::string delimiter, std::set<int> *physicalQubits) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        llvm::DenseMap<std::pair<int, int>, bool> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr (pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;

            size_t commaPos = token.find(',');
            int u = std::stoi(token.substr(1, commaPos));
            int v = std::stoi(token.substr(commaPos + 1, token.size() - 2));
            (*physicalQubits).insert(u);
            (*physicalQubits).insert(v);
            res[std::make_pair(u,v)] = true;
            res[std::make_pair(v,u)] = true;
        }
        return res;
    }

    std::vector<int> generateRandomInitialMapping(std::set<int> *physicalQubits) {
        std::vector<int> randomInitialMapping((*physicalQubits).begin(), (*physicalQubits).end());
        // TODO: Generating completely random mapping is inefficient
        // Replace this with some initial mapping algorithm like BFS or Simulated Annealing

        // Random number generator
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(randomInitialMapping.begin(), randomInitialMapping.end(), g);
        return randomInitialMapping;
    }

    int getRegisterIndexOfOp(Value inQubit) {
        // TODO: There's a better way of doing this
        // instead of backtracking every single CustomOp back to its ExtractOp
        // Use DenseMap [inQubit] -> ExtractOp
        // this way, for any CustomOp, we can go to its definingOp's inQubits/outQubits 
        // and get the ExtractOp from the DenseMap
        Operation *prevOp = inQubit.getDefiningOp();
        if (isa<quantum::ExtractOp>(prevOp)) 
            return (cast<quantum::ExtractOp>(prevOp)).getIdxAttr().value();
        else {
            auto iteratePrevOpOutQubit = cast<quantum::CustomOp>(prevOp).getOutQubits();
            auto iteratePrevOpInQubit = cast<quantum::CustomOp>(prevOp).getInQubits();
            for (auto iter = 0; iter < iteratePrevOpOutQubit.size() ; iter++ ) {
                if (iteratePrevOpOutQubit[iter] == inQubit) 
                    return getRegisterIndexOfOp(iteratePrevOpInQubit[iter]);
            }
        }
    }
    
    void runOnOperation() override {

        std::set<int> physicalQubits;
        llvm::DenseMap<std::pair<int, int>, bool> couplingMap = parseHardwareGraph(hardwareGraph, ";", &physicalQubits);
        int numPhysicalQubits = physicalQubits.size();
        int numLogicalQubits = physicalQubits.size(); // works with automatic qubit management

        llvm::outs() << "Number of Physical Qubits on the Hardware : " << numPhysicalQubits << "\n";
        llvm::outs() << "Physical Qubits on the Hardware :\n";
        for (auto i : physicalQubits) llvm::outs() << i << "\n";

        llvm::outs() << "Hardware Topology :\n";
        for (auto& entry : couplingMap) {
            std::pair<int, int>& key = entry.first;
            bool value = entry.second;
            llvm::outs() << "(" << key.first << ", " << key.second << ") => " 
                    << (value ? "true" : "false") << "\n";
        }

        std::vector<int> randomInitialMapping;

        getOperation()->walk([&](Operation *op) {
            if (isa<quantum::AllocOp>(op)) {
                randomInitialMapping =  generateRandomInitialMapping(&physicalQubits);

                llvm::outs() << "Number of Logical Qubits in the Circuit : " << numLogicalQubits << "\n";
                llvm::outs() << "Random Initial Mapping:\n";
                for (auto i = 0; i < randomInitialMapping.size(); i++)
                    llvm::outs() << i << "->" << randomInitialMapping[i] << "\n";
            }
            else if (isa<quantum::CustomOp>(op)) {

                StringRef gateName = cast<quantum::CustomOp>(op).getGateName();
                int nQubits = cast<quantum::CustomOp>(op).getInQubits().size();
                auto inQubits = cast<quantum::CustomOp>(op).getInQubits();
                
                if (nQubits == 2) {
                    llvm::outs() << "Gate name: " << gateName << "\n";
                    int logical_Qubit_0 = getRegisterIndexOfOp(inQubits[0]);
                    int logical_Qubit_1 = getRegisterIndexOfOp(inQubits[1]);
                    std::pair<int, int> logical_Edge = std::make_pair(logical_Qubit_0,logical_Qubit_1);
                    llvm::outs() << "Logical qubit: (" << logical_Qubit_0 << "," << logical_Qubit_1 << ")\n";
                    if (couplingMap[logical_Edge])
                        // If logical qubits are connected, no need to change gate
                        llvm::outs() << "Physical qubit connected: (" << randomInitialMapping[logical_Qubit_0] << "," << randomInitialMapping[logical_Qubit_1] << ")\n";
                    else
                        // ig logical qubits not connected, call Z3 solver to find what SWAPs to insert
                        llvm::outs() << "Physical qubit not connected: (" << randomInitialMapping[logical_Qubit_0] << "," << randomInitialMapping[logical_Qubit_1] << ")\n";
                    
                }


                // Rewrite gate
                // mlir::OpBuilder builder(op); // Create OpBuilder
                // builder.setInsertionPoint(op);

                // auto loc = op->getLoc();
                // auto newOp = builder.create<quantum::CustomOp>(loc,
                //     /*out_qubits=*/mlir::TypeRange({cast<quantum::CustomOp>(op).getOutQubits().getTypes()}),
                //     /*out_ctrl_qubits=*/mlir::TypeRange(),
                //     /*params=*/mlir::ValueRange({cast<quantum::CustomOp>(op).getParams()}),
                //     /*in_qubits=*/mlir::ValueRange({cast<quantum::CustomOp>(op).getInQubits()}),
                //     /*gate_name=*/gateName,
                //     /*adjoint=*/false,
                //     /*in_ctrl_qubits=*/mlir::ValueRange(),
                //     /*in_ctrl_values=*/mlir::ValueRange());

                // op->replaceAllUsesWith(newOp->getResults());
                // op->erase();
                    
            }
        });
        // context c;

        // expr x = c.bool_const("x");
        // expr y = c.bool_const("y");
        // expr conjecture = (!(x && y)) == (!x || !y);
        
        // solver s(c);
        // s.add(!conjecture);
        // // llvm::outs() <<  "Z3 result " << s.to_smt2() << "\n";
        // llvm::outs() <<  "Z3 check " << s.check() << "\n";
        // llvm::outs() <<  "sat " << z3::sat << "\n";
        // llvm::outs() <<  "unsat " << z3::unsat << "\n";
        // llvm::outs() <<  "unknown " << z3::unknown << "\n";

    }
};

std::unique_ptr<Pass> createRoutingPass()
{
    return std::make_unique<RoutingPass>();
}

} // namespace catalyst
