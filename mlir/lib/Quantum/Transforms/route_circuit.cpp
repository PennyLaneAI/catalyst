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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"


using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_ROUTINGPASS
#define GEN_PASS_DECL_ROUTINGPASS
#include "Quantum/Transforms/Passes.h.inc"

const int MAXIMUM = 1e9;


struct RoutingPass : public impl::RoutingPassBase<RoutingPass> {
    using impl::RoutingPassBase<RoutingPass>::RoutingPassBase;

    int countLogicalQubit(Operation *op) {
        int numQubits = cast<quantum::AllocOp>(op).getNqubitsAttr().value_or(-1);
        assert(numQubits != -1 && "PPM specs with dynamic number of qubits is not implemented");
        return numQubits;
    }

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

    quantum::ExtractOp getRegisterIndexOfOp(Value inQubit) {
        Operation *prevOp = inQubit.getDefiningOp();
        if (isa<quantum::ExtractOp>(prevOp)) 
            return (cast<quantum::ExtractOp>(prevOp));
        else {
            auto iteratePrevOpOutQubit = cast<quantum::CustomOp>(prevOp).getOutQubits();
            auto iteratePrevOpInQubit = cast<quantum::CustomOp>(prevOp).getInQubits();
            for (size_t iter = 0; iter < (iteratePrevOpOutQubit.size()) ; iter++ ) {
                if (iteratePrevOpOutQubit[iter] == inQubit) 
                    return getRegisterIndexOfOp(iteratePrevOpInQubit[iter]);
            }
        }
        return nullptr;
    }

    void preProcessing(
        std::set<int> *physicalQubits, std::vector<int> *randomInitialMapping, 
        std::set<quantum::CustomOp> *frontLayer, 
        llvm::DenseMap<quantum::CustomOp, std::vector<quantum::ExtractOp>> &OpToExtractMap, 
        llvm::DenseMap<quantum::ExtractOp, int> &ExtractOpToQubitMap, 
        int *dagLogicalQubits) {
        auto logicalQubitIndex = 0;
        getOperation()->walk([&](Operation *op) {
            if (isa<quantum::AllocOp>(op)) {
                *dagLogicalQubits = countLogicalQubit(op);
                *randomInitialMapping =  generateRandomInitialMapping(physicalQubits);
            }
            else if (isa<quantum::ExtractOp>(op)) {
                ExtractOpToQubitMap[cast<quantum::ExtractOp>(op)] = logicalQubitIndex;
                logicalQubitIndex = logicalQubitIndex + 1;
            }
            else if (isa<quantum::CustomOp>(op)) {
                // StringRef gateName = cast<quantum::CustomOp>(op).getGateName();
                int nQubits = cast<quantum::CustomOp>(op).getInQubits().size();
                auto inQubits = cast<quantum::CustomOp>(op).getInQubits();

                for (auto inQubit :inQubits) {
                    OpToExtractMap[cast<quantum::CustomOp>(op)].push_back(getRegisterIndexOfOp(inQubit));
                }
            
                if (nQubits == 2) {
                    Operation *prevOp_0 = inQubits[0].getDefiningOp();
                    Operation *prevOp_1 = inQubits[1].getDefiningOp();
                    if (isa<quantum::ExtractOp>(prevOp_0) && isa<quantum::ExtractOp>(prevOp_1)) 
                        (*frontLayer).insert(cast<quantum::CustomOp>(op));
                }
                else if (nQubits == 1) {
                    Operation *prevOp_0 = inQubits[0].getDefiningOp();
                    if (isa<quantum::ExtractOp>(prevOp_0)) 
                        (*frontLayer).insert(cast<quantum::CustomOp>(op));
                }
            }
        }
    );
        return;
    }

    void distanceMatrices(std::set<int> *physicalQubits, llvm::DenseMap<std::pair<int,int>, int> &distanceMatrix, llvm::DenseMap<std::pair<int,int>, int> &predecessorMatrix, llvm::DenseMap<std::pair<int, int>, bool> &couplingMap) {
        // initial distances between non-connected physical qubits maximum
        for (auto i_itr = (*physicalQubits).begin(); i_itr != (*physicalQubits).end(); i_itr++)
        {
            for (auto j_itr = (*physicalQubits).begin(); j_itr != (*physicalQubits).end(); j_itr++)
            {
                distanceMatrix[std::make_pair(*i_itr, *j_itr)] = MAXIMUM;
                predecessorMatrix[std::make_pair(*i_itr, *j_itr)] = -1; 
            }
        }

        // distance from self to self -> 0
        for (auto i : (*physicalQubits))
        {
            predecessorMatrix[std::make_pair(i,i)] = i;
            distanceMatrix[std::make_pair(i, i)] = 0;
        }
        
        // distance between physical qubits connected by edge => 1s
        for (auto& entry : couplingMap) {
            std::pair<int, int>& key = entry.first;
            if(entry.second)
            {
                distanceMatrix[std::make_pair(key.first, key.second)] = 1;
                predecessorMatrix[std::make_pair(key.first, key.second)] = key.first;
            }
        }

        // All-pair-shortest-path
        for (auto i_itr = (*physicalQubits).begin(); i_itr != (*physicalQubits).end(); i_itr++)
        {
            for (auto j_itr = (*physicalQubits).begin(); j_itr != (*physicalQubits).end(); j_itr++ ) 
            {
                for (auto k_itr = (*physicalQubits).begin(); k_itr != (*physicalQubits).end(); k_itr++ ) 
                {
                    if (distanceMatrix[std::make_pair(*j_itr,*i_itr)] + distanceMatrix[std::make_pair(*i_itr,*k_itr)] < distanceMatrix[std::make_pair(*j_itr,*k_itr)] )
                    {
                        distanceMatrix[std::make_pair(*j_itr,*k_itr)] = distanceMatrix[std::make_pair(*j_itr,*i_itr)] + distanceMatrix[std::make_pair(*i_itr,*k_itr)];
                        predecessorMatrix[std::make_pair(*j_itr,*k_itr)] = predecessorMatrix[std::make_pair(*i_itr,*k_itr)];
                    }
                }
            }
        }
        return;
    }
    
    std::vector<int> getShortestPath(int source, int target, llvm::DenseMap<std::pair<int,int>, int> &predecessorMatrix) {
        std::vector<int> path;
        if ( predecessorMatrix[std::make_pair(source, target)] == -1 && source != target) {
            return path;
        }

        int current = target;
        while (current != source) {
            path.push_back(current);
            current = predecessorMatrix[std::make_pair(source, current)];
            if (current == -1 && path.size() > 0) 
            { 
                path.clear(); 
                return path;
            }
        }
        path.push_back(source);
        std::reverse(path.begin(), path.end()); 
        return path;
    }
    
    void getExecuteGateList(
            std::set<quantum::CustomOp> *frontLayer, 
            std::set<quantum::CustomOp> *executeGateList, 
            llvm::DenseMap<std::pair<int, int>, bool> &couplingMap, 
            std::vector<int> *randomInitialMapping,
            llvm::DenseMap<quantum::CustomOp, std::vector<quantum::ExtractOp>> &OpToExtractMap,
            llvm::DenseMap<quantum::ExtractOp, int> &ExtractOpToQubitMap) {
        for(auto op : *frontLayer) {
            int nQubits = op.getInQubits().size(); 
            if (nQubits == 1)
                (*executeGateList).insert(op);
            else if (nQubits == 2) {
                auto extractOps = OpToExtractMap[op];
                int physical_Qubit_0 = (*randomInitialMapping)[ExtractOpToQubitMap[extractOps[0]]];
                int physical_Qubit_1 = (*randomInitialMapping)[ExtractOpToQubitMap[extractOps[1]]];

                std::pair<int, int> is_physical_Edge = std::make_pair(physical_Qubit_0,physical_Qubit_1);
                if (couplingMap[is_physical_Edge])
                    (*executeGateList).insert(op);
            }
        }
        return;
    }

    void Heuristic(
        std::set<quantum::CustomOp> *frontLayer, 
        llvm::DenseMap<std::pair<int,int>, int> &swap_candidates, 
        std::vector<int> *randomInitialMapping, 
        llvm::DenseMap<std::pair<int,int>, int> &distanceMatrix,
        llvm::DenseMap<quantum::CustomOp, std::vector<quantum::ExtractOp>> &OpToExtractMap,
        llvm::DenseMap<quantum::ExtractOp, int> &ExtractOpToQubitMap) {
        
        for (auto& entry : swap_candidates) {
            std::pair<int, int> &swap_pair = entry.first;
            std::vector<int> temp_mapping(*randomInitialMapping);
            
            //update temp mapping 
            for (size_t temp_mapping_index = 0; temp_mapping_index < temp_mapping.size(); temp_mapping_index++)
            {
                if(temp_mapping[temp_mapping_index] == swap_pair.first)
                    temp_mapping[temp_mapping_index] = swap_pair.second;
                else if(temp_mapping[temp_mapping_index] == swap_pair.second)
                    temp_mapping[temp_mapping_index] = swap_pair.first;
            }
            int temp_score = 0;
            for(auto op : *frontLayer) {
                auto extractOps = OpToExtractMap[op];
                int physical_Qubit_0 = temp_mapping[ExtractOpToQubitMap[extractOps[0]]];
                int physical_Qubit_1 = temp_mapping[ExtractOpToQubitMap[extractOps[1]]];
                temp_score = temp_score + distanceMatrix[std::make_pair(physical_Qubit_0,physical_Qubit_1)];
            }
            swap_candidates[swap_pair] = std::min(swap_candidates[swap_pair], temp_score);
        }
    }
    
    void runOnOperation() override {

        std::set<int> physicalQubits;
        llvm::DenseMap<std::pair<int, int>, bool> couplingMap = parseHardwareGraph(hardwareGraph, ";", &physicalQubits);
        int dagLogicalQubits;
        int numLogicalQubits = physicalQubits.size(); // works with automatic qubit management

        // distance matrix
        llvm::DenseMap<std::pair<int,int>, int> distanceMatrix;
        llvm::DenseMap<std::pair<int,int>, int> predecessorMatrix;
        distanceMatrices(&physicalQubits, distanceMatrix, predecessorMatrix, couplingMap);

        std::vector<int> randomInitialMapping;
        std::set<quantum::CustomOp> frontLayer;
        llvm::DenseMap<quantum::CustomOp, std::vector<quantum::ExtractOp>> OpToExtractMap;
        llvm::DenseMap<quantum::ExtractOp, int> ExtractOpToQubitMap;
        preProcessing(&physicalQubits, &randomInitialMapping, &frontLayer, OpToExtractMap, ExtractOpToQubitMap, &dagLogicalQubits);
        
        // print init mapping
        llvm::outs() << "Random Initial Mapping: \n";
        for (size_t logical_qubit_index = 0; logical_qubit_index < randomInitialMapping.size(); logical_qubit_index++) 
            llvm::outs() << logical_qubit_index << "->" << randomInitialMapping[logical_qubit_index] << "\n";
        
        std::vector<StringRef> compiledGateNames;
        std::vector<std::vector<int>> compiledGateQubits;
        std::set<quantum::CustomOp> executeGateList;
        int search_steps = 0;
        int max_iterations_without_progress = 10 * dagLogicalQubits;
        while( frontLayer.size() ) {
            getExecuteGateList(&frontLayer, &executeGateList, couplingMap, &randomInitialMapping, OpToExtractMap, ExtractOpToQubitMap);
            if (executeGateList.size()) {
                for (auto op : executeGateList)
                {
                    compiledGateNames.push_back(op.getGateName());
                    std::vector<int> currOpPhysicalQubits;
                    for(auto currOpExtract : OpToExtractMap[op])
                        currOpPhysicalQubits.push_back(randomInitialMapping[ExtractOpToQubitMap[currOpExtract]]);
                    compiledGateQubits.push_back(currOpPhysicalQubits);

                    // remove the executed op from front layer
                    frontLayer.erase(op);
                    // get successor of op
                    auto outQubits = op.getOutQubits();
                    for (auto outQubit : outQubits) {
                        for (auto &use : outQubit.getUses()) {
                            Operation *successorOp = use.getOwner();
                            if (isa<quantum::CustomOp>(*successorOp)) 
                                frontLayer.insert(cast<quantum::CustomOp>(*successorOp));
                        }
                    }
                }
                executeGateList.clear(); // clear execute gate list
            }
            else if (search_steps >= max_iterations_without_progress) {
                search_steps = 0;
                while (compiledGateNames.back() == "SWAP")
                {
                    compiledGateNames.pop_back();
                    compiledGateQubits.pop_back();
                }   
                auto greedyGate = *(frontLayer.begin());
                auto inExtract = OpToExtractMap[greedyGate];
                int physical_Qubit_0 = randomInitialMapping[ExtractOpToQubitMap[inExtract[0]]];
                int physical_Qubit_1 = randomInitialMapping[ExtractOpToQubitMap[inExtract[1]]];
                std::vector<int> swapPath = getShortestPath(physical_Qubit_0, physical_Qubit_1, predecessorMatrix);
                for(size_t i = 1; i<swapPath.size()-1; i++)
                {
                    int u = swapPath[i-1];
                    int v = swapPath[i];
                    compiledGateNames.push_back("SWAP");
                    compiledGateQubits.push_back({u,v});
                    //update mapping 
                    for (size_t random_init_mapping_index = 0; random_init_mapping_index < randomInitialMapping.size(); random_init_mapping_index++)
                    {
                        if(randomInitialMapping[random_init_mapping_index] == u)
                            randomInitialMapping[random_init_mapping_index] = v;
                        else if(randomInitialMapping[random_init_mapping_index] == v)
                            randomInitialMapping[random_init_mapping_index] = u;
                    }
                }
            }
            else {
                llvm::DenseMap<std::pair<int,int>, int> swap_candidates;
                for(auto op : frontLayer) {
                    for (auto logivalQubitExtractToBeRouted : OpToExtractMap[op]) {
                        int firstPhysicalQubitToBeRouted = randomInitialMapping[ExtractOpToQubitMap[logivalQubitExtractToBeRouted]];
                        for (auto secondPhysicalQubitToBeRouted : physicalQubits)
                            if (distanceMatrix[std::make_pair(firstPhysicalQubitToBeRouted, secondPhysicalQubitToBeRouted)] == 1)
                                swap_candidates[std::make_pair(firstPhysicalQubitToBeRouted, secondPhysicalQubitToBeRouted)] = MAXIMUM;
                    }
                }
                Heuristic(&frontLayer, swap_candidates, &randomInitialMapping, distanceMatrix, OpToExtractMap, ExtractOpToQubitMap);
                int min_dist_swap = MAXIMUM;
                std::pair<int, int> min_swap;
                for (auto& entry : swap_candidates) {
                    std::pair<int, int> &key = entry.first;
                    if (entry.second < min_dist_swap) {
                        min_swap = key;
                        min_dist_swap = entry.second;
                    }
                }
                // add the min SWAP
                compiledGateNames.push_back("SWAP");
                compiledGateQubits.push_back({min_swap.first,min_swap.second});
                //update mapping 
                for (size_t random_init_mapping_index = 0; random_init_mapping_index < randomInitialMapping.size(); random_init_mapping_index++)
                {
                    if(randomInitialMapping[random_init_mapping_index] == min_swap.first)
                        randomInitialMapping[random_init_mapping_index] = min_swap.second;
                    else if(randomInitialMapping[random_init_mapping_index] == min_swap.second)
                        randomInitialMapping[random_init_mapping_index] = min_swap.first;
                }
                search_steps++;
            }
        }
        for (size_t compile_gate_index = 0; compile_gate_index < compiledGateNames.size(); compile_gate_index++) {
            llvm::outs() << compiledGateNames[compile_gate_index] << ": ";
            for (auto compile_qubits : compiledGateQubits[compile_gate_index]) 
                llvm::outs() << compile_qubits << " ";
            llvm::outs() << "\n";
        }
        mlir::func::FuncOp func;
        getOperation()->walk([&](Operation *op) {
            if (isa<quantum::AllocOp>(op)) 
                func = op->getParentOfType<func::FuncOp>();
        });

        mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
        mlir::MLIRContext *context = &getContext();
        mlir::OpBuilder builder(context);
        builder.setInsertionPointToEnd(module.getBody());
        mlir::FunctionType funcType = builder.getFunctionType(
            /*inputs=*/{}, /*results=*/{});
        mlir::func::FuncOp newFunc = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), (func.getName().str() + "_routed"), funcType);

        // changing signature of new func
        // mlir::FunctionType originalFuncType = func.getFunctionType();
        // mlir::TypeRange originalResultTypes = originalFuncType.getResults();
        // mlir::TypeRange newFuncInputTypes = newFunc.getFunctionType().getInputs();
        // mlir::FunctionType newFuncType = builder.getFunctionType(
        //     func.getFunctionType(),      
        //     originalResultTypes
        // );
        // newFunc.setType(newFuncType);

        // insert allocs
        newFunc.addEntryBlock();
        builder.setInsertionPointToStart(&newFunc.getBody().front());

        // 3. Create the AllocOp and other operations for the new function's body.
        Type quregType = builder.getType<catalyst::quantum::QuregType>();
        IntegerAttr numQubitsAttr = builder.getI64IntegerAttr(numLogicalQubits);
        mlir::Operation *allocOp = builder.create<quantum::AllocOp>(builder.getUnknownLoc(), quregType, mlir::Value{}, numQubitsAttr);

        builder.setInsertionPointAfter(allocOp);
        mlir::Operation *extractOp = builder.create<quantum::ExtractOp>(
            builder.getUnknownLoc(), 
            builder.getType<quantum::QubitType>(), 
            allocOp->getResult(0),
            nullptr,
            builder.getI64IntegerAttr(1)
        );

        mlir::Value qubitValue = extractOp->getResult(0);
        llvm::SmallVector<mlir::Type, 4> resultTypes = {builder.getType<quantum::QubitType>()};
        builder.create<quantum::CustomOp>(
            builder.getUnknownLoc(), 
            resultTypes, 
            TypeRange(),
            ValueRange(), qubitValue, "Hadamard",
            false, ValueRange(), ValueRange()
        );

        // Use the builder to insert operations *after* allocOp.
        builder.create<quantum::DeallocOp>(builder.getUnknownLoc(), allocOp->getResult(0));
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    }
};

        // change type signature of a func
        // mlir::FunctionType originalFuncType = func.getFunctionType();
        // mlir::TypeRange originalResultTypes = originalFuncType.getResults();
        // mlir::TypeRange newFuncInputTypes = newFunc.getFunctionType().getInputs();
        // mlir::FunctionType newFuncNewType = builder.getFunctionType(
        //     func.getFunctionType(),      
        //     originalResultTypes
        // );
        // newFunc.setType(newFuncNewType);

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

std::unique_ptr<Pass> createRoutingPass()
{
    return std::make_unique<RoutingPass>();
}

} // namespace catalyst
