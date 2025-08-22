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

#include <bitset>
#include <cstdarg>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace Catalyst::Runtime {

class RoutingPass final {
  private:
    const int MAXIMUM = 1e9;
    std::set<QubitIdType> physicalQubits;
    std::map<QubitIdType, QubitIdType> wireMap;
    std::map<std::pair<QubitIdType, QubitIdType>, bool> couplingMap;
    std::map<std::pair<QubitIdType, QubitIdType>, int> distanceMatrix;
    std::map<std::pair<QubitIdType, QubitIdType>, int> predecessorMatrix;

  public:
    RoutingPass(std::string_view coupling_map_str)
    {
        auto string_index = 1;
        while (string_index < coupling_map_str.size() - 1) {
            size_t next_closing_bracket = coupling_map_str.find(")", string_index);
            std::string curr_coupling_map_str = std::string(
                coupling_map_str.substr(string_index + 1, next_closing_bracket - string_index - 1));
            std::istringstream iss(curr_coupling_map_str);
            QubitIdType first_qubit_id, second_qubit_id;
            char comma;
            iss >> first_qubit_id >> comma &&comma == ',' && iss >> second_qubit_id;
            this->physicalQubits.insert(first_qubit_id);
            this->physicalQubits.insert(second_qubit_id);
            this->couplingMap[std::make_pair(first_qubit_id, second_qubit_id)] = true;
            this->couplingMap[std::make_pair(second_qubit_id, first_qubit_id)] = true;
            string_index = next_closing_bracket + 3;
        }
        for (auto i_itr = this->physicalQubits.begin(); i_itr != this->physicalQubits.end();
             i_itr++) {
            // initial mapping i->i
            this->wireMap[*i_itr] = *i_itr;
            // self-distances : 0
            this->distanceMatrix[std::make_pair(*i_itr, *i_itr)] = 0;
            // parent(self) = self
            this->predecessorMatrix[std::make_pair(*i_itr, *i_itr)] = *i_itr;
        }

        // initial distances maximum
        for (auto i_itr = this->physicalQubits.begin(); i_itr != this->physicalQubits.end();
             i_itr++) {
            for (auto j_itr = this->physicalQubits.begin(); j_itr != this->physicalQubits.end();
                 j_itr++) {
                this->distanceMatrix[std::make_pair(*i_itr, *j_itr)] = MAXIMUM;
                this->predecessorMatrix[std::make_pair(*i_itr, *j_itr)] = -1;
            }
        }

        // edge-distances : 1
        for (auto &entry : this->couplingMap) {
            const std::pair<QubitIdType, QubitIdType> &key = entry.first;
            bool value = entry.second;
            if (value) {
                this->distanceMatrix[std::make_pair(key.first, key.second)] = 1;
                this->predecessorMatrix[std::make_pair(key.first, key.second)] = key.first;
            }
        }
        // run floyd-warshall
        for (auto i_itr = this->physicalQubits.begin(); i_itr != this->physicalQubits.end();
             i_itr++) {
            for (auto j_itr = this->physicalQubits.begin(); j_itr != this->physicalQubits.end();
                 j_itr++) {
                for (auto k_itr = this->physicalQubits.begin(); k_itr != this->physicalQubits.end();
                     k_itr++) {
                    if (this->distanceMatrix[std::make_pair(*j_itr, *i_itr)] +
                            this->distanceMatrix[std::make_pair(*i_itr, *k_itr)] <
                        this->distanceMatrix[std::make_pair(*j_itr, *k_itr)]) {
                        this->distanceMatrix[std::make_pair(*j_itr, *k_itr)] =
                            this->distanceMatrix[std::make_pair(*j_itr, *i_itr)] +
                            this->distanceMatrix[std::make_pair(*i_itr, *k_itr)];
                        this->predecessorMatrix[std::make_pair(*j_itr, *k_itr)] =
                            this->predecessorMatrix[std::make_pair(*i_itr, *k_itr)];
                    }
                }
            }
        }
    }

    std::vector<QubitIdType> getShortestPath(QubitIdType source, QubitIdType target)
    {
        std::vector<QubitIdType> path;
        if (this->predecessorMatrix.at(std::make_pair(source, target)) == -1 && source != target) {
            return path;
        }

        QubitIdType current = target;
        while (current != source) {
            path.push_back(current);
            current = this->predecessorMatrix.at(std::make_pair(source, current));
            if (current == -1 && path.size() > 0) {
                path.clear();
                return path;
            }
        }
        path.push_back(source);
        std::reverse(path.begin(), path.end());
        return path;
    }

    QubitIdType getMappedWire(QubitIdType wire) { return this->wireMap[wire]; }

    std::tuple<QubitIdType, QubitIdType, std::vector<QubitIdType>>
    getRoutedQubits(QUBIT *control, QUBIT *target)
    {
        // Similar to qml.transpile implementation
        // https://docs.pennylane.ai/en/stable/_modules/pennylane/transforms/transpile.html

        QubitIdType firstQubit = reinterpret_cast<QubitIdType>(control);
        QubitIdType secondQubit = reinterpret_cast<QubitIdType>(target);
        std::vector<QubitIdType> swapPath = {};

        if (this->couplingMap[std::make_pair(firstQubit, secondQubit)]) {
            // since in each iteration, we adjust indices of each op,
            // we reset logical -> phyiscal mapping
            for (auto it = this->wireMap.begin(); it != this->wireMap.end(); ++it) {
                this->wireMap[it->second] = it->second;
            }
        }
        else {
            swapPath = this->getShortestPath(firstQubit, secondQubit);
            //  i<swapPath.size()-1 since last qubit is already our target
            for (auto i = 1; i < swapPath.size() - 1; i++) {
                QubitIdType u = swapPath[i - 1];
                QubitIdType v = swapPath[i];
                for (auto it = this->wireMap.begin(); it != this->wireMap.end(); ++it) {
                    // update logical -> phyiscal mapping
                    if (this->wireMap[it->first] == u)
                        this->wireMap[it->first] = v;
                    else if (this->wireMap[it->first] == v)
                        this->wireMap[it->first] = u;
                }
            }
            firstQubit = this->wireMap[firstQubit];
            secondQubit = this->wireMap[secondQubit];
        }
        return std::make_tuple(firstQubit, secondQubit, swapPath);
    }

    ~RoutingPass() = default;
};
} // namespace Catalyst::Runtime
