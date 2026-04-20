// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include <algorithm>
#include <map>

#include "Exception.hpp"
#include "Types.h"

namespace Catalyst::Runtime {

/**
 * Qubit Manager
 *
 * @brief That maintains mapping of qubit IDs between runtime and device
 * ids (e.g., Lightning-Dynamic). When user allocates a qubit, the
 * `QubitManager` adds the qubit as an active qubit that operations
 * can act on. When user releases a qubit, the `QubitManager` removes
 * that qubit from the list of active wires.
 */
template <typename SimQubitIdType = QubitIdType, typename DevQubitIdType = size_t>
class QubitManager {
  private:
    using LQMapT = std::map<SimQubitIdType, DevQubitIdType>;

    SimQubitIdType next_idx{0};
    LQMapT qubits_map{};

    template <class OIter = typename LQMapT::iterator>
    [[nodiscard]] inline OIter _remove_simulator_qubit_id(SimQubitIdType s_idx)
    {
        const auto &&s_idx_iter = this->qubits_map.find(s_idx);
        RT_FAIL_IF(s_idx_iter == this->qubits_map.end(), "Invalid simulator qubit index");

        return this->qubits_map.erase(s_idx_iter);
    }

    template <class IIter = typename LQMapT::iterator>
    inline void _update_qubits_mapfrom(IIter s_idx_iter)
    {
        for (; s_idx_iter != this->qubits_map.end(); s_idx_iter++) {
            s_idx_iter->second--;
        }
    }

  public:
    QubitManager() = default;
    ~QubitManager() = default;

    QubitManager(const QubitManager &) = delete;
    QubitManager &operator=(const QubitManager &) = delete;
    QubitManager(QubitManager &&) = delete;
    QubitManager &operator=(QubitManager &&) = delete;

    [[nodiscard]] auto isValidQubitId(SimQubitIdType s_idx) -> bool
    {
        return this->qubits_map.contains(s_idx);
    }

    [[nodiscard]] auto isValidQubitId(const std::vector<SimQubitIdType> &ss_idx) -> bool
    {
        return std::all_of(ss_idx.begin(), ss_idx.end(),
                           [this](SimQubitIdType s) { return isValidQubitId(s); });
    }

    [[nodiscard]] auto getAllQubitIds() -> std::vector<SimQubitIdType>
    {
        std::vector<SimQubitIdType> ids;
        ids.reserve(this->qubits_map.size());
        for (const auto &it : this->qubits_map) {
            ids.push_back(it.first);
        }

        return ids;
    }

    [[nodiscard]] auto getDeviceId(SimQubitIdType s_idx) -> DevQubitIdType
    {
        RT_FAIL_IF(!isValidQubitId(s_idx), "Invalid device qubit index");

        return this->qubits_map[s_idx];
    }

    auto getDeviceIds(const std::vector<SimQubitIdType> &ss_idx) -> std::vector<DevQubitIdType>
    {
        std::vector<DevQubitIdType> dd_idx;
        dd_idx.reserve(ss_idx.size());
        for (const auto &s : ss_idx) {
            dd_idx.push_back(getDeviceId(s));
        }
        return dd_idx;
    }

    [[nodiscard]] auto getSimulatorId(DevQubitIdType d_idx) -> SimQubitIdType
    {
        auto s_idx = std::find_if(this->qubits_map.begin(), this->qubits_map.end(),
                                  [&d_idx](auto &&p) { return p.second == d_idx; });

        RT_FAIL_IF(s_idx == this->qubits_map.end(), "Invalid simulator qubit index");

        return s_idx->first;
    }

    [[nodiscard]] auto Allocate(DevQubitIdType d_next_idx) -> SimQubitIdType
    {
        this->qubits_map[this->next_idx++] = d_next_idx;
        return this->next_idx - 1;
    }

    auto AllocateRange(DevQubitIdType start_idx, size_t size) -> std::vector<SimQubitIdType>
    {
        std::vector<SimQubitIdType> ids;
        ids.reserve(size);
        for (DevQubitIdType i = start_idx; i < start_idx + size; i++) {
            ids.push_back(this->next_idx);
            this->qubits_map[this->next_idx++] = i;
        }
        return ids;
    }

    void Release(SimQubitIdType s_idx)
    {
        _update_qubits_mapfrom(_remove_simulator_qubit_id(s_idx));
    }

    void ReleaseAll()
    {
        // Release all qubits by clearing the map.
        this->qubits_map.clear();
    }
};
} // namespace Catalyst::Runtime
