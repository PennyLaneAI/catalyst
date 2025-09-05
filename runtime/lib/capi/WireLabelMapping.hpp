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

// This file contains the helper functions to remap device wire indices
// back into user wire labels for the non-scalar terminal measurements.

#pragma once

#include <algorithm>
#include <vector>

#include "Exception.hpp"
#include "ExecutionContext.hpp"

namespace Catalyst::Runtime {

template <typename T> void Remap1DResultWires(T *data_aligned, RTDevice *RTD_PTR)
{
    // Need to remap labels to device's cannonical order
    // When user writes qml.device(wires=3), qml.state()
    // They mean qml.state(wires=[0,1,2])
    // However, it is not guaranteed that the labels (0,1,2) will correspond
    // to the device wire indices (0,1,2).
    // We need to find the indices that these labels actually correspond to.

    // Let's walk through with an example.
    // Say the array of qubit IDs on the device is [100, 200, 300].
    // This means calling just `State()` on the device,
    // returns an array where the i-th entry in the array,
    // i = "abc" in binary (e.g. i=6 means abc=110)
    // gives the amplitude when device qubit 100, 200, 300 are in |a>, |b> and |c>

    // Let's also say the label-to-index map is {0:1, 1:0, 2:2}.
    // The user wants qml.state(wires=[0,1,2]), with 0, 1, 2 being labels
    // So the indices should be [1,0,2], or device qubit IDs [200, 100, 300]
    // i.e. the user wants an array where the i-th entry gives the amplitude
    // when qubit 200, 100, 300 are in |a>, |b> and |c>
    int64_t capacity = RTD_PTR->getDeviceCapacity();
    RTD_PTR->fillWireLabelMapUpToCapacity();
    std::vector<T> remappedState(1 << capacity);

    for (int64_t i = 0; i < (1 << capacity); i++) {
        // say i=6
        // 1. parse 6 into binary, i.e. abc=110
        // Use standard int size
        constexpr int64_t MAX_NUM_QUBITS = 64;

        std::bitset<MAX_NUM_QUBITS> labelBits(i);

        std::string labelBinaryStringLeftPadded = labelBits.to_string();
        std::string labelBinaryString =
            labelBinaryStringLeftPadded.substr(MAX_NUM_QUBITS - capacity);

        // 2. This means user wants label 0 in |1>, label 1 in |1>, label 2 in |0>
        // Get the corresponding indices
        std::vector<uint64_t> indices(capacity);
        for (int64_t label = 0; label < static_cast<int64_t>(labelBinaryString.size()); label++) {
            char bit = labelBinaryString[label];
            uint64_t index = RTD_PTR->getWireLabelMap().at(label);
            // user wants label 0 in |1>
            // Map is {0:1}
            // This means we want qubit index 1, or qubit ID 200, in |1>
            indices[index] = bit - '0';
        }

        // 3. Get the index number back from the index bitstream
        int64_t index_decimal = 0;
        for (int64_t p = 0; p < capacity; p++) {
            index_decimal += (1 << (capacity - 1 - p)) * indices[p];
        }
        remappedState[i] = data_aligned[index_decimal];
    }

    for (int64_t i = 0; i < static_cast<int64_t>(remappedState.size()); i++) {
        data_aligned[i] = remappedState[i];
    }
}

void RemapStateResultWires(std::complex<double> *data_aligned, RTDevice *RTD_PTR)
{
    Remap1DResultWires<std::complex<double>>(data_aligned, RTD_PTR);
}

void RemapProbsResultWires(double *data_aligned, RTDevice *RTD_PTR)
{
    Remap1DResultWires<double>(data_aligned, RTD_PTR);
}

// sample() is a lot easier since no binary bit mappings involved
void RemapSampleResultWires(double *data_aligned, int64_t Nrows, int64_t Ncols, RTDevice *RTD_PTR)
{
    RTD_PTR->fillWireLabelMapUpToCapacity();
    std::vector<double> remappedSamples(Nrows * Ncols);

    for (int64_t i = 0; i < Nrows; i++) {
        // Each shot is the same
        for (int64_t label = 0; label < Ncols; label++) {
            uint64_t index = RTD_PTR->getWireLabelMap().at(label);
            remappedSamples[i * Ncols + label] = data_aligned[i * Ncols + index];
        }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(remappedSamples.size()); i++) {
        data_aligned[i] = remappedSamples[i];
    }
}

void RemapCountsResultWires(double *eigvals_data_aligned, int64_t *counts_data_aligned,
                            RTDevice *RTD_PTR)
{
    Remap1DResultWires<double>(eigvals_data_aligned, RTD_PTR);
    Remap1DResultWires<int64_t>(counts_data_aligned, RTD_PTR);

    // After remapping the order will be some random order
    // So we need to sort the compbasis counts again
    int64_t capacity = RTD_PTR->getDeviceCapacity();
    std::vector<std::pair<double, int64_t>> countsPairs(1 << capacity);

    for (int64_t i = 0; i < (1 << capacity); i++) {
        countsPairs[i] = std::make_pair(eigvals_data_aligned[i], counts_data_aligned[i]);
    }

    std::sort(countsPairs.begin(), countsPairs.end());

    for (int64_t i = 0; i < (1 << capacity); i++) {
        eigvals_data_aligned[i] = countsPairs[i].first;
        counts_data_aligned[i] = countsPairs[i].second;
    }
}

} // namespace Catalyst::Runtime
