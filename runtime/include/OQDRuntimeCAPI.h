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

#pragma once
#ifndef OQDRUNTIMECAPI_H
#define OQDRUNTIMECAPI_H

#include <array>
#include <cstdint>

#include "Exception.hpp"
#include "Types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Beam {
    int64_t transition_index;
    double rabi;
    double detuning;
    std::array<int64_t, 3> polarization;
    std::array<int64_t, 3> wavevector;
};

struct Pulse {
    Beam *beam;
    size_t target;
    double duration;
    double phase;
};

// OQD Runtime Instructions
void __catalyst__oqd__rt__initialize();
void __catalyst__oqd__rt__finalize(const std::string &openapl_file_name);
void __catalyst__oqd__ion(const std::string &ion_specs);
void __catalyst__oqd__modes(const std::vector<std::string> &phonon_specs);
Pulse *__catalyst__oqd__pulse(QUBIT *qubit, double duration, double phase, Beam *beam);
void __catalyst__oqd__ParallelProtocol(Pulse **pulses, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
