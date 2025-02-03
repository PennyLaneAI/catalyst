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

#include "Types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Level {
    char *label;
    int64_t principal;
    double spin;
    double orbital;
    double nuclear;
    double spin_orbital;
    double spin_orbital_nuclear;
    double spin_orbital_nuclear_magnetization;
    double energy;
};

struct Transition {
    char *level1;
    char *level2;
    double einstein_a;
};

struct Ion {
    char *name;
    double mass;
    double charge;
    std::array<int64_t, 3> position;
    Level *levels;
    int64_t num_of_levels;
    Transition *transitions;
    int64_t num_of_transitions;
};

struct Beam {
    int64_t transition_index;
    double rabi;
    double detuning;
    std::array<int64_t, 3> polarization;
    std::array<int64_t, 3> wavevector;
};

// OQD Runtime Instructions
void __catalyst__oqd__greetings();

void __catalyst__oqd__rt__initialize();
void __catalyst__oqd__rt__finalize();
void __catalyst__oqd__ion(Ion *);
void __catalyst__oqd__pulse(QUBIT *qubit, double duration, double phase, Beam *beam);


#ifdef __cplusplus
} // extern "C"
#endif

#endif
