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

#include <cmath>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct Level {
    char *label;
    size_t principal;
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
    size_t position[3];
    Level *levels;
    Transition *transitions;
};

// OQD Runtime Instructions
void __catalyst__oqd__greetings();

void __catalyst__oqd__ion(Ion *);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
