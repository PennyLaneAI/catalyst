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

#include "OQDRuntimeCAPI.h"

#include "TestUtils.hpp"

TEST_CASE("Test hello world", "[OQD]") { __catalyst__oqd__greetings(); }

TEST_CASE("Test ion generation", "[OQD]")
{
    char name[] = "Yb171";
    char downstate_name[] = "downstate";
    char estate_name[] = "estate";
    char upstate_name[] = "upstate";

    Level downstate = {downstate_name, 6, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 0.0};
    Level estate = {estate_name, 6, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0, 1.264300e+10};
    Level upstate = {upstate_name, 5, 2.4, 2.5, 2.6, 2.8, 2.9, 3.0, 8.115200e+14};
    Level levels[] = {downstate, estate, upstate};

    Transition de = {downstate_name, estate_name, 2.200000e+00};
    Transition du = {downstate_name, upstate_name, 1.100000e+00};
    Transition eu = {estate_name, upstate_name, 3.300000e+00};
    Transition transitions[] = {de, du, eu};

    Ion ion = {name, 171.0, 42.42, {1, 2, 3}, levels, transitions};

    __catalyst__oqd__ion(&ion);
}
