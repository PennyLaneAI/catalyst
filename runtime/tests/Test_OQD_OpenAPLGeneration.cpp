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

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "OQDRuntimeCAPI.h"
#include "TestUtils.hpp"

using json = nlohmann::json;

TEST_CASE("Test hello world", "[OQD]") { __catalyst__oqd__greetings(); }

TEST_CASE("Test ion generation", "[OQD]")
{
    char name[] = "Yb171";
    char l0_label[] = "l0";
    char l1_label[] = "l1";
    char l2_label[] = "l2";
    char l3_label[] = "l3";

    Level l0 = {l0_label, 6, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0};
    Level l1 = {l1_label, 6, 0.5, 0.0, 0.5, 0.5, 1.0, 0.0, 62.83185307179586};
    Level l2 = {l2_label, 5, 0.5, 1.0, 0.5, 0.5, 1.0, -1.0, 628.3185307179587};
    Level l3 = {l3_label, 5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1256.6370614359173};
    Level levels[] = {l0, l1, l2, l3};

    Transition tr0 = {l0_label, l2_label, 1.0};
    Transition tr1 = {l0_label, l3_label, 1.0};
    Transition tr2 = {l1_label, l2_label, 1.0};
    Transition tr3 = {l1_label, l3_label, 1.0};
    Transition transitions[] = {tr0, tr1, tr2, tr3};

    Ion ion = {name, 171.0, 1.0, {0, 0, 0}, levels, 4, transitions, 4};

    // Example ion taken from opentoml example of one qubit RX(pi/2)
    // https://github.com/OpenQuantumDesign/TrICal/blob/dev_yihong/examples/2a_example_rx_pi_by_2.json
    json expected = json::parse(R"(
      {
        "class_": "Ion",
        "mass": 171.0,
        "charge": 1.0,
        "levels": [
          {
            "class_": "Level",
            "principal": 6,
            "spin": 0.5,
            "orbital": 0.0,
            "nuclear": 0.5,
            "spin_orbital": 0.5,
            "spin_orbital_nuclear": 0.0,
            "spin_orbital_nuclear_magnetization": 0.0,
            "energy": 0.0
          },
          {
            "class_": "Level",
            "principal": 6,
            "spin": 0.5,
            "orbital": 0.0,
            "nuclear": 0.5,
            "spin_orbital": 0.5,
            "spin_orbital_nuclear": 1.0,
            "spin_orbital_nuclear_magnetization": 0.0,
            "energy": 62.83185307179586
          },
          {
            "class_": "Level",
            "principal": 5,
            "spin": 0.5,
            "orbital": 1.0,
            "nuclear": 0.5,
            "spin_orbital": 0.5,
            "spin_orbital_nuclear": 1.0,
            "spin_orbital_nuclear_magnetization": -1.0,
            "energy": 628.3185307179587
          },
          {
            "class_": "Level",
            "principal": 5,
            "spin": 0.5,
            "orbital": 1.0,
            "nuclear": 0.5,
            "spin_orbital": 0.5,
            "spin_orbital_nuclear": 1.0,
            "spin_orbital_nuclear_magnetization": 1.0,
            "energy": 1256.6370614359173
          }
        ],
        "transitions": [
          {
            "class_": "Transition",
            "level1": {
              "class_": "Level",
              "principal": 6,
              "spin": 0.5,
              "orbital": 0.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 0.0,
              "spin_orbital_nuclear_magnetization": 0.0,
              "energy": 0.0
            },
            "level2": {
              "class_": "Level",
              "principal": 5,
              "spin": 0.5,
              "orbital": 1.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 1.0,
              "spin_orbital_nuclear_magnetization": -1.0,
              "energy": 628.3185307179587
            },
            "einsteinA": 1.0
          },
          {
            "class_": "Transition",
            "level1": {
              "class_": "Level",
              "principal": 6,
              "spin": 0.5,
              "orbital": 0.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 0.0,
              "spin_orbital_nuclear_magnetization": 0.0,
              "energy": 0.0
            },
            "level2": {
              "class_": "Level",
              "principal": 5,
              "spin": 0.5,
              "orbital": 1.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 1.0,
              "spin_orbital_nuclear_magnetization": 1.0,
              "energy": 1256.6370614359173
            },
            "einsteinA": 1.0
          },
          {
            "class_": "Transition",
            "level1": {
              "class_": "Level",
              "principal": 6,
              "spin": 0.5,
              "orbital": 0.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 1.0,
              "spin_orbital_nuclear_magnetization": 0.0,
              "energy": 62.83185307179586
            },
            "level2": {
              "class_": "Level",
              "principal": 5,
              "spin": 0.5,
              "orbital": 1.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 1.0,
              "spin_orbital_nuclear_magnetization": -1.0,
              "energy": 628.3185307179587
            },
            "einsteinA": 1.0
          },
          {
            "class_": "Transition",
            "level1": {
              "class_": "Level",
              "principal": 6,
              "spin": 0.5,
              "orbital": 0.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 1.0,
              "spin_orbital_nuclear_magnetization": 0.0,
              "energy": 62.83185307179586
            },
            "level2": {
              "class_": "Level",
              "principal": 5,
              "spin": 0.5,
              "orbital": 1.0,
              "nuclear": 0.5,
              "spin_orbital": 0.5,
              "spin_orbital_nuclear": 1.0,
              "spin_orbital_nuclear_magnetization": 1.0,
              "energy": 1256.6370614359173
            },
            "einsteinA": 1.0
          }
        ],
        "position": [
          0.0,
          0.0,
          0.0
        ]
      }
)");

    __catalyst__oqd__ion(&ion);
    json observed = json::parse(std::ifstream("ion_output.json"));

    CHECK(expected == observed);

    std::filesystem::remove("ion_output.json");
}
