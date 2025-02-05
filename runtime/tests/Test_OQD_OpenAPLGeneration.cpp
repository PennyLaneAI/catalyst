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
#include "RuntimeCAPI.h"
#include "TestUtils.hpp"

using json = nlohmann::json;

TEST_CASE("Test OpenAPL Program generation", "[OQD]")
{
    json expected = json::parse(R"(
{
  "class_": "AtomicCircuit",
  "system": {
    "class_": "System",
    "ions": [
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
      },
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
    ]
  },
  "protocol": {
    "class_": "SequentialProtocol",
    "sequence": [
      {
        "class_": "ParallelProtocol",
        "sequence": [
          {
            "class_": "Pulse",
            "beam": {
              "class_": "Beam",
              "transition": {
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
              "rabi": {
                "class_": "MathNum",
                "value": 31.41592653589793
              },
              "detuning": {
                "class_": "MathNum",
                "value": 157.07963267948966
              },
              "phase": {
                "class_": "MathNum",
                "value": 0
              },
              "polarization": [
                1.0,
                0.0,
                0.0
              ],
              "wavevector": [
                0.0,
                1.0,
                0.0
              ],
              "target": 0
            },
            "duration": 2.0
          },
          {
            "class_": "Pulse",
            "beam": {
              "class_": "Beam",
              "transition": {
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
              "rabi": {
                "class_": "MathNum",
                "value": 31.41592653589793
              },
              "detuning": {
                "class_": "MathNum",
                "value": 157.07963267948966
              },
              "phase": {
                "class_": "MathNum",
                "value": 3.14
              },
              "polarization": [
                1.0,
                0.0,
                0.0
              ],
              "wavevector": [
                0.0,
                1.0,
                0.0
              ],
              "target": 0
            },
            "duration": 2.0
          }
        ]
      },
      {
        "class_": "ParallelProtocol",
        "sequence": [
          {
            "class_": "Pulse",
            "beam": {
              "class_": "Beam",
              "transition": {
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
              "rabi": {
                "class_": "MathNum",
                "value": 31.41592653589793
              },
              "detuning": {
                "class_": "MathNum",
                "value": 157.07963267948966
              },
              "phase": {
                "class_": "MathNum",
                "value": 0
              },
              "polarization": [
                0.0,
                0.0,
                1.0
              ],
              "wavevector": [
                0.0,
                0.0,
                1.0
              ],
              "target": 0
            },
            "duration": 2.0
          },
          {
            "class_": "Pulse",
            "beam": {
              "class_": "Beam",
              "transition": {
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
              },
              "rabi": {
                "class_": "MathNum",
                "value": 31.41592653589793
              },
              "detuning": {
                "class_": "MathNum",
                "value": 157.07963267948966
              },
              "phase": {
                "class_": "MathNum",
                "value": -3.14
              },
              "polarization": [
                0.0,
                0.0,
                1.0
              ],
              "wavevector": [
                1.0,
                0.0,
                0.0
              ],
              "target": 1
            },
            "duration": 2.0
          }
        ]
      }
    ]
  }
}
)");

    size_t num_qubits = 2;

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

    Beam beam1 = {0, 31.41592653589793, 157.07963267948966, {1, 0, 0}, {0, 1, 0}};
    Beam beam2 = {2, 31.41592653589793, 157.07963267948966, {1, 0, 0}, {0, 1, 0}};

    Beam beam3 = {1, 31.41592653589793, 157.07963267948966, {0, 0, 1}, {0, 0, 1}};
    Beam beam4 = {3, 31.41592653589793, 157.07963267948966, {0, 0, 1}, {1, 0, 0}};

    const auto [rtd_lib, rtd_name, rtd_kwargs] =
        std::array<std::string, 3>{"null.qubit", "null_qubit", ""};
    __catalyst__rt__initialize(nullptr);
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str(), nullptr, 1000);
    __catalyst__oqd__rt__initialize();

    QirArray *qs = __catalyst__rt__qubit_allocate_array(num_qubits);

    QUBIT **target0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
    QUBIT **target1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

    for (size_t i = 0; i < num_qubits; i++) {
        __catalyst__oqd__ion(&ion);
    }

    Pulse *pulse1 = __catalyst__oqd__pulse(*target0, 2.0, 0.00, &beam1);
    Pulse *pulse2 = __catalyst__oqd__pulse(*target0, 2.0, 3.14, &beam2);
    Pulse *pulses12[] = {pulse1, pulse2};
    __catalyst__oqd__ParallelProtocol(pulses12, 2);

    Pulse *pulse3 = __catalyst__oqd__pulse(*target0, 2.0, 0.00, &beam3);
    Pulse *pulse4 = __catalyst__oqd__pulse(*target1, 2.0, -3.14, &beam4);
    Pulse *pulses34[] = {pulse3, pulse4};
    __catalyst__oqd__ParallelProtocol(pulses34, 2);

    __catalyst__rt__qubit_release_array(qs);
    __catalyst__oqd__rt__finalize();
    __catalyst__rt__device_release();
    __catalyst__rt__finalize();

    json observed = json::parse(std::ifstream("__openapl__output.json"));
    CHECK(expected == observed);
    CHECK(1 == 0);

    std::filesystem::remove("__openapl__output.json");
}
