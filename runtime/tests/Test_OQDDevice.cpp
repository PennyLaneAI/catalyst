// Copyright 2023-2025 Xanadu Quantum Technologies Inc.

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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include "RuntimeCAPI.h"
#include "TestUtils.hpp"

#include "OQDDevice.hpp"
#include "OQDRuntimeCAPI.h"

using namespace Catch::Matchers;
using namespace Catalyst::Runtime::Device;

using json = nlohmann::json;

TEST_CASE("Test the OQDDevice constructor", "[oqd]")
{
    auto device = OQDDevice("{shots : 100}");

    REQUIRE_THROWS_WITH(device.GetNumQubits(), ContainsSubstring("unsupported by device"));
    REQUIRE_THROWS_WITH(device.Measure(0), ContainsSubstring("unsupported by device"));
}

TEST_CASE("Test the OQDDevice qubit allocation and release", "[oqd]")
{
    auto device = OQDDevice(R"({shots : 100}ION:{"name":"Yb171"}PHONON:{"class_":"Phonon"})");

    CHECK(device.getOutputFile() == "__openapl__output.json");
    CHECK(device.getIonSpecs() == "{\"name\":\"Yb171\"}");
    CHECK(device.getPhononSpecs()[0] == "{\"class_\":\"Phonon\"}");

    std::vector<QubitIdType> allocaedQubits = device.AllocateQubits(3);
    CHECK(allocaedQubits[0] == 0);
    CHECK(allocaedQubits[1] == 1);
    CHECK(allocaedQubits[2] == 2);

    device.ReleaseAllQubits();
    CHECK(device.getIonSpecs() == "");
    CHECK(device.getPhononSpecs().empty());

    std::filesystem::remove("__openapl__output.json");
}

TEST_CASE("Test the OQDDevice ion index out of range", "[oqd]")
{
    auto device = OQDDevice(R"({shots : 100}ION:{"name":"Yb171"})");
    std::vector<QubitIdType> allocaedQubits = device.AllocateQubits(3);

    Beam beam = {0, 1.1, 2.2, {1, 0, 0}, {0, 1, 0}};
    Pulse p = {&beam, /*target=*/100, 1.5, 3.14};
    Pulse *pulses[] = {&p, &p};

    REQUIRE_THROWS_WITH(__catalyst__oqd__ParallelProtocol(pulses, 2),
                        ContainsSubstring("ion index out of range"));
}

TEST_CASE("Test the OQDDevice transition index out of range", "[oqd]")
{
    auto device = OQDDevice(R"({shots : 100}ION:{"transitions":[]})");
    std::vector<QubitIdType> allocaedQubits = device.AllocateQubits(1);

    Beam beam = {/*transition_index=*/100, 1.1, 2.2, {1, 0, 0}, {0, 1, 0}};
    Pulse p = {&beam, 0, 1.5, 3.14};
    Pulse *pulses[] = {&p, &p};

    REQUIRE_THROWS_WITH(__catalyst__oqd__ParallelProtocol(pulses, 2),
                        ContainsSubstring("transition index out of range"));
}

TEST_CASE("Test OpenAPL Program generation", "[oqd]")
{
    json expected = json::parse(R"(
{
  "class_": "AtomicCircuit",
  "system": {
    "class_": "System",
    "modes":[
      {
        "class_": "Phonon",
        "eigenvector": [1.0,0.0,0.0],
        "energy": 3.3
      },
      {
        "class_": "Phonon",
        "eigenvector": [0.0,1.0,0.0],
        "energy": 4.4
      },
      {
        "class_": "Phonon",
        "eigenvector": [0.0,0.0,1.0],
        "energy": 5.5
      }
    ],
    "ions": [
      {
        "class_": "Ion",
        "mass": 171.0,
        "charge": 1.0,
        "levels": [
          {
            "class_": "Level",
            "label": "l0",
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
            "label": "l1",
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
            "label": "l2",
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
            "label": "l3",
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
            "label": "l0->l2",
            "level1": {
              "class_": "Level",
              "label": "l0",
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
              "label": "l2",
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
            "label": "l0->l3",
            "level1": {
              "class_": "Level",
              "label": "l0",
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
              "label": "l3",
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
            "label": "l1->l2",
            "level1": {
              "class_": "Level",
              "label": "l1",
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
              "label": "l2",
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
            "label": "l1->l3",
            "level1": {
              "class_": "Level",
              "label": "l1",
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
              "label": "l3",
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
            "label": "l0",
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
            "label": "l1",
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
            "label": "l2",
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
            "label": "l3",
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
            "label": "l0->l2",
            "level1": {
              "class_": "Level",
              "label": "l0",
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
              "label": "l2",
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
            "label": "l0->l3",
            "level1": {
              "class_": "Level",
              "label": "l0",
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
              "label": "l3",
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
            "label": "l1->l2",
            "level1": {
              "class_": "Level",
              "label": "l1",
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
              "label": "l2",
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
            "label": "l1->l3",
            "level1": {
              "class_": "Level",
              "label": "l1",
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
              "label": "l3",
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
                "label": "l0->l2",
                "level1": {
                  "class_": "Level",
                  "label": "l0",
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
                  "label": "l2",
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
                "label": "l1->l2",
                "level1": {
                  "class_": "Level",
                  "label": "l1",
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
                  "label": "l2",
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
                "label": "l0->l3",
                "level1": {
                  "class_": "Level",
                  "label": "l0",
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
                  "label": "l3",
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
                "label": "l1->l3",
                "level1": {
                  "class_": "Level",
                  "label": "l1",
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
                  "label": "l3",
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

    const auto [rtd_lib, rtd_name, rtd_kwargs] =
        std::array<std::string, 3>{"oqd.qubit", "oqd",
                                   R"({'shots': 0, 'mcmc': False}ION:
      {
        "class_": "Ion",
        "mass": 171.0,
        "charge": 1.0,
        "levels": [
          {
            "class_": "Level",
            "label": "l0",
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
            "label": "l1",
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
            "label": "l2",
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
            "label": "l3",
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
            "label": "l0->l2",
            "level1": {
              "class_": "Level",
              "label": "l0",
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
              "label": "l2",
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
            "label": "l0->l3",
            "level1": {
              "class_": "Level",
              "label": "l0",
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
              "label": "l3",
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
            "label": "l1->l2",
            "level1": {
              "class_": "Level",
              "label": "l1",
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
              "label": "l2",
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
            "label": "l1->l3",
            "level1": {
              "class_": "Level",
              "label": "l1",
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
              "label": "l3",
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
      }PHONON:
      {
        "class_": "Phonon",
        "eigenvector": [1.0,0.0,0.0],
        "energy": 3.3
      }PHONON:
      {
        "class_": "Phonon",
        "eigenvector": [0.0,1.0,0.0],
        "energy": 4.4
      }PHONON:
      {
        "class_": "Phonon",
        "eigenvector": [0.0,0.0,1.0],
        "energy": 5.5
      })"};

    size_t num_qubits = 2;

    Beam beam1 = {0, 31.41592653589793, 157.07963267948966, {1, 0, 0}, {0, 1, 0}};
    Beam beam2 = {2, 31.41592653589793, 157.07963267948966, {1, 0, 0}, {0, 1, 0}};

    Beam beam3 = {1, 31.41592653589793, 157.07963267948966, {0, 0, 1}, {0, 0, 1}};
    Beam beam4 = {3, 31.41592653589793, 157.07963267948966, {0, 0, 1}, {1, 0, 0}};

    __catalyst__rt__initialize(nullptr);
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str(), 1000, false);

    QirArray *qs = __catalyst__rt__qubit_allocate_array(num_qubits);

    QUBIT **target0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
    QUBIT **target1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

    CHECK(reinterpret_cast<QubitIdType>(*target0) == 0);
    CHECK(reinterpret_cast<QubitIdType>(*target1) == 1);

    Pulse *pulse1 = __catalyst__oqd__pulse(*target0, 2.0, 0.00, &beam1);
    Pulse *pulse2 = __catalyst__oqd__pulse(*target0, 2.0, 3.14, &beam2);
    Pulse *pulses12[] = {pulse1, pulse2};
    __catalyst__oqd__ParallelProtocol(pulses12, 2);

    Pulse *pulse3 = __catalyst__oqd__pulse(*target0, 2.0, 0.00, &beam3);
    Pulse *pulse4 = __catalyst__oqd__pulse(*target1, 2.0, -3.14, &beam4);
    Pulse *pulses34[] = {pulse3, pulse4};
    __catalyst__oqd__ParallelProtocol(pulses34, 2);

    __catalyst__rt__qubit_release_array(qs);
    __catalyst__rt__device_release();
    __catalyst__rt__finalize();

    json observed = json::parse(std::ifstream("__openapl__output.json"));
    CHECK(expected == observed);

    std::filesystem::remove("__openapl__output.json");
}
