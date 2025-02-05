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

#include <fstream>
#include <functional> // std::reference_wrapper
#include <iostream>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "OQDRuntimeCAPI.h"

using json = nlohmann::json;

static std::unique_ptr<json> JSON = nullptr;
static std::unique_ptr<std::vector<Pulse *>> PulseGarbageCan = nullptr;

template <typename T> json &numerical_json_factory(T value)
{
    static json j;
    j["class_"] = "MathNum";
    j["value"] = value;
    return j;
}

void to_json(json &j, const Level &l)
{
    j = json{{"class_", "Level"},
             {"principal", l.principal},
             {"spin", l.spin},
             {"orbital", l.orbital},
             {"nuclear", l.nuclear},
             {"spin_orbital", l.spin_orbital},
             {"spin_orbital_nuclear", l.spin_orbital_nuclear},
             {"spin_orbital_nuclear_magnetization", l.spin_orbital_nuclear_magnetization},
             {"energy", l.energy}};
}

void to_json(json &j, const Transition &tr)
{
    j = json{{"class_", "Transition"},
             {"einsteinA", tr.einstein_a},
             {"level1", tr.level1},
             {"level2", tr.level2}};
}

void to_json(json &j, const Pulse &p)
{
    j = json{{"class_", "Pulse"}, {"duration", p.duration}};

    json j_beam;
    j_beam["class_"] = "Beam";
    j_beam["target"] = p.target;
    j_beam["polarization"] = p.beam->polarization;
    j_beam["wavevector"] = p.beam->wavevector;
    j_beam["rabi"] = numerical_json_factory<double>(p.beam->rabi);
    j_beam["detuning"] = numerical_json_factory<double>(p.beam->detuning);
    j_beam["phase"] = numerical_json_factory<double>(p.phase);

    const auto &transitions = (*JSON)["system"]["ions"][p.target]["transitions"];
    j_beam["transition"] = transitions[p.beam->transition_index];

    j["beam"] = j_beam;
}

extern "C" {

void __catalyst__oqd__rt__initialize()
{
    PulseGarbageCan = std::make_unique<std::vector<Pulse *>>();

    JSON = std::make_unique<json>();
    (*JSON)["class_"] = "AtomicCircuit";

    json system = R"({
  "class_": "System",
  "ions": []
})"_json;
    (*JSON)["system"] = system;

    // The main openapl program is a sequential protocol
    // Each gate is a parallel protocol in the main sequential protocol
    json protocol = R"({
  "class_": "SequentialProtocol",
  "sequence": []
})"_json;
    (*JSON)["protocol"] = protocol;
}

void __catalyst__oqd__rt__finalize()
{
    for (auto pulse : *PulseGarbageCan) {
        delete pulse;
    }

    std::ofstream out_json("output.json");
    out_json << JSON->dump(2);
    JSON = nullptr;
}

void __catalyst__oqd__ion(Ion *ion)
{
    json j;
    j["class_"] = "Ion";
    j["mass"] = ion->mass;
    j["charge"] = ion->charge;
    j["position"] = ion->position;

    // The json library maps std array-like containers to json lists
    // To avoid copying complex objects like Level and Transition, we
    // wrap them into a reference when putting them into a container
    std::unordered_map<std::string, std::size_t> labeled_levels;
    std::vector<std::reference_wrapper<Level>> levels;
    for (std::size_t i = 0; i < ion->num_of_levels; i++) {
        labeled_levels.insert({(ion->levels)[i].label, i});
        levels.push_back(std::ref((ion->levels)[i]));
    }
    j["levels"] = levels;

    std::vector<std::reference_wrapper<Transition>> transitions;
    for (std::size_t i = 0; i < ion->num_of_transitions; i++) {
        transitions.push_back(std::ref((ion->transitions)[i]));
    }
    j["transitions"] = transitions;

    // Ion dialect keeps transitions' levels as label strings,
    // but openapl spec repeats all the level attributes in the transition object again
    // We have asked them to change it, but for now we conform to the spec
    for (auto &transition_in_json : j["transitions"]) {
        transition_in_json["level1"] = levels[labeled_levels[transition_in_json["level1"]]];
        transition_in_json["level2"] = levels[labeled_levels[transition_in_json["level2"]]];
    }

    (*JSON)["system"]["ions"].push_back(j);
}

Pulse *__catalyst__oqd__pulse(QUBIT *qubit, double duration, double phase, Beam *beam)
{
    size_t wire = reinterpret_cast<QubitIdType>(qubit);

    // Since this is CAPI, we cannot return smart pointer.
    // This means we have to new and delete manually.
    Pulse *pulse = new Pulse({beam, wire, duration, phase});
    PulseGarbageCan->push_back(pulse);
    return pulse;
}

void __catalyst__oqd__ParallelProtocol(Pulse **pulses, size_t num_of_pulses)
{
    json j;
    j["class_"] = "ParallelProtocol";

    std::vector<std::reference_wrapper<Pulse>> pulses_json;
    for (std::size_t i = 0; i < num_of_pulses; i++) {
        pulses_json.push_back(std::ref(*(pulses[i])));
    }
    j["sequence"] = pulses_json;

    (*JSON)["protocol"]["sequence"].push_back(j);
}

} // extern "C"
