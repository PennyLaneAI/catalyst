// Copyright 2024 Xanadu Quantum Technologies Inc.

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
#include <cassert>
#include <set>

#include <toml++/toml.hpp>

#include "oqd_database_types.hpp"
#include "oqd_database_utils.hpp"

namespace catalyst {
namespace ion {

class OQDDatabaseManager {
  public:
    OQDDatabaseManager(const std::string &DeviceTomlLoc, const std::string &QubitTomlLoc,
                       const std::string &Gate2PulseDecompTomlLoc, size_t n_qubits)
    {
        sourceTomlDevice = toml::parse_file(DeviceTomlLoc);
        sourceTomlQubit = toml::parse_file(QubitTomlLoc);
        sourceTomlGateDecomposition = toml::parse_file(Gate2PulseDecompTomlLoc);

        assert(sourceTomlDevice && "Parsing of device toml failed!");
        assert(sourceTomlQubit && "Parsing of qubit toml failed!");
        assert(sourceTomlGateDecomposition && "Parsing of gate decomposition toml failed!");

        loadBeams1Params();
        loadBeams2Params();
        loadDetectionBeamParams();
        loadMeasurementDuration();

        loadPhononParams(n_qubits);

        loadIonParams();
    }

    const std::vector<Beam> &getBeams1Params() const { return beams1; }
    const std::vector<Beam> &getBeams2Params() const { return beams2; }
    const std::vector<DetectionBeam> &getDetectionBeamParams() const { return detectionBeams; }

    const std::vector<Phonon> &getPhononParams() const { return phonons; }

    const std::map<std::string, Ion> &getIonParams() const { return ions; }

    double getMeasurementDuration() const { return measurementDuration; }

  private:
    toml::parse_result sourceTomlDevice;
    toml::parse_result sourceTomlQubit;
    toml::parse_result sourceTomlGateDecomposition;

    std::vector<Beam> beams1;
    std::vector<Beam> beams2;
    std::vector<DetectionBeam> detectionBeams;

    std::vector<Phonon> phonons;

    double measurementDuration = 1e-4;

    std::map<std::string, Ion> ions;

    void loadBeams1Params() { loadBeamsParamsImpl("beams1", beams1); }
    void loadBeams2Params() { loadBeamsParamsImpl("beams2", beams2); }

    void loadMeasurementDuration()
    {
        toml::node_view<toml::node> val = sourceTomlGateDecomposition["measurement_duration"];
        if (val && val.is_floating_point()) {
            measurementDuration = val.as_floating_point()->get();
        }
    }

    static int parseTransitionIndex(const std::string &transition)
    {
        if (transition == "downstate_estate")
            return 0;
        if (transition == "downstate_estate2")
            return 1;
        if (transition == "upstate_estate")
            return 2;
        if (transition == "upstate_estate2")
            return 3;
        return -1;
    }

    static double getTomlDouble(toml::node_view<toml::node> node)
    {
        if (auto fp = node.as_floating_point())
            return fp->get();
        if (auto i = node.as_integer())
            return static_cast<double>(i->get());
        return 0.0;
    }

    void loadDetectionBeamParams()
    {
        toml::node_view<toml::node> beamsToml = sourceTomlGateDecomposition["detection_beam"];
        if (!beamsToml || !beamsToml.is_array()) {
            return;
        }
        size_t numBeams = beamsToml.as_array()->size();
        for (size_t i = 0; i < numBeams; i++) {
            auto beam = beamsToml[i];
            double rabi = getTomlDouble(beam["rabi"]);
            double detuning = getTomlDouble(beam["detuning"]);
            std::vector<int64_t> polarization =
                tomlArray2StdVector<int64_t>(*(beam["polarization"].as_array()));
            std::vector<int64_t> wavevector =
                tomlArray2StdVector<int64_t>(*(beam["wavevector"].as_array()));
            std::string transition = beam["transition"].as_string()->get();
            int transitionIndex = parseTransitionIndex(transition);
            if (transitionIndex < 0) {
                assert(false && "detection_beam: invalid transition string");
            }
            detectionBeams.emplace_back(rabi, detuning, polarization, wavevector, transitionIndex);
        }
    }

    void loadBeamsParamsImpl(const std::string &mode, std::vector<Beam> &collector)
    {
        // Read in the gate decomposition beam parameters from toml file.
        // The toml contains a list of beams, where each beam has the following fields:
        //   rabi = 4.4
        //   detuning = 5.5
        //   polarization = [6,7]
        //   wavevector = [8,9]

        toml::node_view<toml::node> beamsToml = sourceTomlGateDecomposition[mode];
        if (!beamsToml || !beamsToml.is_array()) {
            return;
        }
        size_t numBeams = beamsToml.as_array()->size();

        for (size_t i = 0; i < numBeams; i++) {
            auto beam = beamsToml[i];
            double rabi = beam["rabi"].as_floating_point()->get();
            double detuning = beam["detuning"].as_floating_point()->get();
            std::vector<int64_t> polarization =
                tomlArray2StdVector<int64_t>(*(beam["polarization"].as_array()));
            std::vector<int64_t> wavevector =
                tomlArray2StdVector<int64_t>(*(beam["wavevector"].as_array()));

            collector.push_back(Beam(rabi, detuning, polarization, wavevector));
        }
    }

    void loadPhononParams(size_t n_qubits)
    {
        // TODO: The fact that loading phonons depend on the number of qubits is a bit of a hack.
        // This is not ideal since we want to support dynamic number of qubits in the future.
        // We should find a better way to handle this in the database.
        std::string phonon_str = "phonons" + std::to_string(n_qubits);
        toml::node_view<toml::node> phononsToml = sourceTomlGateDecomposition[phonon_str];
        size_t numPhonons = phononsToml.as_array()->size();

        auto parseSinglePhonon = [](auto direction) {
            double energy = direction["energy"].as_floating_point()->get();
            std::vector<double> eigenvector =
                tomlArray2StdVector<double>(*(direction["eigenvector"].as_array()));
            return Phonon(energy, eigenvector);
        };

        for (size_t i = 0; i < numPhonons; i++) {
            Phonon phonon = parseSinglePhonon(phononsToml[i]);
            phonons.push_back(phonon);
        }
    }

    void loadIonParams()
    {
        toml::node_view<toml::node> ionsToml = sourceTomlQubit["ions"];

        auto parseSingleLevel = [](auto level) {
            std::string label = level["label"].as_string()->get();
            int64_t principal = level["principal"].as_integer()->get();

            std::vector<std::string> properties{"spin",
                                                "orbital",
                                                "nuclear",
                                                "spin_orbital",
                                                "spin_orbital_nuclear",
                                                "spin_orbital_nuclear_magnetization",
                                                "energy"};
            std::vector<double> propertiesData(properties.size());

            std::transform(properties.begin(), properties.end(), propertiesData.begin(),
                           [&level](const std::string &name) {
                               return level[name].as_floating_point()->get();
                           });

            return Level(label, principal, propertiesData[0], propertiesData[1], propertiesData[2],
                         propertiesData[3], propertiesData[4], propertiesData[5],
                         propertiesData[6]);
        };

        auto parseSingleTransition = [](const auto &transition_entry) {
            double einstein_a = transition_entry["einstein_a"].as_floating_point()->get();
            std::string level1 = transition_entry["level1"].as_string()->get();
            std::string level2 = transition_entry["level2"].as_string()->get();
            std::string multipole = transition_entry["multipole"].as_string()->get();

            std::set<std::string> levelEncodings{"downstate", "upstate", "estate", "estate2"};
            assert((levelEncodings.count(level1) & levelEncodings.count(level2)) &&
                   "Only \"downstate\", \"upstate\", \"estate\" and \"estate2\" are allowed in the "
                   "atom's "
                   "transition levels.");

            return Transition(level1, level2, multipole, einstein_a);
        };

        for (auto &ion_it : *(ionsToml.as_table())) {
            std::string name(ion_it.first.str());
            toml::table *data = ion_it.second.as_table();

            double mass = data->at_path("mass").as_floating_point()->get();
            double charge = data->at_path("charge").as_floating_point()->get();

            std::vector<double> position =
                tomlArray2StdVector<double>(*(data->at_path("position").as_array()));

            Level downstate = parseSingleLevel(data->at_path("levels")["downstate"]);
            Level upstate = parseSingleLevel(data->at_path("levels")["upstate"]);
            Level estate = parseSingleLevel(data->at_path("levels")["estate"]);
            Level estate2 = parseSingleLevel(data->at_path("levels")["estate2"]);
            std::vector<Level> levels{downstate, upstate, estate, estate2};

            std::vector<Transition> transitions;
            auto *transitionsTable = data->at_path("transitions").as_table();
            for (auto &transition : *transitionsTable) {
                transitions.push_back(parseSingleTransition(*(transition.second.as_table())));
            }

            Ion ion(name, mass, charge, position, levels, transitions);
            ions.insert({name, ion});
        }
    }
};

} // namespace ion
} // namespace catalyst
