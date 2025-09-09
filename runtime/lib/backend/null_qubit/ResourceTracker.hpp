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

#include <chrono>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utils.hpp"

namespace Catalyst::Runtime {

/**
 * @brief A utility class for tracking quantum circuit resource usage including gate counts, wire
 * usage, and circuit depth
 *
 * This class provides comprehensive tracking of quantum circuit resources during execution,
 * including counting gate types and sizes, tracking maximum wire usage, and optionally
 * computing circuit depth. It can export the collected data to JSON format for analysis.
 */
struct ResourceTracker final {
  private:
    std::unordered_map<std::string, std::size_t> gate_types_;
    std::unordered_map<std::size_t, std::size_t> gate_sizes_;
    std::unordered_map<QubitIdType, std::size_t> wire_depths;
    std::size_t max_num_wires_;
    bool static_fname_;
    bool compute_depth_;
    std::string resources_fname_;

    /**
     * @brief Internal method to record an operation being applied to the device
     *
     * Updates the gate type and size counts, and if depth tracking is enabled,
     * updates the depth of the wires involved in the operation.
     *
     * @param name The name of the operation being applied
     * @param wires The wires the operation is being applied to
     * @param controlled_wires The control wires the operation is being applied to
     */
    void Operation(const std::string &name, const std::vector<QubitIdType> &wires,
                   const std::vector<QubitIdType> &controlled_wires)
    {
        // Sanity check that wire numbers make sense
        std::size_t total_wires = wires.size() + controlled_wires.size();
        std::vector<QubitIdType> combined_wires = {};
        combined_wires.insert(combined_wires.end(), wires.begin(), wires.end());
        combined_wires.insert(combined_wires.end(), controlled_wires.begin(),
                              controlled_wires.end());
        for (const auto &i : combined_wires) {
            RT_FAIL_IF(static_cast<std::size_t>(i) >= max_num_wires_,
                       ("Wire index " + std::to_string(i) + " exceeds allocated wires").c_str());
        }

        gate_types_[name]++;
        gate_sizes_[total_wires]++;
        if (compute_depth_) {
            std::size_t max_depth = 0;
            for (const auto &i : combined_wires) {
                max_depth = std::max(max_depth, wire_depths[i]);
            }
            // All wires used in this operation must now have their depth set based on this max
            max_depth++;
            for (const auto &i : wires) {
                wire_depths[i] = max_depth;
            }
        }
    }

  public:
    /**
     * @brief Default constructor that initializes the ResourceTracker with default settings
     *
     * Initializes the tracker with depth computation disabled and no static filename set.
     * Calls Reset() to ensure all tracking data structures are properly initialized.
     */
    ResourceTracker()
    {
        Reset();
        compute_depth_ = false;
        static_fname_ = false;
    }

    /**
     * @brief Resets all tracked resource data to initial state
     *
     * Clears all gate type counts, gate size counts, wire depth information,
     * and resets the maximum wire count to zero. Does not affect configuration
     * settings like depth computation or filename settings.
     */
    void Reset()
    {
        gate_types_.clear();
        gate_sizes_.clear();
        wire_depths.clear();
        max_num_wires_ = 0;
    }

    /**
     * @brief Returns the number of gates used since the last time this object was reset
     *
     * @param gate_name Optional specific gate name to count. If empty, returns total count of all
     * gates
     * @return The count of gates matching the specified name, or total gate count if no name
     * provided
     */
    auto GetNumGates(const std::string &gate_name = "") -> std::size_t
    {
        if (gate_name != "") {
            return gate_types_[gate_name];
        }

        std::size_t num_gates = 0;
        for (const auto &[gate_type, count] : gate_types_) {
            num_gates += count;
        }
        return num_gates;
    }

    /**
     * @brief Returns the number of gates of a specific size (number of qubits acted upon)
     *
     * @param gate_size The number of qubits that gates of interest act upon
     * @return The count of gates that act on exactly gate_size qubits
     */
    auto GetNumGatesBySize(const std::size_t &gate_size) -> std::size_t
    {
        return gate_sizes_[gate_size];
    }

    /**
     * @brief Returns the maximum number of qubits used since the last time this object was reset
     *
     * @return The highest number of qubits that have been allocated at any point
     */
    auto GetNumWires() -> QubitIdType { return max_num_wires_; }

    /**
     * @brief Returns the filename where resource tracking information is dumped
     *
     * @return The current filename that will be used for writing resource data
     */
    auto GetFilename() const -> std::string { return resources_fname_; }

    /**
     * @brief Returns whether circuit depth computation is currently enabled
     *
     * @return True if depth tracking is enabled, false otherwise
     */
    auto GetComputeDepth() const -> bool { return compute_depth_; }

    /**
     * @brief Returns the current circuit depth if depth computation is enabled
     *
     * @return The maximum depth across all wires if depth tracking is enabled, 0 otherwise
     */
    auto GetDepth() const -> std::size_t
    {
        if (compute_depth_ && !wire_depths.empty()) {
            auto max_pair = std::max_element(wire_depths.begin(), wire_depths.end(),
                                             [](const std::pair<QubitIdType, std::size_t> &p1,
                                                const std::pair<QubitIdType, std::size_t> &p2) {
                                                 return p1.second < p2.second;
                                             });
            return max_pair->second;
        }
        return 0;
    }

    /**
     * @brief Sets a static filename for resource data output.
     *
     * Once this function has been called, dynamic filenames will no longer be generated
     * @param fname The filename to use for writing resource tracking data
     */
    void SetResourcesFname(const std::string &fname)
    {
        resources_fname_ = fname;
        static_fname_ = true;
    }

    /**
     * @brief Updates the maximum number of wires tracked
     *
     * Stores the highest number of qubits allocated at any time since device creation.
     * The maximum can only increase, never decrease.
     *
     * @param max_wires The current number of allocated wires
     */
    void SetMaxWires(std::size_t num_wires)
    {
        max_num_wires_ = std::max(num_wires, max_num_wires_);
    }

    /**
     * @brief Enables or disables circuit depth computation
     *
     * Must be run before any qubits have been allocated and any gates have been executed
     *
     * @param compute_depth Whether to enable depth tracking
     * @throws Runtime error if called after qubits have already been allocated
     */
    void SetComputeDepth(const bool compute_depth)
    {
        if (max_num_wires_ != 0) {
            RT_FAIL("Cannot set depth tracking after qubits have been allocated.");
        }
        this->compute_depth_ = compute_depth;
    }

    /**
     * @brief Records a named quantum operation for resource tracking
     *
     * Tracks a quantum gate with proper handling of controlled and adjoint modifiers.
     * Automatically formats the operation name with appropriate prefixes and suffixes
     * based on the modifiers applied.
     *
     * @param name The base name of the quantum operation
     * @param inverse Whether this is an adjoint (inverse) operation
     * @param wires The target wires the operation acts upon
     * @param controlled_wires The control wires for controlled operations (empty for
     * non-controlled)
     */
    void NamedOperation(const std::string &name, bool inverse,
                        const std::vector<QubitIdType> &wires,
                        const std::vector<QubitIdType> &controlled_wires = {})
    {
        std::string prefix = "";
        std::string suffix = "";
        if (!controlled_wires.empty()) {
            if (controlled_wires.size() > 1) {
                prefix += std::to_string(controlled_wires.size());
            }
            prefix += "C(";
            suffix += ")";
        }
        if (inverse) {
            prefix += "Adjoint(";
            suffix += ")";
        }
        Operation(prefix + name + suffix, wires, controlled_wires);
    }

    /**
     * @brief Records a matrix-based quantum operation for resource tracking
     *
     * Tracks arbitrary unitary matrix operations with proper handling of
     * controlled and adjoint modifiers. Used for operations defined by
     * explicit unitary matrices rather than named gates.
     *
     * @param inverse Whether this is an adjoint (inverse) operation
     * @param wires The target wires the operation acts upon
     * @param controlled_wires The control wires for controlled operations (empty for
     * non-controlled)
     */
    void MatrixOperation(bool inverse, const std::vector<QubitIdType> &wires,
                         const std::vector<QubitIdType> &controlled_wires = {})
    {
        std::string op_name = "QubitUnitary";

        if (!controlled_wires.empty()) {
            op_name = "Controlled" + op_name;
        }
        if (inverse) {
            op_name = "Adjoint(" + op_name + ")";
        }
        Operation(op_name, wires, controlled_wires);
    }

    /**
     * @brief Prints resource usage statistics in JSON format to the specified file
     *
     * Outputs comprehensive resource tracking data including number of wires,
     * total gate count, breakdown by gate types, breakdown by gate sizes,
     * and circuit depth (if enabled) in JSON format.
     *
     * @param resources_file File pointer where JSON data will be written
     * @throws Runtime error if file writing fails
     */
    void PrintResourceUsage(FILE *resources_file)
    {
        std::stringstream resources;

        resources << "{\n";
        resources << "  \"num_wires\": " << max_num_wires_ << ",\n";
        resources << "  \"num_gates\": " << GetNumGates() << ",\n";
        resources << "  \"gate_types\": ";
        pretty_print_dict(gate_types_, 2, resources);
        resources << ",\n";
        resources << "  \"gate_sizes\": ";
        pretty_print_dict(gate_sizes_, 2, resources);
        resources << ",\n";
        if (compute_depth_) {
            resources << "  \"depth\": " << GetDepth();
        }
        else {
            resources << "  \"depth\": null";
        }
        resources << "\n}" << std::endl;

        std::size_t total_bytes = resources.str().size();
        std::size_t bytes_written = fwrite(resources.str().c_str(), 1, total_bytes, resources_file);

        while (bytes_written < total_bytes) {
            if (ferror(resources_file)) {
                RT_FAIL("Error writing resource tracking data to file.");
            }
            bytes_written += fwrite(resources.str().c_str() + bytes_written, 1,
                                    total_bytes - bytes_written, resources_file);
        }
    }

    /**
     * @brief Writes resource tracking data to file and resets the tracker
     *
     * Creates a timestamped filename if no static filename was set, opens the file
     * for exclusive writing, prints the resource usage data in JSON format,
     * and then resets all tracking data. The file is opened with "wx" mode to
     * prevent overwriting existing files.
     *
     * @throws Runtime error if file cannot be opened or written to
     */
    void WriteOut()
    {
        if (!static_fname_) {
            auto time = std::chrono::high_resolution_clock::now();
            auto timestamp =
                std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch())
                    .count();
            std::stringstream new_resources_fname;
            new_resources_fname << "__pennylane_resources_data_" << timestamp << ".json";

            this->resources_fname_ = new_resources_fname.str(); // Update written location
        }

        // Need to use FILE* instead of ofstream since ofstream has no way to atomically open a
        // file only if it does not already exist
        FILE *resources_file = fopen(this->resources_fname_.c_str(), "wx");
        if (resources_file == nullptr) {
            std::string err_msg =
                "Error opening file '" + this->resources_fname_ + "'."; // LCOV_EXCL_LINE
            RT_FAIL(err_msg.c_str());                                   // LCOV_EXCL_LINE
        }
        else {
            PrintResourceUsage(resources_file);
            fclose(resources_file);
        }

        Reset();
    }
};

} // namespace Catalyst::Runtime
