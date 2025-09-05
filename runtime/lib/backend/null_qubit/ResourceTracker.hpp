#include <chrono>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utils.hpp"

namespace Catalyst::Runtime {

struct ResourceTracker final {
  private:
    std::unordered_map<std::string, std::size_t> gate_types_;
    std::unordered_map<std::size_t, std::size_t> gate_sizes_;
    std::unordered_map<QubitIdType, std::size_t> wire_depths;
    std::size_t max_num_wires_;
    bool static_fname_;
    bool compute_depth_;
    std::string resources_fname_;

    void Operation(const std::string &name, const std::vector<QubitIdType> &wires,
                   const std::vector<QubitIdType> &controlled_wires)
    {
        // Sanity check that wire numbers make sense
        size_t total_wires = wires.size() + controlled_wires.size();
        std::vector<QubitIdType> combined_wires = {};
        combined_wires.insert(combined_wires.end(), wires.begin(), wires.end());
        combined_wires.insert(combined_wires.end(), controlled_wires.begin(),
                              controlled_wires.end());
        for (const auto &i : combined_wires) {
            if (i >= max_num_wires_) {
                RT_FAIL(("Wire index " + std::to_string(i) + " exceeds allocated wires").c_str());
            }
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
    ResourceTracker()
    {
        Reset();
        compute_depth_ = false;
        static_fname_ = false;
    }

    void Reset()
    {
        gate_types_.clear();
        gate_sizes_.clear();
        wire_depths.clear();
        max_num_wires_ = 0;
    }

    /**
     * @brief Returns the number of gates used since the last time this object was reset
     */
    auto GetNumGates(const std::string &gate_name = "") -> std::size_t
    {
        if (gate_name != "") {
            return gate_types_[gate_name];
        }

        size_t num_gates = 0;
        for (const auto &[gate_type, count] : gate_types_) {
            // TODO: Probably a nice functional way to do this
            num_gates += count;
        }
        return num_gates;
    }

    auto GetNumGatesBySize(const std::size_t &gate_size) -> std::size_t
    {
        return gate_sizes_[gate_size];
    }

    /**
     * @brief Returns the maximum number of qubits used since the last time this object was reset
     */
    auto GetNumWires() -> std::size_t { return max_num_wires_; }

    /**
     * @brief Returns the filename where resource tracking information is dumped
     */
    auto GetFilename() const -> std::string { return resources_fname_; }

    auto GetComputeDepth() const -> bool { return compute_depth_; }

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

    void SetResourcesFname(const std::string &fname)
    {
        resources_fname_ = fname;
        static_fname_ = true;
    }

    // Store the highest number of qubits allocated at any time since device creation
    void SetMaxWires(std::size_t max_wires)
    {
        max_num_wires_ = std::max(max_wires, max_num_wires_);
    }

    void SetComputeDepth(const bool compute_depth)
    {
        if (max_num_wires_ != 0) {
            RT_FAIL("Cannot set depth tracking after qubits have been allocated.");
        }
        this->compute_depth_ = compute_depth;
    }

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
     * @brief Prints resources that would be used to execute this circuit as a JSON
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