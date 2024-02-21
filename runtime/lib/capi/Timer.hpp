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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace Catalyst::Runtime::Utils {

class Timer {
  private:
    // Toggle the support w.r.t. the value of `ENABLE_DEBUG_TIMER`
    bool debug_timer;

    // Manage the call order of `start` and `stop` methods
    bool running;

    // Start and stop time points using steady_clock
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::time_point<std::chrono::steady_clock> stop_time_;

  public:
    explicit Timer() : debug_timer(false), running(false)
    {
        char *value = getenv("ENABLE_DEBUG_TIMER");
        if (value && std::string(value) == "ON") {
            debug_timer = true;
        }
    }

    void start() noexcept
    {
        if (debug_timer) {
            start_time_ = std::chrono::steady_clock::now();
            running = true;
        }
    }

    void stop() noexcept
    {
        if (debug_timer && running) {
            stop_time_ = std::chrono::steady_clock::now();
            running = false;
        }
    }

    auto elapsed() noexcept
    {
        if (debug_timer) {
            if (running)
                stop();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_ - start_time_);
        }
        else {
            return std::chrono::nanoseconds(0);
        }
    }

    void print(const std::string &name) noexcept
    {
        // Convert nanoseconds (long) to milliseconds (double)
        const auto ms = static_cast<double>(elapsed().count()) / 1e6;
        // Get the hash of id as there is no conversion from id to size_t (or string)
        const auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());

        std::cerr << "[TIMER] Running " << name << " in " << ms << "ms";
        std::cerr << " (thread_id = " << std::to_string(id) << ")" << std::endl;
    }

    void store(const std::string &name, const std::string &key,
               const std::filesystem::path &file_path)
    {
        // Convert nanoseconds (long) to milliseconds (double)
        const auto ms = static_cast<double>(elapsed().count()) / 1e6;
        // Get the hash of id as there is no conversion from id to size_t (or string)
        const auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        // Create YAML headers with key and thread-id conditionally
        const auto header = key + " (thread_id=" + std::to_string(id) + ")";

        if (!std::filesystem::exists(file_path)) {
            std::ofstream ofile(file_path);
            assert(ofile.is_open() && "Invalid file to store timer results");
            ofile << header << ":" << std::endl;
            ofile << "  - " << name << ": " << ms << "ms" << std::endl;
            ofile.close();
            return;
        }
        // else
        // First, check if the header is in the file
        std::ifstream ifile(file_path);
        assert(ifile.is_open() && "Invalid file to store timer results");
        std::string line;
        bool add_header = true;
        while (add_header && std::getline(ifile, line)) {
            if (line.find(header) != std::string::npos) {
                add_header = false;
            }
        }
        ifile.close();

        // Second, update the file
        std::ofstream ofile(file_path, std::ios::app);
        assert(ofile.is_open() && "Invalid file to store timer results");
        if (add_header) {
            ofile << header << ":" << std::endl;
        }
        ofile << "  - " << name << ": " << ms << "ms" << std::endl;
        ofile.close();
    }

    void dump(const std::string &name, const std::string &key = "Runtime")
    {
        if (!debug_timer) {
            return;
        }

        char *file = getenv("DEBUG_TIMER_RESULT_PATH");
        if (!file) {
            print(name);
            return;
        }
        // else
        store(name, key, std::filesystem::path{file});
    }
};
} // namespace Catalyst::Runtime::Utils
