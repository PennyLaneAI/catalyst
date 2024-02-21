// Copyright 2023 Xanadu Quantum Technologies Inc.

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
    explicit Timer() : running(false)
    {
        char *value = getenv("ENABLE_DEBUG_TIMER");
        if (!value || std::string(value) != "ON") {
            debug_timer = false;
        }
        else {
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
            return std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ - start_time_);
        }
        else {
            return std::chrono::milliseconds(0);
        }
    }

    void print(const std::string &name) noexcept
    {
        if (!debug_timer)
            return;
        std::cerr << "[TIMER] Running " << name << " in " << elapsed().count() << "ms";
        std::cerr << " (thread_id = " << std::this_thread::get_id() << ")" << std::endl;
    }

    void dump(const std::string &name, const std::string &key = "Runtime") noexcept
    {
        if (!debug_timer)
            return;

        char *file = getenv("DEBUG_TIMER_RESULT_PATH");
        if (!file) {
            print(name);
            return;
        }

        // Path to where the results should be stored
        // If not provided, results will be dumped (stderr)
        std::filesystem::path file_ = std::filesystem::path{file};
        if (!std::filesystem::exists(file_)) {
            std::ofstream ofile(file_);
            RT_FAIL_IF(!ofile.is_open(), "Invalid file to store timer results");
            ofile << key << ":" << std::endl;
            ofile << "  - " << name << ": " << elapsed().count() << "ms" << std::endl;
            ofile.close();
        }
        else {
            std::ofstream ofile(file_, std::ios::app); // Open file_ in 'append' mode
            RT_FAIL_IF(!ofile.is_open(), "Invalid file to store timer results");
            ofile << "  - " << name << ": " << elapsed().count() << "ms" << std::endl;
            ofile.close();
        }
    }
};
} // namespace Catalyst::Runtime::Utils
