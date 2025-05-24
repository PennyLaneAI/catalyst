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

#include <cassert>
#include <cstdlib>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility> // std::forward

#include <ctime>

// Note that this method returns CPU time on Linux/Unix-like systems,
// and returns wall-clock time on Windows.
static inline double getClock()
{
    // Convert results in ms
    return static_cast<double>(1000.0 * std::clock() / CLOCKS_PER_SEC) * 0.001;
}

namespace catalyst::utils {

/**
 * Timer: A utility class to measure the wall-time and CPU-time of code blocks.
 *
 * To display results, run the driver with the `ENABLE_DIAGNOSTICS=ON` variable.
 * To store results in YAML format, use `DIAGNOSTICS_RESULTS_PATH=/path/to/file.yml`
 * along with `ENABLE_DIAGNOSTICS=ON`.
 */
class Timer {
  private:
    // Toggle the support w.r.t. the value of `ENABLE_DIAGNOSTICS`
    bool debug_timer;

    // Manage the call order of `start` and `stop` methods
    bool running;

    // Start and stop time points using steady_clock
    std::chrono::time_point<std::chrono::steady_clock> start_wall_time_;
    std::chrono::time_point<std::chrono::steady_clock> stop_wall_time_;

    // Start and stop CPU Time using the system clock
    double start_cpu_time_;
    double stop_cpu_time_;

    static inline bool enable_debug_timer() noexcept
    {
        char *value = getenv("ENABLE_DIAGNOSTICS");
        return value && std::string(value) == "ON";
    }

  public:
    explicit Timer() : debug_timer(enable_debug_timer()), running(false) {}

    [[nodiscard]] bool is_active() const noexcept { return running; }

    void start() noexcept
    {
        if (debug_timer) {
            start_wall_time_ = std::chrono::steady_clock::now();
            start_cpu_time_ = getClock();
            running = true;
        }
    }

    void stop() noexcept
    {
        if (debug_timer && running) {
            stop_cpu_time_ = getClock();
            stop_wall_time_ = std::chrono::steady_clock::now();
            running = false;
        }
    }

    [[nodiscard]] auto elapsed() noexcept
    {
        if (debug_timer) {
            if (running) {
                stop();
            }
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop_wall_time_ -
                                                                        start_wall_time_);
        }
        else {
            return std::chrono::nanoseconds(0);
        }
    }

    void print(const std::string &name, bool add_endl = true) noexcept
    {
        // Convert nanoseconds (long) to milliseconds (double)
        const auto wall_elapsed = static_cast<double>(elapsed().count()) / 1e6;
        const auto cpu_elapsed = (stop_cpu_time_ - start_cpu_time_) * 1e+3;

        std::cerr << "[DIAGNOSTICS] Running " << std::setw(23) << std::left << name;
        std::cerr << "\t" << std::fixed << "walltime: " << std::setprecision(3) << wall_elapsed
                  << std::fixed << " ms";
        std::cerr << "\t" << std::fixed << "cputime: " << std::setprecision(3) << cpu_elapsed
                  << std::fixed << " ms";
        if (add_endl) {
            std::cerr << "\n";
        }
    }

    void store(const std::string &name, const std::filesystem::path &file_path)
    {
        // Convert nanoseconds (long) to milliseconds (double)
        const auto wall_elapsed = static_cast<double>(elapsed().count()) / 1e6;
        const auto cpu_elapsed = (stop_cpu_time_ - start_cpu_time_) * 1e+3;

        const std::string_view key_padding = "          ";
        const std::string_view val_padding = "              ";

        if (!std::filesystem::exists(file_path)) {
            std::ofstream ofile(file_path);
            assert(ofile.is_open() && "Invalid file to store timer results");
            ofile << key_padding << "- " << name << ":\n";
            ofile << val_padding << "walltime: " << wall_elapsed << "\n";
            ofile << val_padding << "cputime: " << cpu_elapsed << "\n";
            ofile.close();
            return;
        }
        // else

        // Second, update the file
        std::ofstream ofile(file_path, std::ios::app);
        assert(ofile.is_open() && "Invalid file to store timer results");
        ofile << key_padding << "- " << name << ":\n";
        ofile << val_padding << "walltime: " << wall_elapsed << "\n";
        ofile << val_padding << "cputime: " << cpu_elapsed << "\n";
        ofile.close();
    }

    void dump(const std::string &name, bool add_endl = true)
    {
        if (!debug_timer) {
            return;
        }

        char *file = getenv("DIAGNOSTICS_RESULTS_PATH");
        if (!file) {
            print(name, add_endl);
            return;
        }
        // else
        store(name, std::filesystem::path{file});
    }

    template <typename Function, typename... Args>
    static auto timer(Function func, const std::string &name, bool add_endl, Args &&...args)
    {
        if (!enable_debug_timer()) {
            return func(std::forward<Args>(args)...);
        }

        Timer timer{};

        timer.start();
        auto result = func(std::forward<Args>(args)...);
        timer.dump(name, add_endl);

        return result;
    }
};
} // namespace catalyst::utils
