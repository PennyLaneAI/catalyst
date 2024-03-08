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
#include <iostream>
#include <string>
#include <thread>

#ifdef __linux__
#include <ctime>
#include <sys/time.h>

static inline double _getCPUTime()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)) {
        return .0;
    }
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
#else
static inline double _getCPUTime() { return .0 }
#endif

namespace catalyst::utils {

/**
 * Timer: A utility class to measure the wall-time and CPU-time of code blocks.
 *
 * To display results, run the driver with the `ENABLE_DEBUG_TIMER=ON` variable.
 * To store results in YAML format, use `DEBUG_RESULTS_FILE=/path/to/file.yml`
 * along with `ENABLE_DEBUG_TIMER=ON`.
 *
 * Note that using both `ENABLE_DEBUG_INFO=ON` and `ENABLE_DEBUG_TIMER=ON` will
 * introduce noise to the timing results.
 */
class Timer {
  private:
    // Toggle the support w.r.t. the value of `ENABLE_DEBUG_TIMER`
    bool debug_timer;

    // Manage the call order of `start` and `stop` methods
    bool running;

    // Start and stop time points using steady_clock
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::time_point<std::chrono::steady_clock> stop_time_;

    // Start and stop CPU Time using the system clock
    double start_cpu_time_;
    double stop_cpu_time_;

    static inline bool enable_debug_timer() noexcept
    {
        char *value = getenv("ENABLE_DEBUG_TIMER");
        if (value && std::string(value) == "ON") {
            return true;
        }
        return false;
    }

  public:
    explicit Timer() : debug_timer(enable_debug_timer()), running(false) {}

    void start() noexcept
    {
        if (debug_timer) {
            start_time_ = std::chrono::steady_clock::now();
            start_cpu_time_ = _getCPUTime();
            running = true;
        }
    }

    void stop() noexcept
    {
        if (debug_timer && running) {
            stop_cpu_time_ = _getCPUTime();
            stop_time_ = std::chrono::steady_clock::now();
            running = false;
        }
    }

    [[nodiscard]] auto elapsed() noexcept
    {
        if (debug_timer) {
            if (running) {
                stop();
            }
            return std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_ - start_time_);
        }
        else {
            return std::chrono::nanoseconds(0);
        }
    }

    void print(const std::string &name) noexcept
    {
        // Convert nanoseconds (long) to milliseconds (double)
        const auto wall_elapsed = static_cast<double>(elapsed().count()) / 1e6;
        const auto cpu_elapsed = (stop_cpu_time_ - start_cpu_time_) * 1e+3;

        std::cerr << "[TIMER] Running " << name;
        std::cerr << "\t walltime: " << wall_elapsed << "ms";
        std::cerr << "\t cputime: " << cpu_elapsed << "ms";
        std::cerr << std::endl;
    }

    void store(const std::string &name, const std::filesystem::path &file_path)
    {
        // Convert nanoseconds (long) to milliseconds (double)
        const auto wall_elapsed = static_cast<double>(elapsed().count()) / 1e6;
        const auto cpu_elapsed = (stop_cpu_time_ - start_cpu_time_) * 1e+3;

        if (!std::filesystem::exists(file_path)) {
            std::ofstream ofile(file_path);
            assert(ofile.is_open() && "Invalid file to store timer results");
            ofile << "        - " << name << "\n";
            ofile << "          walltime: " << wall_elapsed << "\n";
            ofile << "          cputime: " << cpu_elapsed << "\n";
            ofile.close();
            return;
        }
        // else

        // Second, update the file
        std::ofstream ofile(file_path, std::ios::app);
        assert(ofile.is_open() && "Invalid file to store timer results");
        ofile << "        - " << name << "\n";
        ofile << "          walltime: " << wall_elapsed << "\n";
        ofile << "          cputime: " << cpu_elapsed << "\n";
        ofile.close();
    }

    void dump(const std::string &name)
    {
        if (!debug_timer) {
            return;
        }

        char *file = getenv("DEBUG_RESULTS_FILE");
        if (!file) {
            print(name);
            return;
        }
        // else
        store(name, std::filesystem::path{file});
    }

    template <typename Function, typename... Args>
    static auto timer(Function func, const std::string &name, Args &&...args)
    {
        if (!enable_debug_timer()) {
            return func(std::forward<Args>(args)...);
        }

        Timer timer{};

        timer.start();
        auto &&result = func(std::forward<Args>(args)...);
        timer.dump(name);

        return result;
    }
};
} // namespace catalyst::utils
