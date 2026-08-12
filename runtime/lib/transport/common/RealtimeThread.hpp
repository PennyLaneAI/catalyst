// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file
 * Put the calling thread on a dedicated core with the interruptions turned off.
 */

#pragma once

#include <cstdint>
#include <string>

#if defined(__linux__)
#include <sys/mman.h>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace catalyst::transport::common {

/**
 * @brief Pin the calling thread to @p cpu; if @p realtime, also request SCHED_FIFO, mlockall, and
 * low DMA latency. Best-effort (no-op on non-Linux). Call once from the target thread before the
 * loop.
 *
 * @param cpu Core to pin to; negative leaves affinity alone.
 * @param realtime Request realtime scheduling / locked memory / low latency.
 * @return Short description of what was granted (for logging).
 */
inline std::string pin_thread(int cpu, bool realtime) {
#if defined(__linux__)
    std::string got;
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
            got += "cpu=" + std::to_string(cpu);
        } else {
            got += "cpu=unavailable";
        }
    }
    if (!realtime) {
        return got.empty() ? "unpinned" : got;
    }

    sched_param sp{};
    sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
    got += (sched_setscheduler(0, SCHED_FIFO, &sp) == 0)
               ? " sched=FIFO/" + std::to_string(sp.sched_priority)
               : " sched=default(needs root)";

    got += (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) ? " mlock=on" : " mlock=off";

    static int dma_latency_fd = -1;
    if (dma_latency_fd < 0) {
        dma_latency_fd = ::open("/dev/cpu_dma_latency", O_RDWR);
        if (dma_latency_fd >= 0) {
            const std::int32_t zero = 0;
            if (::write(dma_latency_fd, &zero, sizeof(zero)) != sizeof(zero)) {
                ::close(dma_latency_fd);
                dma_latency_fd = -1;
            }
        }
    }
    got += (dma_latency_fd >= 0) ? " cstates=off" : " cstates=default";
    return got;
#else
    (void)cpu;
    (void)realtime;
    return "unpinned (not Linux)";
#endif
}

/**
 * @brief Spin-wait hint.
 *
 * Marks a busy-wait so the CPU backs off speculated loads and yields the SMT sibling.
 */
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

/**
 * @brief Interval between stop-token polls in a coprocessor's arrival spin.
 *
 * The count for how often the arrival spin looks at the stop token.
 */
inline constexpr std::uint32_t STOP_CHECK_SPINS = 600'000'000;

} // namespace catalyst::transport::common
