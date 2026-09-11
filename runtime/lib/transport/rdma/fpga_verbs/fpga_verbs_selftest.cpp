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

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <string>
#include <vector>

#include "FpgaControllerSession.hpp"
#include "WireProtocol.hpp"

#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>

using namespace catalyst::transport;
using namespace catalyst::transport::common;

// Controller RTT self-test: connect to the coprocessor and, per round, write a salted
// request via data_slot()+kick(), then collect() the echoed reply and check it
// against SALT. The coprocessor listens; start it first. Locally, prefer run_roundtrip.sh.

namespace {

// clang-format off
struct Args {
    std::string dev  = "xib_0";
    int gid          = 3;
    std::string peer = "192.168.1.2";
    std::uint16_t port    = 18560;
    std::uint64_t iters   = 1000;
    std::uint64_t warmup  = 0;
    std::uint32_t ring    = static_cast<std::uint32_t>(K_RING_SLOTS);
    std::uint32_t stride_log2      = 6;
    std::uint32_t syndrome_bytes   = sizeof(std::uint64_t);
    std::uint32_t correction_bytes = sizeof(std::uint64_t);
    int cpu_pin      = -1;
    const char *csv  = nullptr;
};
// clang-format on

std::uint64_t now_ns() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

bool parse(int argc, char **argv, Args &a) {
    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char *f) { return std::strcmp(argv[i], f) == 0; };
        auto next = [&]() { return argv[++i]; };
        if (eq("--dev")) {
            a.dev = next();
        } else if (eq("--gid")) {
            a.gid = std::atoi(next());
        } else if (eq("--peer")) {
            a.peer = next();
        } else if (eq("--port")) {
            a.port = static_cast<std::uint16_t>(std::atoi(next()));
        } else if (eq("--iters")) {
            a.iters = std::strtoull(next(), nullptr, 10);
        } else if (eq("--warmup")) {
            a.warmup = std::strtoull(next(), nullptr, 10);
        } else if (eq("--ring")) {
            a.ring = static_cast<std::uint32_t>(std::strtoul(next(), nullptr, 0));
        } else if (eq("--stride-log2")) {
            a.stride_log2 = static_cast<std::uint32_t>(std::strtoul(next(), nullptr, 0));
        } else if (eq("--syndrome-bytes")) {
            a.syndrome_bytes = static_cast<std::uint32_t>(std::strtoul(next(), nullptr, 0));
        } else if (eq("--correction-bytes")) {
            a.correction_bytes = static_cast<std::uint32_t>(std::strtoul(next(), nullptr, 0));
        } else if (eq("--cpu-pin")) {
            a.cpu_pin = std::atoi(next());
        } else if (eq("--csv")) {
            a.csv = next();
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

// Print the fault addr, backtrace, and /proc/self/maps
// (for offline addr2line), then re-raise. Async-signal-safe calls only.
void crash_handler(int sig, siginfo_t *si, void * /*ucontext*/) {
    char buf[160];
    int len =
        std::snprintf(buf, sizeof(buf), "\n[controller] *** FATAL signal %d at fault addr %p ***\n",
                      sig, si != nullptr ? si->si_addr : nullptr);
    if (len > 0) {
        (void)!write(STDERR_FILENO, buf, static_cast<size_t>(len));
    }
    void *frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    const char hdr[] = "[controller] ----- /proc/self/maps -----\n";
    (void)!write(STDERR_FILENO, hdr, sizeof(hdr) - 1);
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd >= 0) {
        char m[1024];
        ssize_t r;
        while ((r = read(fd, m, sizeof(m))) > 0) {
            (void)!write(STDERR_FILENO, m, static_cast<size_t>(r));
        }
        close(fd);
    }

    std::signal(sig, SIG_DFL);
    raise(sig);
}

void install_crash_handler() {
    struct sigaction sa = {};
    sa.sa_sigaction = crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
}

// RTT percentiles (ns) over samples, dropping the leading `warmup` shots.
void report_rtt(std::vector<std::uint64_t> s, std::uint64_t warmup, const char *clock) {
    if (warmup > 0 && s.size() > warmup) {
        s.erase(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(warmup));
    }
    if (s.empty()) {
        std::fprintf(stderr, "[controller] no timed samples to report\n");
        return;
    }
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) { return s[static_cast<std::size_t>(p * (double)(s.size() - 1))]; };
    long double sum = 0;
    for (std::uint64_t v : s) {
        sum += static_cast<long double>(v);
    }
    double mean = static_cast<double>(sum / static_cast<long double>(s.size()));
    auto row = [&](const char *tag, double ns) {
        std::fprintf(stderr, "  %-6s %10.0f ns   (%8.3f us)\n", tag, ns, ns / 1000.0);
    };
    std::fprintf(stderr, "\n=== per-round RTT  (n=%zu, excl. %llu warmup, %s) ===\n", s.size(),
                 static_cast<unsigned long long>(warmup), clock);
    row("min", (double)s.front());
    row("p50", (double)pct(0.50));
    row("p95", (double)pct(0.95));
    row("p99", (double)pct(0.99));
    row("p99.9", (double)pct(0.999));
    row("max", (double)s.back());
    row("mean", mean);
}

void write_csv(const char *path, const std::vector<std::uint64_t> &s, std::uint64_t warmup) {
    std::size_t start = (warmup > 0) ? static_cast<std::size_t>(warmup) : 0;
    if (s.size() <= start) {
        std::fprintf(stderr, "[controller] no post-warmup samples; csv not written\n");
        return;
    }
    FILE *f = std::fopen(path, "w");
    if (!f) {
        std::fprintf(stderr, "[controller] cannot open csv %s\n", path);
        return;
    }
    std::fprintf(f, "sample,rtt_ns,rtt_us\n");
    for (std::size_t i = start; i < s.size(); ++i) {
        std::fprintf(f, "%zu,%llu,%.4f\n", i - start, (unsigned long long)s[i],
                     (double)s[i] / 1000.0);
    }
    std::fclose(f);
    std::fprintf(stderr, "[controller] wrote %zu samples to %s\n", s.size() - start, path);
}

} // namespace

int main(int argc, char **argv) {
    install_crash_handler();

    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(
            stderr,
            "usage: fpga_verbs_selftest [--dev DEV] [--gid 3] --peer <ip> [--port 18560]\n"
            "   [--iters 1000] [--warmup 0] [--ring 256] [--stride-log2 6]\n"
            "   [--syndrome-bytes 8] [--correction-bytes 8] [--cpu-pin N] [--csv rtt.csv]\n");
        return 2;
    }
    if (a.iters == 0) {
        a.iters = 1;
    }

    std::fprintf(stderr,
                 "[controller] dev=%s gid=%d peer=%s port=%u iters=%llu warmup=%llu ring=%u "
                 "stride=%uB syn=%uB corr=%uB\n",
                 a.dev.c_str(), a.gid, a.peer.c_str(), a.port,
                 static_cast<unsigned long long>(a.iters),
                 static_cast<unsigned long long>(a.warmup), a.ring, 1u << a.stride_log2,
                 a.syndrome_bytes, a.correction_bytes);

    // RdmaError -> catch; reply timeout -> collect() returns rc<0 (NO-DATA).
    // Real faults still hit SIGSEGV via the crash handler.
    try {
        rdma::devices::fpga_verbs::FpgaControllerSession s(a.dev, a.gid, a.ring, a.stride_log2);
        ConnectInfo ci{
            .peer = a.peer,
            .oob_port = a.port,
        };
        s.connect(ci);
        MemRegion m = s.alloc_memory(REGION_BYTES, MemKind::CpuRam);
        PeerRef p = s.exchange_keys(m);
        ChannelDesc desc{
            .transport = "rdma",
        };
        s.establish_channel(desc, m, p);
        s.set_cpu_pin(a.cpu_pin);
        s.commit_work_item(0, a.syndrome_bytes, a.correction_bytes);
        s.start();

        std::fprintf(
            stderr, "[controller] channel up (peer rkey=0x%x addr=0x%llx) -- running %llu rounds\n",
            p.rkey, static_cast<unsigned long long>(p.remote_addr),
            static_cast<unsigned long long>(a.iters));

        std::vector<std::uint64_t> samples;
        samples.reserve(a.iters);
        std::vector<std::uint8_t> corr(a.correction_bytes, 0);

        std::uint64_t completed = 0;
        bool pass = true;
        bool used_hw = false;
        for (std::uint64_t cursor = 0; cursor < a.iters; ++cursor) {
            // (cursor << 32) | SALT: the coprocessor validates the low 32 bits (SALT).
            // kick() stamps seq_num; here we only fill the data word.
            auto *slot = reinterpret_cast<Payload *>(s.data_slot());
            if (slot == nullptr) {
                std::fprintf(stderr, "[controller] round %llu: data_slot() returned NULL\n",
                             static_cast<unsigned long long>(cursor));
                pass = false;
                break;
            }
            slot->value = (static_cast<std::uint64_t>(cursor) << 32) | SALT;
            std::fill(corr.begin(), corr.end(), 0);

            // Wall-clock timing; prefer last_rtt_ns() if the transport reports it.
            std::uint64_t t0 = now_ns();
            if (s.kick(0) != 0) {
                std::fprintf(stderr, "[controller] kick failed at cursor=%llu\n",
                             static_cast<unsigned long long>(cursor));
                pass = false;
                break;
            }
            void *outs[1] = {corr.data()};
            const std::size_t outn[1] = {a.correction_bytes};
            const int rc = s.collect(outs, outn, 1);
            std::uint64_t sw_rtt = now_ns() - t0;
            if (rc != 0) {
                std::fprintf(stderr,
                             "[controller] collect rc=%d at cursor=%llu -> NO-DATA (no reply; the "
                             "RDMA_WRITE likely never egressed the NIC)\n",
                             rc, static_cast<unsigned long long>(cursor));
                pass = false;
                break;
            }

            std::uint64_t got = 0;
            std::memcpy(&got, corr.data(), std::min<std::size_t>(a.correction_bytes, sizeof(got)));
            if (static_cast<std::uint32_t>(got) != SALT) {
                std::fprintf(stderr,
                             "[controller] bad reply at cursor=%llu: got=0x%llx (SALT=0x%x)\n",
                             static_cast<unsigned long long>(cursor),
                             static_cast<unsigned long long>(got), SALT);
                pass = false;
                break;
            }

            std::uint64_t hw_rtt = s.last_rtt_ns();
            used_hw = used_hw || (hw_rtt != 0);
            samples.push_back(hw_rtt != 0 ? hw_rtt : sw_rtt);
            ++completed;
            if ((cursor & 63) == 0 || cursor == a.iters - 1) {
                std::uint64_t rtt = samples.back();
                std::fprintf(stderr,
                             "[controller] round %4llu ok  rtt=%llu ns (%.3f us) [%s]  "
                             "got=0x%016llx\n",
                             static_cast<unsigned long long>(cursor),
                             static_cast<unsigned long long>(rtt), (double)rtt / 1000.0,
                             hw_rtt ? "hw" : "sw", static_cast<unsigned long long>(got));
            }
        }
        s.stop();

        report_rtt(samples, a.warmup, used_hw ? "software engine RTT" : "software wall-clock");
        if (a.csv != nullptr) {
            write_csv(a.csv, samples, a.warmup);
        }
        std::fprintf(stderr, "[controller] completed %llu/%llu round-trips -> %s\n",
                     static_cast<unsigned long long>(completed),
                     static_cast<unsigned long long>(a.iters), pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[controller] FAILED: %s\n", e.what());
        return 1;
    }
}
