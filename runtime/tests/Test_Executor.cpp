// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// End-to-end tests for the catalyst-executor server. Each test spawns the real
// binary on a loopback port, drives it with the host-side session API, and
// asserts on what the server process actually did.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>

#include "catch2/catch_test_macros.hpp"

#include "ExecutorSession.hpp"

#include <netinet/in.h>
#include <signal.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

// These bounds are only hit when something is broken.
constexpr auto Timeout = 10s;
constexpr auto PollInterval = 20ms;

// Bind a loopback socket to a kernel-chosen port, never listening on it. The
// caller owns the returned descriptor.
int reservePort(int &Port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
    Port = ntohs(addr.sin_port);
    return fd;
}

// A port the executor can bind. The socket is released before returning.
int freePort() {
    int Port = 0;
    ::close(reservePort(Port));
    return Port;
}

// Runs `catalyst-executor --bind 127.0.0.1:<port>` for the lifetime of the
// scope. Killing the listener leaves any already-forked connection process to
// finish its own shutdown, which is what the cleanup test relies on.
class ExecutorProcess {
  public:
    ExecutorProcess() : Address("127.0.0.1:" + std::to_string(freePort())) {
        Pid = ::fork();
        REQUIRE(Pid >= 0);
        if (Pid == 0) {
            ::execl(CATALYST_EXECUTOR_PATH, "catalyst-executor", "--bind", Address.c_str(),
                    static_cast<char *>(nullptr));
            std::fprintf(stderr, "exec(%s) failed: %s\n", CATALYST_EXECUTOR_PATH,
                         std::strerror(errno));
            ::_exit(127);
        }
    }

    ~ExecutorProcess() {
        ::kill(Pid, SIGTERM);
        int status = 0;
        ::waitpid(Pid, &status, 0);
    }

    ExecutorProcess(const ExecutorProcess &) = delete;
    ExecutorProcess &operator=(const ExecutorProcess &) = delete;

    const std::string &address() const { return Address; }

  private:
    std::string Address;
    pid_t Pid{-1};
};

// A private directory for one test's input files, removed on scope exit.
class ScratchDir {
  public:
    explicit ScratchDir(const std::string &Tag)
        : Path(fs::temp_directory_path() /
               ("catalyst-executor-test-" + Tag + "-" + std::to_string(::getpid()))) {
        std::error_code EC;
        fs::remove_all(Path, EC);
        fs::create_directories(Path, EC);
    }

    ~ScratchDir() {
        std::error_code EC;
        fs::remove_all(Path, EC);
    }

    ScratchDir(const ScratchDir &) = delete;
    ScratchDir &operator=(const ScratchDir &) = delete;

    fs::path write(const std::string &Name, const std::string &Contents) const {
        fs::path P = Path / Name;
        std::ofstream Out(P, std::ios::binary);
        Out.write(Contents.data(), static_cast<std::streamsize>(Contents.size()));
        Out.close();
        REQUIRE(fs::exists(P));
        return P;
    }

  private:
    fs::path Path;
};

// The listener needs a moment to bind, so retry until it answers.
catalyst::executor::ExecutorSession *openWithRetry(const std::string &Address) {
    auto Deadline = std::chrono::steady_clock::now() + Timeout;
    for (;;) {
        if (auto *S = catalyst::executor::open(Address.c_str())) {
            return S;
        }
        if (std::chrono::steady_clock::now() >= Deadline) {
            return nullptr;
        }
        std::this_thread::sleep_for(PollInterval);
    }
}

// Poll until `Pred` holds, or give up after the timeout.
template <typename PredT> bool waitFor(PredT Pred) {
    auto Deadline = std::chrono::steady_clock::now() + Timeout;
    while (std::chrono::steady_clock::now() < Deadline) {
        if (Pred()) {
            return true;
        }
        std::this_thread::sleep_for(PollInterval);
    }
    return Pred();
}

// Asset names carry this test process's PID, so a staged file can be traced
// back to the connection that sent it without depending on what else is
// running on the machine.
std::string assetName(const std::string &Tag) {
    return Tag + "-" + std::to_string(::getpid()) + ".asset";
}

// The per-connection staging directories currently holding `Name`.
std::set<fs::path> dirsHolding(const std::string &Name) {
    std::set<fs::path> Dirs;
    std::error_code EC;
    // A missing root just yields an end iterator, i.e. the empty set.
    for (const auto &Entry :
         fs::directory_iterator(fs::temp_directory_path() / "catalyst-assets", EC)) {
        if (fs::exists(Entry.path() / Name)) {
            Dirs.insert(Entry.path());
        }
    }
    return Dirs;
}

size_t countEntries(const fs::path &Dir) {
    return static_cast<size_t>(
        std::distance(fs::directory_iterator(Dir), fs::directory_iterator{}));
}

std::string readFile(const fs::path &P) {
    std::ifstream In(P, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(In), std::istreambuf_iterator<char>());
}

// All 256 byte values, so a truncated or text-mangled transfer cannot pass.
std::string binaryPayload(size_t Size) {
    std::string Payload;
    Payload.reserve(Size);
    for (size_t I = 0; I < Size; ++I) {
        Payload.push_back(static_cast<char>(I % 256));
    }
    return Payload;
}

} // namespace

TEST_CASE("the executor serves one connection after another", "[executor]") {
    // The listener forks per connection and keeps accepting, so a second
    // session must succeed once the first one is gone.
    ExecutorProcess Executor;

    auto *First = openWithRetry(Executor.address());
    REQUIRE(First != nullptr);
    catalyst::executor::close(First);

    auto *Second = openWithRetry(Executor.address());
    REQUIRE(Second != nullptr);
    catalyst::executor::close(Second);
}

TEST_CASE("open fails when nothing is listening", "[executor]") {
    // Hold the port bound for the whole test so nothing else can start
    // listening on it.
    int Port = 0;
    int Held = reservePort(Port);
    std::string Address = "127.0.0.1:" + std::to_string(Port);

    CHECK(catalyst::executor::open(Address.c_str()) == nullptr);
    CHECK(std::strlen(catalyst::executor::last_error()) > 0);

    ::close(Held);
}

TEST_CASE("a staged asset is stored byte for byte", "[executor]") {
    ExecutorProcess Executor;
    ScratchDir Scratch("stage");
    const std::string Name = assetName("stage");
    const std::string Payload = binaryPayload(4096);
    fs::path Src = Scratch.write(Name, Payload);

    auto *S = openWithRetry(Executor.address());
    REQUIRE(S != nullptr);
    REQUIRE(catalyst::executor::load_asset_path(S, Src.c_str()) == 0);

    std::set<fs::path> Dirs = dirsHolding(Name);
    REQUIRE(Dirs.size() == 1);
    const fs::path &Dir = *Dirs.begin();

    // The staging subdirectory is keyed on the serving process's PID.
    CHECK(Dir.filename().string().find_first_not_of("0123456789") == std::string::npos);
    CHECK(readFile(Dir / Name) == Payload);
    // Write-then-rename leaves no temporary alongside the asset.
    CHECK(countEntries(Dir) == 1);

    catalyst::executor::close(S);
}

TEST_CASE("each connection stages a same-named asset in its own directory", "[executor]") {
    ExecutorProcess Executor;
    // Same asset name sent over two open connections. The executor stages by
    // basename, so these would land on top of each other if connections
    // shared a staging directory.
    ScratchDir ScratchA("collide-a");
    ScratchDir ScratchB("collide-b");
    const std::string Name = assetName("collide");
    const std::string PayloadA(1024, 'A');
    const std::string PayloadB(2048, 'B');
    fs::path SrcA = ScratchA.write(Name, PayloadA);
    fs::path SrcB = ScratchB.write(Name, PayloadB);

    auto *A = openWithRetry(Executor.address());
    REQUIRE(A != nullptr);
    auto *B = openWithRetry(Executor.address());
    REQUIRE(B != nullptr);

    REQUIRE(catalyst::executor::load_asset_path(A, SrcA.c_str()) == 0);
    REQUIRE(catalyst::executor::load_asset_path(B, SrcB.c_str()) == 0);

    // Both payloads survive intact: neither connection clobbered the other.
    std::set<fs::path> Dirs = dirsHolding(Name);
    REQUIRE(Dirs.size() == 2);
    std::multiset<std::string> Staged;
    for (const auto &Dir : Dirs) {
        Staged.insert(readFile(Dir / Name));
    }
    CHECK(Staged.count(PayloadA) == 1);
    CHECK(Staged.count(PayloadB) == 1);

    catalyst::executor::close(A);
    catalyst::executor::close(B);
}

TEST_CASE("a staging directory is removed when its connection ends", "[executor]") {
    ExecutorProcess Executor;
    ScratchDir Scratch("cleanup");
    const std::string Name = assetName("cleanup");
    fs::path Src = Scratch.write(Name, "payload");

    auto *S = openWithRetry(Executor.address());
    REQUIRE(S != nullptr);
    REQUIRE(catalyst::executor::load_asset_path(S, Src.c_str()) == 0);

    std::set<fs::path> Dirs = dirsHolding(Name);
    REQUIRE(Dirs.size() == 1);
    fs::path Dir = *Dirs.begin();

    // The serving process clears its staging directory before exiting.
    catalyst::executor::close(S);
    CHECK(waitFor([&] { return !fs::exists(Dir); }));
}
