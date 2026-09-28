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
 * @brief Remote ORC v2 EPC executor for cross-arch JIT execution of
 * catalyst-compiled programs.
 *
 * Listens on one or more TCP addresses for connections from host
 * orchestrators and runs ORC's SimpleRemoteEPCServer over each. The host's
 * LLJIT JITLinks the kernel object into the address space and invokes the
 * catalyst entry function (`_catalyst_pyface_*`) on it.
 *
 * Each `--bind` spawns an independent listener process. The listener loops
 * on accept(), forking a grandchild per connection so that concurrent
 * clients to the same address are served independently (each grandchild
 * gets its own catalyst CTX and process memory). Listeners run until the
 * parent (and thus the whole group) is terminated by the user
 * (SIGINT/SIGTERM).
 *
 * Usage: catalyst-executor --bind <host:port> [--bind <host:port>...]
 *                          [--plugin <path>]...
 *
 *   --bind <h:p>     bind address (repeatable; one server process per address)
 *   --plugin <p>     extra .so to dlopen at startup (repeatable)
 *
 * Also setup the LD_LIBRARY_PATH to include the libraries that should be found
 * for this binary.
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/DefaultHostBootstrapValues.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorDylibManager.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleRemoteEPCServer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/UnwindInfoManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <unistd.h>

using namespace llvm;
using namespace llvm::orc;

namespace catalyst_services {
// TODO: provide a slab-based alloc.
// Returns a null ExecutorAddr on OOM (host checks `if (!ret)`).
ExecutorAddr _catalyst_remote_alloc(uint64_t size) {
    return ExecutorAddr::fromPtr(std::calloc(1, size));
}

void _catalyst_remote_free(ExecutorAddr addr) { std::free(addr.toPtr<void *>()); }

void _catalyst_remote_invoke(ExecutorAddr fn, std::vector<ExecutorAddr> args) {
    if (!fn) {
        return;
    }
    if (args.size() != 2) {
        llvm::report_fatal_error(
            llvm::formatv("kernel entry function must have 2 arguments, got {0}", args.size()));
    }
    using pyface_t = void (*)(void *, void *);
    fn.toPtr<pyface_t>()(args[0].toPtr<void *>(), args[1].toPtr<void *>());
}

std::filesystem::path assetDir() {
    return std::filesystem::temp_directory_path() / "catalyst-assets" / std::to_string(::getpid());
}

void clearAssets() {
    std::error_code ec;
    std::filesystem::remove_all(assetDir(), ec);
}

int32_t _catalyst_remote_store_asset(std::vector<char> bytes, std::string name) {
    namespace fs = std::filesystem;
    fs::path dir = assetDir();
    fs::path dst = dir / fs::path(name).filename();
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "catalyst-executor: mkdir %s failed: %s\n", dir.c_str(),
                     ec.message().c_str());
        return -1;
    }

    // Write to a unique temporary file and rename it into place, so a later
    // load can never observe a half-written asset.
    static std::atomic<uint64_t> tmpCounter{0};
    fs::path tmp = dst;
    tmp += ".tmp." + std::to_string(tmpCounter++);

    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "catalyst-executor: fopen %s failed: %s\n", tmp.c_str(),
                     std::strerror(errno));
        return -1;
    }
    size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
    // fclose flushes the stream, so a failure here can also mean lost bytes.
    int closeRc = std::fclose(f);

    // check if the write size is matched
    if (wrote != bytes.size()) {
        std::fprintf(stderr,
                     "catalyst-executor: Unmatched write size for %s (wrote=%zu, expected=%zu)\n",
                     tmp.c_str(), wrote, bytes.size());
        fs::remove(tmp, ec);
        return -1;
    }

    // check if the flush succeeded
    if (closeRc != 0) {
        std::fprintf(stderr, "catalyst-executor: fclose %s failed: %s\n", tmp.c_str(),
                     std::strerror(errno));
        fs::remove(tmp, ec);
        return -1;
    }

    fs::rename(tmp, dst, ec);
    if (ec) {
        std::fprintf(stderr, "catalyst-executor: rename %s -> %s failed: %s\n", tmp.c_str(),
                     dst.c_str(), ec.message().c_str());
        fs::remove(tmp, ec);
        return -1;
    }
    return 0;
}

} // namespace catalyst_services

extern "C" {
// This class will be renamed to `WrapperFunctionBuffer` in this PR in the
// future: https://github.com/llvm/llvm-project/pull/172633
llvm::orc::shared::CWrapperFunctionResult catalyst_remote_alloc(const char *ArgData,
                                                                size_t ArgSize) {
    auto result = shared::WrapperFunction<shared::SPSExecutorAddr(uint64_t)>::handle(
        ArgData, ArgSize, &catalyst_services::_catalyst_remote_alloc);
    return result.release();
}

llvm::orc::shared::CWrapperFunctionResult catalyst_remote_free(const char *ArgData,
                                                               size_t ArgSize) {
    return shared::WrapperFunction<void(shared::SPSExecutorAddr)>::handle(
               ArgData, ArgSize, &catalyst_services::_catalyst_remote_free)
        .release();
}

llvm::orc::shared::CWrapperFunctionResult catalyst_remote_invoke(const char *ArgData,
                                                                 size_t ArgSize) {
    return shared::WrapperFunction<void(shared::SPSExecutorAddr,
                                        shared::SPSSequence<shared::SPSExecutorAddr>)>::
        handle(ArgData, ArgSize, &catalyst_services::_catalyst_remote_invoke)
            .release();
}

llvm::orc::shared::CWrapperFunctionResult catalyst_remote_store_asset(const char *ArgData,
                                                                      size_t ArgSize) {
    return shared::WrapperFunction<int32_t(shared::SPSSequence<char>, shared::SPSString)>::handle(
               ArgData, ArgSize, &catalyst_services::_catalyst_remote_store_asset)
        .release();
}
} // extern "C"

namespace {

// dlopen a .so with RTLD_GLOBAL
bool load_lib_global(const char *path) {
    if (!path || !*path) {
        return false;
    }
    void *h = ::dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        std::fprintf(stderr, "catalyst-executor: dlopen(%s) failed: %s\n", path, ::dlerror());
        return false;
    }
    std::fprintf(stderr, "catalyst-executor: loaded %s\n", path);
    return true;
}

// Bind + listen on Host:PortStr. Returns the listening FD, or -1 on error.
// Originally adapted from
// `llvm/tools/llvm-jitlink/llvm-jitlink-executor/llvm-jitlink-executor.cpp`,
int openListening(const std::string &Host, const std::string &PortStr) {
    addrinfo Hints{};
    Hints.ai_family = AF_INET;
    Hints.ai_socktype = SOCK_STREAM;
    if (Host.empty()) {
        Hints.ai_flags = AI_PASSIVE;
    }
    const char *node = Host.empty() ? nullptr : Host.c_str();
    addrinfo *AI = nullptr;
    if (int EC = getaddrinfo(node, PortStr.c_str(), &Hints, &AI)) {
        errs() << "Error setting up bind address " << Host << ":" << PortStr << ": "
               << gai_strerror(EC) << "\n";
        return -1;
    }

    int SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol);
    if (SockFD < 0) {
        errs() << "Error creating socket: " << std::strerror(errno) << "\n";
        freeaddrinfo(AI);
        return -1;
    }

    const int Yes = 1;
    if (setsockopt(SockFD, SOL_SOCKET, SO_REUSEADDR, &Yes, sizeof(int)) == -1) {
        errs() << "Error calling setsockopt: " << std::strerror(errno) << "\n";
        ::close(SockFD);
        freeaddrinfo(AI);
        return -1;
    }

    if (bind(SockFD, AI->ai_addr, AI->ai_addrlen) < 0) {
        errs() << "Error on binding " << Host << ":" << PortStr << ": " << std::strerror(errno)
               << "\n";
        ::close(SockFD);
        freeaddrinfo(AI);
        return -1;
    }

    static constexpr int ConnectionQueueLen = 64;
    if (listen(SockFD, ConnectionQueueLen) < 0) {
        errs() << "Error on listen: " << std::strerror(errno) << "\n";
        ::close(SockFD);
        freeaddrinfo(AI);
        return -1;
    }
    freeaddrinfo(AI);
    return SockFD;
}

// Parse "host:port"
bool parseHostPort(const std::string &Spec, std::string &Host, std::string &Port) {
    auto Colon = Spec.rfind(':');
    if (Colon == std::string::npos) {
        return false;
    }
    Host = Spec.substr(0, Colon);
    Port = Spec.substr(Colon + 1);
    return !Host.empty() && !Port.empty();
}

// Bootstrap the SimpleRemoteEPCServer with the catalyst service symbols.
Error setupCatalystServer(SimpleRemoteEPCServer::Setup &S) {
    S.setDispatcher(std::make_unique<SimpleRemoteEPCServer::ThreadDispatcher>());

    S.bootstrapSymbols() = SimpleRemoteEPCServer::defaultBootstrapSymbols();
    addDefaultBootstrapValuesForHostProcess(S.bootstrapMap(), S.bootstrapSymbols());

    S.bootstrapSymbols()["catalyst_remote_alloc"] = ExecutorAddr::fromPtr(&catalyst_remote_alloc);
    S.bootstrapSymbols()["catalyst_remote_free"] = ExecutorAddr::fromPtr(&catalyst_remote_free);
    S.bootstrapSymbols()["catalyst_remote_invoke"] = ExecutorAddr::fromPtr(&catalyst_remote_invoke);
    S.bootstrapSymbols()["catalyst_remote_store_asset"] =
        ExecutorAddr::fromPtr(&catalyst_remote_store_asset);

    UnwindInfoManager::TryEnable();
    UnwindInfoManager::addBootstrapSymbols(S.bootstrapSymbols());

    S.services().push_back(std::make_unique<rt_bootstrap::SimpleExecutorMemoryManager>());
    S.services().push_back(std::make_unique<rt_bootstrap::SimpleExecutorDylibManager>());
    return Error::success();
}

// Render an accepted peer address as "host:port" for logging.
std::string formatPeer(const sockaddr_storage &Peer, socklen_t Len) {
    char Host[NI_MAXHOST];
    char Port[NI_MAXSERV];
    if (::getnameinfo(reinterpret_cast<const sockaddr *>(&Peer), Len, Host, sizeof(Host), Port,
                      sizeof(Port), NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
        return "<unknown>";
    }
    return std::string(Host) + ":" + Port;
}

template <typename FnT> FnT lookupRuntimeSymbol(const char *Name, const std::string &GLabel) {
    auto *Fn = reinterpret_cast<FnT>(::dlsym(RTLD_DEFAULT, Name));
    if (!Fn) {
        std::fprintf(stderr, "[%s] Warning: %s not found in any loaded plugin\n", GLabel.c_str(),
                     Name);
    }
    return Fn;
}

void initializeCatalystRuntime(const std::string &GLabel) {
    auto *initFn = lookupRuntimeSymbol<void (*)(uint32_t *)>("__catalyst__rt__initialize", GLabel);
    if (initFn) {
        initFn(nullptr);
    }
}

void finalizeCatalystRuntime(const std::string &GLabel) {
    auto *finalizeFn = lookupRuntimeSymbol<void (*)()>("__catalyst__rt__finalize", GLabel);
    if (finalizeFn) {
        finalizeFn();
    }
}

// Listener loop
[[noreturn]] void runServerLoop(int ListenFD, const std::string &Label) {
    {
        struct sigaction sa{};
        sa.sa_handler = SIG_IGN;
        sa.sa_flags = SA_NOCLDWAIT;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGCHLD, &sa, nullptr);
    }

    std::fprintf(stderr, "[%s] executor ready, waiting for connections\n", Label.c_str());

    for (;;) {
        sockaddr_storage Peer{};
        socklen_t Len = sizeof(Peer);
        int CSock = ::accept(ListenFD, reinterpret_cast<sockaddr *>(&Peer), &Len);
        if (CSock < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::fprintf(stderr, "[%s] accept failed: %s\n", Label.c_str(), std::strerror(errno));
            ::_exit(1);
        }

        pid_t pid = ::fork();
        if (pid < 0) {
            std::fprintf(stderr, "[%s] fork-per-connection failed: %s\n", Label.c_str(),
                         std::strerror(errno));
            ::close(CSock);
            continue;
        }
        if (pid == 0) {
            ::close(ListenFD);
            std::string GLabel = Label + "#" + std::to_string(::getpid());

            initializeCatalystRuntime(GLabel);

            ExitOnError ExitOnErr;
            ExitOnErr.setBanner("CatalystExecutor[" + GLabel + "]: ");
            {
                std::unique_ptr<SimpleRemoteEPCServer> Server =
                    ExitOnErr(SimpleRemoteEPCServer::Create<FDSimpleRemoteEPCTransport>(
                        setupCatalystServer, CSock, CSock));
                ExitOnErr(Server->waitForDisconnect());
            }
            finalizeCatalystRuntime(GLabel);
            catalyst_services::clearAssets();
            std::fprintf(stderr, "[%s] client disconnected, exiting\n", GLabel.c_str());
            ::_exit(0);
        }

        ::close(CSock);
        std::fprintf(stderr,
                     "[%s] accepted connection from %s on pid %d, waiting for next connection\n",
                     Label.c_str(), formatPeer(Peer, Len).c_str(), static_cast<int>(pid));
    }
}

volatile sig_atomic_t g_shutdown_requested = 0;
void parentSignalHandler(int) { g_shutdown_requested = 1; }

cl::OptionCategory ExecutorCat("catalyst-executor options");

cl::list<std::string> BindOpt("bind", cl::value_desc("host:port"),
                              cl::desc("Bind address (repeatable; one "
                                       "independent server process per entry)"),
                              cl::cat(ExecutorCat));

cl::list<std::string> PluginsOpt("plugin", cl::value_desc("path"),
                                 cl::desc("Extra .so to dlopen at startup "
                                          "(may be repeated)"),
                                 cl::cat(ExecutorCat));

} // namespace

int main(int argc, char **argv) {
    cl::HideUnrelatedOptions(ExecutorCat);
    cl::ParseCommandLineOptions(argc, argv,
                                "Remote ORC v2 EPC executor for catalyst-compiled programs\n");

    if (BindOpt.empty()) {
        errs() << "error: at least one --bind host:port is required\n";
        return 1;
    }

    for (const auto &p : PluginsOpt) {
        load_lib_global(p.c_str());
    }

    ::signal(SIGPIPE, SIG_IGN);

    struct Listener {
        std::string label;
        int listenFD;
    };
    std::vector<Listener> Listeners;
    Listeners.reserve(BindOpt.size());
    for (const auto &Spec : BindOpt) {
        std::string Host;
        std::string Port;
        if (!parseHostPort(Spec, Host, Port)) {
            errs() << "error: --bind " << Spec << " is not 'host:port'\n";
            for (auto &L : Listeners) {
                ::close(L.listenFD);
            }
            return 1;
        }
        int Fd = openListening(Host, Port);
        if (Fd < 0) {
            for (auto &L : Listeners) {
                ::close(L.listenFD);
            }
            return 1;
        }
        Listeners.push_back({Host + ":" + Port, Fd});
        std::fprintf(stderr, "Listening on %s:%s\n", Host.c_str(), Port.c_str());
    }

    std::vector<pid_t> Pids;
    Pids.reserve(Listeners.size());
    for (size_t i = 0; i < Listeners.size(); ++i) {
        pid_t pid = ::fork();
        if (pid < 0) {
            std::fprintf(stderr, "fork failed: %s\n", std::strerror(errno));
            for (pid_t p : Pids) {
                ::kill(p, SIGTERM);
            }
            return 1;
        }
        if (pid == 0) {
            for (size_t j = 0; j < Listeners.size(); ++j) {
                if (j != i) {
                    ::close(Listeners[j].listenFD);
                }
            }
            runServerLoop(Listeners[i].listenFD, Listeners[i].label);
        }
        Pids.push_back(pid);
    }

    for (auto &L : Listeners) {
        ::close(L.listenFD);
    }

    struct sigaction sa{};
    sa.sa_handler = parentSignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    while (!Pids.empty()) {
        int status = 0;
        pid_t r = ::waitpid(-1, &status, 0);
        if (r < 0) {
            if (errno == EINTR) {
                if (g_shutdown_requested) {
                    g_shutdown_requested = 0;
                    std::fprintf(stderr, "Received signal, terminating %zu child process(es)\n",
                                 Pids.size());
                    for (pid_t p : Pids) {
                        ::kill(p, SIGTERM);
                    }
                }
                continue;
            }
            std::fprintf(stderr, "waitpid failed: %s\n", std::strerror(errno));
            break;
        }
        Pids.erase(std::remove(Pids.begin(), Pids.end(), r), Pids.end());
        std::fprintf(stderr, "Child %d exited (status=0x%x)\n", (int)r, status);
    }

    return 0;
}
