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

#include "ExecutorSession.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/MemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteMemoryMapper.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include "WholeMessageFDTransport.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>

using namespace llvm;
using namespace llvm::orc;

// Public CAPI from librt_capi (declared in runtime/include/RuntimeCAPI.h).
// Allocates a host buffer and registers it with CTX->getMemoryManager().
extern "C" void *__catalyst__rt__alloc_managed(size_t size);

namespace {

// The connection behaviours below are mirrored from `llvm/tools/llvm-jitlink/llvm-jitlink.cpp`,
// and the Session class is mirrored from the official LLVM JIT tutorial:
// `https://llvm.org/docs/tutorial/BuildingAJIT1.html`.

// Memref descriptor layout: allocated*, aligned*, offset, sizes[], strides[].
// Offsets of the fixed-size prefix are independent of rank.
// The information can be obtained from `mlir/ExecutionEngine/CRunnerUtils.h`.
constexpr size_t kAllocatedOff = 0;
constexpr size_t kAlignedOff = sizeof(void *);
constexpr size_t kOffsetOff = sizeof(void *) * 2;
constexpr size_t kShapeOff = sizeof(void *) * 2 + sizeof(size_t);

void initialize_targets() {
    static const bool inited = []() {
        InitializeAllTargets();
        InitializeAllTargetMCs();
        InitializeAllAsmPrinters();
        return true;
    }();
    (void)inited;
}

// For avoiding the error message being overwritten by subsequent errors in async jobs.
// We use thread_local to store the error message.
thread_local std::string g_last_error;
void set_error(const std::string &msg) { g_last_error = msg; }
void clear_error() { g_last_error.clear(); }

void check(Error E, const Twine &what) {
    if (E) {
        throw std::runtime_error((what + ": " + toString(std::move(E))).str());
    }
}

// unwrap LLVM Expected to C++ exception
template <typename T> T unwrap(Expected<T> v, const Twine &what) {
    check(v.takeError(), what);
    return std::move(*v);
}

// TCP connect (mirrored from llvm-jitlink)
std::string OutOfProcessExecutorConnect;

Error createTCPSocketError(Twine Details) {
    return make_error<StringError>("Failed to connect TCP socket '" +
                                       Twine(OutOfProcessExecutorConnect) + "': " + Details,
                                   inconvertibleErrorCode());
}

// Seconds to wait for the catalyst-executor ORC bootstrap handshake before giving up. Overridable
// via CATALYST_REMOTE_CONNECT_TIMEOUT; defaults to 10s. A value of 0 disables the timeout.
unsigned remoteConnectTimeoutSeconds() {
    if (const char *env = std::getenv("CATALYST_REMOTE_CONNECT_TIMEOUT")) {
        char *end = nullptr;
        unsigned long secs = std::strtoul(env, &end, 10);
        if (end != env) {
            return static_cast<unsigned>(secs);
        }
    }
    return 10;
}

Expected<int> connectTCPSocket(std::string Host, std::string PortStr) {
    addrinfo *AI;
    addrinfo Hints{};
    Hints.ai_family = AF_INET;
    Hints.ai_socktype = SOCK_STREAM;
    Hints.ai_flags = AI_NUMERICSERV;

    if (int EC = getaddrinfo(Host.c_str(), PortStr.c_str(), &Hints, &AI)) {
        return createTCPSocketError("Address resolution failed (" + StringRef(gai_strerror(EC)) +
                                    ")");
    }

    // Cycle through the returned addrinfo structures and connect to the first
    // reachable endpoint.
    int SockFD;
    addrinfo *Server;
    for (Server = AI; Server != nullptr; Server = Server->ai_next) {
        // socket might fail, e.g. if the address family is not supported. Skip to
        // the next addrinfo structure in such a case.
        if ((SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol)) < 0) {
            continue;
        }

        // If connect returns null, we exit the loop with a working socket.
        if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0) {
            break;
        }

        close(SockFD);
    }
    freeaddrinfo(AI);

    // If we reached the end of the loop without connecting to a valid endpoint,
    // dump the last error that was logged in socket() or connect().
    if (Server == nullptr) {
        return createTCPSocketError(std::strerror(errno));
    }

    // Each EPC message is a small header write, then its payload. Without TCP_NODELAY, Nagle's
    // algorithm holds the payload until the header is acknowledged, which a delayed ACK stalls for
    // up to 40 ms, several times per remote call.
    int NoDelay = 1;
    setsockopt(SockFD, IPPROTO_TCP, TCP_NODELAY, &NoDelay, sizeof(NoDelay));

    return SockFD;
}

// Slab-based JIT-link memory manager: reserve a 1 GB slab on the remote once
Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
createSimpleRemoteMemoryManager(SimpleRemoteEPC &SREPC) {
    SimpleRemoteMemoryMapper::SymbolAddrs SAs;
    if (auto Err = SREPC.getBootstrapSymbols(
            {{SAs.Instance, rt::SimpleExecutorMemoryManagerInstanceName},
             {SAs.Reserve, rt::SimpleExecutorMemoryManagerReserveWrapperName},
             {SAs.Initialize, rt::SimpleExecutorMemoryManagerInitializeWrapperName},
             {SAs.Deinitialize, rt::SimpleExecutorMemoryManagerDeinitializeWrapperName},
             {SAs.Release, rt::SimpleExecutorMemoryManagerReleaseWrapperName}})) {
        return std::move(Err);
    }
    // 1 GB for object's sections (e.g .text, .rodata, ...)
    // It will be released once the Session is destroyed.
    // TODO: we might want to make this configurable.
    size_t SlabSize = 1024 * 1024 * 1024;
    return MapperJITLinkMemoryManager::CreateWithMapper<SimpleRemoteMemoryMapper>(SlabSize, SREPC,
                                                                                  SAs);
}

Expected<std::unique_ptr<MemoryBuffer>> getFile(const Twine &filename) {
    auto F = MemoryBuffer::getFile(filename);
    if (F) {
        return std::move(*F);
    }
    return createFileError(filename, F.getError());
}

void discardEPC(Expected<std::unique_ptr<SimpleRemoteEPC>> &EPC) {
    consumeError(EPC ? (*EPC)->disconnect() : EPC.takeError());
}

} // namespace

namespace catalyst::executor {

struct ExecutorSession {
    std::unique_ptr<ExecutionSession> ES;

    DataLayout DL;

    MangleAndInterner Mangle;
    ObjectLinkingLayer ObjectLayer;

    // Shared namespace: the process-symbol generator (QIR runtime, libc) plus library-call objects.
    JITDylib &MainJD;
    // One JITDylib per shipped kernel object, keyed by its object-file path. Each links against
    // MainJD for its external deps, but keeps its own entry symbol isolated — so two objects can
    // export the same `_catalyst_pyface_<entry>` without colliding.
    StringMap<JITDylib *> KernelJDs;

    ExecutorAddr alloc_fn{0};
    ExecutorAddr free_fn{0};
    ExecutorAddr invoke_fn{0};
    ExecutorAddr store_asset_fn{0};

    ExecutorSession(std::unique_ptr<ExecutionSession> es, DataLayout dl)
        : ES(std::move(es)), DL(std::move(dl)), Mangle(*this->ES, this->DL), ObjectLayer(*this->ES),
          MainJD(this->ES->createBareJITDylib("<main>")) {
        MainJD.addGenerator(
            cantFail(EPCDynamicLibrarySearchGenerator::GetForTargetProcess(*this->ES)));
    }

    ~ExecutorSession() {
        if (auto Err = ES->endSession()) {
            ES->reportError(std::move(Err));
        }
    }

    static Expected<std::unique_ptr<ExecutorSession>> Create(StringRef remote_addr) {
        initialize_targets();

        OutOfProcessExecutorConnect = remote_addr.str();
        auto [Host, PortStr] = remote_addr.split(':');
        if (Host.empty()) {
            return createTCPSocketError("Host name for -" + OutOfProcessExecutorConnect +
                                        " can not be empty");
        }
        if (PortStr.empty()) {
            return createTCPSocketError("Port number in -" + OutOfProcessExecutorConnect +
                                        " can not be empty");
        }

        auto SockFD = connectTCPSocket(Host.str(), PortStr.str());
        if (!SockFD) {
            return SockFD.takeError();
        }

        auto setup = SimpleRemoteEPC::Setup();
        setup.CreateMemoryManager = createSimpleRemoteMemoryManager;
        // The ORC bootstrap handshake inside SimpleRemoteEPC::Create is an unbounded blocking read
        // on the socket. If the peer is not a live catalyst-executor, it would hang forever.
        // A watchdog shuts the socket down after the timeout, which forces the blocked read to
        // fail and Create to return an error instead of hanging.
        const int sockFd = *SockFD;
        const unsigned timeoutSecs = remoteConnectTimeoutSeconds();
        std::mutex mtx;
        std::condition_variable cv;
        bool handshakeDone = false;
        bool timedOut = false;
        std::thread watchdog;
        if (timeoutSecs > 0) {
            watchdog = std::thread([&] {
                std::unique_lock<std::mutex> lock(mtx);
                if (!cv.wait_for(lock, std::chrono::seconds(timeoutSecs),
                                 [&] { return handshakeDone; })) {
                    timedOut = true;
                    ::shutdown(sockFd, SHUT_RDWR);
                }
            });
        }

        // SimpleRemoteEPC::disconnect() (via ExecutionSession::endSession()) closes the transport,
        // then waits for the listener thread to see EOF, so the transport shutdown()s the socket on
        // disconnect: close() alone would leave the listener's read() blocked and teardown hung.
        auto EPC = SimpleRemoteEPC::Create<catalyst::executor::WholeMessageFDTransport>(
            std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt), std::move(setup),
            sockFd, sockFd, /*shutdownOnDisconnect=*/true);

        if (watchdog.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                handshakeDone = true;
            }
            cv.notify_all();
            watchdog.join();
        }

        if (timedOut) {
            discardEPC(EPC);
            return createTCPSocketError("handshake with catalyst-executor timed out after " +
                                        Twine(timeoutSecs) +
                                        "s (is a catalyst-executor actually listening there?). "
                                        "Override with CATALYST_REMOTE_CONNECT_TIMEOUT");
        }
        if (!EPC) {
            return EPC.takeError();
        }

        JITTargetMachineBuilder JTMB((*EPC)->getTargetTriple());
        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL) {
            discardEPC(EPC);
            return joinErrors(
                createStringError(llvm::inconvertibleErrorCode(),
                                  "no data layout for the catalyst-executor's target triple '" +
                                      (*EPC)->getTargetTriple().str() +
                                      "'; this runtime may lack the LLVM backend for it"),
                DL.takeError());
        }

        auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));
        return std::make_unique<ExecutorSession>(std::move(ES), std::move(*DL));
    }

    Error addObjectFile(StringRef path, std::unique_ptr<MemoryBuffer> Buf) {
        if (KernelJDs.count(path)) {
            return make_error<StringError>("object already loaded: " + path,
                                           inconvertibleErrorCode());
        }
        JITDylib &jd = ES->createBareJITDylib(("kernel:" + path).str());
        jd.addToLinkOrder(MainJD);
        KernelJDs[path] = &jd;
        return ObjectLayer.add(jd, std::move(Buf));
    }

    ExecutorAddr lookupSym(StringRef path, StringRef Name) {
        auto it = KernelJDs.find(path);
        JITDylib *jd = (it != KernelJDs.end()) ? it->second : &MainJD;
        auto Sym = unwrap(ES->lookup({jd}, Mangle(Name.str())), "lookup(" + Name + ")");
        return Sym.getAddress();
    }

    // Resolve `Name` in the shared process namespace (library-call symbols).
    ExecutorAddr lookupSym(StringRef Name) {
        auto Sym = unwrap(ES->lookup({&MainJD}, Mangle(Name.str())), "lookup(" + Name + ")");
        return Sym.getAddress();
    }

    ExecutorProcessControl &getEPC() { return ES->getExecutorProcessControl(); }
};

// ---------------------------------------------------------------------------
// Memref Marshalling Helpers
// ---------------------------------------------------------------------------

namespace {

ExecutorAddr remote_alloc(ExecutorSession *s, size_t size) {
    ExecutorAddr ret;
    auto &epc = s->getEPC();
    std::string error_prefix = "alloc(" + std::to_string(size) + ")";
    check(epc.callSPSWrapper<shared::SPSExecutorAddr(uint64_t)>(s->alloc_fn, ret,
                                                                static_cast<uint64_t>(size)),
          error_prefix);
    if (!ret) {
        throw std::runtime_error(error_prefix + " out of memory");
    }
    return ret;
}

void remote_free(ExecutorSession *s, ExecutorAddr addr) {
    auto &epc = s->getEPC();
    if (auto err = epc.callSPSWrapper<void(shared::SPSExecutorAddr)>(s->free_fn, addr)) {
        consumeError(std::move(err));
    }
}

void remote_write(ExecutorSession *s, ArrayRef<tpctypes::BufferWrite> writes) {
    check(s->getEPC().getMemoryAccess().writeBuffers(writes), "write");
}

void remote_read(ExecutorSession *s, ExecutorAddr addr, void *data, size_t size) {
    ExecutorAddrRange r(addr, addr + size);
    auto out = unwrap(s->getEPC().getMemoryAccess().readBuffers({r}), "read");
    if (out.empty()) {
        throw std::runtime_error("read: empty read");
    }
    if (out[0].size() != size) {
        throw std::runtime_error("read: size mismatch");
    }
    std::memcpy(data, out[0].data(), size);
}

void remote_invoke(ExecutorSession *s, ExecutorAddr entry, ArrayRef<ExecutorAddr> arg_addrs) {
    auto &epc = s->getEPC();
    check(epc.callSPSWrapper<void(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>)>(s->invoke_fn,
                                                                                 entry, arg_addrs),
          "remote_invoke");
}

// Memref descriptors are layout-equivalent to:
// { void *allocated; void *aligned; size_t offset; size_t sizes[rank]; size_t strides[rank]; }
size_t memref_desc_size(size_t rank) {
    return sizeof(void *)          // void *allocated
           + sizeof(void *)        // void *aligned
           + sizeof(size_t)        // size_t offset
           + sizeof(size_t) * rank // size_t sizes[rank]
           + sizeof(size_t) * rank // size_t strides[rank]
        ;
}

size_t memref_data_size(const char *desc, size_t rank, size_t elem_size) {
    if (rank == 0) {
        return elem_size;
    }
    const size_t *shape = reinterpret_cast<const size_t *>(desc + kShapeOff);
    return std::accumulate(shape, shape + rank, elem_size, std::multiplies<size_t>());
}

// One remote allocation, freed when the block goes out of scope.
class RemoteBlock {
  private:
    ExecutorSession *sess;
    ExecutorAddr addr;

  public:
    RemoteBlock(ExecutorSession *s, size_t size) : sess(s), addr(remote_alloc(s, size)) {}
    ~RemoteBlock() { remote_free(sess, addr); }
    RemoteBlock(const RemoteBlock &) = delete;
    RemoteBlock &operator=(const RemoteBlock &) = delete;

    ExecutorAddr at(size_t offset) const { return addr + offset; }
};

// `size` rounded up to the alignment of every buffer in a launch's remote block.
size_t align_up(size_t size) {
    constexpr size_t kAlign = 64;
    return (size + kAlign - 1) / kAlign * kAlign;
}

} // namespace

// ---------------------------------------------------------------------------
// ExecutorSession's Exported APIs
// ---------------------------------------------------------------------------

/**
 * @brief Open a session to the remote device.
 *
 * @param remote_addr the remote address
 * @return ExecutorSession * the session object
 */
ExecutorSession *open(const char *remote_addr) {
    clear_error();
    try {
        auto s = unwrap(ExecutorSession::Create(remote_addr), "open(" + Twine(remote_addr) + ")");
        check(s->getEPC().getBootstrapSymbols({{s->alloc_fn, "catalyst_remote_alloc"},
                                               {s->free_fn, "catalyst_remote_free"},
                                               {s->invoke_fn, "catalyst_remote_invoke"},
                                               {s->store_asset_fn, "catalyst_remote_store_asset"}}),
              "getBootstrapSymbols");
        return s.release();
    } catch (const std::exception &e) {
        set_error(e.what());
        return nullptr;
    }
}

/**
 * @brief Close the session to the remote device.
 *        The remote session's destructor will handle the cleanup of the session.
 *
 * @param s the session object
 */
void close(ExecutorSession *s) { delete s; }

/**
 * @brief Load an object file (cross-compiled for the remote arch) into the remote JIT.
 *
 * @param s the session object
 * @param path the path to the object file
 * @return int 0 on success, non-zero on error
 */
int load_object_path(ExecutorSession *s, const char *path) {
    clear_error();
    try {
        auto buf = unwrap(getFile(path), "getFile(" + Twine(path) + ")");
        check(s->addObjectFile(path, std::move(buf)), "addObjectFile");
        return 0;
    } catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

/**
 * @brief Load an asset file into the remote JIT.
 *
 * @param s the session object
 * @param path the path to the asset file
 * @return int 0 on success, -1 on error
 */
int load_asset_path(ExecutorSession *s, const char *path) {
    clear_error();
    try {
        auto buf = unwrap(getFile(path), "getFile(" + Twine(path) + ")");
        ArrayRef<char> bytes(buf->getBufferStart(), buf->getBufferSize());

        int32_t rc = 0;
        check(s->getEPC().callSPSWrapper<int32_t(shared::SPSSequence<char>, shared::SPSString)>(
                  s->store_asset_fn, rc, bytes, std::string(path)),
              "store_asset");
        if (rc != 0) {
            throw std::runtime_error("Got non-zero status from store_asset(" + std::string(path) +
                                     "): " + std::to_string(rc));
        }
        return 0;
    } catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

/**
 * @brief Generic raw ORC wrapper-function call. Looks `sym` up on the executor, invokes its wrapper
 * with `(args_buf, args_size)`, and copies the resulting byte buffer into a `out_buf` with the size
 * of `out_size`. The caller is responsible for `free()` the result buffer.
 *
 * @param s The session object.
 * @param sym The symbol of the function to call.
 * @param args_buf The buffer of the arguments.
 * @param args_size The size of the arguments.
 * @param out_buf The buffer of the result.
 * @param out_size The size of the result.
 * @return int 0 on success, -1 on error.
 */
int call_wrapper_raw(ExecutorSession *s, const char *sym, const char *args_buf, size_t args_size,
                     char **out_buf, size_t *out_size) {
    clear_error();
    *out_buf = nullptr;
    *out_size = 0;
    try {
        ExecutorAddr fn = s->lookupSym(sym);
        if (!fn) {
            throw std::runtime_error(std::string("symbol not found: ") + sym);
        }
        auto result = s->getEPC().callWrapper(fn, ArrayRef<char>(args_buf, args_size));
        size_t n = result.size();
        char *buf = nullptr;
        if (n > 0) {
            buf = static_cast<char *>(std::malloc(n));
            if (!buf) {
                throw std::runtime_error("malloc failed for wrapper result");
            }
            std::memcpy(buf, result.data(), n);
        }
        *out_buf = buf;
        *out_size = n;
        return 0;
    } catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

/**
 * @brief Lookup the address of a symbol in the remote device.
 *
 * @param s the session object
 * @param name the name of the symbol
 * @return uint64_t the address of the symbol
 */
uint64_t lookup(ExecutorSession *s, const char *name, const char *object) {
    clear_error();
    try {
        if (object && *object) {
            return s->lookupSym(object, name).getValue();
        }
        return s->lookupSym(name).getValue();
    } catch (const std::exception &e) {
        set_error(e.what());
        return 0;
    }
}

/**
 * @brief Invoke a remote kernel.
 *
 * A launch costs a fixed number of round trips to the remote, whatever the number of memrefs: one
 * allocation holding every input descriptor and buffer, the argument array and the result
 * descriptors, one write of all the inputs, the call, one read of the result descriptors, one read
 * of all the output buffers, and one free.
 *
 * @param s the session object
 * @param entry_addr the address of the kernel entry function
 * @param num_inputs the number of input memrefs
 * @param input_descs the input memref descriptors
 * @param input_ranks the ranks of the input memrefs
 * @param input_elem_sizes the element sizes of the input memrefs
 * @param num_outputs the number of output memrefs
 * @param output_descs the output memref descriptors
 * @param output_ranks the ranks of the output memrefs
 * @param output_elem_sizes the element sizes of the output memrefs
 * @return int 0 on success, non-zero on error
 */
int invoke_kernel(ExecutorSession *s, uint64_t entry_addr, size_t num_inputs,
                  void *const *input_descs, const size_t *input_ranks,
                  const size_t *input_elem_sizes, size_t num_outputs, void *const *output_descs,
                  const size_t *output_ranks, const size_t *output_elem_sizes) {
    clear_error();
    try {
        // The remote executor's catalyst_remote_invoke calls the entry as Catalyst's pyface ABI:
        // `void(rv*, av*)`.

        // Layout (av):
        // av (argument) is a struct whose Nth field is a pointer to the Nth input memref
        // descriptor. So av = [N x uintptr_t] (array of remote descriptor addresses).
        //
        // av ──►┌───────────┐     ┌──────────────────────┐   ┌──────┐
        //       │slot0 (ptr)┼────►│memref      .allocated┼┬─►│buffer│
        //       │slot1      │     │descriptor  .aligned ─┼┘  └──────┘
        //       │slot2      │     │            .offset   │
        //       │  ...      │     │            .shape    │
        //       │           │     │            .strides  │
        //       └───────────┘     └──────────────────────┘

        // Layout (rv):
        // rv is a struct whose Nth field is the Nth output memref descriptor (not a pointer)
        //
        // rv ──►┌─────┬─────┬─────┬─────┬─────────┐
        //       │desc0│desc1│desc2│desc3│  ...    │
        //       └─────┴─────┴─────┴─────┴─────────┘
        // Each slot maps to a output memref descriptor

        // The remote block, as offsets: each input's data then its descriptor, then av, then rv.
        struct Input {
            size_t data_off, data_size, desc_off;
        };
        std::vector<Input> inputs(num_inputs);
        size_t total = 0;
        for (size_t i = 0; i < num_inputs; ++i) {
            const char *desc = static_cast<const char *>(input_descs[i]);
            inputs[i].data_size = memref_data_size(desc, input_ranks[i], input_elem_sizes[i]);
            inputs[i].data_off = total;
            total += align_up(inputs[i].data_size);
            inputs[i].desc_off = total;
            total += align_up(memref_desc_size(input_ranks[i]));
        }
        const size_t av_off = total;
        total += align_up(sizeof(uintptr_t) * num_inputs);
        std::vector<size_t> output_offsets(num_outputs);
        size_t rv_total = 0;
        for (size_t i = 0; i < num_outputs; ++i) {
            output_offsets[i] = rv_total;
            rv_total += memref_desc_size(output_ranks[i]);
        }
        const size_t rv_off = total;
        total += align_up(rv_total);
        RemoteBlock block(s, std::max<size_t>(total, 1));

        // Every input's data and descriptor, with the descriptor pointing at the remote data, and
        // av pointing at the descriptors, in one write.
        std::vector<std::vector<char>> descs(num_inputs);
        std::vector<uintptr_t> av(num_inputs);
        std::vector<tpctypes::BufferWrite> writes;
        for (size_t i = 0; i < num_inputs; ++i) {
            const char *desc_host = static_cast<const char *>(input_descs[i]);
            const ExecutorAddr data_remote =
                inputs[i].data_size > 0 ? block.at(inputs[i].data_off) : ExecutorAddr(0);
            void *aligned_host = *reinterpret_cast<void *const *>(desc_host + kAlignedOff);
            if (inputs[i].data_size > 0 && aligned_host) {
                int64_t host_offset = 0;
                std::memcpy(&host_offset, desc_host + kOffsetOff, sizeof(int64_t));
                const char *src = static_cast<const char *>(aligned_host) +
                                  host_offset * static_cast<int64_t>(input_elem_sizes[i]);
                writes.push_back({data_remote, ArrayRef<char>(src, inputs[i].data_size)});
            }
            descs[i].assign(desc_host, desc_host + memref_desc_size(input_ranks[i]));
            const uintptr_t data_value = data_remote.getValue();
            std::memcpy(descs[i].data() + kAllocatedOff, &data_value, sizeof(uintptr_t));
            std::memcpy(descs[i].data() + kAlignedOff, &data_value, sizeof(uintptr_t));
            std::memset(descs[i].data() + kOffsetOff, 0, sizeof(int64_t));
            writes.push_back({block.at(inputs[i].desc_off), ArrayRef<char>(descs[i])});
            av[i] = block.at(inputs[i].desc_off).getValue();
        }
        const ExecutorAddr av_remote = num_inputs > 0 ? block.at(av_off) : ExecutorAddr(0);
        if (num_inputs > 0) {
            writes.push_back({av_remote, ArrayRef<char>(reinterpret_cast<const char *>(av.data()),
                                                        sizeof(uintptr_t) * num_inputs)});
        }
        if (!writes.empty()) {
            remote_write(s, writes);
        }

        // Invoke the kernel remotely.
        const ExecutorAddr rv_remote = rv_total > 0 ? block.at(rv_off) : ExecutorAddr(0);
        std::vector<ExecutorAddr> arg_addrs = {rv_remote, av_remote};
        remote_invoke(s, ExecutorAddr(entry_addr), arg_addrs);

        // The output descriptors, then every output's data in one read.
        if (rv_total > 0) {
            std::vector<char> rv_buf(rv_total);
            remote_read(s, rv_remote, rv_buf.data(), rv_total);
            std::vector<ExecutorAddrRange> ranges;
            std::vector<void *> destinations;
            for (size_t i = 0; i < num_outputs; ++i) {
                char *desc = rv_buf.data() + output_offsets[i];
                uintptr_t aligned_remote;
                std::memcpy(&aligned_remote, desc + kAlignedOff, sizeof(uintptr_t));
                const size_t data_size =
                    memref_data_size(desc, output_ranks[i], output_elem_sizes[i]);
                void *aligned_host = __catalyst__rt__alloc_managed(std::max<size_t>(data_size, 1));
                if (data_size && aligned_remote) {
                    ranges.emplace_back(ExecutorAddr(aligned_remote),
                                        ExecutorAddr(aligned_remote) + data_size);
                    destinations.push_back(aligned_host);
                }
                uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(aligned_host);
                std::memcpy(desc + kAllocatedOff, &aligned_addr, sizeof(uintptr_t));
                std::memcpy(desc + kAlignedOff, &aligned_addr, sizeof(uintptr_t));
                std::memcpy(output_descs[i], desc, memref_desc_size(output_ranks[i]));
            }
            if (!ranges.empty()) {
                auto data = unwrap(s->getEPC().getMemoryAccess().readBuffers(ranges), "read");
                if (data.size() != ranges.size()) {
                    throw std::runtime_error("read: size mismatch");
                }
                for (size_t i = 0; i < ranges.size(); ++i) {
                    if (data[i].size() != ranges[i].size()) {
                        throw std::runtime_error("read: size mismatch");
                    }
                    std::memcpy(destinations[i], data[i].data(), data[i].size());
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

const char *last_error() { return g_last_error.c_str(); }

} // namespace catalyst::executor
