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

#include "RemoteSession.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <vector>

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

#include <netdb.h>
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

void initialize_targets()
{
    static const bool inited = []() {
        InitializeAllTargets();
        InitializeAllTargetMCs();
        InitializeAllAsmPrinters();
        return true;
    }();
    (void)inited;
}

// A SimpleRemoteEPC transport that wraps the stock FDSimpleRemoteEPCTransport and adds a
// shutdown() in disconnect().
//
// Teardown of a socket-backed session otherwise deadlocks: SimpleRemoteEPC::disconnect() (called
// from ExecutionSession::endSession()) does `T->disconnect()` then waits for the transport's
// listenLoop to observe end-of-stream and call handleDisconnect. FDSimpleRemoteEPCTransport's
// disconnect() only close()s the socket fd — but on Linux close() neither wakes a read() already
// in flight on the listener thread nor sends a FIN while that read holds a reference, so the
// listener never returns and the wait blocks forever (and the peer never sees EOF either).
//
// shutdown(SHUT_RDWR) forces the in-flight read() to return and sends FIN to the peer. Doing it
// here — inside disconnect(), which runs only after endSession() has finished its executor
// messaging (e.g. deallocating JIT'd memory) — avoids tearing the socket down prematurely.
class ShutdownFDTransport : public SimpleRemoteEPCTransport {
  public:
    static Expected<std::unique_ptr<ShutdownFDTransport>>
    Create(SimpleRemoteEPCTransportClient &C, int InFD, int OutFD)
    {
        auto Inner = FDSimpleRemoteEPCTransport::Create(C, InFD, OutFD);
        if (!Inner) {
            return Inner.takeError();
        }
        return std::make_unique<ShutdownFDTransport>(std::move(*Inner), InFD, OutFD);
    }

    ShutdownFDTransport(std::unique_ptr<FDSimpleRemoteEPCTransport> inner, int inFD, int outFD)
        : Inner(std::move(inner)), InFD(inFD), OutFD(outFD)
    {
    }

    Error start() override { return Inner->start(); }

    Error sendMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo, ExecutorAddr TagAddr,
                      ArrayRef<char> ArgBytes) override
    {
        return Inner->sendMessage(OpC, SeqNo, TagAddr, ArgBytes);
    }

    void disconnect() override
    {
        ::shutdown(InFD, SHUT_RDWR);
        if (OutFD != InFD) {
            ::shutdown(OutFD, SHUT_RDWR);
        }
        Inner->disconnect();
    }

  private:
    std::unique_ptr<FDSimpleRemoteEPCTransport> Inner;
    int InFD, OutFD;
};

// For avoiding the error message being overwritten by subsequent errors in async jobs.
// We use thread_local to store the error message.
thread_local std::string g_last_error;
void set_error(const std::string &msg) { g_last_error = msg; }
void clear_error() { g_last_error.clear(); }

void check(Error E, const Twine &what)
{
    if (E) {
        throw std::runtime_error((what + ": " + toString(std::move(E))).str());
    }
}

// unwrap LLVM Expected to C++ exception
template <typename T> T unwrap(Expected<T> v, const Twine &what)
{
    check(v.takeError(), what);
    return std::move(*v);
}

// TCP connect (mirrored from llvm-jitlink)
std::string OutOfProcessExecutorConnect;

Error createTCPSocketError(Twine Details)
{
    return make_error<StringError>("Failed to connect TCP socket '" +
                                       Twine(OutOfProcessExecutorConnect) + "': " + Details,
                                   inconvertibleErrorCode());
}

// Seconds to wait for the catalyst-executor ORC bootstrap handshake before giving up. Overridable
// via CATALYST_REMOTE_CONNECT_TIMEOUT; defaults to 10s. A value of 0 disables the timeout (the
// handshake then blocks indefinitely, matching the upstream llvm-jitlink behaviour).
unsigned remoteConnectTimeoutSeconds()
{
    if (const char *env = std::getenv("CATALYST_REMOTE_CONNECT_TIMEOUT")) {
        char *end = nullptr;
        unsigned long secs = std::strtoul(env, &end, 10);
        if (end != env) {
            return static_cast<unsigned>(secs);
        }
    }
    return 10;
}

Expected<int> connectTCPSocket(std::string Host, std::string PortStr)
{
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
        if ((SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol)) < 0)
            continue;

        // If connect returns null, we exit the loop with a working socket.
        if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0)
            break;

        close(SockFD);
    }
    freeaddrinfo(AI);

    // If we reached the end of the loop without connecting to a valid endpoint,
    // dump the last error that was logged in socket() or connect().
    if (Server == nullptr) {
        return createTCPSocketError(std::strerror(errno));
    }

    return SockFD;
}

// Slab-based JIT-link memory manager: reserve a 1 GB slab on the remote once
Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
createSimpleRemoteMemoryManager(SimpleRemoteEPC &SREPC)
{
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

Expected<std::unique_ptr<MemoryBuffer>> getFile(const Twine &filename)
{
    auto F = MemoryBuffer::getFile(filename);
    if (F) {
        return std::move(*F);
    }
    return createFileError(filename, F.getError());
}

} // namespace

namespace catalyst::remote {

struct RemoteSession {
    std::unique_ptr<ExecutionSession> ES;

    DataLayout DL;

    MangleAndInterner Mangle;
    ObjectLinkingLayer ObjectLayer;

    // Shared namespace: the process-symbol generator (QIR runtime, libc) plus library-call objects.
    JITDylib &MainJD;
    // One JITDylib per shipped kernel object, keyed by its object-file path. Each links against
    // MainJD for its external deps, but keeps its own entry symbol isolated — so two objects can
    // export the same `_catalyst_pyface_<entry>` without colliding.
    std::map<std::string, JITDylib *> KernelJDs;

    ExecutorAddr alloc_fn{0};
    ExecutorAddr free_fn{0};
    ExecutorAddr invoke_fn{0};
    ExecutorAddr store_asset_fn{0};

    RemoteSession(std::unique_ptr<ExecutionSession> es, DataLayout dl)
        : ES(std::move(es)), DL(std::move(dl)), Mangle(*this->ES, this->DL), ObjectLayer(*this->ES),
          MainJD(this->ES->createBareJITDylib("<main>"))
    {
        MainJD.addGenerator(
            cantFail(EPCDynamicLibrarySearchGenerator::GetForTargetProcess(*this->ES)));
    }

    ~RemoteSession()
    {
        if (auto Err = ES->endSession()) {
            ES->reportError(std::move(Err));
        }
    }

    static Expected<std::unique_ptr<RemoteSession>> Create(StringRef remote_addr)
    {
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
        // The ORC bootstrap handshake inside SimpleRemoteEPC::Create is an unbounded blocking read on
        // the socket. If the peer is not a live catalyst-executor (e.g. a stale port-forward or a
        // dead SSH tunnel accepted the connection), it would hang forever. A watchdog shuts the
        // socket down after the timeout, which forces the blocked read to fail and Create to return
        // an error instead of hanging.
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

        // ShutdownFDTransport instead of FDSimpleRemoteEPCTransport so teardown shutdown()s the
        // socket and doesn't deadlock — see the class comment above.
        auto EPC = SimpleRemoteEPC::Create<ShutdownFDTransport>(
            std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt), std::move(setup), sockFd,
            sockFd);

        if (watchdog.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                handshakeDone = true;
            }
            cv.notify_all();
            watchdog.join();
        }

        if (timedOut) {
            // Consume any error the forced socket shutdown produced before reporting the timeout.
            if (!EPC) {
                consumeError(EPC.takeError());
            }
            return createTCPSocketError("handshake with catalyst-executor timed out after " +
                                        Twine(timeoutSecs) +
                                        "s (is a catalyst-executor actually listening there?). "
                                        "Override with CATALYST_REMOTE_CONNECT_TIMEOUT");
        }
        if (!EPC) {
            return EPC.takeError();
        }

        // A remote session only needs the executor's symbol-mangling convention (its global symbol
        // prefix), which is fixed by the target's object format. Deriving it from the executor triple
        // avoids building a target machine, so the client does not need the executor's codegen
        // backend in its own LLVM — a client can drive an executor of a different architecture.
        const Triple &executorTriple = (*EPC)->getTargetTriple();
        StringRef manglingSpec = executorTriple.isOSBinFormatMachO()   ? "m:o"
                                 : executorTriple.isOSBinFormatCOFF()  ? "m:w"
                                                                       : "m:e";
        DataLayout DL(manglingSpec);

        auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));
        return std::make_unique<RemoteSession>(std::move(ES), std::move(DL));
    }

    // Load a kernel object into its own JITDylib (keyed by `path`), linked against MainJD for deps.
    Error addObjectFile(StringRef path, std::unique_ptr<MemoryBuffer> Buf)
    {
        JITDylib &jd = ES->createBareJITDylib(("kernel:" + path).str());
        jd.addToLinkOrder(MainJD);
        KernelJDs[path.str()] = &jd;
        return ObjectLayer.add(jd, std::move(Buf));
    }

    // Resolve `Name` in the JITDylib of the object identified by `path` (kernel entry points). Falls
    // back to MainJD if the object is unknown.
    ExecutorAddr lookupSym(StringRef path, StringRef Name)
    {
        auto it = KernelJDs.find(path.str());
        JITDylib *jd = (it != KernelJDs.end()) ? it->second : &MainJD;
        auto Sym = unwrap(ES->lookup({jd}, Mangle(Name.str())), "lookup(" + Name + ")");
        return Sym.getAddress();
    }

    // Resolve `Name` in the shared process namespace (library-call symbols).
    ExecutorAddr lookupSym(StringRef Name)
    {
        auto Sym = unwrap(ES->lookup({&MainJD}, Mangle(Name.str())), "lookup(" + Name + ")");
        return Sym.getAddress();
    }

    ExecutorProcessControl &getEPC() { return ES->getExecutorProcessControl(); }
};

// ---------------------------------------------------------------------------
// Memref Marshalling Helpers
// ---------------------------------------------------------------------------

namespace {

ExecutorAddr remote_alloc(RemoteSession *s, size_t size)
{
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

void remote_free(RemoteSession *s, ExecutorAddr addr)
{
    auto &epc = s->getEPC();
    if (auto err = epc.callSPSWrapper<void(shared::SPSExecutorAddr)>(s->free_fn, addr)) {
        consumeError(std::move(err));
    }
}

void remote_write(RemoteSession *s, ExecutorAddr addr, const void *data, size_t size)
{
    auto &epc = s->getEPC();
    tpctypes::BufferWrite w{addr, ArrayRef<char>(static_cast<const char *>(data), size)};
    check(epc.getMemoryAccess().writeBuffers({w}), "write");
}

void remote_read(RemoteSession *s, ExecutorAddr addr, void *data, size_t size)
{
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

void remote_invoke(RemoteSession *s, ExecutorAddr entry, ArrayRef<ExecutorAddr> arg_addrs)
{
    auto &epc = s->getEPC();
    check(epc.callSPSWrapper<void(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>)>(s->invoke_fn,
                                                                                 entry, arg_addrs),
          "remote_invoke");
}

// Memref descriptors are layout-equivalent to:
// { void *allocated; void *aligned; size_t offset; size_t sizes[rank]; size_t strides[rank]; }
size_t memref_desc_size(size_t rank)
{
    return sizeof(void *)          // void *allocated
           + sizeof(void *)        // void *aligned
           + sizeof(size_t)        // size_t offset
           + sizeof(size_t) * rank // size_t sizes[rank]
           + sizeof(size_t) * rank // size_t strides[rank]
        ;
}

size_t memref_data_size(const char *desc, size_t rank, size_t elem_size)
{
    if (rank == 0) {
        return elem_size;
    }
    const size_t *shape = reinterpret_cast<const size_t *>(desc + kShapeOff);
    return std::accumulate(shape, shape + rank, elem_size, std::multiplies<size_t>());
}

class RemoteAllocator {
  private:
    RemoteSession *sess;
    std::vector<ExecutorAddr> addrs;

  public:
    explicit RemoteAllocator(RemoteSession *s) : sess(s) {}
    ~RemoteAllocator()
    {
        for (ExecutorAddr a : addrs) {
            remote_free(sess, a);
        }
    }
    RemoteAllocator(const RemoteAllocator &) = delete;
    RemoteAllocator &operator=(const RemoteAllocator &) = delete;

    ExecutorAddr alloc(size_t size)
    {
        ExecutorAddr addr = remote_alloc(sess, size);
        addrs.push_back(addr);
        return addr;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// RemoteSession's Exported APIs
// ---------------------------------------------------------------------------

/**
 * @brief Open a session to the remote device.
 *
 * @param remote_addr the remote address
 * @return RemoteSession * the session object
 */
RemoteSession *open(const char *remote_addr)
{
    clear_error();
    try {
        auto s = unwrap(RemoteSession::Create(remote_addr), "open(" + Twine(remote_addr) + ")");
        check(s->getEPC().getBootstrapSymbols({{s->alloc_fn, "catalyst_remote_alloc"},
                                               {s->free_fn, "catalyst_remote_free"},
                                               {s->invoke_fn, "catalyst_remote_invoke"},
                                               {s->store_asset_fn, "catalyst_remote_store_asset"}}),
              "getBootstrapSymbols");
        return s.release();
    }
    catch (const std::exception &e) {
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
void close(RemoteSession *s) { delete s; }

/**
 * @brief Load an object file (cross-compiled for the remote arch) into the remote JIT.
 *
 * @param s the session object
 * @param path the path to the object file
 * @return int 0 on success, non-zero on error
 */
int load_object_path(RemoteSession *s, const char *path)
{
    clear_error();
    try {
        auto buf = unwrap(getFile(path), "getFile(" + Twine(path) + ")");
        check(s->addObjectFile(path, std::move(buf)), "addObjectFile");
        return 0;
    }
    catch (const std::exception &e) {
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
int load_asset_path(RemoteSession *s, const char *path)
{
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
    }
    catch (const std::exception &e) {
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
int call_wrapper_raw(RemoteSession *s, const char *sym, const char *args_buf, size_t args_size,
                     char **out_buf, size_t *out_size)
{
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
    }
    catch (const std::exception &e) {
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
uint64_t lookup(RemoteSession *s, const char *name, const char *object)
{
    clear_error();
    try {
        if (object && *object) {
            return s->lookupSym(object, name).getValue();
        }
        return s->lookupSym(name).getValue();
    }
    catch (const std::exception &e) {
        set_error(e.what());
        return 0;
    }
}

/**
 * @brief Run the kernel as a main function (take argv as arguments, argc is the length of argv).
 *
 * @param s the session object
 * @param entry the entry function address
 * @param argv the command line arguments
 * @return int32_t the exit code
 */
int32_t run_as_main(RemoteSession *s, uint64_t entry_addr, int argc, const char *const *argv)
{
    clear_error();
    try {
        std::vector<std::string> args;
        args.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
        return unwrap(s->getEPC().runAsMain(ExecutorAddr(entry_addr), args), "run_as_main");
    }
    catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

/**
 * @brief Push one host memref to the remote:
 *        1. allocates the data buffer on the remote
 *        2. allocates the descriptor on the remote (which has a pointer to the data buffer)
 *        3. returns the descriptor's remote addr.
 *        4. If `copy_data` is true, the data will be copied to the remote.
 *           It's used for input memrefs like arguments.
 *           In the case of output memrefs, the data will be copied back to the host,
 *           so we don't need to copy the data to the remote.
 *
 * @param s the session object
 * @param alloc the remote allocator
 * @param host_desc the host memref descriptor
 * @param rank the rank of the memref
 * @param elem_size the element size of the memref
 * @param copy_data whether to copy the data to the remote
 * @return ExecutorAddr the remote address of the memref descriptor
 */
ExecutorAddr push_memref(RemoteSession *s, RemoteAllocator &alloc, void *host_desc, size_t rank,
                         size_t elem_size, bool copy_data)
{
    char *desc_host = static_cast<char *>(host_desc);
    size_t desc_size = memref_desc_size(rank);
    size_t data_size = memref_data_size(desc_host, rank, elem_size);

    // memref descriptor layout:
    // ┌──────────────────────┐   ┌──────┐
    // │memref      .allocated┼┬─►│buffer│
    // │descriptor  .aligned ─┼┘  └──────┘
    // │            .offset   │
    // │            .shape    │
    // │            .strides  │
    // └──────────────────────┘

    std::vector<char> desc(desc_size);
    std::memcpy(desc.data(), desc_host, desc_size);

    ExecutorAddr data_remote = ExecutorAddr(0);
    if (data_size > 0) {
        data_remote = alloc.alloc(data_size);
        if (copy_data) {
            void *aligned_host = *reinterpret_cast<void **>(desc_host + kAlignedOff);
            if (aligned_host) {
                remote_write(s, data_remote, aligned_host, data_size);
            }
        }
    }
    std::memcpy(desc.data() + kAllocatedOff, &data_remote, sizeof(uintptr_t));
    std::memcpy(desc.data() + kAlignedOff, &data_remote, sizeof(uintptr_t));
    std::memset(desc.data() + kOffsetOff, 0, sizeof(int64_t));

    ExecutorAddr desc_remote = alloc.alloc(desc_size);
    remote_write(s, desc_remote, desc.data(), desc.size());
    return desc_remote;
}

/**
 * @brief Pull a remote memref descriptor + its data back into the host descriptor.
 *
 * @param s the session object
 * @param remote_desc the remote address of the memref descriptor
 * @param host_desc the host memref descriptor
 * @param rank the rank of the memref
 * @param elem_size the element size of the memref
 */
void pull_memref(RemoteSession *s, ExecutorAddr remote_desc, void *host_desc, size_t rank,
                 size_t elem_size)
{
    size_t desc_size = memref_desc_size(rank);
    std::vector<char> desc(desc_size);
    remote_read(s, remote_desc, desc.data(), desc.size());

    uintptr_t aligned_remote;
    std::memcpy(&aligned_remote, desc.data() + kAlignedOff, sizeof(uintptr_t));

    size_t data_size = memref_data_size(desc.data(), rank, elem_size);
    size_t alloc_size = std::max<size_t>(data_size, 1);
    void *aligned_host = __catalyst__rt__alloc_managed(alloc_size);
    if (data_size && aligned_remote) {
        remote_read(s, ExecutorAddr(aligned_remote), aligned_host, data_size);
    }
    uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(aligned_host);
    std::memcpy(desc.data() + kAllocatedOff, &aligned_addr, sizeof(uintptr_t));
    std::memcpy(desc.data() + kAlignedOff, &aligned_addr, sizeof(uintptr_t));
    std::memcpy(host_desc, desc.data(), desc_size);
}

/**
 * @brief Invoke a remote kernel.
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
int invoke_kernel(RemoteSession *s, uint64_t entry_addr, size_t num_inputs,
                  void *const *input_descs, const size_t *input_ranks,
                  const size_t *input_elem_sizes, size_t num_outputs, void *const *output_descs,
                  const size_t *output_ranks, const size_t *output_elem_sizes)
{
    clear_error();
    RemoteAllocator allocator(s);
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

        // Step 1. Push the input memref descriptors (with data) to the remote.
        std::vector<ExecutorAddr> input_remote_descs(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            input_remote_descs[i] = push_memref(s, allocator, input_descs[i], input_ranks[i],
                                                input_elem_sizes[i], /*copy_data=*/true);
        }

        // Step 2. Allocate a remote buffer holding the input memref descriptors' pointers.
        ExecutorAddr av_remote = ExecutorAddr(0);
        if (num_inputs > 0) {
            av_remote = allocator.alloc(sizeof(uintptr_t) * num_inputs);
            std::vector<uintptr_t> av_buf(num_inputs);
            for (size_t i = 0; i < num_inputs; ++i) {
                av_buf[i] = input_remote_descs[i].getValue();
            }
            remote_write(s, av_remote, av_buf.data(), sizeof(uintptr_t) * num_inputs);
        }

        // Step 3. Allocate a remote buffer for kernel to write the output memref descriptors.
        std::vector<size_t> output_offsets(num_outputs);
        size_t rv_total = 0;
        for (size_t i = 0; i < num_outputs; ++i) {
            output_offsets[i] = rv_total;
            rv_total += memref_desc_size(output_ranks[i]);
        }
        ExecutorAddr rv_remote = ExecutorAddr(0);
        if (rv_total > 0) {
            rv_remote = allocator.alloc(rv_total);
        }

        // Step 4. Invoke the kernel remotely.
        std::vector<ExecutorAddr> arg_addrs = {rv_remote, av_remote};
        remote_invoke(s, ExecutorAddr(entry_addr), arg_addrs);

        // Step 5. Pull each output descriptor back from rv buffer.
        if (rv_total > 0) {
            std::vector<char> rv_buf(rv_total);
            remote_read(s, rv_remote, rv_buf.data(), rv_total);
            for (size_t i = 0; i < num_outputs; ++i) {
                size_t desc_size = memref_desc_size(output_ranks[i]);
                size_t elem_size = output_elem_sizes[i];
                char *desc = rv_buf.data() + output_offsets[i];

                uintptr_t aligned_remote;
                std::memcpy(&aligned_remote, desc + kAlignedOff, sizeof(uintptr_t));

                size_t data_size = memref_data_size(desc, output_ranks[i], elem_size);
                size_t alloc_size = std::max<size_t>(data_size, 1);
                void *aligned_host = __catalyst__rt__alloc_managed(alloc_size);
                if (data_size && aligned_remote) {
                    remote_read(s, ExecutorAddr(aligned_remote), aligned_host, data_size);
                }
                uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(aligned_host);
                std::memcpy(desc + kAllocatedOff, &aligned_addr, sizeof(uintptr_t));
                std::memcpy(desc + kAlignedOff, &aligned_addr, sizeof(uintptr_t));
                std::memcpy(output_descs[i], desc, desc_size);
            }
        }
        return 0;
    }
    catch (const std::exception &e) {
        set_error(e.what());
        return -1;
    }
}

const char *last_error() { return g_last_error.c_str(); }

} // namespace catalyst::remote
