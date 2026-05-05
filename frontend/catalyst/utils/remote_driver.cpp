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
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/socket.h>

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
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include <netdb.h>
#include <unistd.h>

namespace nb = nanobind;
using namespace llvm;
using namespace llvm::orc;

// The connection behaviours below are mirrored from `llvm/tools/llvm-jitlink/llvm-jitlink.cpp`,
// and the Session class is mirrored from the official LLVM JIT tutorial:
// `https://llvm.org/docs/tutorial/BuildingAJIT1.html`.
namespace {
// Any N >= 1 yields identical offsets for the four prefix fields below.
using DescLayout = StridedMemRefType<char, 1>;
constexpr size_t kAllocatedOff = offsetof(DescLayout, basePtr);
constexpr size_t kAlignedOff = offsetof(DescLayout, data);
constexpr size_t kOffsetOff = offsetof(DescLayout, offset);
constexpr size_t kShapeOff = offsetof(DescLayout, sizes);

void initialize_targets()
{
    static const bool inited = []() {
#ifdef CATALYST_HAS_X86
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmPrinter();
#endif
#ifdef CATALYST_HAS_AARCH64
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
#endif
        return true;
    }();
    (void)inited;
}

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

static std::string OutOfProcessExecutorConnect;

static Error createTCPSocketError(Twine Details)
{
    return make_error<StringError>("Failed to connect TCP socket '" +
                                       Twine(OutOfProcessExecutorConnect) + "': " + Details,
                                   inconvertibleErrorCode());
}

static Expected<int> connectTCPSocket(std::string Host, std::string PortStr)
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
static Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
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
    size_t SlabSize = 1024 * 1024 * 1024;
    return MapperJITLinkMemoryManager::CreateWithMapper<SimpleRemoteMemoryMapper>(SlabSize, SREPC,
                                                                                  SAs);
}

Expected<std::unique_ptr<MemoryBuffer>> getFile(const Twine &FileName)
{
    auto F = MemoryBuffer::getFile(FileName);
    if (F) {
        return std::move(*F);
    }
    return createFileError(FileName, F.getError());
}

} // namespace

class Session {
  private:
    std::unique_ptr<ExecutionSession> ES;

    DataLayout DL;

    MangleAndInterner Mangle;
    ObjectLinkingLayer ObjectLayer;

    JITDylib &MainJD;

  public:
    ExecutorAddr alloc_fn{0};
    ExecutorAddr free_fn{0};
    ExecutorAddr invoke_fn{0};

    Session(std::unique_ptr<ExecutionSession> ES, DataLayout DL)
        : ES(std::move(ES)), DL(std::move(DL)), Mangle(*this->ES, this->DL), ObjectLayer(*this->ES),
          MainJD(this->ES->createBareJITDylib("<main>"))
    {
        MainJD.addGenerator(
            cantFail(EPCDynamicLibrarySearchGenerator::GetForTargetProcess(*this->ES)));
    }

    ~Session()
    {
        if (auto Err = ES->endSession()) {
            ES->reportError(std::move(Err));
        }
    }

    static Expected<std::unique_ptr<Session>> Create(StringRef remote_addr)
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
        auto EPC = SimpleRemoteEPC::Create<FDSimpleRemoteEPCTransport>(
            std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt), std::move(setup),
            *SockFD, *SockFD);
        if (!EPC) {
            return EPC.takeError();
        }

        JITTargetMachineBuilder JTMB((*EPC)->getTargetTriple());
        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL) {
            return DL.takeError();
        }

        auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));
        return std::make_unique<Session>(std::move(ES), std::move(*DL));
    }

    Error addObjectFile(std::unique_ptr<MemoryBuffer> Buf)
    {
        return ObjectLayer.add(MainJD, std::move(Buf));
    }

    const ExecutorAddr lookup(StringRef Name)
    {
        auto Sym = unwrap(ES->lookup({&MainJD}, Mangle(Name.str())), "lookup(" + Name + ")");
        return Sym.getAddress();
    }

    ExecutorProcessControl &getEPC() { return ES->getExecutorProcessControl(); }
};

/**
 * @brief A wrapper class for a remote function.
 *
 * @param session the session object holding the JIT session to the remote device.
 * @param pyface_addr the address of the entry of the kernel on the remote device.
 */
struct RemoteFunc {
    Session *session;
    uint64_t pyface_addr;
};

// ---------------------------------------------------------------------------
// Remote Driver Helper Functions
// ---------------------------------------------------------------------------

ExecutorAddr remote_alloc(Session *s, size_t size)
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

void remote_free(Session *s, ExecutorAddr addr)
{
    auto &epc = s->getEPC();
    auto err = epc.callSPSWrapper<void(shared::SPSExecutorAddr)>(s->free_fn, addr);
    if (err) {
        consumeError(std::move(err));
    }
}

void remote_write(Session *s, ExecutorAddr addr, const void *data, size_t size)
{
    auto &epc = s->getEPC();
    tpctypes::BufferWrite w{addr, ArrayRef<char>(static_cast<const char *>(data), size)};
    check(epc.getMemoryAccess().writeBuffers({w}), "write");
}

void remote_read(Session *s, ExecutorAddr addr, void *data, size_t size)
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

void remote_invoke(Session *s, ExecutorAddr entry, ArrayRef<ExecutorAddr> arg_addrs)
{
    auto &epc = s->getEPC();
    check(epc.callSPSWrapper<void(shared::SPSExecutorAddr,
                                  shared::SPSSequence<shared::SPSExecutorAddr>)>(s->invoke_fn,
                                                                                 entry, arg_addrs),
          "remote_invoke");
}

enum class StructKind { ArgStruct, ResultStruct };

struct MemrefSlot {
    size_t offset;    // offset of the slot within the outer struct
    size_t desc_size; // sizeof(memref descriptor)
    size_t rank;      // memref rank
    size_t elem_size; // sizeof(element)
};

struct MemrefStructLayout {
    size_t total_size = 0;
    StructKind kind = StructKind::ArgStruct;
    std::vector<MemrefSlot> memrefs;
};

// TODO: copied from `wrapper.cpp` do we need to share them in a common header file?
size_t memref_size_based_on_rank(size_t rank)
{
    size_t allocated = sizeof(void *);
    size_t aligned = sizeof(void *);
    size_t offset = sizeof(size_t);
    size_t sizes = rank * sizeof(size_t);
    size_t strides = rank * sizeof(size_t);
    return allocated + aligned + offset + sizes + strides;
}

size_t memref_data_size(const char *desc, const MemrefSlot &slot)
{
    if (slot.rank == 0) {
        return slot.elem_size;
    }

    const size_t *shape = reinterpret_cast<const size_t *>(desc + kShapeOff);
    return std::accumulate(shape, shape + slot.rank, slot.elem_size, std::multiplies<size_t>());
}

class RemoteAllocator {
  private:
    Session *sess;
    std::vector<ExecutorAddr> addrs;

  public:
    explicit RemoteAllocator(Session *sess) : sess(sess) {}
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

// Convert a ctypes Structure into a MeStructLayout.
// It takes a ctypes Structure subclass which is defined in
// `compiled_functions.py:CompiledFunctionArgValue` and
// `compiled_functions.py:CompiledFunctionReturnValue`. and convert it into a MemrefStructLayout.
MemrefStructLayout from_layout(nb::object cls, StructKind kind)
{
    auto ctypes = nb::module_::import_("ctypes");
    MemrefStructLayout layout{
        .kind = kind,
        .total_size = nb::cast<size_t>(ctypes.attr("sizeof")(cls)),
    };

    /* Data from the descriptor class */
    auto ranks = cls.attr("_ranks_");
    auto etypes = cls.attr("_etypes_");
    auto sizes = cls.attr("_sizes_");

    nb::object fields = cls.attr("_fields_");
    size_t n = nb::cast<size_t>(fields.attr("__len__")());
    layout.memrefs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        nb::object field = fields.attr("__getitem__")(i);
        std::string fname = nb::cast<std::string>(field.attr("__getitem__")(0));
        size_t rank = nb::cast<size_t>(ranks.attr("__getitem__")(i));
        size_t elem_size = nb::cast<size_t>(sizes.attr("__getitem__")(i));
        size_t desc_size = memref_size_based_on_rank(rank);
        size_t off = nb::cast<size_t>(cls.attr(fname.c_str()).attr("offset"));

        layout.memrefs.push_back({off, desc_size, rank, elem_size});
    }

    return layout;
}

struct ShimState {
    // original RemoteFunc members
    Session *sess;
    ExecutorAddr pyface_addr;

    // argument/result struct layouts
    MemrefStructLayout av_layout;
    MemrefStructLayout rv_layout;

    // Error state
    bool had_error = false;
    std::string error_msg;
};

// The "function pointer" abstraction: marshal av to remote, alloc/zero rv on
// remote, invoke the kernel remotely, unmarshal rv back. Cleans up on throw.
void do_remote_invoke(ShimState &s, void *rv_host, void *av_host)
{
    RemoteAllocator allocator(s.sess);
    try {
        ExecutorAddr av_remote = ExecutorAddr(0);
        ExecutorAddr rv_remote = ExecutorAddr(0);

        // Before invoking the kernel, we need to marshal the argument struct to the remote,
        // and pre-allocate the space for the result struct on the remote.
        // Once the kernel is invoked, we can unmarshal the result struct back to the host.

        // STEP 1: Processing the argument struct
        if (av_host) {
            //                              v desc_host for each slot
            // av_host ──►┌───────────┐     ┌──────────────────────┐   ┌──────┐
            //            │slot0 (ptr)┼────►│memref      .allocated┼┬─►│buffer│
            //            │slot1      │     │descriptor  .aligned ─┼┘  └──────┘
            //            │slot2      │     │            .offset   │
            //            │  ...      │     │            .shape    │
            //            │           │     │            .strides  │
            //            └───────────┘     └──────────────────────┘
            // 1. prepare the argument struct on the host
            std::vector<char> struct_buf(s.av_layout.total_size);
            std::memcpy(struct_buf.data(), av_host, s.av_layout.total_size);

            for (const auto &slot : s.av_layout.memrefs) {
                char *slot_addr = static_cast<char *>(av_host) + slot.offset;
                char *desc_host = *reinterpret_cast<char **>(slot_addr);
                if (!desc_host) {
                    continue;
                }

                // 2. prepare the descriptor for pushing to the remote
                std::vector<char> desc(slot.desc_size);
                std::memcpy(desc.data(), desc_host, slot.desc_size);

                // 3. allocate the data buffer on the remote
                size_t data_size = memref_data_size(desc_host, slot);
                if (data_size == 0) {
                    // allocated/aligned/offset fields are set to 0
                    std::memset(desc.data(), 0, kShapeOff);
                }
                else {
                    // 4. allocate the data buffer and write the data to the remote
                    ExecutorAddr data_remote = ExecutorAddr(0);
                    void *aligned_host = *reinterpret_cast<void **>(desc_host + kAlignedOff);
                    if (aligned_host) {
                        data_remote = allocator.alloc(data_size);
                        remote_write(s.sess, data_remote, aligned_host, data_size);
                    }

                    // 5. Copy the descriptor and update the allocated and aligned fields to point
                    // to the remote data
                    std::memcpy(desc.data(), desc_host, slot.desc_size);
                    std::memcpy(desc.data() + kAllocatedOff, &data_remote, sizeof(uintptr_t));
                    std::memcpy(desc.data() + kAlignedOff, &data_remote, sizeof(uintptr_t));
                    std::memset(desc.data() + kOffsetOff, 0, sizeof(int64_t));
                }

                // 6. push the descriptor to the remote and update the slot in the argument struct
                ExecutorAddr desc_remote = allocator.alloc(slot.desc_size);
                remote_write(s.sess, desc_remote, desc.data(), desc.size());
                std::memcpy(struct_buf.data() + slot.offset, &desc_remote, sizeof(uintptr_t));
            }

            av_remote = allocator.alloc(s.av_layout.total_size);
            remote_write(s.sess, av_remote, struct_buf.data(), struct_buf.size());
        }

        // Allocate the space for the result struct on the remote and invoke the kernel remotely.
        if (s.rv_layout.total_size) {
            rv_remote = allocator.alloc(s.rv_layout.total_size);
        }

        std::vector<ExecutorAddr> args = {rv_remote, av_remote};
        remote_invoke(s.sess, s.pyface_addr, args);

        // STEP 2: Processing the result struct to get the result back from the remote
        if (rv_remote && rv_host) {
            // The allocated and aligned fields in the descriptor are pointers to the data on
            // the remote, we need to:
            // 1. Read the descriptor from the remote -> store in struct_buf
            std::vector<char> struct_buf(s.rv_layout.total_size);
            remote_read(s.sess, rv_remote, struct_buf.data(), struct_buf.size());

            // v struct_buf
            // ┌─────┬─────┬─────┬─────┬─────────┐
            // │desc0│desc1│desc2│desc3│  ...    │
            // └─────┴─────┴─────┴─────┴─────────┘
            // Each slot maps to a descriptor in struct_buf
            for (const auto &slot : s.rv_layout.memrefs) {
                char *desc = struct_buf.data() + slot.offset;
                uintptr_t aligned_remote;
                std::memcpy(&aligned_remote, desc + kAlignedOff, sizeof(uintptr_t));

                // 2. Allocate a host buffer for the data
                size_t aligned_size = memref_data_size(desc, slot);
                size_t alloc_size = std::max<size_t>(aligned_size, 1);
                void *aligned_host = std::malloc(alloc_size);

                // 3. Read the data from the remote for each slot to the allocated buffer
                if (aligned_size && aligned_remote) {
                    remote_read(s.sess, ExecutorAddr(aligned_remote), aligned_host, aligned_size);
                }

                // 4. Update the descriptor to point to the host allocated buffer
                // Since the allocated and aligned fields are pointers to the data on the remote,
                // which are created by the remote kernel via `_mlir_memref_to_llvm_alloc`.
                uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(aligned_host);
                std::memcpy(desc + kAllocatedOff, &aligned_addr, sizeof(uintptr_t));
                std::memcpy(desc + kAlignedOff, &aligned_addr, sizeof(uintptr_t));
            }

            // 5. Copy the updated struct_buf back to the host
            std::memcpy(rv_host, struct_buf.data(), struct_buf.size());
        }
    }
    catch (std::exception &e) {
        s.had_error = true;
        s.error_msg = e.what();
    }
    catch (...) {
        s.had_error = true;
        s.error_msg = "unknown C++ exception in remote invoke";
    }
}

// ---------------------------------------------------------------------------
// Remote Driver Exported Functions
// ---------------------------------------------------------------------------

/**
 * @brief Open a session to the remote device and load the kernel object file.
 *
 * @param kernel_path the path to the kernel object file
 * @param remote the remote address
 * @return Session * the session object
 */
Session *open(const std::string &kernel_path, const std::string &remote)
{
    auto s = unwrap(Session::Create(remote), "open(" + remote + ")");

    // Load remote symbols
    check(s->getEPC().getBootstrapSymbols({{s->alloc_fn, "catalyst_remote_alloc"},
                                           {s->free_fn, "catalyst_remote_free"},
                                           {s->invoke_fn, "catalyst_remote_invoke"}}),
          "getBootstrapSymbols(catalyst_remote_*)");

    auto buf = unwrap(getFile(kernel_path), "getFile(" + kernel_path + ")");
    check(s->addObjectFile(std::move(buf)), "addObjectFile(" + kernel_path + ")");
    return s.release();
}

/**
 * @brief Lookup the address of a symbol in the remote device.
 *
 * @param s the session object
 * @param name the name of the symbol
 * @return uint64_t the address of the symbol
 */
uint64_t lookup(Session *s, const std::string &name) { return s->lookup(name).getValue(); }

/**
 * @brief Run the kernel as a main function (take argv as arguments, argc is the length of argv).
 *
 * @param s the session object
 * @param entry the entry function address
 * @param argv the command line arguments
 * @return int32_t the exit code
 */
int32_t run_as_main(Session *s, uint64_t entry, std::vector<std::string> &argv)
{
    auto &epc = s->getEPC();
    return unwrap(epc.runAsMain(ExecutorAddr(entry), argv), "run_as_main");
}

/**
 * @brief Build a CFUNCTYPE wrapped shim for RemoteFunc
 *
 * @param func the RemoteFunc object
 * @param av_class the ctypes Structure subclass for the argument struct
 * @param rv_class the ctypes Structure subclass for the result struct
 * @return nb::tuple the shim function and the state object
 */
nb::tuple make_remote_callable(RemoteFunc *func, nb::object av_class, nb::object rv_class)
{
    auto state = std::make_unique<ShimState>();
    state->sess = func->session;
    state->pyface_addr = ExecutorAddr(func->pyface_addr);
    state->av_layout = from_layout(av_class, StructKind::ArgStruct);
    if (!rv_class.is_none()) {
        state->rv_layout = from_layout(rv_class, StructKind::ResultStruct);
    }

    // Wrap the shim in a CFUNCTYPE with the signature `(void(*)(void*, void*)) -> void`.
    // This function wrap the remote invoke logic and will be taken by the `wrapper.wrap` to
    // formalize as `void (*)(void *, void *)` for execution.
    auto ctypes = nb::module_::import_("ctypes");
    nb::object func_type =
        ctypes.attr("CFUNCTYPE")(nb::none(), ctypes.attr("c_void_p"), ctypes.attr("c_void_p"));
    nb::object callable =
        nb::cpp_function([raw = state.get()](uintptr_t rv_addr, uintptr_t av_addr) {
            do_remote_invoke(*raw, reinterpret_cast<void *>(rv_addr),
                             reinterpret_cast<void *>(av_addr));
        });
    nb::object shim = func_type(callable);

    nb::object state_py = nb::cast(state.release(), nb::rv_policy::take_ownership);
    return nb::make_tuple(shim, state_py);
}

NB_MODULE(remote_driver, m)
{
    m.doc() = "remote_driver module";
    nb::class_<Session>(m, "Session");
    nb::class_<RemoteFunc>(m, "RemoteFunc")
        .def(nb::init<Session *, uint64_t>(), nb::arg("session"), nb::arg("pyface_addr"));
    nb::class_<ShimState>(m, "ShimState")
        .def_ro("had_error", &ShimState::had_error)
        .def_ro("error_msg", &ShimState::error_msg);

    m.def("open", &open, nb::rv_policy::take_ownership, nb::arg("kernel_path"), nb::arg("remote"));
    m.def("lookup", &lookup, nb::arg("session"), nb::arg("name"));
    m.def("run_as_main", &run_as_main, nb::arg("session"), nb::arg("entry_addr"), nb::arg("argv"));
    m.def("make_remote_callable", &make_remote_callable, nb::arg("func"), nb::arg("av_class"),
          nb::arg("rv_class").none());
}
