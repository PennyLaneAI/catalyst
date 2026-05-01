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
//

#include "CatalystRemoteDriver.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <sys/socket.h>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/MemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include <netdb.h>
#include <unistd.h>

using namespace llvm;
using namespace llvm::orc;

struct CatalystRemoteSession_ {
    std::unique_ptr<LLJIT> J;
    ExecutorAddr alloc_fn{0};
    ExecutorAddr free_fn{0};
    ExecutorAddr invoke_fn{0};
    ExecutorAddr result_slot{0};
};

namespace {

void copy_err_msg(char *buf, size_t buflen, llvm::StringRef msg)
{
    if (!buf || buflen == 0) {
        return;
    }
    size_t n = std::min<size_t>(buflen - 1, msg.size());
    std::memcpy(buf, msg.data(), n);
    buf[n] = '\0';
}

void capture_err(char *buf, size_t buflen, Error E)
{
    std::string Out;
    raw_string_ostream OS(Out);
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &I) { OS << I.message() << "\n"; });
    copy_err_msg(buf, buflen, OS.str());
}

Expected<int> connectTCP(StringRef remoteAddr)
{
    auto Colon = remoteAddr.rfind(':');
    if (Colon == StringRef::npos) {
        return createStringError(inconvertibleErrorCode(),
                                 "remote spec must be host:port, got '%s'",
                                 remoteAddr.str().c_str());
    }
    std::string Host = remoteAddr.substr(0, Colon).str();
    std::string Port = remoteAddr.substr(Colon + 1).str();

    addrinfo Hints{};
    Hints.ai_family = AF_UNSPEC;
    Hints.ai_socktype = SOCK_STREAM;

    addrinfo *Res = nullptr;
    int Gai = getaddrinfo(Host.c_str(), Port.c_str(), &Hints, &Res);
    if (Gai != 0) {
        return createStringError(inconvertibleErrorCode(), "getaddrinfo(%s:%s): %s", Host.c_str(),
                                 Port.c_str(), gai_strerror(Gai));
    }

    int Sock = -1;
    int LastErr = 0;
    for (auto *AI = Res; AI; AI = AI->ai_next) {
        Sock = ::socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol);
        if (Sock < 0) {
            LastErr = errno;
            continue;
        }
        if (::connect(Sock, AI->ai_addr, AI->ai_addrlen) == 0) {
            break;
        }
        LastErr = errno;
        ::close(Sock);
        Sock = -1;
    }
    freeaddrinfo(Res);

    if (Sock < 0) {
        return createStringError(inconvertibleErrorCode(), "connect %s:%s: %s", Host.c_str(),
                                 Port.c_str(), std::strerror(LastErr));
    }
    return Sock;
}

Expected<std::unique_ptr<ExecutorProcessControl>> makeRemoteEPC(StringRef remoteAddr)
{
    auto Sock = connectTCP(remoteAddr);
    if (!Sock) {
        return Sock.takeError();
    }
    return SimpleRemoteEPC::Create<FDSimpleRemoteEPCTransport>(
        std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt), SimpleRemoteEPC::Setup(),
        *Sock, *Sock);
}

Expected<ExecutorAddr> lookup_in(LLJIT &J, StringRef name)
{
    auto &ES = J.getExecutionSession();
    auto Sym = ES.lookup({&J.getMainJITDylib()}, ES.intern(name));
    if (!Sym) {
        return Sym.takeError();
    }
    return Sym->getAddress();
}

bool ensure_helpers(CatalystRemoteSession *session)
{
    if (!session) {
        return false;
    }
    if (session->invoke_fn.getValue() != 0) {
        return true;
    }
    auto a = lookup_in(*session->J, "catalyst_remote_alloc");
    auto f = lookup_in(*session->J, "catalyst_remote_free");
    auto i = lookup_in(*session->J, "catalyst_remote_invoke");
    auto r = lookup_in(*session->J, "catalyst_remote_result_slot");
    if (!a || !f || !i || !r) {
        consumeError(a.takeError());
        consumeError(f.takeError());
        consumeError(i.takeError());
        consumeError(r.takeError());
        return false;
    }
    session->alloc_fn = *a;
    session->free_fn = *f;
    session->invoke_fn = *i;
    session->result_slot = *r;
    return true;
}

void initialize_targets_once()
{
    static const bool inited = []() {
#ifdef CATALYST_REMOTE_HAS_X86
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmPrinter();
#endif
#ifdef CATALYST_REMOTE_HAS_AARCH64
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
#endif
        return true;
    }();
    (void)inited;
}

} // namespace

extern "C" CatalystRemoteSession *catalyst_remote_open(const char *kernel_path,
                                                       const char *remoteAddr, char *error_buf,
                                                       size_t error_buf_size)
{
    if (!kernel_path || !remoteAddr) {
        copy_err_msg(error_buf, error_buf_size, "kernel_path and remoteAddr are required");
        return nullptr;
    }

    initialize_targets_once();

    auto EPC = makeRemoteEPC(remoteAddr);
    if (!EPC) {
        capture_err(error_buf, error_buf_size, EPC.takeError());
        return nullptr;
    }

    auto J = LLJITBuilder().setExecutorProcessControl(std::move(*EPC)).create();
    if (!J) {
        capture_err(error_buf, error_buf_size, J.takeError());
        return nullptr;
    }

    // Resolve any unresolved extern symbols
    auto G = EPCDynamicLibrarySearchGenerator::GetForTargetProcess((*J)->getExecutionSession());
    if (!G) {
        capture_err(error_buf, error_buf_size, G.takeError());
        return nullptr;
    }
    (*J)->getMainJITDylib().addGenerator(std::move(*G));

    auto Buf = MemoryBuffer::getFile(kernel_path);
    if (!Buf) {
        copy_err_msg(error_buf, error_buf_size,
                     ("open '" + std::string(kernel_path) + "': " + Buf.getError().message()));
        return nullptr;
    }
    if (auto E = (*J)->addObjectFile(std::move(*Buf))) {
        capture_err(error_buf, error_buf_size, std::move(E));
        return nullptr;
    }

    auto *S = new CatalystRemoteSession_;
    S->J = std::move(*J);
    return S;
}

extern "C" void catalyst_remote_close(CatalystRemoteSession *s)
{
    if (!s) {
        return;
    }
    delete s;
}

extern "C" int catalyst_remote_lookup(CatalystRemoteSession *s, const char *name,
                                      uint64_t *out_addr)
{
    if (!s || !name || !out_addr) {
        return -EINVAL;
    }
    auto &ES = s->J->getExecutionSession();
    auto Sym = ES.lookup({&s->J->getMainJITDylib()}, ES.intern(name));
    if (!Sym) {
        consumeError(Sym.takeError());
        return -ENOENT;
    }
    *out_addr = Sym->getAddress().getValue();
    return 0;
}

extern "C" int catalyst_remote_alloc(CatalystRemoteSession *s, size_t size, uint64_t *out_addr)
{
    if (!s || !out_addr) {
        return -EINVAL;
    }
    if (!ensure_helpers(s)) {
        return -ENOSYS;
    }
    auto &EPC = s->J->getExecutionSession().getExecutorProcessControl();
    char size_buf[24];
    std::snprintf(size_buf, sizeof(size_buf), "%zu", size);
    std::vector<std::string> Args{"catalyst_remote_alloc", size_buf};
    auto Rc = EPC.runAsMain(s->alloc_fn, Args);
    if (!Rc || *Rc != 0) {
        if (!Rc) {
            consumeError(Rc.takeError());
        }
        return -EIO;
    }
    // The executor wrote the new pointer into the result_slot, read it back
    auto R = EPC.getMemoryAccess().readUInt64s({s->result_slot});
    if (!R || R->empty()) {
        if (!R) {
            consumeError(R.takeError());
        }
        return -EIO;
    }
    *out_addr = (*R)[0];
    return *out_addr ? 0 : -ENOMEM;
}

extern "C" int catalyst_remote_free(CatalystRemoteSession *s, uint64_t addr, size_t size)
{
    (void)size;
    if (!s) {
        return -EINVAL;
    }
    if (!ensure_helpers(s)) {
        return -ENOSYS;
    }
    auto &EPC = s->J->getExecutionSession().getExecutorProcessControl();
    char addr_buf[24];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long)addr);
    std::vector<std::string> Args{"catalyst_remote_free", addr_buf};
    auto Rc = EPC.runAsMain(s->free_fn, Args);
    if (!Rc) {
        consumeError(Rc.takeError());
        return -EIO;
    }
    return 0;
}

extern "C" int catalyst_remote_write(CatalystRemoteSession *s, uint64_t remoteAddr,
                                     const void *host_data, size_t size)
{
    if (!s || !host_data) {
        return -EINVAL;
    }
    auto &EPC = s->J->getExecutionSession().getExecutorProcessControl();
    tpctypes::BufferWrite W{
        ExecutorAddr(remoteAddr),
        ArrayRef<char>(static_cast<const char *>(host_data), size),
    };
    if (auto E = EPC.getMemoryAccess().writeBuffers({W})) {
        consumeError(std::move(E));
        return -EIO;
    }
    return 0;
}

extern "C" int catalyst_remote_read(CatalystRemoteSession *s, uint64_t remoteAddr, void *host_data,
                                    size_t size)
{
    if (!s || !host_data) {
        return -EINVAL;
    }
    auto &EPC = s->J->getExecutionSession().getExecutorProcessControl();
    ExecutorAddrRange R(ExecutorAddr(remoteAddr), ExecutorAddr(remoteAddr + size));
    auto Result = EPC.getMemoryAccess().readBuffers({R});
    if (!Result || Result->empty() || (*Result)[0].size() != size) {
        if (!Result) {
            consumeError(Result.takeError());
        }
        return -EIO;
    }
    std::memcpy(host_data, (*Result)[0].data(), size);
    return 0;
}

extern "C" int catalyst_remote_invoke_pyface(CatalystRemoteSession *s, uint64_t entry_addr,
                                             const uint64_t *arg_addrs, int n_args)
{
    if (!s) {
        return -EINVAL;
    }
    if (n_args < 0 || n_args > 8) {
        return -EINVAL;
    }
    if (!ensure_helpers(s)) {
        return -ENOSYS;
    }
    auto &EPC = s->J->getExecutionSession().getExecutorProcessControl();

    std::vector<std::string> Args;
    Args.reserve(2 + n_args);
    Args.emplace_back("catalyst_remote_invoke");
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)entry_addr);
    Args.emplace_back(buf);
    for (int i = 0; i < n_args; ++i) {
        std::snprintf(buf, sizeof(buf), "0x%llx",
                      (unsigned long long)(arg_addrs ? arg_addrs[i] : 0));
        Args.emplace_back(buf);
    }
    auto Rc = EPC.runAsMain(s->invoke_fn, Args);
    if (!Rc || *Rc != 0) {
        if (!Rc) {
            consumeError(Rc.takeError());
        }
        return -EIO;
    }
    return 0;
}

extern "C" int catalyst_remote_run_as_main(CatalystRemoteSession *s, uint64_t entry_addr, int argc,
                                           const char *const *argv, int32_t *out_rc)
{
    if (!s || !out_rc) {
        return -EINVAL;
    }
    std::vector<std::string> Args;
    Args.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        Args.emplace_back(argv && argv[i] ? argv[i] : "");
    }
    auto Rc = s->J->getExecutionSession().getExecutorProcessControl().runAsMain(
        ExecutorAddr(entry_addr), Args);
    if (!Rc) {
        consumeError(Rc.takeError());
        return -EIO;
    }
    *out_rc = static_cast<int32_t>(*Rc);
    return 0;
}
