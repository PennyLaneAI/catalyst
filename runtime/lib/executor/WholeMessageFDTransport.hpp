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

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <sys/socket.h>
#include <system_error>
#include <vector>

#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include <unistd.h>

namespace catalyst::executor {

/**
 * @brief A SimpleRemoteEPC transport over file descriptors that writes each message in one write.
 *
 * Messages are read as llvm::orc::FDSimpleRemoteEPCTransport reads them, and keep its wire format:
 * a 32-byte header of four little-endian u64 fields (message size, opcode, sequence number, tag
 * address), then the argument bytes. Header and argument bytes go out in a single write(), so a TCP
 * hop without TCP_NODELAY, such as the last leg of an ssh port forward, sends them as one segment
 * instead of holding the argument bytes for the peer's delayed ACK, which costs up to 40 ms per
 * message.
 *
 * With `shutdownOnDisconnect`, disconnect() also shutdown()s the descriptors. A read() the
 * listener thread is blocked in then returns, and the peer sees the FIN. Without it, close() alone
 * neither wakes that read() nor sends the FIN while the read() holds the socket open.
 */
class WholeMessageFDTransport : public llvm::orc::SimpleRemoteEPCTransport {
  public:
    static llvm::Expected<std::unique_ptr<WholeMessageFDTransport>>
    Create(llvm::orc::SimpleRemoteEPCTransportClient &C, int InFD, int OutFD,
           bool shutdownOnDisconnect = false) {
        auto Inner = llvm::orc::FDSimpleRemoteEPCTransport::Create(C, InFD, OutFD);
        if (!Inner) {
            return Inner.takeError();
        }
        return std::make_unique<WholeMessageFDTransport>(std::move(*Inner), InFD, OutFD,
                                                         shutdownOnDisconnect);
    }

    WholeMessageFDTransport(std::unique_ptr<llvm::orc::FDSimpleRemoteEPCTransport> inner, int inFD,
                            int outFD, bool shutdownOnDisconnect)
        : Inner(std::move(inner)), InFD(inFD), OutFD(outFD),
          ShutdownOnDisconnect(shutdownOnDisconnect) {}

    llvm::Error start() override { return Inner->start(); }

    llvm::Error sendMessage(llvm::orc::SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                            llvm::orc::ExecutorAddr TagAddr,
                            llvm::ArrayRef<char> ArgBytes) override {
        std::vector<char> Message(HeaderSize + ArgBytes.size());
        llvm::support::endian::write64le(Message.data(), Message.size());
        llvm::support::endian::write64le(Message.data() + 8, static_cast<uint64_t>(OpC));
        llvm::support::endian::write64le(Message.data() + 16, SeqNo);
        llvm::support::endian::write64le(Message.data() + 24, TagAddr.getValue());
        if (!ArgBytes.empty()) {
            std::memcpy(Message.data() + HeaderSize, ArgBytes.data(), ArgBytes.size());
        }

        std::lock_guard<std::mutex> Lock(M);
        if (Disconnected) {
            return llvm::make_error<llvm::StringError>("FD-transport disconnected",
                                                       llvm::inconvertibleErrorCode());
        }
        const char *Src = Message.data();
        size_t Left = Message.size();
        while (Left > 0) {
            ssize_t Written = ::write(OutFD, Src, Left);
            if (Written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return llvm::errorCodeToError(std::error_code(errno, std::generic_category()));
            }
            Src += Written;
            Left -= static_cast<size_t>(Written);
        }
        return llvm::Error::success();
    }

    void disconnect() override {
        {
            std::lock_guard<std::mutex> Lock(M);
            Disconnected = true;
        }
        if (ShutdownOnDisconnect) {
            ::shutdown(InFD, SHUT_RDWR);
            if (OutFD != InFD) {
                ::shutdown(OutFD, SHUT_RDWR);
            }
        }
        Inner->disconnect();
    }

  private:
    // The message header: size, opcode, sequence number and tag address, each a little-endian u64.
    static constexpr size_t HeaderSize = 4 * sizeof(uint64_t);

    std::unique_ptr<llvm::orc::FDSimpleRemoteEPCTransport> Inner;
    int InFD;
    int OutFD;
    bool ShutdownOnDisconnect;
    std::mutex M;
    bool Disconnected = false;
};

} // namespace catalyst::executor
