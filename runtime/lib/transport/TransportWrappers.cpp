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

// TransportWrappers.cpp
//
//   __wrapper   dispatched. The executor resolves the symbol through LLVM ORC and calls it with one
//               flat argument buffer; the result goes back as another flat buffer. Neither buffer
//               is framed, so the layout comes from the signature the call site declared.
//   __call      in-process, through catalyst.custom_call. Each argument arrives as its own encoded
//               memref, read through its data pointer.
//
// Both layouts have to agree with pennylane/runtime/operands.py, which is what builds the
// arguments.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "TransportABI.h"
#include "TransportCAPI.h"

namespace {

//===----------------------------------------------------------------------===//
// In-process: an array of pointers to encoded memrefs
//===----------------------------------------------------------------------===//

// Where operand `i` keeps its data. Results are addressed the same way
void *data_of(void **slots, unsigned i) {
    return static_cast<CatalystEncodedMemref *>(slots[i])->data_aligned;
}

template <typename T> T arg(void **args, unsigned i) {
    return *static_cast<const T *>(data_of(args, i));
}

// A str operand is already NUL-terminated bytes, so its data pointer is the C string.
const char *str_arg(void **args, unsigned i) { return static_cast<const char *>(data_of(args, i)); }

// A session handle crosses as a plain integer and is never dereferenced on the way here.
CatalystTransportSession *session_arg(void **args, unsigned i) {
    return reinterpret_cast<CatalystTransportSession *>(arg<std::uintptr_t>(args, i));
}

template <typename T> void put(void **results, unsigned i, T value) {
    *static_cast<T *>(data_of(results, i)) = value;
}

//===----------------------------------------------------------------------===//
// Dispatched: one flat argument buffer in, one flat result buffer out
//===----------------------------------------------------------------------===//

// Reads a flat argument buffer, one argument after another.
class Flat {
  public:
    Flat(const char *buf, std::size_t size) : buf_(buf), size_(size) {}

    template <typename T> T get() {
        T value{};
        if (take(sizeof(T))) {
            std::memcpy(&value, buf_ + offset_ - sizeof(T), sizeof(T));
        }
        return value;
    }

    // A str occupies a fixed NUL-padded field, however short the string in it is.
    const char *str() {
        if (!take(CATALYST_TRANSPORT_STR_BYTES)) {
            return "";
        }
        const char *field = buf_ + offset_ - CATALYST_TRANSPORT_STR_BYTES;
        if (std::memchr(field, '\0', CATALYST_TRANSPORT_STR_BYTES) == nullptr) {
            ok_ = false;
            return "";
        }
        return field;
    }

    CatalystTransportSession *session() {
        return reinterpret_cast<CatalystTransportSession *>(get<std::uint64_t>());
    }

    bool ok() const { return ok_; }

  private:
    bool take(std::size_t nbytes) {
        if (!ok_ || offset_ + nbytes > size_) {
            ok_ = false;
            return false;
        }
        offset_ += nbytes;
        return true;
    }

    const char *buf_;
    std::size_t size_;
    std::size_t offset_ = 0;
    bool ok_ = true;
};

// Builds the flat result buffer, in the order the results are declared. Its size is what the
// caller's result buffers add up to, which each adapter knows from its own arguments.
class Out {
  public:
    explicit Out(std::size_t bytes) : bytes_(bytes) {
        buf_ = static_cast<char *>(std::calloc(1, bytes ? bytes : 1));
    }
    ~Out() { std::free(buf_); }
    Out(const Out &) = delete;
    Out &operator=(const Out &) = delete;

    explicit operator bool() const { return buf_ != nullptr; }

    // The next `nbytes` of the result buffer, to be written through.
    void *reserve(std::size_t nbytes) {
        void *slot = buf_ + offset_;
        offset_ += nbytes;
        return slot;
    }

    template <typename T> T *slot() { return static_cast<T *>(reserve(sizeof(T))); }

    CatalystWrapperResult release() {
        CatalystWrapperResult result;
        result.size = bytes_;
        if (bytes_ <= sizeof(result.data.value)) {
            std::memset(result.data.value, 0, sizeof(result.data.value));
            std::memcpy(result.data.value, buf_, bytes_);
            std::free(buf_);
        } else {
            result.data.value_ptr = buf_;
        }
        buf_ = nullptr;
        return result;
    }

  private:
    std::size_t bytes_;
    std::size_t offset_ = 0;
    char *buf_ = nullptr;
};

CatalystWrapperResult wrapper_error(const char *message) {
    CatalystWrapperResult result;
    result.size = 0;
    std::size_t nbytes = std::strlen(message) + 1;
    char *copy = static_cast<char *>(std::malloc(nbytes));
    if (copy != nullptr) {
        std::memcpy(copy, message, nbytes);
    }
    result.data.value_ptr = copy;
    return result;
}

CatalystWrapperResult finish(const Flat &in, Out &out) {
    if (!in.ok()) {
        return wrapper_error("transport: argument buffer is short of what the operation takes");
    }
    if (!out) {
        return wrapper_error("transport: out of memory building a result");
    }
    return out.release();
}

constexpr std::size_t I32 = sizeof(std::int32_t);
constexpr std::size_t I64 = sizeof(std::int64_t);
constexpr std::size_t U64 = sizeof(std::uint64_t);

} // namespace

extern "C" {

//===----------------------------------------------------------------------===//
// Dispatched wrappers: sessions
//===----------------------------------------------------------------------===//

CatalystWrapperResult __catalyst__transport__create__wrapper(const char *buf, std::size_t size) {
    Flat in(buf, size);
    const char *library = in.str();
    const char *config = in.str();
    auto role = in.get<std::int32_t>();
    const char *key = in.str();
    Out out(U64);
    if (in.ok() && out) {
        *out.slot<std::uint64_t>() = reinterpret_cast<std::uintptr_t>(
            __catalyst__transport__create(library, config, role, key));
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__get_session__wrapper(const char *buf,
                                                                  std::size_t size) {
    Flat in(buf, size);
    auto role = in.get<std::int32_t>();
    const char *key = in.str();
    Out out(U64);
    if (in.ok() && out) {
        *out.slot<std::uint64_t>() =
            reinterpret_cast<std::uintptr_t>(__catalyst__transport__get_session(role, key));
    }
    return finish(in, out);
}

//===----------------------------------------------------------------------===//
// Dispatched wrappers: connecting and key exchange
//===----------------------------------------------------------------------===//

CatalystWrapperResult __catalyst__transport__connect__wrapper(const char *buf, std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    const char *peer = in.str();
    auto port = in.get<std::uint16_t>();
    Out out(I32);
    if (in.ok() && out) {
        *out.slot<std::int32_t>() = __catalyst__transport__connect(session, peer, port);
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__connect_async__wrapper(const char *buf,
                                                                    std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    const char *peer = in.str();
    auto port = in.get<std::uint16_t>();
    Out out(I64);
    if (in.ok() && out) {
        *out.slot<std::int64_t>() = __catalyst__transport__connect_async(session, peer, port);
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__exchange_keys_async__wrapper(const char *buf,
                                                                          std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    Out out(I64);
    if (in.ok() && out) {
        *out.slot<std::int64_t>() = __catalyst__transport__exchange_keys_async(session);
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__await__wrapper(const char *buf, std::size_t size) {
    Flat in(buf, size);
    auto token = in.get<std::int64_t>();
    Out out(I32);
    if (in.ok() && out) {
        *out.slot<std::int32_t>() = __catalyst__transport__await(token);
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__exchange_keys__wrapper(const char *buf,
                                                                    std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    Out out(I32);
    if (in.ok() && out) {
        *out.slot<std::int32_t>() = __catalyst__transport__exchange_keys(session);
    }
    return finish(in, out);
}

//===----------------------------------------------------------------------===//
// Dispatched wrappers: channel setup
//===----------------------------------------------------------------------===//

CatalystWrapperResult __catalyst__transport__establish_channel__wrapper(const char *buf,
                                                                        std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    const char *transport = in.str();
    Out out(I32);
    if (in.ok() && out) {
        *out.slot<std::int32_t>() = __catalyst__transport__establish_channel(session, transport);
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__set_coprocessor_fn__wrapper(const char *buf,
                                                                         std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    const char *symbol = in.str();
    Out out(I32);
    if (in.ok() && out) {
        *out.slot<std::int32_t>() = __catalyst__transport__set_coprocessor_fn(session, symbol);
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__set_message_sizes__wrapper(const char *buf,
                                                                        std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    auto work_item = in.get<std::uint32_t>();
    auto in_bytes = in.get<std::uint64_t>();
    auto out_bytes = in.get<std::uint64_t>();
    Out out(I32);
    if (in.ok() && out) {
        *out.slot<std::int32_t>() =
            __catalyst__transport__set_message_sizes(session, work_item, in_bytes, out_bytes);
    }
    return finish(in, out);
}

//===----------------------------------------------------------------------===//
// Dispatched wrappers: benchmark
//===----------------------------------------------------------------------===//

CatalystWrapperResult __catalyst__transport__start_benchmark__wrapper(const char *buf,
                                                                      std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    auto iters = in.get<std::uint32_t>();
    auto decoder_id = in.get<std::uint32_t>();
    auto flags = in.get<std::uint32_t>();
    auto samples_bytes = in.get<std::uint64_t>();

    if (samples_bytes > SIZE_MAX - I32 - U64) {
        return wrapper_error("transport: start_benchmark was given a samples buffer size that is "
                             "not a size");
    }

    Out out(I32 + static_cast<std::size_t>(samples_bytes) + U64);
    if (in.ok() && out) {
        std::int32_t *status = out.slot<std::int32_t>();
        // reserve the samples buffer with the caller's capacity
        auto *samples = static_cast<std::uint64_t *>(out.reserve(samples_bytes));
        auto *rounds = out.slot<std::uint64_t>();
        *status = __catalyst__transport__start_benchmark(session, iters, decoder_id, flags, samples,
                                                         samples_bytes, rounds);
    }
    return finish(in, out);
}

//===----------------------------------------------------------------------===//
// Dispatched wrappers: lifecycle
//===----------------------------------------------------------------------===//

CatalystWrapperResult __catalyst__transport__start__wrapper(const char *buf, std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    Out out(I32);
    if (in.ok() && out) {
        __catalyst__transport__start(session);
        *out.slot<std::int32_t>() = CATALYST_TRANSPORT_OK;
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__stop__wrapper(const char *buf, std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    Out out(I32);
    if (in.ok() && out) {
        __catalyst__transport__stop(session);
        *out.slot<std::int32_t>() = CATALYST_TRANSPORT_OK;
    }
    return finish(in, out);
}

CatalystWrapperResult __catalyst__transport__destroy__wrapper(const char *buf, std::size_t size) {
    Flat in(buf, size);
    auto *session = in.session();
    Out out(I32);
    if (in.ok() && out) {
        __catalyst__transport__destroy(session);
        *out.slot<std::int32_t>() = CATALYST_TRANSPORT_OK;
    }
    return finish(in, out);
}

//===----------------------------------------------------------------------===//
// In-process adapters: sessions
//===----------------------------------------------------------------------===//

void __catalyst__transport__create__call(void **args, void **results) {
    put<std::uint64_t>(
        results, 0,
        reinterpret_cast<std::uintptr_t>(__catalyst__transport__create(
            str_arg(args, 0), str_arg(args, 1), arg<std::int32_t>(args, 2), str_arg(args, 3))));
}

void __catalyst__transport__get_session__call(void **args, void **results) {
    put<std::uint64_t>(results, 0,
                       reinterpret_cast<std::uintptr_t>(__catalyst__transport__get_session(
                           arg<std::int32_t>(args, 0), str_arg(args, 1))));
}

//===----------------------------------------------------------------------===//
// In-process adapters: connecting and key exchange
//===----------------------------------------------------------------------===//

void __catalyst__transport__connect__call(void **args, void **results) {
    put<std::int32_t>(results, 0,
                      __catalyst__transport__connect(session_arg(args, 0), str_arg(args, 1),
                                                     arg<std::uint16_t>(args, 2)));
}

void __catalyst__transport__connect_async__call(void **args, void **results) {
    put<std::int64_t>(results, 0,
                      __catalyst__transport__connect_async(session_arg(args, 0), str_arg(args, 1),
                                                           arg<std::uint16_t>(args, 2)));
}

void __catalyst__transport__exchange_keys__call(void **args, void **results) {
    put<std::int32_t>(results, 0, __catalyst__transport__exchange_keys(session_arg(args, 0)));
}

void __catalyst__transport__exchange_keys_async__call(void **args, void **results) {
    put<std::int64_t>(results, 0, __catalyst__transport__exchange_keys_async(session_arg(args, 0)));
}

void __catalyst__transport__await__call(void **args, void **results) {
    put<std::int32_t>(results, 0, __catalyst__transport__await(arg<std::int64_t>(args, 0)));
}

//===----------------------------------------------------------------------===//
// In-process adapters: channel setup
//===----------------------------------------------------------------------===//

void __catalyst__transport__establish_channel__call(void **args, void **results) {
    put<std::int32_t>(
        results, 0,
        __catalyst__transport__establish_channel(session_arg(args, 0), str_arg(args, 1)));
}

void __catalyst__transport__set_coprocessor_fn__call(void **args, void **results) {
    put<std::int32_t>(
        results, 0,
        __catalyst__transport__set_coprocessor_fn(session_arg(args, 0), str_arg(args, 1)));
}

void __catalyst__transport__set_message_sizes__call(void **args, void **results) {
    put<std::int32_t>(results, 0,
                      __catalyst__transport__set_message_sizes(
                          session_arg(args, 0), arg<std::uint32_t>(args, 1),
                          arg<std::uint64_t>(args, 2), arg<std::uint64_t>(args, 3)));
}

//===----------------------------------------------------------------------===//
// In-process adapters: data path
//===----------------------------------------------------------------------===//

void __catalyst__transport__request_slot__call(void **args, void **results) {
    put<std::uint64_t>(results, 0,
                       reinterpret_cast<std::uintptr_t>(
                           __catalyst__transport__request_slot(session_arg(args, 0))));
}

void __catalyst__transport__reply_slot__call(void **args, void **results) {
    put<std::uint64_t>(
        results, 0,
        reinterpret_cast<std::uintptr_t>(__catalyst__transport__reply_slot(session_arg(args, 0))));
}

// The source is a buf: its data pointer, with the byte count as its own argument.
void __catalyst__transport__stage_payload__call(void **args, void **results) {
    put<std::int32_t>(results, 0,
                      __catalyst__transport__stage_payload(session_arg(args, 0), data_of(args, 1),
                                                           arg<std::uint64_t>(args, 2),
                                                           arg<std::uint32_t>(args, 3)));
}

void __catalyst__transport__post__call(void **args, void **results) {
    put<std::int32_t>(
        results, 0, __catalyst__transport__post(session_arg(args, 0), arg<std::uint32_t>(args, 1)));
}

// The reply is an out buffer, so it comes from results[1] while its size is argument 1.
void __catalyst__transport__collect__call(void **args, void **results) {
    put<std::int32_t>(results, 0,
                      __catalyst__transport__collect(session_arg(args, 0), data_of(results, 1),
                                                     arg<std::uint64_t>(args, 1)));
}

void __catalyst__transport__last_rtt_ns__call(void **args, void **results) {
    put<std::uint64_t>(results, 0, __catalyst__transport__last_rtt_ns(session_arg(args, 0)));
}

//===----------------------------------------------------------------------===//
// In-process adapters: benchmark
//===----------------------------------------------------------------------===//

void __catalyst__transport__start_benchmark__call(void **args, void **results) {
    put<std::int32_t>(
        results, 0,
        __catalyst__transport__start_benchmark(
            session_arg(args, 0), arg<std::uint32_t>(args, 1), arg<std::uint32_t>(args, 2),
            arg<std::uint32_t>(args, 3), static_cast<std::uint64_t *>(data_of(results, 1)),
            arg<std::uint64_t>(args, 4), static_cast<std::uint64_t *>(data_of(results, 2))));
}

//===----------------------------------------------------------------------===//
// In-process adapters: lifecycle
//===----------------------------------------------------------------------===//

void __catalyst__transport__start__call(void **args, void **results) {
    __catalyst__transport__start(session_arg(args, 0));
    put<std::int32_t>(results, 0, CATALYST_TRANSPORT_OK);
}

void __catalyst__transport__stop__call(void **args, void **results) {
    __catalyst__transport__stop(session_arg(args, 0));
    put<std::int32_t>(results, 0, CATALYST_TRANSPORT_OK);
}

void __catalyst__transport__destroy__call(void **args, void **results) {
    __catalyst__transport__destroy(session_arg(args, 0));
    put<std::int32_t>(results, 0, CATALYST_TRANSPORT_OK);
}

} // extern "C"
