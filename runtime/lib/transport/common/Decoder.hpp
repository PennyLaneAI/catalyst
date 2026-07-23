#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>

#include "DecoderPlugin.hpp"

namespace rdma::devices::common {

// A per-shot decode compute: reads the syndrome from `in` (in_len bytes) and
// writes the correction into `out` (out_len bytes), in place on the caller's
// buffers. Must not throw.
class Decoder {
  public:
    virtual ~Decoder() = default;
    virtual void run(const void *in, std::size_t in_len, void *out, std::size_t out_len) = 0;
};

// Passthrough: out = in for min(in_len, out_len) bytes. Default / self-test.
class EchoDecoder : public Decoder {
  public:
    void run(const void *in, std::size_t in_len, void *out, std::size_t out_len) override
    {
        std::memcpy(out, in, std::min(in_len, out_len));
    }
};

// Adapts a dlopen'd DecoderPlugin (see DecoderPlugin.hpp) to the Decoder API.
class PluginDecoder : public Decoder {
  public:
    explicit PluginDecoder(std::unique_ptr<DecoderPlugin> plugin) : plugin_(std::move(plugin)) {}
    void run(const void *in, std::size_t in_len, void *out, std::size_t out_len) override
    {
        plugin_->fn()(plugin_->ctx(), in, in_len, out, out_len);
    }

  private:
    std::unique_ptr<DecoderPlugin> plugin_;
};

} // namespace rdma::devices::common
