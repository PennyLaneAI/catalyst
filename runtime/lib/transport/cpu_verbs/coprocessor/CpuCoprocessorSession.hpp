#pragma once
#include <cstddef>
#include <memory>

#include "CpuSessionBase.hpp"

namespace rdma::devices::cpu_libibverbs {

// Coprocessor role: receives messages, runs the coprocessor function, and
// returns the result. The function is bound via set_coprocessor_fn; nullptr
// selects the built-in echo (passthrough self-test).
class CpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit CpuCoprocessorSession(std::string dev = "rxe0", int gid_idx = 1)
        : base_(std::move(dev), gid_idx)
    {
    }

    int connect(const ConnectInfo &info) override { return base_.connect(info); }
    MemRegion alloc_memory(std::size_t size, MemKind kind, std::uint32_t access) override
    {
        return base_.alloc_memory(size, kind, access);
    }
    PeerRef exchange_keys(const MemRegion &local) override { return base_.exchange_keys(local); }
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override
    {
        base_.establish_channel(desc, local, peer);
    }
    void start() override { base_.start(); }
    int collect(void *const *outputs, const std::uint64_t *output_bytes, std::size_t n) override
    {
        return base_.collect(outputs, output_bytes, n);
    }
    void stop() override { base_.stop(); }

    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

  private:
    class Impl : public CpuSessionBase {
      public:
        using CpuSessionBase::CpuSessionBase;
        ~Impl() { stop(); }
        CoprocessorFn coproc_fn_ = nullptr; // nullptr -> built-in echo
        void *coproc_ctx_ = nullptr;
      protected:
        void run(std::stop_token st) override;
        bool oob_listens() const override { return true; }
    };
    Impl base_;
};

} // namespace rdma::devices::cpu_libibverbs
