#include "CpuControllerSession.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stop_token>

#include "WireProtocol.hpp"

namespace rdma::devices::cpu_libibverbs {
using namespace rdma::devices::common;

namespace {
std::uint64_t now_ns()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
} // namespace

void CpuControllerSession::Impl::start()
{
    stop(); // drain any leftovers + reset (idempotent)
    next_send_ = 0;
    next_recv_ = 0;
    signaled_outstanding_ = 0;
    rtt_ns_ = 0;
}

void CpuControllerSession::Impl::stop()
{
    if (fwd_cq_)
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/true);
    CpuSessionBase::stop(); // no engine thread runs for the controller; harmless join
}

void CpuControllerSession::Impl::commit_work_item(std::uint32_t /*work_item_idx*/,
                                                  std::uint64_t in_bytes, std::uint64_t out_bytes)
{
    in_bytes_ = in_bytes;
    out_bytes_ = out_bytes;
}

void *CpuControllerSession::Impl::data_slot()
{
    // Current round's outbound slot: the caller writes up to in_bytes_ here, then kick()s.
    return &send_payload()->value;
}

int CpuControllerSession::Impl::kick(std::uint32_t /*work_item_idx*/)
{
    // The payload was written into data_slot() by the caller; fire one round.
    kick_ns_ = now_ns();
    const bool sig = (next_send_ % SIGNAL_EVERY == 0);
    post_write(fwd_qp_->get(), next_send_, /*inline_data=*/true, sig);
    if (sig) {
        ++signaled_outstanding_;
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/false); // lazy: free the SQ
    }
    ++next_send_;
    return 0;
}

int CpuControllerSession::Impl::collect(void *const *outputs, const std::uint64_t *output_bytes,
                                        std::size_t n)
{
    std::stop_token none; // blocking wait for this round's reply
    Payload *r = poll_message_arrival(next_recv_, none);
    if (!r)
        return -1;
    rtt_ns_ = now_ns() - kick_ns_;
    ++next_recv_;
    if (n > 0 && outputs && outputs[0]) {
        const std::size_t cap = output_bytes ? output_bytes[0] : out_bytes_;
        const std::size_t nb = std::min<std::size_t>(cap, sizeof(r->value));
        std::memcpy(outputs[0], &r->value, nb);
    }
    return 0;
}

} // namespace rdma::devices::cpu_libibverbs
