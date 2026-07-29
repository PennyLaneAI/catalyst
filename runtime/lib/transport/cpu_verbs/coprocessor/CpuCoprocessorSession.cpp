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

#include "CpuCoprocessorSession.hpp"

#include <cstring>
#include <thread>

#include "WireProtocol.hpp"

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport::common;

void CpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx)
{
    base_.coproc_fn_ = fn;
    base_.coproc_ctx_ = ctx;
}

// Coprocessor: wait for a message, run the coprocessor function into the send
// buffer in place, then send the result. A null fn is the built-in echo
// (passthrough). Replies are inline + selectively signaled; the bwd CQ is
// reaped in batches at signal points.
void CpuCoprocessorSession::Impl::run(std::stop_token st)
{
    int signaled_outstanding = 0;
    for (std::uint64_t c = 0; !st.stop_requested(); c++) {
        Payload *r = poll_message_arrival(c, st); // the incoming message
        if (!r) {
            reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/true);
            return;
        }
        last_word_.store(r->value, std::memory_order_relaxed);
        completed_.fetch_add(1, std::memory_order_release);
        Payload *send = send_payload();
        send->value = 0; // deterministic high bytes when the result is shorter
        if (coproc_fn_) {
            coproc_fn_(&r->value, sizeof(r->value), &send->value, sizeof(send->value), coproc_ctx_);
        }
        else {
            send->value = r->value; // built-in echo
        }
        const bool sig = (c % SIGNAL_EVERY == 0);
        post_write(bwd_qp_->get(), c, /*inline_data=*/true, sig);
        if (sig) {
            ++signaled_outstanding;
            reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/false);
        }
    }
    reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/true);
}

} // namespace catalyst::transport::cpu_verbs
