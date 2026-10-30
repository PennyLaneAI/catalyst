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

#include "ProtectionDomain.hpp"

#include <utility>

#include "Error.hpp"

namespace catalyst::transport::common {
ProtectionDomain::ProtectionDomain(std::shared_ptr<Context> ctx) : ctx_(std::move(ctx)) {
    pd_ = ibv_alloc_pd(ctx_->get());
    TP_CHECK_ERRNO(pd_, "ibv_alloc_pd");
}
ProtectionDomain::~ProtectionDomain() {
    if (pd_) {
        ibv_dealloc_pd(pd_);
    }
}
ibv_pd *ProtectionDomain::get() const { return pd_; }
} // namespace catalyst::transport::common
