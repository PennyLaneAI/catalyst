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
#include <memory>

#include "Context.hpp"

#include <infiniband/verbs.h>

namespace catalyst::transport::common {

/**
 * @class CompletionQueue class.
 *
 * @brief An RAII wrapper for an RDMA Completion Queue (`ibv_cq`).
 */
class CompletionQueue {
  public:
    CompletionQueue() = delete;
    CompletionQueue(std::shared_ptr<Context> ctx, int depth);
    ~CompletionQueue();
    CompletionQueue(const CompletionQueue &) = delete;
    CompletionQueue &operator=(const CompletionQueue &) = delete;
    ibv_cq *get() const;

  private:
    std::shared_ptr<Context> ctx_;
    ibv_cq *cq_ = nullptr;
};
} // namespace catalyst::transport::common
