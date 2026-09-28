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
#include <cstdint>
#include <string>

#include <infiniband/verbs.h>

namespace catalyst::transport::common {
/**
 * @class Context
 * @brief RAII wrapper for an RDMA device context (`ibv_context`).
 *
 * Manages unique ownership of a physical NIC handle. Exposes network
 * attributes, port status, and GID table properties.
 */
class Context {
  public:
    /**
     * @brief Opens an RDMA device by its system name.
     */
    explicit Context(const std::string &dev_name); // open by name

    /**
     * @brief Closes the physical RDMA device handle.
     */
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    /**
     * @brief Returns the raw underlying device context pointer.
     */
    ibv_context *get() const;

    /**
     * @brief Accesses hardware attributes for a specific physical port.
     */
    ibv_port_attr port_attr(std::uint8_t port) const;

    /**
     * @brief Retrieves a Global Identifier (GID) from the device table.
     */
    ibv_gid gid(std::uint8_t port, int idx) const;

  private:
    ibv_context *ctx_ = nullptr;
};
} // namespace catalyst::transport::common
