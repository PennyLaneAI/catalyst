#pragma once
#include <cstdint>
#include <string>

#include <infiniband/verbs.h>

namespace rdma::devices::common {
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
     * @brief Opens an RDMA device by its system name (e.g., "mlx5_0").
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
} // namespace rdma::devices::common
