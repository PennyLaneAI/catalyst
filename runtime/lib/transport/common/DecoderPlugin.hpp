#pragma once
#include <cstddef>
#include <string>

namespace rdma::devices::common {

/**
 * @class DecoderPlugin
 * @brief RAII manager for a dynamically loaded decoder shared library.
 *
 * Handles the automatic loading (@c dlopen), symbol resolution, optional
 * context lifecycle (@c decoder_create / @c decoder_destroy), and resource
 * cleanup upon destruction or movement.
 */
class DecoderPlugin {
  public:
    /**
     * @brief Function pointer signature for the decoding operation.
     */
    using Fn = void (*)(void *ctx, const void *in, std::size_t in_len, void *out,
                        std::size_t out_len);

    /**
     * @brief Loads the shared library and resolves the decoder symbols.
     * @param path The filesystem path to the shared library (.so).
     * @throw Throw runtime errors if loading or symbol resolution fails.
     */
    explicit DecoderPlugin(const std::string &path); // dlopen + resolve; throws on failure

    /**
     * @brief Destructor. Automatically releases context and unloads the
     * library.
     */
    ~DecoderPlugin();
    DecoderPlugin(DecoderPlugin &&o) noexcept;
    DecoderPlugin &operator=(DecoderPlugin &&o) noexcept;
    DecoderPlugin(const DecoderPlugin &) = delete;
    DecoderPlugin &operator=(const DecoderPlugin &) = delete;

    /**
     * @brief Retrieves the resolved decoding function pointer.
     * @return The decoding function pointer, or @c nullptr if uninitialized.
     */
    Fn fn() const noexcept { return fn_; }

    /**
     * @brief Retrieves the optional plugin context pointer.
     * @return Pointer to the internal context instance, or @c nullptr if none
     * exists.
     */
    void *ctx() const noexcept { return ctx_; }

  private:
    /**
     * @brief Safely releases all held resources and resets pointers to @c
     * nullptr.
     */
    void reset() noexcept;
    void *handle_ = nullptr; // Dynamic library handle returned by dlopen.
    Fn fn_ = nullptr;        // Function pointer to the 'decode' symbol.
    void *ctx_ = nullptr;    // Optional plugin context instance.
    void (*destroy_)(void *) = nullptr;
};

} // namespace rdma::devices::common
