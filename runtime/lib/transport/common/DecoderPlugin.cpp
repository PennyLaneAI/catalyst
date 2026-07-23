#include "DecoderPlugin.hpp"

#include <dlfcn.h>
#include <utility>

#include "Error.hpp"

namespace rdma::devices::common {

DecoderPlugin::DecoderPlugin(const std::string &path)
{
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    RDMA_CHECK(handle_, "dlopen(%s): %s", path.c_str(), dlerror());
    fn_ = reinterpret_cast<Fn>(dlsym(handle_, "decode"));
    if (!fn_) {
        const char *e = dlerror();
        dlclose(handle_);
        handle_ = nullptr;
        RDMA_FAIL("decoder %s missing 'decode' symbol: %s", path.c_str(), e ? e : "?");
    }
    // Optional ctx lifecycle (a pure decoder omits both).
    auto create = reinterpret_cast<void *(*)()>(dlsym(handle_, "decoder_create"));
    destroy_ = reinterpret_cast<void (*)(void *)>(dlsym(handle_, "decoder_destroy"));
    if (create) {
        ctx_ = create();
    }
}

void DecoderPlugin::reset() noexcept
{
    if (destroy_ && ctx_) {
        destroy_(ctx_);
    }
    if (handle_) {
        dlclose(handle_);
    }
    handle_ = nullptr;
    fn_ = nullptr;
    ctx_ = nullptr;
    destroy_ = nullptr;
}

DecoderPlugin::~DecoderPlugin() { reset(); }

DecoderPlugin::DecoderPlugin(DecoderPlugin &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), fn_(std::exchange(other.fn_, nullptr)),
      ctx_(std::exchange(other.ctx_, nullptr)), destroy_(std::exchange(other.destroy_, nullptr))
{
}

DecoderPlugin &DecoderPlugin::operator=(DecoderPlugin &&other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        fn_ = std::exchange(other.fn_, nullptr);
        ctx_ = std::exchange(other.ctx_, nullptr);
        destroy_ = std::exchange(other.destroy_, nullptr);
    }
    return *this;
}

} // namespace rdma::devices::common
