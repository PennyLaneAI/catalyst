// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include <dlfcn.h>

#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "Exception.hpp"
#include "QuantumDevice.hpp"
#include "Types.h"

extern void callbackCall(int64_t, int64_t, int64_t, va_list);

namespace Catalyst::Runtime {

extern "C" void __catalyst_inactive_callback(int64_t identifier, int64_t argc, int64_t retc, ...);

class MemoryManager // NOLINT(cppcoreguidelines-special-member-functions,
                    // hicpp-special-member-functions)
    final {
  private:
    std::unordered_set<void *> _impl;
    std::mutex mu; // To guard the memory manager

  public:
    explicit MemoryManager() { _impl.reserve(1024); };

    ~MemoryManager()
    {
        // Lock the mutex to protect _impl free
        std::lock_guard<std::mutex> lock(mu);
        for (auto *allocation : _impl) {
            free(allocation); // NOLINT(cppcoreguidelines-no-malloc, hicpp-no-malloc)
        }
    }

    void insert(void *ptr)
    {
        // Lock the mutex to protect _impl update
        std::lock_guard<std::mutex> lock(mu);
        _impl.insert(ptr);
    }
    void erase(void *ptr)
    {
        // Lock the mutex to protect _impl update
        std::lock_guard<std::mutex> lock(mu);
        _impl.erase(ptr);
    }
    bool contains(void *ptr)
    {
        // Lock the mutex to protect _impl update
        std::lock_guard<std::mutex> lock(mu);
        return _impl.contains(ptr);
    }
};

class SharedLibraryManager final {
  private:
    void *_handler{nullptr};

  public:
    SharedLibraryManager() = delete;
    explicit SharedLibraryManager(const std::string &filename)
    {
#ifdef __APPLE__
        auto rtld_flags = RTLD_LAZY;
#else
        // Closing the dynamic library of Lightning simulators with dlclose() where OpenMP
        // directives (in Lightning simulators) are in use would raise memory segfaults.
        // Note that we use RTLD_NODELETE as a workaround to fix the issue.
        auto rtld_flags = RTLD_LAZY | RTLD_NODELETE;
#endif

        _handler = dlopen(filename.c_str(), rtld_flags);
        RT_FAIL_IF(!_handler, dlerror());
    }

    ~SharedLibraryManager()
    {
        // dlopen and dlclose increment and decrement reference counters.
        // Since we have a guaranteed _handler in a valid SharedLibraryManager instance
        // then we don't really need to worry about dlclose.
        // In other words, there is an one to one correspondence between an instance
        // of SharedLibraryManager and an increase in the reference count for the dynamic library.
        // dlclose returns non-zero on error.
        //
        // Errors in dlclose are implementation dependent.
        // There are two possible errors during dlclose in glibc: "shared object not open"
        // and "cannot create scope list". Look for _dl_signal_error in:
        //
        //     https://codebrowser.dev/glibc/glibc/elf/dl-close.c.html
        //
        // This means that at the very least, one could trigger an error in the following line by
        // doing the following: dlopen the same library and closing it multiple times in a different
        // location.
        //
        // This would mean that the reference count would be less than the number of instances
        // of SharedLibraryManager.
        //
        // There really is no way to protect against this error, except to always use
        // SharedLibraryManager to manage shared libraries.
        //
        // Exercise for the reader, how could one trigger the "cannot create scope list" error?
        dlclose(_handler);
    }

    SharedLibraryManager(const SharedLibraryManager &other) = delete;
    SharedLibraryManager &operator=(const SharedLibraryManager &other) = delete;
    SharedLibraryManager(SharedLibraryManager &&other) = delete;
    SharedLibraryManager &operator=(SharedLibraryManager &&other) = delete;

    void *getSymbol(const std::string &symbol)
    {
        void *sym = dlsym(_handler, symbol.c_str());
        RT_FAIL_IF(!sym, dlerror());
        return sym;
    }
};

/**
 * This indicates the various stages a device can be in:
 * - `Active`   : The device is added to the device pool and the `ExecutionContext` device pointer
 *                (`RTD_PTR`) points to this device instance. The CAPI routines have only access to
 *                one single active device per thread via `RTD_PTR`.
 * - `Inactive`  : The device is deactivated meaning `RTD_PTR` does not point to this device.
 *                 The device is not removed from the pool, allowing the `ExecutionContext` manager
 *                 to reuse this device in a multi-qnode workflow when another device with identical
 *                 specifications is requested.
 */
enum class RTDeviceStatus : uint8_t {
    Active = 0,
    Inactive,
};

extern "C" Catalyst::Runtime::QuantumDevice *GenericDeviceFactory(const char *kwargs);

/**
 * Runtime Device data-class.
 *
 * This class introduces an interface for constructed devices by the `ExecutionContext`
 * manager. This includes the device name, library, kwargs, and a shared pointer to the
 * `QuantumDevice` entry point.
 */
class RTDevice {
  private:
    std::string rtd_lib;
    std::string rtd_name;
    std::string rtd_kwargs;

    std::unique_ptr<SharedLibraryManager> rtd_dylib{nullptr};
    std::unique_ptr<QuantumDevice> rtd_qdevice{nullptr};

    RTDeviceStatus status{RTDeviceStatus::Inactive};

    static void _complete_dylib_os_extension(std::string &rtd_lib, const std::string &name) noexcept
    {
#ifdef __linux__
        rtd_lib = "librtd_" + name + ".so";
#elif defined(__APPLE__)
        rtd_lib = "librtd_" + name + ".dylib";
#endif
    }

    static void _pl2runtime_device_info(std::string &rtd_lib, std::string &rtd_name) noexcept
    {
        // The following if-elif is required for C++ tests where these backend devices
        // are linked in the interface library of the runtime. (check runtime/CMakeLists.txt)
        // Besides, this provides support for runtime device (RTD) libraries added to the system
        // path. This maintains backward compatibility for specifying a device using its name.
        // TODO: This support may need to be removed after updating the C++ unit tests.
        if (rtd_lib == "null.qubit") {
            rtd_name = "NullQubit";
            _complete_dylib_os_extension(rtd_lib, "null_qubit");
        }
        else if (rtd_lib == "lightning.qubit") {
            rtd_name = "LightningSimulator";
            _complete_dylib_os_extension(rtd_lib, "lightning");
        }
        else if (rtd_lib == "braket.aws.qubit" || rtd_lib == "braket.local.qubit") {
            rtd_name = "OpenQasmDevice";
            _complete_dylib_os_extension(rtd_lib, "openqasm");
        }
    }

  public:
    explicit RTDevice(std::string _rtd_lib, std::string _rtd_name = {},
                      std::string _rtd_kwargs = {})
        : rtd_lib(std::move(_rtd_lib)), rtd_name(std::move(_rtd_name)),
          rtd_kwargs(std::move(_rtd_kwargs))
    {
        _pl2runtime_device_info(rtd_lib, rtd_name);
    }

    explicit RTDevice(std::string_view _rtd_lib, std::string_view _rtd_name,
                      std::string_view _rtd_kwargs)
        : rtd_lib(_rtd_lib), rtd_name(_rtd_name), rtd_kwargs(_rtd_kwargs)
    {
        _pl2runtime_device_info(rtd_lib, rtd_name);
    }

    ~RTDevice() = default;
    RTDevice(const RTDevice &other) = delete;
    RTDevice &operator=(const RTDevice &other) = delete;
    RTDevice(RTDevice &&other) = delete;
    RTDevice &operator=(RTDevice &&other) = delete;

    auto operator==(const RTDevice &other) const -> bool
    {
        return (this->rtd_lib == other.rtd_lib && this->rtd_name == other.rtd_name) &&
               this->rtd_kwargs == other.rtd_kwargs;
    }

    [[nodiscard]] auto getQuantumDevicePtr() -> const std::unique_ptr<QuantumDevice> &
    {
        if (rtd_qdevice) {
            return rtd_qdevice;
        }

        rtd_dylib = std::make_unique<SharedLibraryManager>(rtd_lib);
        std::string factory_name{rtd_name + "Factory"};
        void *f_ptr = rtd_dylib->getSymbol(factory_name);
        rtd_qdevice = std::unique_ptr<QuantumDevice>(
            (f_ptr != nullptr)
                ? reinterpret_cast<decltype(GenericDeviceFactory) *>(f_ptr)(rtd_kwargs.c_str())
                : nullptr);
        return rtd_qdevice;
    }

    [[nodiscard]] auto getDeviceInfo() const -> std::tuple<std::string, std::string, std::string>
    {
        return {rtd_lib, rtd_name, rtd_kwargs};
    }

    [[nodiscard]] auto getDeviceName() const -> const std::string & { return rtd_name; }

    void setDeviceStatus(RTDeviceStatus new_status) noexcept { status = new_status; }

    [[nodiscard]] auto getDeviceStatus() const -> RTDeviceStatus { return status; }

    friend std::ostream &operator<<(std::ostream &os, const RTDevice &device)
    {
        os << "RTD, name: " << device.rtd_name << " lib: " << device.rtd_lib
           << " kwargs: " << device.rtd_kwargs;
        return os;
    }
};

class ExecutionContext final {
  private:
    // Device pool
    std::vector<std::shared_ptr<RTDevice>> device_pool;
    std::mutex pool_mu; // To protect device_pool

    bool initial_tape_recorder_status{false};

    // ExecutionContext pointers
    std::unique_ptr<MemoryManager> memory_man_ptr{nullptr};

    // PRNG
    uint32_t *seed;
    std::mt19937 gen;

  public:
    explicit ExecutionContext(uint32_t *seed = nullptr) : seed(seed)
    {
        memory_man_ptr = std::make_unique<MemoryManager>();

        if (this->seed != nullptr) {
            this->gen = std::mt19937(*seed);
        }
    }

    ~ExecutionContext() = default;
    ExecutionContext(const ExecutionContext &other) = delete;
    ExecutionContext &operator=(const ExecutionContext &other) = delete;
    ExecutionContext(ExecutionContext &&other) = delete;
    ExecutionContext &operator=(ExecutionContext &&other) = delete;

    void setDeviceRecorderStatus(bool status) noexcept { initial_tape_recorder_status = status; }

    [[nodiscard]] auto getDeviceRecorderStatus() const -> bool
    {
        return initial_tape_recorder_status;
    }

    [[nodiscard]] auto getMemoryManager() const -> const std::unique_ptr<MemoryManager> &
    {
        return memory_man_ptr;
    }

    [[nodiscard]] auto getOrCreateDevice(std::string_view rtd_lib, std::string_view rtd_name,
                                         std::string_view rtd_kwargs)
        -> const std::shared_ptr<RTDevice> &
    {
        std::lock_guard<std::mutex> lock(pool_mu);

        auto device = std::make_shared<RTDevice>(rtd_lib, rtd_name, rtd_kwargs);

        const size_t key = device_pool.size();
        for (size_t i = 0; i < key; i++) {
            if (device_pool[i]->getDeviceStatus() == RTDeviceStatus::Inactive &&
                *device_pool[i] == *device) {
                device_pool[i]->setDeviceStatus(RTDeviceStatus::Active);
                return device_pool[i];
            }
        }

        RT_ASSERT(device->getQuantumDevicePtr());

        // Add a new device
        device->setDeviceStatus(RTDeviceStatus::Active);
        if (this->seed != nullptr) {
            device->getQuantumDevicePtr()->SetDevicePRNG(&(this->gen));
        }
        else {
            device->getQuantumDevicePtr()->SetDevicePRNG(nullptr);
        }
        device_pool.push_back(device);

        return device_pool[key];
    }

    [[nodiscard]] auto getOrCreateDevice(const std::string &rtd_lib,
                                         const std::string &rtd_name = {},
                                         const std::string &rtd_kwargs = {})
        -> const std::shared_ptr<RTDevice> &
    {
        return getOrCreateDevice(std::string_view{rtd_lib}, std::string_view{rtd_name},
                                 std::string_view{rtd_kwargs});
    }

    [[nodiscard]] auto getDevice(size_t device_key) -> const std::shared_ptr<RTDevice> &
    {
        std::lock_guard<std::mutex> lock(pool_mu);
        RT_FAIL_IF(device_key >= device_pool.size(), "Invalid device_key");
        return device_pool[device_key];
    }

    void deactivateDevice(RTDevice *RTD_PTR)
    {
        std::lock_guard<std::mutex> lock(pool_mu);
        RTD_PTR->setDeviceStatus(RTDeviceStatus::Inactive);
    }
};
} // namespace Catalyst::Runtime
