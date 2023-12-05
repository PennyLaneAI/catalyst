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

#include <dlfcn.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#if __has_include("pybind11/embed.h")
#include <pybind11/embed.h>
#define __build_with_pybind11
#endif

#include "Exception.hpp"
#include "QuantumDevice.hpp"

namespace Catalyst::Runtime {

/**
 * A (RAII) class for `pybind11::initialize_interpreter` and `pybind11::finalize_interpreter`.
 *
 * @note This is not copyable or movable and used in C++ tests and the ExecutionContext manager
 * of the runtime to solve the issue with re-initialization of the Python interpreter in `catch2`
 * tests which also enables the runtime to reuse the same interpreter in the scope of the global
 * quantum device unique pointer.
 *
 * @note This is only required for OpenQasmDevice and when CAPI is built with pybind11.
 */
#ifdef __build_with_pybind11
// LCOV_EXCL_START
struct PythonInterpreterGuard {
    // This ensures the guard scope to avoid Interpreter
    // conflicts with runtime calls from the frontend.
    bool _init_by_guard = false;

    PythonInterpreterGuard()
    {
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
            _init_by_guard = true;
        }
    }
    ~PythonInterpreterGuard()
    {
        if (_init_by_guard) {
            pybind11::finalize_interpreter();
        }
    }

    PythonInterpreterGuard(const PythonInterpreterGuard &) = delete;
    PythonInterpreterGuard(PythonInterpreterGuard &&) = delete;
    PythonInterpreterGuard &operator=(const PythonInterpreterGuard &) = delete;
    PythonInterpreterGuard &operator=(PythonInterpreterGuard &&) = delete;
};
// LCOV_EXCL_STOP
#else
struct PythonInterpreterGuard {
    PythonInterpreterGuard() {}
    ~PythonInterpreterGuard() {}
};
#endif

class MemoryManager final {
  private:
    std::unordered_set<void *> _impl;

  public:
    explicit MemoryManager() { _impl.reserve(1024); };

    ~MemoryManager()
    {
        for (auto allocation : _impl) {
            free(allocation);
        }
    }

    void insert(void *ptr) { _impl.insert(ptr); }
    void erase(void *ptr) { _impl.erase(ptr); }
    bool contains(void *ptr) { return _impl.contains(ptr); }
};

class SharedLibraryManager final {
  private:
    void *_handler{nullptr};

  public:
    SharedLibraryManager() = delete;
    explicit SharedLibraryManager(std::string filename)
    {
        // RTLD_DEEPBIND is incompatible with sanitizers.
        // If you have compiled this file with sanitizers and you reach this line
        // you will get an error.
        // Please re-compile without sanitizers.

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

    void *getSymbol(const std::string &symbol)
    {
        void *sym = dlsym(_handler, symbol.c_str());
        RT_FAIL_IF(!sym, dlerror());
        return sym;
    }
};

/**
 * This indicates the various stages a device can be in:
 * - `Zero`     : The device is added to the device pool and it's partially created
 *                but hasn't been initialized yet. At this stage, `_device_so_ptr == nullptr`.
 * - `Init`     : The device gets all information including device name, lib, and kwargs
 *                so that it can be initialized: `_device_so_ptr <> nullptr`.
 * - `Active`   : The device is activated, this is the stage after initializing the device
 *                showing that the `_device_ptr` points to this device. The CAPI routines
 *                have only access to the active device via `_device_ptr`.
 * - `Release`  : The device is released but not removed from the pool. The ECM can
 *                reuse this device if requested. (TODO)
 */
enum class RTDeviceStatusT : uint8_t {
    Zero = 0,
    Init,
    Active,
    Release,
};

extern "C" Catalyst::Runtime::QuantumDevice *GenericDeviceFactory(const std::string &kwargs);

/**
 * Runtime Device data-class.
 *
 * This class introduces an interface for initialized devices by the execution
 * context manager of the runtime. This includes the device name, library, kwargs,
 * and a shared pointer to the `QuantumDevice` entry point.
 */
class RTDeviceInfoT {
  private:
    std::string rtd_lib;
    std::string rtd_name;
    std::string rtd_kwargs;

    std::shared_ptr<QuantumDevice> rtd_qdevice{nullptr};

    size_t rtd_hash;
    bool tape_recording{false};

    void _complete_dylib_os_extension(std::string &rtd_lib, const std::string &name) noexcept
    {
#ifdef __linux__
        rtd_lib = "librtd_" + name + ".so";
#elif defined(__APPLE__)
        rtd_lib = "librtd_" + name + ".dylib";
#endif
    }

    void _pl2runtime_device_info(std::string &rtd_lib, std::string &rtd_name) noexcept
    {
        // The following if-elif is required for C++ tests where these backend devices
        // are linked in the interface library of the runtime. (check runtime/CMakeLists.txt)
        // Besides, this provides support for runtime device (RTD) libraries added to the system
        // path. This maintains backward compatibility for specifying a device using its name.
        // TODO: This support may need to be removed after updating the C++ unit tests.
        if (rtd_lib == "lightning.qubit" || rtd_lib == "lightning.kokkos") {
            rtd_name =
                (rtd_lib == "lightning.qubit") ? "LightningSimulator" : "LightningKokkosSimulator";
            _complete_dylib_os_extension(rtd_lib, "lightning");
        }
        else if (rtd_lib == "braket.aws.qubit" || rtd_lib == "braket.local.qubit") {
            rtd_name = "OpenQasmDevice";
            _complete_dylib_os_extension(rtd_lib, "openqasm");
        }
    }

  public:
    explicit RTDeviceInfoT(std::string _rtd_lib, std::string _rtd_name = {},
                           std::string _rtd_kwargs = {})
        : rtd_lib(std::move(_rtd_lib)), rtd_name(std::move(_rtd_name)),
          rtd_kwargs(std::move(_rtd_kwargs))
    {
        _pl2runtime_device_info(rtd_lib, rtd_name);
        rtd_hash = std::hash<std::string>{}(rtd_lib + rtd_name + rtd_kwargs);
    }

    ~RTDeviceInfoT() = default;

    [[nodiscard]] auto getHash() const -> size_t { return rtd_hash; }

    auto operator==(const RTDeviceInfoT &other) const -> bool
    {
        return this->getHash() == other.getHash();
    }

    [[nodiscard]] auto getQuantumDevicePtr() -> std::shared_ptr<QuantumDevice>
    {
        if (rtd_qdevice.get() != nullptr) {
            return rtd_qdevice;
        }

        std::unique_ptr<SharedLibraryManager> rtd_dylib =
            std::make_unique<SharedLibraryManager>(rtd_lib);
        std::string factory_name{rtd_name + "Factory"};
        void *f_ptr = rtd_dylib->getSymbol(factory_name);
        QuantumDevice *impl =
            f_ptr ? reinterpret_cast<decltype(GenericDeviceFactory) *>(f_ptr)(rtd_kwargs) : nullptr;

        rtd_qdevice.reset(impl);
        return rtd_qdevice;
    }

    void setTapeRecorderStatus(bool status) noexcept { tape_recording = status; }

    [[nodiscard]] auto getTapeRecorderStatus() const -> bool { return tape_recording; }

    [[nodiscard]] auto getDeviceInfo() const -> std::tuple<std::string, std::string, std::string>
    {
        return {rtd_lib, rtd_name, rtd_kwargs};
    }

    [[nodiscard]] auto getDeviceName() const -> const std::string & { return rtd_name; }

    friend std::ostream &operator<<(std::ostream &os, const RTDeviceInfoT &device)
    {
        os << "RTD, name: " << device.rtd_name << " lib: " << device.rtd_lib
           << " kwargs: " << device.rtd_kwargs;
        return os;
    }
};

class ExecutionContext final {
  private:
    // Device pool
    std::unordered_map<size_t, std::pair<RTDeviceStatusT, std::shared_ptr<RTDeviceInfoT>>>
        device_pool;
    size_t pool_counter{0}; // Counter for generating unique keys for devices
    std::mutex pool_mutex;  // Mutex to guard the device pool

    // Device info before initialization to support the current frontend-backend pipeline
    // TODO: remove this after the end-to-end adaptation of the async support in Catalyst
    std::string zero_rtd_kwargs{};
    std::string zero_rtd_name{};
    bool zero_rtd_tape_recorder_status;

    // ExecutionContext pointers
    std::shared_ptr<QuantumDevice> active_rtd_ptr{nullptr};
    std::unique_ptr<MemoryManager> memory_man_ptr{nullptr};
    std::unique_ptr<PythonInterpreterGuard> py_guard{nullptr};

  public:
    explicit ExecutionContext() : zero_rtd_tape_recorder_status(false)
    {
        memory_man_ptr = std::make_unique<MemoryManager>();
    };

    ~ExecutionContext()
    {
        memory_man_ptr.reset(nullptr);
        py_guard.reset(nullptr);

        RT_ASSERT(getMemoryManager() == nullptr);
    };

    void setDeviceRecorder(bool status) noexcept { zero_rtd_tape_recorder_status = status; }

    void setDeviceKwArgs(std::string_view kwargs) noexcept { zero_rtd_kwargs = kwargs; }

    void setDeviceName(std::string_view name) noexcept { zero_rtd_name = name; }

    [[nodiscard]] auto addDevice(const std::string &rtd_lib, const std::string &rtd_name = {},
                                 const std::string &rtd_kwargs = {})
        -> std::pair<size_t, std::shared_ptr<QuantumDevice>>
    {
        // Lock the mutex to protect device_pool updates
        std::lock_guard<std::mutex> lock(pool_mutex);
        size_t key = pool_counter++;
        auto device = std::make_shared<RTDeviceInfoT>(rtd_lib, rtd_name, rtd_kwargs);
        device_pool[key] = std::make_pair(RTDeviceStatusT::Init, device);
        return {key, device->getQuantumDevicePtr()};
    }

    [[nodiscard]] auto getDevice(size_t device_key) -> std::shared_ptr<QuantumDevice>
    {
        // Lock the mutex to protect device_pool info
        std::lock_guard<std::mutex> lock(pool_mutex);
        auto it = device_pool.find(device_key);
        return (it != device_pool.end()) ? it->second.second->getQuantumDevicePtr() : nullptr;
    }

    void removeDevice(size_t device_key)
    {
        // Lock the mutex to protect device_pool updates
        std::lock_guard<std::mutex> lock(pool_mutex);
        device_pool.erase(device_key);
    }

    [[nodiscard]] auto getPoolSize() -> size_t
    {
        // Lock the mutex to protect device_pool info
        std::lock_guard<std::mutex> lock(pool_mutex);
        return device_pool.size();
    }

    [[nodiscard]] auto getDeviceRecorderStatus() const -> bool
    {
        return zero_rtd_tape_recorder_status;
    }

    [[nodiscard]] bool initDevice(std::string_view rtd_lib)
    {
        auto &&[key, rtd_ptr] =
            this->addDevice(std::string(rtd_lib), zero_rtd_name, zero_rtd_kwargs);
        this->active_rtd_ptr = rtd_ptr;

#ifdef __build_with_pybind11
        if (device_pool[key].second->getDeviceName() == "OpenQasmDevice" && !Py_IsInitialized()) {
            py_guard = std::make_unique<PythonInterpreterGuard>(); // LCOV_EXCL_LINE
        }
#endif

        return true;
    }

    [[nodiscard]] auto getDevice() -> std::shared_ptr<QuantumDevice> { return active_rtd_ptr; }

    [[nodiscard]] auto getMemoryManager() const -> const std::unique_ptr<MemoryManager> &
    {
        return memory_man_ptr;
    }
};
} // namespace Catalyst::Runtime
