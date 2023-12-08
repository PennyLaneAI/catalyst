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
#include <string>
#include <string_view>
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

extern "C" Catalyst::Runtime::QuantumDevice *GenericDeviceFactory(const char *kwargs);

class ExecutionContext final {
  private:
    using DeviceInitializer =
        std::function<std::unique_ptr<QuantumDevice>(bool, const std::string &)>;

    // Device specifications
    std::string _device_kwargs{};
    std::string _device_name;
    bool _tape_recording;

    // ExecutionContext pointers
    std::unique_ptr<QuantumDevice> _driver_ptr{nullptr};
    std::unique_ptr<MemoryManager> _driver_mm_ptr{nullptr};
    std::unique_ptr<SharedLibraryManager> _driver_so_ptr{nullptr};
    std::unique_ptr<PythonInterpreterGuard> _py_guard{nullptr};

    // Helper methods
    void update_device_pl2runtime(std::string &pl_name, const std::string &rtd_name)
    {
#ifdef __linux__
        pl_name = "librtd_" + rtd_name + ".so";
#elif defined(__APPLE__)
        pl_name = "librtd_" + rtd_name + ".dylib";
#endif
    }

  public:
    explicit ExecutionContext(std::string_view default_device = "lightning.qubit")
        : _device_name(default_device), _tape_recording(false)
    {
        _driver_mm_ptr = std::make_unique<MemoryManager>();
    };

    ~ExecutionContext()
    {
        _driver_ptr.reset(nullptr);
        _driver_mm_ptr.reset(nullptr);
        _driver_so_ptr.reset(nullptr);
        _py_guard.reset(nullptr);

        RT_ASSERT(getDevice() == nullptr);
        RT_ASSERT(getMemoryManager() == nullptr);
    };

    void setDeviceRecorder(bool status) noexcept { _tape_recording = status; }

    void setDeviceKwArgs(std::string_view kwargs) noexcept { _device_kwargs = kwargs; }

    void setDeviceName(std::string_view name) noexcept { _device_name = name; }

    [[nodiscard]] auto getDeviceName() const -> std::string_view { return _device_name; }

    [[nodiscard]] auto getDeviceKwArgs() const -> std::string { return _device_kwargs; }

    [[nodiscard]] auto getDeviceRecorderStatus() const -> bool { return _tape_recording; }

    [[nodiscard]] QuantumDevice *loadDevice(std::string filename)
    {
        _driver_so_ptr = std::make_unique<SharedLibraryManager>(filename);
        std::string factory_name{_device_name + "Factory"};
        void *f_ptr = _driver_so_ptr->getSymbol(factory_name);
        return f_ptr ? reinterpret_cast<decltype(GenericDeviceFactory) *>(f_ptr)(
                           _device_kwargs.c_str())
                     : nullptr;
    }

    [[nodiscard]] bool initDevice(std::string_view rtd_lib)
    {
        // Reset the driver pointer
        _driver_ptr.reset(nullptr);

        // Convert to a mutable string
        std::string rtd_lib_str{rtd_lib};

        // LCOV_EXCL_START
        // The following if-elif is required for C++ tests where these backend devices
        // are linked in the interface library of the runtime. (check runtime/CMakeLists.txt)
        // Besides, this provides support for runtime device (RTD) libraries added to the system
        // path. This maintains backward compatibility for specifying a device using its name.
        // TODO: This support may need to be removed after updating the C++ unit tests.
        if (rtd_lib == "lightning.qubit" || rtd_lib == "lightning.kokkos") {
            _device_name =
                (rtd_lib == "lightning.qubit") ? "LightningSimulator" : "LightningKokkosSimulator";
            update_device_pl2runtime(rtd_lib_str, "lightning");
        }
        else if (rtd_lib == "braket.aws.qubit" || rtd_lib == "braket.local.qubit") {
            _device_name = "OpenQasmDevice";
            update_device_pl2runtime(rtd_lib_str, "openqasm");
        }
        // LCOV_EXCL_STOP

#ifdef __build_with_pybind11
        if (_device_name == "OpenQasmDevice" && !Py_IsInitialized()) {
            _py_guard = std::make_unique<PythonInterpreterGuard>(); // LCOV_EXCL_LINE
        }
#endif

        QuantumDevice *impl = loadDevice(rtd_lib_str);
        _driver_ptr.reset(impl);
        return true;
    }

    [[nodiscard]] auto getDevice() const -> const std::unique_ptr<QuantumDevice> &
    {
        return _driver_ptr;
    }

    [[nodiscard]] auto getMemoryManager() const -> const std::unique_ptr<MemoryManager> &
    {
        return _driver_mm_ptr;
    }
};
} // namespace Catalyst::Runtime
