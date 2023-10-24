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

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#if __has_include("LightningSimulator.hpp")
// device: lightning.qubit
#include "LightningSimulator.hpp"
#endif

#if __has_include("LightningKokkosSimulator.hpp")
// device: lightning.kokkos
#include "LightningKokkosSimulator.hpp"
#endif

#if __has_include("OpenQasmDevice.hpp")
// device: openqasm
#include "OpenQasmDevice.hpp"
#include <pybind11/embed.h>
#endif

namespace Catalyst::Runtime {

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
    void *_handler{NULL};

  public:
    SharedLibraryManager() = delete;
    SharedLibraryManager(std::string filename)
    {
        _handler = dlopen(filename.c_str(), RTLD_LAZY | RTLD_DEEPBIND);
        if (!_handler) {
            char *error_msg = dlerror();
            throw RuntimeException(error_msg);
        }
    }

    ~SharedLibraryManager()
    {
        // dlopen and dlclose increment and decrement reference counters.
        // Since we have a guaranteed _handler in a valid SharedLibraryManager instance
        // then we don't really need to worry about dlclose.
        // In other words, there is a one to one correspondance between an instance
        // of SharedLibraryManager and an increase in the reference count for the dynamic library.
        // dlclose returns non-zero on error.
        //
        // Errors in dlclose are implementation dependent.
        // There are two possible errors during dlclose in glibc: "shared object not open"
        // and "cannot create scope list". Look for _dl_signal_error in:
        //
        //     codebrowser.dev/glibc/glibc/elf/dl-close.c.html/
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

    void *getSymbol(std::string symbol)
    {
        void *sym = dlsym(_handler, symbol.c_str());
        if (!sym) {
            char *error_msg = dlerror();
            throw RuntimeException(error_msg);
        }
        return sym;
    }
};

extern "C" Catalyst::Runtime::QuantumDevice *getCustomDevice();

class ExecutionContext final {
  private:
    using DeviceInitializer =
        std::function<std::unique_ptr<QuantumDevice>(bool, const std::string &)>;
    std::unordered_map<std::string_view, DeviceInitializer> _device_map{
        {"lightning.qubit",
         [](bool tape_recording, const std::string &device_kwargs) {
             return std::make_unique<Simulator::LightningSimulator>(tape_recording, device_kwargs);
         }},
    };

    // Device specifications
    std::string _device_kwargs{};
    size_t _device_shots{1000};

    std::string _device_name;
    bool _tape_recording;

    // ExecutionContext pointers
    std::unique_ptr<QuantumDevice> _driver_ptr{nullptr};
    std::unique_ptr<MemoryManager> _driver_mm_ptr{nullptr};
    std::unique_ptr<SharedLibraryManager> _driver_so_ptr{nullptr};

#ifdef __device_openqasm
    std::unique_ptr<Device::OpenQasm::PythonInterpreterGuard> _py_guard{nullptr};
#endif

  public:
    explicit ExecutionContext(std::string_view default_device = "lightning.qubit")
        : _device_name(default_device), _tape_recording(false)
    {
#ifdef __device_lightning_kokkos
        _device_map.emplace("lightning.kokkos",
                            [](bool tape_recording, const std::string &device_kwargs) {
                                return std::make_unique<Simulator::LightningKokkosSimulator>(
                                    tape_recording, device_kwargs);
                            });
#endif
#ifdef __device_openqasm
        _device_map.emplace("openqasm", [](bool tape_recording, const std::string &device_kwargs) {
            return std::make_unique<Device::OpenQasmDevice>(tape_recording, device_kwargs);
        });
#endif
        _driver_mm_ptr = std::make_unique<MemoryManager>();
    };

    ~ExecutionContext()
    {
        _driver_ptr.reset(nullptr);
        _driver_mm_ptr.reset(nullptr);
        _driver_so_ptr.reset(nullptr);

#ifdef __device_openqasm
        _py_guard.reset(nullptr);
#endif

        RT_ASSERT(getDevice() == nullptr);
        RT_ASSERT(getMemoryManager() == nullptr);
    };

    void setDeviceRecorder(bool status) noexcept { _tape_recording = status; }

    void setDeviceKwArgs(std::string_view info) noexcept { _device_kwargs = info; }

    [[nodiscard]] auto getDeviceName() const -> std::string_view { return _device_name; }

    [[nodiscard]] auto getDeviceKwArgs() const -> std::string { return _device_kwargs; }

    [[nodiscard]] auto getDeviceRecorderStatus() const -> bool { return _tape_recording; }

    [[nodiscard]] QuantumDevice *loadDevice(std::string filename)
    {
        _driver_so_ptr = std::make_unique<SharedLibraryManager>(filename);
        void *f_ptr = _driver_so_ptr->getSymbol("getCustomDevice");
        return f_ptr ? reinterpret_cast<decltype(getCustomDevice) *>(f_ptr)() : NULL;
    }

    [[nodiscard]] bool initDevice(std::string_view name)
    {
        if (name != "default") {
            _device_name = name;
        }

        if (_device_name == "braket.aws.qubit" || _device_name == "braket.local.qubit") {
            _device_kwargs = "device_type : " + _device_name + "," + _device_kwargs;
            _device_name = "openqasm";
        }

        _driver_ptr.reset(nullptr);

        auto iter = _device_map.find(_device_name);
        if (iter != _device_map.end()) {
            _driver_ptr = iter->second(_tape_recording, _device_kwargs);

#ifdef __device_openqasm
            if (_device_name == "openqasm" && !Py_IsInitialized()) {
                _py_guard =
                    std::make_unique<Device::OpenQasm::PythonInterpreterGuard>(); // LCOV_EXCL_LINE
            }
#endif

            return true;
        }

        try {
            // TODO: Once all devices are shared libraries, they all need to be loaded.
            // During this transition period, there are several ways in which we can do this.
            // This try catch is just for allowing the previous mechanism to still succeed
            // while keeping the implementation of SharedLibraryManager as a minimal as possible.
            // Once all devices are shared libraries, we can replace initDevice with loadDevice.
            //
            // Yes, I know there is a performance impact. But this try-catch will be removed once
            // all devices are shared libraries.
            QuantumDevice *impl = loadDevice(std::string(name));
            _driver_ptr.reset(impl);
            return true;
        }
        catch (RuntimeException &e) {
            // fall-through
        }

        return false;
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
