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

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#if __has_include("StateVectorDynamicCPU.hpp")
// device: lightning.qubit
#include "LightningSimulator.hpp"
#endif

#if __has_include("StateVectorKokkos.hpp")
// device: lightning.kokkos
#include "LightningKokkosSimulator.hpp"
#endif

#if __has_include("openqasm/OpenQasmDevice.hpp")
// device: openqasm
#include "openqasm/OpenQasmDevice.hpp"
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

class ExecutionContext final {
  private:
    using DeviceInitializer =
        std::function<std::unique_ptr<QuantumDevice>(bool, size_t, std::string)>;
    std::unordered_map<std::string_view, DeviceInitializer> _device_map{
        {"lightning.qubit",
         [](bool tape_recording, size_t shots, [[maybe_unused]] std::string hw_name) {
             return std::make_unique<Simulator::LightningSimulator>(tape_recording, shots);
         }},
    };

    static constexpr size_t _default_device_shots{1000}; // tidy: readability-magic-numbers

    // Device specifications
    std::string _name;
    bool _tape_recording;

    // ExecutionContext pointers
    std::unique_ptr<QuantumDevice> _driver_ptr{nullptr};
    std::unique_ptr<MemoryManager> _driver_mm_ptr{nullptr};

#ifdef __device_openqasm
    std::unique_ptr<Device::OpenQasm::PythonInterpreterGuard> _py_guard{nullptr};
#endif

  public:
    explicit ExecutionContext(std::string_view default_device = "lightning.qubit")
        : _name(default_device), _tape_recording(false)
    {
#ifdef __device_lightning_kokkos
        _device_map.emplace("lightning.kokkos", [](bool tape_recording, size_t shots,
                                                   [[maybe_unused]] std::string hw_name) {
            return std::make_unique<Simulator::LightningKokkosSimulator>(tape_recording, shots);
        });
#endif
#ifdef __device_openqasm
        _device_map.emplace("openqasm", [](bool tape_recording, size_t shots, std::string hw_name) {
            return std::make_unique<Device::OpenQasmDevice>(tape_recording, shots,
                                                            std::move(hw_name));
        });
#endif

        _driver_mm_ptr = std::make_unique<MemoryManager>();
    };

    ~ExecutionContext()
    {
        _driver_ptr.reset(nullptr);
        _driver_mm_ptr.reset(nullptr);

#ifdef __device_openqasm
        _py_guard.reset(nullptr);
#endif

        RT_ASSERT(getDevice() == nullptr);
        RT_ASSERT(getMemoryManager() == nullptr);
    };

    void setDeviceRecorder(bool status) noexcept { _tape_recording = status; }

    [[nodiscard]] auto getDeviceName() const -> std::string_view { return _name; }

    [[nodiscard]] auto getDeviceRecorderStatus() const -> bool { return _tape_recording; }

    [[nodiscard]] bool initDevice(std::string_view name) noexcept
    {
        std::string hw_name;
        if (_name.find("aws::braket") != std::string::npos) {
            hw_name = name;
            _name = "openqasm";
        }
        else {
            hw_name = "";
            _name = name != "default" ? name : _name;
        }

        _driver_ptr.reset(nullptr);

        auto iter = _device_map.find(_name);
        if (iter != _device_map.end()) {
            _driver_ptr = iter->second(_tape_recording, _default_device_shots, hw_name);

#ifdef __device_openqasm
            if (_name == "openqasm") {
                _py_guard = std::make_unique<Device::OpenQasm::PythonInterpreterGuard>();
            }
#endif

            return true;
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
