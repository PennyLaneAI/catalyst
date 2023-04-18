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

// device: lightning.qubit
#include "LightningSimulator.hpp"

#if __has_include("StateVectorKokkos.hpp")
// device: lightning.kokkos
#include "LightningKokkosSimulator.hpp"
#endif

namespace Catalyst::Runtime::CAPI {

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
};

class Driver final {
  private:
    using DeviceInitializer = std::function<std::unique_ptr<QuantumDevice>(bool, size_t)>;
    std::unordered_map<std::string_view, DeviceInitializer> _device_map{
        {"lightning.qubit",
         [](bool tape_recording, size_t shots) {
             return std::make_unique<Catalyst::Runtime::Simulator::LightningSimulator>(
                 tape_recording, shots);
         }},
    };

    // Device Info
    std::string _name;
    bool _tape_recording;
    size_t _shots;

    // Driver pointers
    std::unique_ptr<Catalyst::Runtime::QuantumDevice> _driver_ptr = nullptr;
    std::unique_ptr<MemoryManager> _driver_mm_ptr = nullptr;

  public:
    explicit Driver(std::string_view default_device, bool status, size_t shots)
        : _name(default_device), _tape_recording(status), _shots(shots)
    {
#ifdef __device_lightning_kokkos
        _device_map.emplace("lightning.kokkos", [](bool tape_recording, size_t shots) {
            return std::make_unique<Catalyst::Runtime::Simulator::LightningKokkosSimulator>(
                tape_recording, shots);
        });
#endif
        this->_driver_mm_ptr = std::make_unique<MemoryManager>();
    };

    ~Driver()
    {
        _driver_ptr.reset(nullptr);
        _driver_mm_ptr.reset(nullptr);

        RT_ASSERT(getDevice() == nullptr);
        RT_ASSERT(getMemoryManager() == nullptr);
    };

    void setDeviceName(std::string_view name) noexcept
    {
        if (name != "default") {
            this->_name = name;
        }
    }

    void setDeviceShots(size_t shots) noexcept { this->_shots = shots; }

    void toggleDeviceRecorder(bool status) noexcept { this->_tape_recording = status; }

    [[nodiscard]] auto getDeviceName() const -> std::string_view { return this->_name; }

    [[nodiscard]] auto getDeviceShots() const -> size_t { return this->_shots; }

    [[nodiscard]] auto getDeviceRecorderStatus() const -> bool { return this->_tape_recording; }

    [[nodiscard]] bool initDevice() noexcept
    {
        this->_driver_ptr.reset(nullptr);

        auto iter = _device_map.find(this->_name);
        if (iter != _device_map.end()) {
            this->_driver_ptr = iter->second(_tape_recording, _shots);
            return true;
        }
        return false;
    }

    [[nodiscard]] auto getDevice() const
        -> const std::unique_ptr<Catalyst::Runtime::QuantumDevice> &
    {
        return _driver_ptr;
    }

    [[nodiscard]] auto getMemoryManager() const -> const std::unique_ptr<MemoryManager> &
    {
        return _driver_mm_ptr;
    }
};
} // namespace Catalyst::Runtime::CAPI
