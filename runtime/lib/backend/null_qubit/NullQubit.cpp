#include "NullQubit.hpp"

#include "QuantumDevice.hpp"

GENERATE_DEVICE_FACTORY(NullQubit, Catalyst::Runtime::Devices::NullQubit);

// Optional device extensions for classical control, looked up by the runtime in the device
// library. The runtime only calls them with the device created by `NullQubitFactory`.
extern "C" int64_t NullQubitMeasurementToken(Catalyst::Runtime::QuantumDevice *device) {
    return static_cast<Catalyst::Runtime::Devices::NullQubit *>(device)->GetMeasurementToken();
}

extern "C" void NullQubitBeginClassicalControl(Catalyst::Runtime::QuantumDevice *device,
                                               int64_t token) {
    static_cast<Catalyst::Runtime::Devices::NullQubit *>(device)->BeginClassicalControl(token);
}

extern "C" void NullQubitEndClassicalControl(Catalyst::Runtime::QuantumDevice *device) {
    static_cast<Catalyst::Runtime::Devices::NullQubit *>(device)->EndClassicalControl();
}
