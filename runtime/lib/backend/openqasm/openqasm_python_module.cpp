// Copyright 2024 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <string.h>

extern "C" {
[[gnu::visibility("default")]] char *runCircuit(const char *_circuit, const char *_device,
                                                size_t shots, const char *_kwargs)
{
    namespace py = pybind11;
    py::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    using namespace py::literals;

    auto locals = py::dict("circuit"_a = circuit, "braket_device"_a = device, "kwargs"_a = kwargs,
                           "shots"_a = shots, "msg"_a = "");

    py::exec(
        R"(
            from braket.aws import AwsDevice
            from braket.devices import LocalSimulator
            from braket.ir.openqasm import Program as OpenQasmProgram

            try:
                if braket_device in ["default", "braket_sv", "braket_dm"]:
                    device = LocalSimulator(braket_device)
                elif "arn:aws:braket" in braket_device:
                    device = AwsDevice(braket_device)
                else:
                    raise ValueError(
                        "device must be either 'braket.devices.LocalSimulator' or 'braket.aws.AwsDevice'"
                    )
                if kwargs != "":
                    kwargs = kwargs.replace("'", "")
                    kwargs = kwargs[1:-1].split(", ") if kwargs[0] == "(" else kwargs.split(", ")
                    if len(kwargs) != 2:
                        raise ValueError(
                            "s3_destination_folder must be of size 2 with a 'bucket' and 'key' respectively."
                        )
                    result = device.run(
                        OpenQasmProgram(source=circuit),
                        shots=int(shots),
                        s3_destination_folder=tuple(kwargs),
                    ).result()
                else:
                    result = device.run(OpenQasmProgram(source=circuit), shots=int(shots)).result()
                result = str(result)
            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
              )",
        py::globals(), locals);

    auto &&msg = locals["msg"].cast<std::string>();
    if (!msg.empty()) {
        throw std::runtime_error(msg);
    }

    std::string retval = locals["result"].cast<std::string>();
    char *retptr = (char *)malloc(sizeof(char *) * retval.size() + 1);
    memcpy(retptr, retval.c_str(), sizeof(char *) * retval.size());
    return retptr;
}
}

PYBIND11_MODULE(catalyst_callback_registry, m) {}
