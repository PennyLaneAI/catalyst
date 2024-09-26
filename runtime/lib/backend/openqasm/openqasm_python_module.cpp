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
[[gnu::visibility("default")]] double expval(const char *_circuit, const char *_device,
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
                expval = result.values
            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
              )",
        py::globals(), locals);

    auto &&msg = locals["msg"].cast<std::string>();

    if (!msg.empty()) {
        throw std::runtime_error(msg);
    }

    py::list results = locals["expval"];

    return results[0].cast<double>();
}
[[gnu::visibility("default")]] void samples(const char *_circuit, const char *_device, size_t shots,
                                            size_t num_qubits, const char *_kwargs, void *_vector)
{
    namespace py = pybind11;
    py::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    std::vector<size_t> *samples = reinterpret_cast<std::vector<size_t> *>(_vector);

    using namespace py::literals;

    auto locals = py::dict("circuit"_a = circuit, "braket_device"_a = device, "kwargs"_a = kwargs,
                           "shots"_a = shots, "msg"_a = "");
    py::exec(
        R"(
            import numpy as np
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
                samples = np.array(result.measurements).flatten()
            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
              )",
        py::globals(), locals);

    auto &&msg = locals["msg"].cast<std::string>();

    if (!msg.empty()) {
        throw std::runtime_error(msg);
    }

    py::list results = locals["samples"];

    samples->reserve(shots * num_qubits);
    for (py::handle item : results) {
        samples->push_back(item.cast<size_t>());
    }

    return;
}
[[gnu::visibility("default")]] void probs(const char *_circuit, const char *_device, size_t shots,
                                          size_t num_qubits, const char *_kwargs, void *_vector)
{
    namespace py = pybind11;
    py::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    std::vector<double> *probs = reinterpret_cast<std::vector<double> *>(_vector);

    using namespace py::literals;

    auto locals = py::dict("circuit"_a = circuit, "braket_device"_a = device, "kwargs"_a = kwargs,
                           "shots"_a = shots, "num_qubits"_a = num_qubits, "msg"_a = "");

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
                probs_dict = {int(s, 2): p for s, p in result.measurement_probabilities.items()}
                probs_list = []
                for i in range(2 ** int(num_qubits)):
                    probs_list.append(probs_dict[i] if i in probs_dict else 0)
            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
              )",
        py::globals(), locals);

    auto &&msg = locals["msg"].cast<std::string>();
    if (!msg.empty()) {
        throw std::runtime_error(msg);
    }

    py::list results = locals["probs_list"];

    probs->reserve(std::pow(2, num_qubits));
    for (py::handle item : results) {
        probs->push_back(item.cast<double>());
    }

    return;
}
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

PYBIND11_MODULE(openqasm_python_module, m) { m.doc() = "openqasm"; }
