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

std::string program = R"(
import numpy as np
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program as OpenQasmProgram

def py_sanitize_device(user_submitted_device):
    if user_submitted_device in {"default", "braket_sv", "braket_dm"}:
        return LocalSimulator(user_submitted_device)
    elif "arn:aws:braket" in user_submitted_device:
        return AwsDevice(user_submitted_device)
    
    msg = "device must be either 'braket.devices.LocalSimulator' or 'braket.aws.AwsDevice'" 
    raise ValueError(msg)

def py_sanitize_kwargs(user_submitted_kwargs):
    if user_submitted_kwargs == "":
        return None
    kwargs = user_submitted_kwargs.replace("'", "")
    kwargs = kwargs[1:-1].split(", ") if kwargs[0] == "(" else kwargs.split(", ")
    if len(kwargs) != 2:
        msg = "s3_destination_folder must be of size 2 with a 'bucket' and 'key' respectively."
        raise ValueError(msg)
    return kwargs

def py_run_circuit(circuit, braket_device, kwargs, shots):
    device = py_sanitize_device(braket_device)
    kwargs = py_sanitize_kwargs(kwargs)
    if not kwargs:
        result = device.run(OpenQasmProgram(source=circuit), shots=int(shots)).result()
    else:
        result = device.run(OpenQasmProgram(source=circuit), shots=int(shots), s3_desination_folder=tuple(kwargs)).result()
    return result
    

def py_var(circuit, braket_device, kwargs, shots):
    return py_run_circuit(circuit, braket_device, kwargs, shots).values

def py_expval(circuit, braket_device, kwargs, shots):
    return py_run_circuit(circuit, braket_device, kwargs, shots).values
    
def py_samples(circuit, braket_device, kwargs, shots):
    result = py_run_circuit(circuit, braket_device, kwargs, shots)
    return np.array(result.measurements).flatten()
    
def py_probs(circuit, braket_device, kwargs, shots, num_qubits):
    result = py_run_circuit(circuit, braket_device, kwargs, shots)
    probs_dict = {int(s, 2): p for s, p in result.measurement_probabilities.items()}
    probs_list = []
    for i in range(2 ** int(num_qubits)):
        probs_list.append(probs_dict[i] if i in probs_dict else 0)
    return probs_list

def py_get_results(circuit, braket_device, kwargs, shots):
    return str(py_run_circuit(circuit, braket_device, kwargs, shots))
)";

extern "C" {
[[gnu::visibility("default")]] double var(const char *_circuit, const char *_device, size_t shots,
                                          const char *_kwargs)
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
                var = result.values
            except Exception as e:
                print(f"circuit: {circuit}")
                msg = str(e)
              )",
        py::globals(), locals);

    auto &&msg = locals["msg"].cast<std::string>();
    if (!msg.empty()) {
        throw std::runtime_error(msg);
    }

    py::list results = locals["var"];

    return results[0].cast<double>();
}
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

    py::exec(program, py::globals(), py::globals());
    auto results = py::globals()["py_samples"](circuit, device, kwargs, shots);

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

    py::exec(program, py::globals(), py::globals());
    auto results = py::globals()["py_probs"](circuit, device, kwargs, shots, num_qubits);

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

    py::exec(program, py::globals(), py::globals());
    auto retval =
        py::globals()["py_get_results"](circuit, device, kwargs, shots).cast<std::string>();
    char *retptr = (char *)malloc(sizeof(char *) * retval.size() + 1);
    memcpy(retptr, retval.c_str(), sizeof(char *) * retval.size());
    return retptr;
}
}

PYBIND11_MODULE(openqasm_python_module, m) { m.doc() = "openqasm"; }
