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

#include <cstring>
#include <string>
#include <vector>

#include <nanobind/eval.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

const std::string program = R"(
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

extern "C" NB_EXPORT double var(const char *_circuit, const char *_device, size_t shots,
                                const char *_kwargs)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);
    return nb::cast<double>(scope["py_var"](circuit, device, kwargs, shots).attr("__getitem__")(0));
}

extern "C" NB_EXPORT double expval(const char *_circuit, const char *_device, size_t shots,
                                   const char *_kwargs)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);
    return nb::cast<double>(
        scope["py_expval"](circuit, device, kwargs, shots).attr("__getitem__")(0));
}

extern "C" NB_EXPORT void samples(const char *_circuit, const char *_device, size_t shots,
                                  size_t num_qubits, const char *_kwargs, void *_vector)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    std::vector<size_t> *samples = reinterpret_cast<std::vector<size_t> *>(_vector);

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);
    auto results = scope["py_samples"](circuit, device, kwargs, shots);

    samples->reserve(shots * num_qubits);
    for (nb::handle item : results) {
        samples->push_back(nb::cast<size_t>(item));
    }

    return;
}

extern "C" NB_EXPORT void probs(const char *_circuit, const char *_device, size_t shots,
                                size_t num_qubits, const char *_kwargs, void *_vector)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    std::vector<double> *probs = reinterpret_cast<std::vector<double> *>(_vector);

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);
    auto results = scope["py_probs"](circuit, device, kwargs, shots, num_qubits);

    probs->reserve(std::pow(2, num_qubits));
    for (nb::handle item : results) {
        probs->push_back(nb::cast<double>(item));
    }

    return;
}

extern "C" NB_EXPORT char *runCircuit(const char *_circuit, const char *_device, size_t shots,
                                      const char *_kwargs)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    std::string circuit(_circuit);
    std::string device(_device);
    std::string kwargs(_kwargs);

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);
    auto retval = nb::cast<std::string>(scope["py_get_results"](circuit, device, kwargs, shots));
    char *retptr = static_cast<char *>(malloc(sizeof(char *) * retval.size() + 1));
    std::memcpy(retptr, retval.c_str(), sizeof(char *) * retval.size());
    return retptr;
}

NB_MODULE(openqasm_python_module, m) { m.doc() = "openqasm"; }
