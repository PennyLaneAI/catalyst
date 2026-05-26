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

// This nanobind module is dlopen'd at runtime by PLPythonRunner.
// It bridges the C++ runtime back into the Python interpreter to execute
// PennyLane QuantumScript tapes on arbitrary PennyLane Python devices.
//
// Pattern: identical to openqasm_python_module.cpp (Braket backend), except
// we deserialize JSON tape -> PennyLane QuantumScript -> device.execute().

#include <cmath>
#include <cstring>
#include <complex>
#include <string>
#include <vector>

#include "nanobind/eval.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/nanobind.h"

#include <nanobind/make_iterator.h>

// The Python code that reconstructs a PennyLane tape from JSON and executes it.
const std::string program = R"(
import json
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript


def _get_op(name, params, wires, inverse, ctrl_wires, ctrl_values):
    """Instantiate a PennyLane operator from its serialized form."""
    op_cls = getattr(qml, name, None)
    if op_cls is None:
        op_cls = getattr(qml.ops, name, None)
    if op_cls is None:
        raise ValueError(f"Unknown PennyLane gate: {name}")

    if params:
        op = op_cls(*params, wires=wires)
    else:
        op = op_cls(wires=wires)

    if inverse:
        op = qml.adjoint(op)

    if ctrl_wires:
        cv = [bool(v) for v in ctrl_values] if ctrl_values else None
        op = qml.ctrl(op, control=ctrl_wires, control_values=cv)

    return op


def _build_tape(tape_json):
    """Deserialize JSON into a list of PennyLane operations."""
    data = json.loads(tape_json)
    ops = []

    for g in data["ops"]:
        if g["name"] == "MidCircuitMeasure":
            # Mid-circuit measurement
            ops.append(qml.measure(g["wires"][0]))
        else:
            op = _get_op(g["name"], g["params"], g["wires"],
                         g["inverse"], g["ctrl_wires"], g["ctrl_values"])
            ops.append(op)

    # Matrix operations (QubitUnitary)
    for mg in data.get("matrix_ops", []):
        mat_re = np.array(mg["matrix_re"])
        mat_im = np.array(mg["matrix_im"])
        mat = (mat_re + 1j * mat_im).reshape(
            int(np.sqrt(len(mat_re))), int(np.sqrt(len(mat_re))))
        op = qml.QubitUnitary(mat, wires=mg["wires"])
        if mg["inverse"]:
            op = qml.adjoint(op)
        if mg["ctrl_wires"]:
            cv = [bool(v) for v in mg["ctrl_values"]] if mg["ctrl_values"] else None
            op = qml.ctrl(op, control=mg["ctrl_wires"], control_values=cv)
        ops.append(op)

    return ops, data["num_qubits"]


def _build_obs(obs_json, num_qubits):
    """Deserialize JSON observables array into PennyLane observable objects."""
    obs_list = json.loads(obs_json)
    pl_obs = []

    for obs_data in obs_list:
        kind = obs_data["kind"]
        if kind == "named":
            obs_cls = getattr(qml, obs_data["name"])
            pl_obs.append(obs_cls(wires=obs_data["wires"]))
        elif kind == "hermitian":
            mat_re = np.array(obs_data["matrix_re"])
            mat_im = np.array(obs_data["matrix_im"])
            n = int(np.sqrt(len(mat_re)))
            mat = (mat_re + 1j * mat_im).reshape(n, n)
            pl_obs.append(qml.Hermitian(mat, wires=obs_data["wires"]))
        elif kind == "tensor":
            sub = [pl_obs[i] for i in obs_data["obs_ids"]]
            pl_obs.append(qml.prod(*sub))
        elif kind == "hamiltonian":
            sub = [pl_obs[i] for i in obs_data["obs_ids"]]
            coeffs = obs_data["coeffs"]
            pl_obs.append(qml.Hamiltonian(coeffs, sub))
        else:
            raise ValueError(f"Unknown observable kind: {kind}")

    return pl_obs


def _get_device(device_kwargs_json):
    """Instantiate the target PennyLane device from serialized kwargs."""
    kwargs = json.loads(device_kwargs_json)
    device_name = kwargs.pop("pl_device_name", "default.qubit")
    # Convert numeric strings back
    wires_str = kwargs.pop("wires", None)
    shots_str = kwargs.pop("shots", None)

    wires = int(wires_str) if wires_str else None
    shots = int(shots_str) if shots_str and shots_str != "0" else None

    return qml.device(device_name, wires=wires, shots=shots)


def py_probs(tape_json, obs_json, device_kwargs, shots, num_qubits):
    """Execute tape, return probabilities over all qubits."""
    ops, nq = _build_tape(tape_json)
    meas = [qml.probs(wires=list(range(nq)))]
    tape = QuantumScript(ops, meas)
    dev = _get_device(device_kwargs)
    result = dev.execute(tape)
    return np.asarray(result).flatten().tolist()


def py_partial_probs(tape_json, obs_json, device_kwargs, shots, wires):
    """Execute tape, return probabilities on specified wires."""
    ops, nq = _build_tape(tape_json)
    meas = [qml.probs(wires=list(wires))]
    tape = QuantumScript(ops, meas)
    dev = _get_device(device_kwargs)
    result = dev.execute(tape)
    return np.asarray(result).flatten().tolist()


def py_sample(tape_json, obs_json, device_kwargs, shots, num_qubits):
    """Execute tape, return shot samples flat array."""
    ops, nq = _build_tape(tape_json)
    meas = [qml.sample(wires=list(range(nq)))]
    tape = QuantumScript(ops, meas, shots=int(shots))
    dev = _get_device(device_kwargs)
    result = dev.execute(tape)
    return np.asarray(result).flatten().astype(int).tolist()


def py_expval(tape_json, obs_json, device_kwargs, shots, obs_idx):
    """Execute tape, return expectation value of observable."""
    ops, nq = _build_tape(tape_json)
    pl_obs = _build_obs(obs_json, nq)
    meas = [qml.expval(pl_obs[obs_idx])]
    tape = QuantumScript(ops, meas)
    dev = _get_device(device_kwargs)
    result = dev.execute(tape)
    return float(np.asarray(result).flatten()[0])


def py_var(tape_json, obs_json, device_kwargs, shots, obs_idx):
    """Execute tape, return variance of observable."""
    ops, nq = _build_tape(tape_json)
    pl_obs = _build_obs(obs_json, nq)
    meas = [qml.var(pl_obs[obs_idx])]
    tape = QuantumScript(ops, meas)
    dev = _get_device(device_kwargs)
    result = dev.execute(tape)
    return float(np.asarray(result).flatten()[0])


def py_state(tape_json, obs_json, device_kwargs, shots, num_qubits):
    """Execute tape, return statevector."""
    ops, nq = _build_tape(tape_json)
    meas = [qml.state()]
    tape = QuantumScript(ops, meas)
    dev = _get_device(device_kwargs)
    result = dev.execute(tape)
    sv = np.asarray(result).flatten()
    # Return interleaved real/imag for C++ reconstruction
    out = []
    for c in sv:
        out.append(float(c.real))
        out.append(float(c.imag))
    return out
)";


// ============================================================================
// extern "C" functions called by PLPythonRunner via dlopen + getSymbol
// ============================================================================

extern "C" NB_EXPORT void pl_probs(const char *_tape, const char *_obs, const char *_kwargs,
                                   size_t shots, size_t num_qubits, void *_vector)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);

    auto results = scope["py_probs"](std::string(_tape), std::string(_obs), std::string(_kwargs),
                                     shots, num_qubits);

    auto *probs = reinterpret_cast<std::vector<double> *>(_vector);
    probs->reserve(std::pow(2, num_qubits));
    for (nb::handle item : results) {
        probs->push_back(nb::cast<double>(item));
    }
}

extern "C" NB_EXPORT void pl_partial_probs(const char *_tape, const char *_obs,
                                           const char *_kwargs, size_t shots,
                                           const size_t *wires_ptr, size_t num_wires,
                                           void *_vector)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);

    // Convert wires to python list
    nb::list py_wires;
    for (size_t i = 0; i < num_wires; i++) {
        py_wires.append(wires_ptr[i]);
    }

    auto results = scope["py_partial_probs"](std::string(_tape), std::string(_obs),
                                             std::string(_kwargs), shots, py_wires);

    auto *probs = reinterpret_cast<std::vector<double> *>(_vector);
    probs->reserve(std::pow(2, num_wires));
    for (nb::handle item : results) {
        probs->push_back(nb::cast<double>(item));
    }
}

extern "C" NB_EXPORT void pl_sample(const char *_tape, const char *_obs, const char *_kwargs,
                                    size_t shots, size_t num_qubits, void *_vector)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);

    auto results = scope["py_sample"](std::string(_tape), std::string(_obs), std::string(_kwargs),
                                      shots, num_qubits);

    auto *samples = reinterpret_cast<std::vector<size_t> *>(_vector);
    samples->reserve(shots * num_qubits);
    for (nb::handle item : results) {
        samples->push_back(nb::cast<size_t>(item));
    }
}

extern "C" NB_EXPORT double pl_expval(const char *_tape, const char *_obs, const char *_kwargs,
                                      size_t shots, size_t obs_idx)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);

    return nb::cast<double>(scope["py_expval"](std::string(_tape), std::string(_obs),
                                               std::string(_kwargs), shots, obs_idx));
}

extern "C" NB_EXPORT double pl_var(const char *_tape, const char *_obs, const char *_kwargs,
                                   size_t shots, size_t obs_idx)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);

    return nb::cast<double>(scope["py_var"](std::string(_tape), std::string(_obs),
                                            std::string(_kwargs), shots, obs_idx));
}

extern "C" NB_EXPORT void pl_state(const char *_tape, const char *_obs, const char *_kwargs,
                                   size_t shots, size_t num_qubits, void *_vector)
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);

    auto results = scope["py_state"](std::string(_tape), std::string(_obs), std::string(_kwargs),
                                     shots, num_qubits);

    auto *state = reinterpret_cast<std::vector<std::complex<double>> *>(_vector);
    state->reserve(std::pow(2, num_qubits));

    // Results come as interleaved real/imag pairs
    auto it = nb::iter(results);
    while (true) {
        nb::handle re_item;
        nb::handle im_item;
        try {
            re_item = nb::steal(nb::detail::obj_iter_next(it.ptr()));
        }
        catch (nb::python_error &) {
            break;
        }
        try {
            im_item = nb::steal(nb::detail::obj_iter_next(it.ptr()));
        }
        catch (nb::python_error &) {
            break;
        }
        state->emplace_back(nb::cast<double>(re_item), nb::cast<double>(im_item));
    }
}

NB_MODULE(pennylane_python_module, m) { m.doc() = "PennyLane Python device compatibility layer"; }
