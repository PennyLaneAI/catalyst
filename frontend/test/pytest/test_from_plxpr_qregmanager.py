# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module tests the from_plxpr QregManager object.

Quoted from the object's docstring:

    With the above in mind, this `QregManager` class promises the following:
    1. An instance of this class will always be tied to one root qreg value.
    2. At any moment during the from_plxpr conversion:
       - `QregManager.get()` returns the current catalyst qreg SSA value for the managed
          root register on that instance;
       - `QregManager[i]` returns the current catalyst qubit SSA value for the i-th index
          on the managed root register on that instance. If none exists, a new qubit will be
          extracted.

    To achieve the above, users of this class are expected to:
    1. Initialize an instance with the qreg SSA value from a new allocation or a new block argument;
    2. Whenever a new meta-op/op is bind-ed, update the current qreg/qubit SSA value with:
       - `QubitManager.set(new_qreg_value)`
       - `QubitManager[i] = new_qubit_value`
"""

import jax
from jax.core import set_current_trace
from jax.interpreters.partial_eval import DynamicJaxprTrace
import numpy as np
import pennylane as qml
import pytest

from catalyst.from_plxpr.qreg_manager import QregManager
from catalyst.jax_primitives import (
    MeasurementPlane,
    compbasis_p,
    cond_p,
    counts_p,
    device_init_p,
    device_release_p,
    expval_p,
    gphase_p,
    measure_in_basis_p,
    measure_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qinst_p,
    quantum_kernel_p,
    sample_p,
    set_basis_state_p,
    set_state_p,
    state_p,
    unitary_p,
    var_p,
)

def aloha():
    trace = DynamicJaxprTrace(debug_info=None)
    with set_current_trace(trace):
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)

        wires = [0,1]
        in_qubits = [qreg_manager[w] for w in wires]
        out_qubits = qinst_p.bind(
            *[*in_qubits],
            op="my_gate",
            qubits_len=len(wires),
            params_len=0,
            ctrl_len=0,
            adjoint=False,
        )


    breakpoint()
aloha()
