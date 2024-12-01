# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s

import os

import numpy as np
import pennylane as qml

from catalyst import qjit
from catalyst.debug import instrumentation

filename = "__tmp_test_timing_results.txt"

num_layers = 1
num_wires = 1
dev = qml.device("lightning.qubit", wires=num_wires)


@qml.qnode(dev)
def circuit(weights):
    qml.StronglyEntanglingLayers(weights=weights, wires=range(num_wires))
    return qml.expval(qml.PauliZ(0))


shape = qml.StronglyEntanglingLayers.shape(num_layers, num_wires)
weights = np.random.random(size=shape)

# -----

# CHECK:      [[timestamp:[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+]]:
# CHECK-NEXT: name: circuit
# CHECK-NEXT: system:
# CHECK-NEXT:   os: {{Linux|macOS}}
# CHECK-NEXT:   arch: {{amd64|x86_64|aarch64|arm64}}
# CHECK-NEXT:   python: {{[0-9\.]+}}
# CHECK-NEXT: results:
# CHECK-NEXT:   - pre_compilation:
# CHECK-NEXT:       walltime: {{[0-9\.]+}}
# CHECK-NEXT:       cputime: {{[0-9\.]+}}
# CHECK-NEXT:   - capture:
# CHECK-NEXT:       walltime: {{[0-9\.]+}}
# CHECK-NEXT:       cputime: {{[0-9\.]+}}
# CHECK-NEXT:       programsize: {{[0-9]+}}
# CHECK-NEXT:   - generate_ir:
# CHECK-NEXT:       walltime: {{[0-9\.]+}}
# CHECK-NEXT:       cputime: {{[0-9\.]+}}
# CHECK-NEXT:       programsize: {{[0-9]+}}
# CHECK-NEXT:   - get_func_loc:
# CHECK-NEXT:       walltime: {{[0-9\.]+}}
# CHECK-NEXT:       cputime: {{[0-9\.]+}}
# CHECK-NEXT:   - compile:
# CHECK-NEXT:       walltime: {{[0-9\.]+}}
# CHECK-NEXT:       cputime: {{[0-9\.]+}}
# CHECK-NEXT:       programsize: {{[0-9]+}}
# CHECK-NEXT:   - run:
# CHECK-NEXT:       walltime: {{[0-9\.]+}}
# CHECK-NEXT:       cputime: {{[0-9\.]+}}

with instrumentation(circuit.__name__, filename=filename, detailed=False):
    qjit(circuit)(weights)

with open(filename, mode="r", encoding="UTF-8") as f:
    print(f.read())

# -----

# COM: Check that the previous run is still in the file.
# CHECK:      [[timestamp]]:
# CHECK-NEXT: name: circuit
# CHECK-NEXT: system:
# CHECK:      results:
# CHECK:        - pre_compilation:
# CHECK:        - capture:
# CHECK:        - generate_ir:
# CHECK:        - get_func_loc:
# CHECK:        - compile:
# CHECK:        - run:

# CHECK:      {{[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+}}:
# CHECK-NEXT: name: circuit
# CHECK-NEXT: system:
# CHECK:      results:
# CHECK:        - pre_compilation:
# CHECK-NOT:        finegrained:
# CHECK:        - capture:
# CHECK-NOT:        finegrained:
# CHECK:        - generate_ir:
# CHECK-NEXT:       walltime:
# CHECK-NEXT:       cputime:
# CHECK-NEXT:       programsize:
# CHECK-NEXT:       finegrained:
# COM: Do not check detailed output from the first call to the compiler, e.g. 0_Canonicalize,
# COM: as we may want to remove that in the future.
# CHECK:        - compile:
# CHECK-NEXT:       walltime:
# CHECK-NEXT:       cputime:
# CHECK-NEXT:       programsize:
# CHECK-NEXT:       finegrained:
# CHECK-NEXT:         - parseMLIRSource:
# CHECK-NEXT:             walltime: {{[0-9\.]+}}
# CHECK-NEXT:             cputime: {{[0-9\.]+}}
# CHECK-NEXT:             programsize: {{[0-9\]+}}
# CHECK-NEXT:         - {{[a-zA-Z]+}}Pass:
# CHECK-NEXT:             walltime: {{[0-9\.]+}}
# CHECK-NEXT:             cputime: {{[0-9\.]+}}
# CHECK-NEXT:             programsize: {{[0-9\]+}}
# CHECK:        - run:
# CHECK-NEXT:       walltime:
# CHECK-NEXT:       cputime:
# CHECK-NEXT:       finegrained:
# CHECK-NEXT:         - device_init:
# CHECK-NEXT:             walltime: {{[0-9\.]+}}
# CHECK-NEXT:             cputime: {{[0-9\.]+}}
# CHECK-NEXT:         - qubit_allocate_array:
# CHECK-NEXT:             walltime: {{[0-9\.]+}}
# CHECK-NEXT:             cputime: {{[0-9\.]+}}
# CHECK-NEXT:         - qubit_release_array:
# CHECK-NEXT:             walltime: {{[0-9\.]+}}
# CHECK-NEXT:             cputime: {{[0-9\.]+}}
# CHECK-NEXT:         - device_release:
# CHECK-NEXT:             walltime: {{[0-9\.]+}}
# CHECK-NEXT:             cputime: {{[0-9\.]+}}

with instrumentation(circuit.__name__, filename=filename, detailed=True):
    qjit(circuit)(weights)

with open(filename, mode="r", encoding="UTF-8") as f:
    print(f.read())

os.remove(filename)
