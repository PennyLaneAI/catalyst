# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Redirect stderr to stdout to make the output visible to FileCheck.
# RUN: %PYTHON %s 2>&1 | FileCheck %s

import jax
import numpy as np
import pennylane as qml

from catalyst import qjit
from catalyst.debug import instrumentation

# Test only on CPU execution platform
original_jax_platforms = jax.config.jax_platforms if hasattr(jax.config, "jax_platforms") else None
jax.config.update("jax_platforms", "cpu")

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

# CHECK:      [DIAGNOSTICS] Running pre_compilation{{\s*}}
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms
# CHECK-NEXT: [DIAGNOSTICS] Running capture{{\s*}}
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms{{\s*}} programsize: {{[0-9]+}} lines
# CHECK-NEXT: [DIAGNOSTICS] Running generate_ir
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms{{\s*}} programsize: {{[0-9]+}} lines
# CHECK-NEXT: [DIAGNOSTICS] Running compile
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms{{\s*}} programsize: {{[0-9]+}} lines
# CHECK-NEXT: [DIAGNOSTICS] Running run
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms

with instrumentation(circuit.__name__, filename=None, detailed=False):
    qjit(circuit)(weights)

# -----

# COM: Note that finegrained output is produced *before* the high-level output in console mode.
# CHECK:      [DIAGNOSTICS] > Total pre_compilation
# CHECK-NEXT: [DIAGNOSTICS] > Total capture
# COM: Do not check detailed output from the first call to the compiler, e.g. 0_Canonicalize,
# COM: as we may want to remove that in the future.
# CHECK:      [DIAGNOSTICS] > Total generate_ir
# COM: Check for "compile" exactly, otherwise we could match things like "compileObjFile".
# CHECK:      [DIAGNOSTICS] > Total compile{{ }}
# CHECK-NEXT: [DIAGNOSTICS] Running device_init
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms
# CHECK-NEXT: [DIAGNOSTICS] Running qubit_allocate_array
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms
# CHECK-NEXT: [DIAGNOSTICS] Running qubit_release_array
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms
# CHECK-NEXT: [DIAGNOSTICS] Running device_release
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms
# CHECK:      [DIAGNOSTICS] > Total run
# COM: As the output below is generated by the Catalyst CLI, checking the correct order may cause flaky results.
# CHECK: [DIAGNOSTICS] Running parseMLIRSource
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms{{\s*}} programsize: {{[0-9]+}} lines
# CHECK: [DIAGNOSTICS] Running {{[a-zA-Z]+}}Pass
# CHECK-SAME:   walltime: {{[0-9\.]+}} ms{{\s*}} cputime: {{[0-9\.]+}} ms{{\s*}} programsize: {{[0-9]+}} lines

with instrumentation(circuit.__name__, filename=None, detailed=True):
    qjit(circuit)(weights)

# Restore original execution platforms
jax.config.update("jax_platforms", original_jax_platforms)
