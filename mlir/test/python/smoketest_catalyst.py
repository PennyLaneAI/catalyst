# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s

from mlir_quantum._mlir_libs._quantumDialects import catalyst as catalyst_d
from mlir_quantum.ir import *

with Context() as ctx:
    catalyst_d.register_dialect()
    module = Module.parse(
        """
        %0 = "catalyst.list_init"() : () -> !catalyst.arraylist<f64>
        "catalyst.list_dealloc"(%0) : (!catalyst.arraylist<f64>) -> ()
        """
    )

    print(module)
