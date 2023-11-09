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

# RUN: %PYTHON %s

from mlir_quantum.dialects import gradient as gradient_d
from mlir_quantum.ir import *

with Context() as ctx:
    gradient_d.register_dialect()
    module = Module.parse(
        """
        "func.func"() ({}) {function_type = (f64) -> f64, sym_name = "funcScalarScalar", sym_visibility = "private"} : () -> ()
        "func.func"() ({
        ^bb0(%arg0: f64):
            %0 = "gradient.grad"(%arg0) {callee = @funcScalarScalar, method = "fd"} : (f64) -> f64
            "func.return"(%0) : (f64) -> ()
        }) {function_type = (f64) -> f64, sym_name = "gradCallScalarScalar"} : () -> ()
        """
    )

    print(str(module))
