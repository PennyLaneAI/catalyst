# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""xDSL API for qecl transforms"""

from .convert_quantum_to_qecl import ConvertQuantumToQecLogicalPass, convert_quantum_to_qecl_pass
from .inject_noise_to_qecl import InjectNoiseToQECLPass, inject_noise_to_qecl_pass

__all__ = [
    "ConvertQuantumToQecLogicalPass",
    "convert_quantum_to_qecl_pass",
    "InjectNoiseToQECLPass",
    "inject_noise_to_qecl_pass",
]
