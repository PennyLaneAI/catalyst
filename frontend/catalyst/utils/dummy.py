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
"""Dummy module for testing"""


def dummy_func(x):
    """Simple function with if statements for testing the 'auto_include' option of @qjit.
    The parent 'catalayst' module is excluded for autograph conversion by default, hence
    adding this module explicitely to the inclusion list will override that restriction"""

    if x > 5:
        y = x**2
    else:
        y = x**3
    return y
