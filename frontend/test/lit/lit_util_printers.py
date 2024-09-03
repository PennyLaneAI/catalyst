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


def print_attr(f, attr, *args, aot: bool = False, **kwargs):
    """Print function attribute"""
    name = f"TEST {f.__name__}"
    print("\n" + "-" * len(name))
    print(f"{name}\n")
    res = None
    if not aot:
        res = f(*args, **kwargs)
    print(getattr(f, attr))
    return res


def print_jaxpr(f, *args, **kwargs):
    """Print jaxpr code of a function"""
    return print_attr(f, "jaxpr", *args, **kwargs)


def print_mlir(f, *args, **kwargs):
    """Print mlir code of a function"""
    return print_attr(f, "mlir", *args, **kwargs)
