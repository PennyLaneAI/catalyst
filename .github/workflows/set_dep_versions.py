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
"""
This module computes commit hashes for LLVM and MLIR-HLO based on a given JAX version.
"""

import os
import re
import sys

import requests

jax_version = sys.argv[1]

url = f"https://raw.githubusercontent.com/google/jax/jaxlib-v{jax_version}/WORKSPACE"
response = requests.get(url)
match = re.search(r'strip_prefix = "tensorflow-([a-zA-Z0-9]*)"', response.text)
tf_commit = match.group(1)

url = f"https://raw.githubusercontent.com/tensorflow/tensorflow/{tf_commit}/third_party/llvm/workspace.bzl"
response = requests.get(url)
match = re.search(r'LLVM_COMMIT = "([a-zA-Z0-9]*)"', response.text)
llvm_commit = match.group(1)

url = f"https://api.github.com/repos/tensorflow/tensorflow/commits?sha={tf_commit}&path=tensorflow/compiler/xla/mlir_hlo&per_page=1"
response = requests.get(url).json()
tf_hlo_commit = response[0]["sha"]
match = re.search(r"PiperOrigin-RevId: ([0-9]*)", response[0]["commit"]["message"])
piper_id = match.group(1)

url = f"https://api.github.com/search/commits?q=repo:tensorflow/mlir-hlo+{piper_id}"
response = requests.get(url).json()
hlo_commit = response["items"][0]["sha"]

with open(
    os.path.join(os.path.dirname(__file__), "../../.dep-versions"), "w", encoding="UTF-8"
) as f:
    f.write(
        f"""\
jax={jax_version}
mhlo={hlo_commit}
llvm={llvm_commit}
"""
    )
