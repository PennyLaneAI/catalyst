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

# pylint: disable=line-too-long
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=consider-using-with

import os
import re
import sys

import requests

jax_version = sys.argv[1]
dep_versions_path = os.path.join(os.path.dirname(__file__), "../../.dep-versions")
catalyst_init_path = os.path.join(os.path.dirname(__file__), "../../frontend/catalyst/__init__.py")

assert os.path.isfile(dep_versions_path)
assert os.path.isfile(catalyst_init_path)

url = f"https://raw.githubusercontent.com/google/jax/jaxlib-v{jax_version}/WORKSPACE"
response = requests.get(url)
match = re.search(r'strip_prefix = "xla-([a-zA-Z0-9]*)"', response.text)
if not match:
    url = f"https://raw.githubusercontent.com/google/jax/jaxlib-v{jax_version}/third_party/xla/workspace.bzl"
    response = requests.get(url)
    match = re.search(r'XLA_COMMIT = "([a-zA-Z0-9]*)"', response.text)
xla_commit = match.group(1)

url = f"https://raw.githubusercontent.com/openxla/xla/{xla_commit}/third_party/llvm/workspace.bzl"
response = requests.get(url)
match = re.search(r'LLVM_COMMIT = "([a-zA-Z0-9]*)"', response.text)
llvm_commit = match.group(1)

# If the XLA commit is an "Integrate LLVM" commit we need to get the piper_id directly from there
# to look up the corresponding mlir-hlo commit.
url = f"https://api.github.com/repos/openxla/xla/commits?sha={xla_commit}&per_page=1"
response = requests.get(url).json()
match = re.search(r"Integrate LLVM", response[0]["commit"]["message"])
if match:
    match = re.search(r"PiperOrigin-RevId: ([0-9]*)", response[0]["commit"]["message"])
    piper_id = match.group(1)
else:
    # Otherwise, we get the last commit in the XLA repository that touched the mlir-hlo files, and
    # get its piper_id to get the same commit in the standalone mlir-hlo repo.
    url = f"https://api.github.com/repos/openxla/xla/commits?sha={xla_commit}&path=xla/mlir_hlo&per_page=1"
    response = requests.get(url).json()
    xla_hlo_commit = response[0]["sha"]
    match = re.search(r"PiperOrigin-RevId: ([0-9]*)", response[0]["commit"]["message"])
    piper_id = match.group(1)

url = f"https://api.github.com/search/commits?q=repo:tensorflow/mlir-hlo+{piper_id}"
response = requests.get(url).json()
hlo_commit = response["items"][0]["sha"]

existing_text = open(dep_versions_path, "r", encoding="UTF-8").read()
match = re.search(r"enzyme=([a-zA-Z0-9]*)", existing_text)
enzyme_commit = match.group(1)

with open(dep_versions_path, "w", encoding="UTF-8") as f:
    f.write(
        f"""\
jax={jax_version}
mhlo={hlo_commit}
llvm={llvm_commit}
enzyme={enzyme_commit}
"""
    )

quote = '"'
cmd = f"sed -i 's/_jaxlib_version = {quote}\([0-9.]\+\){quote}/_jaxlib_version = {quote}{jax_version}{quote}/g' {catalyst_init_path}"
res = os.system(cmd)
assert res == 0
