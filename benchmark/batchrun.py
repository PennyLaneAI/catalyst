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
"""Measurement cycle management entry point"""
import sys

from catalyst_benchmark.toplevel import (
    AP,
    SYSHASH_ORIG,
    SYSINFO,
    collect,
    load,
    plot,
    syshash,
)

a = AP.parse_args(sys.argv[1:])
if a.verbose:
    print(f"Machine: {SYSINFO.toString()}\nHash {SYSHASH_ORIG}\nEffective {syshash(a)}")

if "collect" in a.actions:
    collect(a)
else:
    print("Skipping the 'collect' action")
if "plot" in a.actions:
    df, sysinfo = load(a)
    plot(a, df, sysinfo)
else:
    print("Skipping the 'plot' action")
