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
"""Single measurement entry point"""
import sys
from argparse import ArgumentParser
from json import dump as json_dump
from json import load as json_load
from os import makedirs
from os.path import dirname
from traceback import print_exc

import numpy as np
from catalyst_benchmark.measurements import (
    REGISTRY,
    BenchmarkResult,
    parse_args,
    parse_implementation,
    selfcheck,
    with_alarm,
)
from catalyst_benchmark.types import BooleanOptionalAction

# fmt: off
ap = ArgumentParser(prog="python3 benchmark.py")
apcmds = ap.add_subparsers(help="command help", dest="command")
sccmd = apcmds.add_parser("selfcheck",
                          help="Check the numeric equality of all the implementations")
sccmd.add_argument("--nqubits", type=int, default=11,
                   help="Number of qubits, should be odd and >=3 (default - 11)")
runcmd = apcmds.add_parser("run",
                           help="Run the benchmark",
                           epilog="Exit codes: 0 - success, 2 - timeout")
runcmd.add_argument("--timeout", type=str, metavar="SEC", default="inf",
                    help="Timeout (default - not set)")
runcmd.add_argument("-p", "--problem", type=str, required=True,
                    help="Problem to run (?|grover|vqe)")
runcmd.add_argument("-m", "--measure", type=str, required=True,
                    help="Value to measure (?|compile|runtime)")
runcmd.add_argument("-i", "--implementation", type=str, required=True,
                    help="Problem implementation (?|catalyst|pennylane[+jax])[/device], "
                    "(default - catalyst)")
runcmd.add_argument("-n", "--niter", type=int, default=10, metavar="INT",
                    help="Number of measurement trials (default - 10)")
runcmd.add_argument("-o", "--output", type=str, default="-", metavar="FILE.json",
                    help="Output *.json filename (default - '-' meaning stdout)")
runcmd.add_argument("--numerical-check", default=False, action=BooleanOptionalAction,
                    help="Whether to do a numerical check or not")
runcmd.add_argument("-N", "--nqubits", type=int, default=11, metavar="INT",
                    help="Number of qubits")
runcmd.add_argument("-L","--nlayers", type=int, default=None, metavar="INT",
                    help="Number of layers, problem-specific (default - auto)")
runcmd.add_argument("--vqe-diff-method", type=str, default="finite-diff",
                    help="VQE-specific: Differentiation method (default - backprop)")
# fmt: on

a = parse_args(ap, sys.argv[1:])

if a.command == "selfcheck":
    selfcheck(ap)
    print("OK")
elif a.command == "run":
    should_exit = False
    if a.problem == "?":
        print("problems:")
        print("\n".join(sorted(["- grover", "- vqe", "- chemvqe"])))
        should_exit = True
    if a.measure == "?":
        print("measurements:")
        print("\n".join(sorted({"- " + x[0] for x in REGISTRY.keys()})))
        should_exit = True
    if a.implementation == "?":
        print("implementations:")
        print("\n".join(sorted({"- " + x[1] for x in REGISTRY.keys()})))
        should_exit = True
    if should_exit:
        sys.exit(1)

    framework, device, _ = parse_implementation(a.implementation)

    try:
        fn = REGISTRY.get((a.measure, f"{framework}/{device}"), None)
        if fn is not None:
            with with_alarm(float(a.timeout)):
                r = fn(a)
        else:
            raise ValueError(
                f"Invalid combination of measure('{a.measure}') and implementation"
                f"('{a.implementation}', deduced as '{framework}/{device}')"
            )

        r2 = BenchmarkResult.from_json(r.to_json())
        r2.numeric_result = None  # Makes screen output readable
        if a.output == "-":
            json_dump(r2.to_dict(), sys.stdout, indent=4)
        else:
            if len(dirname(a.output)) > 0:
                makedirs(dirname(a.output), exist_ok=True)
            with open(a.output, "w", encoding="utf-8") as f:
                json_dump(r2.to_dict(), f, indent=4)
            with open(a.output, "r", encoding="utf-8") as f:
                r3 = BenchmarkResult.from_dict(json_load(f))
            assert np.allclose(r2.measurement_sec, r3.measurement_sec)
    except TimeoutError:
        print_exc()
        sys.exit(2)

else:
    raise ValueError(f"Invalid command {a.command}")
