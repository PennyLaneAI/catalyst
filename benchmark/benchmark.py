""" Single measurement entry point """
import sys
from sys import exit
from argparse import ArgumentParser
from os import makedirs
from os.path import dirname
from json import dump as json_dump, load as json_load
from traceback import print_exc

import numpy as np

from catalyst_benchmark.types import BooleanOptionalAction
from catalyst_benchmark.measurements import (
    parse_args,
    selfcheck,
    REGISTRY,
    parse_implementation,
    BenchmarkResult,
    with_alarm,
)

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
        exit(1)

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
        exit(2)

else:
    raise ValueError(f"Invalid command {a.command}")
