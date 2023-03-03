import sys
import numpy as np
from typing import Optional, Any, List, Tuple
from numpy.testing import assert_allclose
from argparse import ArgumentParser, BooleanOptionalAction
from time import time
from os import makedirs
from os.path import dirname
from json import dump as json_dump, load as json_load
from signal import signal, SIGINT, SIGALRM, setitimer, ITIMER_REAL
from contextlib import contextmanager
from traceback import print_exception

from .types import Problem, BenchmarkResult


@contextmanager
def with_alarm(timeout: float):
    prev = None
    try:
        if timeout > 0 and timeout < float("inf"):

            def _handler(signum, frame):
                raise TimeoutError()

            prev = signal(SIGALRM, _handler)
            setitimer(ITIMER_REAL, timeout)
        yield
    finally:
        if prev is not None:
            setitimer(ITIMER_REAL, 0)
            signal(SIGALRM, prev)


def printerr(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def parse_implementation(implementation: str) -> Tuple[str, str, Optional[str]]:
    tokens = implementation.split("/")
    assert len(tokens) > 0
    framework = tokens[0]

    device: str
    interface: str

    if framework == "pennylane+jax":
        interface = "jax"
    else:
        interface = None

    if len(tokens) == 1:
        if framework == "catalyst":
            device = "lightning.qubit"
        else:
            device = "default.qubit.jax" if interface == "jax" else "default.qubit"
    elif len(tokens) == 2:
        device = tokens[1]
    else:
        raise ValueError(f"Unsupported implementation: {a.implementation}")
    return framework, device, interface


def measure_compile_catalyst(a: Any) -> BenchmarkResult:
    import pennylane as qml
    from catalyst import qjit
    from jax.core import ShapedArray

    t: Problem
    if a.problem == "grover":
        from .grover_catalyst import ProblemC, grover_main as main

        t = ProblemC(qml.device("lightning.qubit", wires=a.nqubits), a.grover_nlayers)
    elif a.problem == "vqe":
        from .vqe_catalyst import ProblemVQE, grad_descent as main

        t = ProblemVQE(qml.device("lightning.qubit", wires=a.nqubits))
    else:
        raise ValueError(f"Unsupported problem {a.problem}")

    weights = t.trial_params(0)

    def _main(weights: ShapedArray(weights.shape, complex)):
        return main(t, weights)

    times = []
    for i in range(a.niter):
        weights = t.trial_params(i)

        b = time()
        jit_main = qjit(_main)
        e = time()
        times.append(e - b)

    r = jit_main(weights).tolist() if a.numerical_check else None

    return BenchmarkResult.fromMeasurements(r, a.argv, prep=None, times=times)


def measure_runtime_catalyst(a: Any) -> BenchmarkResult:
    import pennylane as qml
    from catalyst import qjit
    from jax.core import ShapedArray

    t: Problem
    if a.problem == "grover":
        from .grover_catalyst import ProblemC, grover_main as main

        t = ProblemC(qml.device("lightning.qubit", wires=a.nqubits), a.grover_nlayers)
    elif a.problem == "vqe":
        from .vqe_catalyst import ProblemVQE, grad_descent as main

        t = ProblemVQE(
            qml.device("lightning.qubit", wires=a.nqubits), diff_method=a.vqe_diff_method
        )
    elif a.problem == "chemvqe":
        from .chemvqe_catalyst import ProblemCVQE, workflow as main

        t = ProblemCVQE(
            qml.device("lightning.qubit", wires=a.nqubits), diff_method=a.vqe_diff_method
        )

    else:
        raise ValueError(f"Unsupported problem {a.problem}")

    weights = t.trial_params(0)

    def _main(weights: ShapedArray(weights.shape, weights.dtype)):
        return main(t, weights)

    b = time()
    jit_main = qjit(_main)
    e = time()
    cmptime = e - b

    times = []
    for i in range(a.niter):
        weights = t.trial_params(i)

        b = time()
        r = jit_main(weights)
        e = time()
        times.append(e - b)

    return BenchmarkResult.fromMeasurements(r.tolist(), a.argv, cmptime, times)


def measure_compile_pennylanejax(a: Any) -> BenchmarkResult:
    import pennylane as qml
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import (
            ProblemPL as Problem,
            grover_main as main,
            grover_depth as depth,
        )

        t = Problem(qml.device(device, wires=a.nqubits), a.grover_nlayers, interface=interface)

    elif a.problem == "vqe":
        from .vqe_pennylane import ProblemVQE, grad_descent as main

        def depth(_):
            return None

        t = ProblemVQE(qml.device(device, wires=a.nqubits), interface=interface)
    else:
        raise ValueError(f"Unsupported problem {a.problem}")

    def _main(weights):
        return main(t, weights)

    times: list = []
    for i in range(a.niter):
        weights = t.trial_params(i)

        jax.clear_backends()

        b = time()
        jax_main = jax.jit(_main).lower(weights).compile()
        e = time()
        times.append(e - b)

    r = jax_main(weights).tolist() if a.numerical_check else None

    return BenchmarkResult.fromMeasurements(r, a.argv, None, times, depth=depth(t))


def measure_runtime_pennylanejax(a: Any) -> BenchmarkResult:
    import pennylane as qml
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import (
            ProblemPL as Problem,
            grover_main as main,
            grover_depth as depth,
        )

        t = Problem(qml.device(device, wires=a.nqubits), a.grover_nlayers, interface=interface)
    elif a.problem == "vqe":
        from .vqe_pennylane import ProblemVQE, grad_descent as main

        t = ProblemVQE(qml.device(device, wires=a.nqubits), interface=interface)

        def depth(_):
            return None

    else:
        raise ValueError(f"Unsupported problme {a.problem}")

    def _main(weights):
        return main(t, weights)

    weights = t.trial_params(0)
    b = time()
    jax_main = jax.jit(_main).lower(weights).compile()
    e = time()
    cmptime = e - b

    times: list = []
    for i in range(a.niter):
        jax.clear_backends()
        weights = t.trial_params(i)

        b = time()
        r = jax_main(weights)
        e = time()
        times.append(e - b)

    return BenchmarkResult.fromMeasurements(r.tolist(), a.argv, cmptime, times, depth=depth(t))


def measure_compile_pennylane(a: Any) -> BenchmarkResult:
    import pennylane as qml

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import (
            ProblemPL as Problem,
            grover_mainloop as main,
            grover_depth as depth,
        )

        t = Problem(qml.device(device, wires=a.nqubits), a.grover_nlayers, interface=interface)
    else:
        raise ValueError(f"Unsupported problem {a.problem}")

    def _main(weights):
        return main(t, weights)

    times: list = []
    for i in range(a.niter):
        weights = t.trial_params(i)

        b = time()
        qml_main = qml.QNode(_main, t.dev, **t.qnode_kwargs)
        qml_main.construct([weights], {})
        e = time()
        times.append(e - b)

    r = qml_main(weights).tolist() if a.numerical_check else None

    return BenchmarkResult.fromMeasurements(r, a.argv, None, times, depth=depth(t))


def measure_runtime_pennylane(a: Any) -> BenchmarkResult:
    import pennylane as qml

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import (
            ProblemPL as Problem,
            grover_main as main,
            grover_depth as depth,
        )

        t = Problem(qml.device(device, wires=a.nqubits), a.grover_nlayers)
    elif a.problem == "vqe":
        from .vqe_pennylane import ProblemVQE, grad_descent as main

        t = ProblemVQE(qml.device(device, wires=a.nqubits), diff_method=a.vqe_diff_method)

        def depth(_):
            return None

    elif a.problem == "chemvqe":
        from .chemvqe_pennylane import ProblemCVQE, workflow as main

        t = ProblemCVQE(qml.device(device, wires=a.nqubits), diff_method=a.vqe_diff_method)

        def depth(_):
            return None

    else:
        raise ValueError(f"Unsupported problme {a.problem}")

    def _main(weights):
        return main(t, weights)

    times: list = []
    for i in range(a.niter):
        weights = t.trial_params(i)

        b = time()
        r = _main(weights)
        e = time()
        times.append(e - b)

    return BenchmarkResult.fromMeasurements(r.tolist(), a.argv, None, times, depth=depth(t))


REGISTRY = {
    ("compile", "catalyst/lightning.qubit"): measure_compile_catalyst,
    ("runtime", "catalyst/lightning.qubit"): measure_runtime_catalyst,
    ("compile", "pennylane+jax/lightning.qubit"): measure_compile_pennylanejax,
    ("compile", "pennylane+jax/default.qubit.jax"): measure_compile_pennylanejax,
    ("runtime", "pennylane+jax/lightning.qubit"): measure_runtime_pennylanejax,
    ("runtime", "pennylane+jax/default.qubit.jax"): measure_runtime_pennylanejax,
    ("compile", "pennylane/lightning.qubit"): measure_compile_pennylane,
    ("compile", "pennylane/default.qubit"): measure_compile_pennylane,
    ("runtime", "pennylane/lightning.qubit"): measure_runtime_pennylane,
    ("runtime", "pennylane/default.qubit"): measure_runtime_pennylane,
}


def parse_args(ap, args):
    a = ap.parse_args(args)
    setattr(a, "argv", args)
    return a


def selfcheck(ap):
    # fmt: off
    r1 = measure_runtime_catalyst(parse_args(
        ap, ["run", "-p", "vqe", "-m", "runtime", "-i", "catalyst", "-n", "1",
             "-N", "6", "--numerical-check"]))
    r2 = measure_runtime_pennylane(parse_args(
        ap, ["run", "-p", "vqe", "-m", "runtime", "-i", "pennylane/default.qubit", "-n", "1",
             "-N", "6", "--numerical-check"]))
    # fmt: on
    assert_allclose(np.array(r1.numeric_result), np.array(r2.numeric_result), atol=1e-3)

    r1 = None
    for m, i in REGISTRY.keys():
        # fmt: off
        r = REGISTRY[(m, i)](parse_args(
            ap, ["run", "-p", "grover", "-m", m, "-i", i, "-n", "1", "-N", "9",
                 "--numerical-check"]))
        # fmt: on
        if r1 is None:
            r1 = r
        else:
            assert_allclose(np.array(r1.numeric_result), np.array(r.numeric_result), atol=1e-5)


if __name__ == "__main__":
    # fmt: off
    ap = ArgumentParser(prog="python3 -m catalyst_benchmark.main")
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
    runcmd.add_argument("--grover-nlayers", type=int, default=None, metavar="INT",
                        help="Grover-specific: Number of layers (default - auto)")
    runcmd.add_argument("--vqe-diff-method", type=str, default="backprop",
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
            print("\n".join(sorted(set(["- " + x[0] for x in REGISTRY.keys()]))))
            should_exit = True
        if a.implementation == "?":
            print("implementations:")
            print("\n".join(sorted(set(["- " + x[1] for x in REGISTRY.keys()]))))
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
                    f"Unsupported combination of measure('{a.measure}') and implementation"
                    f"('{a.implementation}', deduced as '{framework}/{device}')"
                )

            r2 = BenchmarkResult.from_json(r.to_json())
            r2.numeric_result = None  # Makes screen output readable
            if a.output == "-":
                json_dump(r2.to_dict(), sys.stdout, indent=4)
            else:
                if len(dirname(a.output)) > 0:
                    makedirs(dirname(a.output), exist_ok=True)
                with open(a.output, "w") as f:
                    json_dump(r2.to_dict(), f, indent=4)
                with open(a.output, "r") as f:
                    r3 = BenchmarkResult.from_dict(json_load(f))
                assert np.allclose(r2.measurement_sec, r3.measurement_sec)
        except TimeoutError as err:
            print_exception(err)
            exit(2)

    else:
        raise ValueError(f"Unsupported command {a.command}")
