import sys
import numpy as np
from typing import Optional, Tuple, List
from numpy.testing import assert_allclose
from argparse import ArgumentParser, Namespace as ParsedArguments
from time import time
from os import makedirs
from os.path import dirname
from json import dump as json_dump, load as json_load
from signal import signal, SIGALRM, setitimer, ITIMER_REAL
from contextlib import contextmanager
from traceback import print_exc
from functools import partial

from .types import Problem, BenchmarkResult, BooleanOptionalAction


def catalyst_version() -> str:
    import catalyst._version
    from subprocess import check_output
    from os.path import dirname

    catalyst_version = catalyst._version.__version__
    if "dev" in catalyst_version:
        try:
            commit = (
                check_output(["git", "rev-parse", "HEAD"], cwd=dirname(catalyst.__file__))
                .decode()
                .strip()[:7]
            )
            catalyst_version += f"+g{commit}"
        except Exception:
            catalyst_version += "+g?"
    return catalyst_version


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


def printerr(*args, **kwargs) -> None:
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
        raise NotImplementedError(f"Unsupported implementation: {a.implementation}")
    return framework, device, interface


def measure_compile_catalyst(a: ParsedArguments) -> BenchmarkResult:
    import pennylane as qml
    import jax.numpy as jnp
    import jax
    from catalyst import qjit
    from jax.core import ShapedArray

    versions = {
        "pennylane": qml.__version__,
        "jax": jax.__version__,
        "catalyst": catalyst_version(),
    }

    p: Problem
    if a.problem == "grover":
        from .grover_catalyst import ProblemC as Problem, qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits),
            a.grover_nlayers,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe":
        from .chemvqe_catalyst import ProblemCVQE as Problem, qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits),
            diff_method=a.vqe_diff_method,
            expansion_strategy="device")
    else:
        raise NotImplementedError(f"Unsupported problem {a.problem}")

    weights = p.trial_params(0)

    def _main(weights: ShapedArray(weights.shape, dtype=jnp.float64)):
        qcompile(p, weights)
        return workflow(p, weights)

    times = []
    for i in range(a.niter):
        weights = p.trial_params(i)

        b = time()
        jit_main = qjit(_main)
        e = time()
        times.append(e - b)

    r = jit_main(weights).tolist() if a.numerical_check else None

    return BenchmarkResult.fromMeasurements(
        r, a.argv, prep=None, times=times, depth=None, versions=versions, timeout=a.timeout
    )


def measure_runtime_catalyst(a: ParsedArguments) -> BenchmarkResult:
    import pennylane as qml
    import jax
    import jax.numpy as jnp
    from catalyst import qjit
    from jax.core import ShapedArray

    versions = {
        "pennylane": qml.__version__,
        "jax": jax.__version__,
        "catalyst": catalyst_version(),
    }

    p: Problem
    if a.problem == "grover":
        from .grover_catalyst import ProblemC as Problem, qcompile, workflow

        p = Problem(qml.device("lightning.qubit", wires=a.nqubits), a.grover_nlayers)

    elif a.problem == "chemvqe":
        from .chemvqe_catalyst import ProblemCVQE as Problem, qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits), diff_method=a.vqe_diff_method
        )

    else:
        raise NotImplementedError(f"Unsupported problem {a.problem}")

    weights = p.trial_params(0)

    def _main(weights: ShapedArray(weights.shape, dtype=jnp.float64)):
        qcompile(p, weights)
        return workflow(p, weights)

    b = time()
    jit_main = qjit(_main)
    e = time()
    cmptime = e - b

    times = []
    for i in range(a.niter):
        weights = p.trial_params(i)

        b = time()
        r = jit_main(weights)
        e = time()
        times.append(e - b)

    return BenchmarkResult.fromMeasurements(
        r.tolist(), a.argv, cmptime, times, depth=None, versions=versions, timeout=a.timeout
    )


def measure_compile_pennylanejax(a: ParsedArguments) -> BenchmarkResult:
    import pennylane as qml
    import jax

    versions = {"pennylane": qml.__version__, "jax": jax.__version__}

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import ProblemPL as Problem, qcompile, workflow, size

        p = Problem(
            qml.device(device, wires=a.nqubits),
            a.grover_nlayers,
            interface=interface,
            expansion_strategy="device",
        )

    elif a.problem == "chemvqe":
        from .chemvqe_pennylane import ProblemCVQE as Problem, qcompile, workflow, size

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=jax.grad,
            interface=interface,
            diff_method=a.vqe_diff_method
        )
    else:
        raise NotImplementedError(f"Unsupported problem {a.problem}")

    def _main(weights):
        qcompile(p, weights)
        return workflow(p, weights)

    times: list = []
    for i in range(a.niter):
        weights = p.trial_params(i)

        # Clean the JAX caches
        jax.clear_backends()

        b = time()
        jax_main = jax.jit(_main).lower(weights).compile()
        e = time()
        times.append(e - b)

    r = jax_main(weights).tolist() if a.numerical_check else None

    return BenchmarkResult.fromMeasurements(
        r, a.argv, None, times, depth=size(p), versions=versions, timeout=a.timeout
    )


def measure_runtime_pennylanejax(a: ParsedArguments) -> BenchmarkResult:
    import pennylane as qml
    import jax

    versions = {"pennylane": qml.__version__, "jax": jax.__version__}

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import ProblemPL as Problem, qcompile, workflow, size

        p = Problem(qml.device(device, wires=a.nqubits), a.grover_nlayers, interface=interface)

    elif a.problem == "chemvqe":
        from .chemvqe_pennylane import ProblemCVQE as Problem, qcompile, workflow, size

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=jax.grad,
            interface=interface,
            diff_method=a.vqe_diff_method
        )

    else:
        raise NotImplementedError(f"Unsupported problme {a.problem}")

    def _main(weights):
        qcompile(p, weights)
        return workflow(p, weights)

    weights = p.trial_params(0)
    b = time()
    jax_main = jax.jit(_main).lower(weights).compile()
    e = time()
    cmptime = e - b

    times: list = []
    for i in range(a.niter):
        weights = p.trial_params(i)

        b = time()
        r = jax_main(weights).block_until_ready()
        e = time()
        times.append(e - b)

    return BenchmarkResult.fromMeasurements(
        r.tolist(), a.argv, cmptime, times, depth=size(p), versions=versions, timeout=a.timeout
    )


def measure_compile_pennylane(a: ParsedArguments) -> BenchmarkResult:
    import pennylane as qml

    versions = {"pennylane": qml.__version__}

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import ProblemPL as Problem, qcompile, workflow, size

        p = Problem(
            qml.device(device, wires=a.nqubits),
            a.grover_nlayers,
            interface=interface,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe":
        from .chemvqe_pennylane import ProblemCVQE as Problem, qcompile, workflow, size

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=partial(qml.grad, argnum=0),
            diff_method=a.vqe_diff_method
        )
    else:
        raise NotImplementedError(f"Unsupported problem {a.problem}")

    times: list = []
    for i in range(a.niter):
        weights = p.trial_params(i)

        b = time()
        qcompile(p, weights)
        e = time()
        times.append(e - b)

    r = workflow(p, weights).tolist() if a.numerical_check else None

    return BenchmarkResult.fromMeasurements(
        r, a.argv, None, times, depth=size(p), versions=versions, timeout=a.timeout
    )


def measure_runtime_pennylane(a: ParsedArguments) -> BenchmarkResult:
    import pennylane as qml

    versions = {"pennylane": qml.__version__}

    _, device, interface = parse_implementation(a.implementation)

    if a.problem == "grover":
        from .grover_pennylane import ProblemPL as Problem, workflow, qcompile, size

        p = Problem(qml.device(device, wires=a.nqubits), a.grover_nlayers)
    elif a.problem == "chemvqe":
        from .chemvqe_pennylane import ProblemCVQE as Problem, qcompile, workflow, size

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=partial(qml.grad, argnum=0),
            diff_method=a.vqe_diff_method
        )

    else:
        raise NotImplementedError(f"Unsupported problme {a.problem}")

    weights = p.trial_params(a.niter + 1)
    b = time()
    qcompile(p, weights)
    e = time()
    preptime = e - b

    times: list = []
    for i in range(a.niter):
        weights = p.trial_params(i)

        b = time()
        r = workflow(p, weights)
        e = time()
        times.append(e - b)

    return BenchmarkResult.fromMeasurements(
        r.tolist(), a.argv, preptime, times, depth=size(p), versions=versions, timeout=a.timeout
    )


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


def parse_args(ap:ArgumentParser, args:List[str]) -> ParsedArguments:
    a = ap.parse_args(args)
    setattr(a, "argv", args)
    return a


def selfcheck(ap:ArgumentParser) -> None:
    def _runall(cmdline_fn, atol=1e-5):
        r1 = None
        for m, i in REGISTRY.keys():
            a = parse_args(ap, cmdline_fn(m, i))
            try:
                r = REGISTRY[(m, i)](a)
                print(f"Checking {(a.problem, m, i)}")
                if r1 is None:
                    r1 = r
                else:
                    assert_allclose(
                        np.array(r1.numeric_result), np.array(r.numeric_result), atol=atol
                    )
            except NotImplementedError as e:
                print(f"Skipping {(a.problem, m, i)} due to: {e}")

    # fmt: off
    _runall(lambda m, i: ["run", "-p", "chemvqe", "-m", m, "-i", i, "-n", "1",
                          "-N", "4", "--numerical-check"])
    _runall(lambda m, i: ["run", "-p", "grover", "-m", m, "-i", i, "-n", "1",
                          "-N", "9", "--numerical-check"])
    # fmt: on


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
                with open(a.output, "w") as f:
                    json_dump(r2.to_dict(), f, indent=4)
                with open(a.output, "r") as f:
                    r3 = BenchmarkResult.from_dict(json_load(f))
                assert np.allclose(r2.measurement_sec, r3.measurement_sec)
        except TimeoutError:
            print_exc()
            exit(2)

    else:
        raise ValueError(f"Invalid command {a.command}")
