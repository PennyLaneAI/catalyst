# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This file contains a libraru of single value measurement procedures plus the self-checking
routine ensuring the numeric equivalence across similar problems."""

# pylint: disable=import-outside-toplevel
# pylint: disable=redefined-outer-name
# pylint: disable=consider-using-dict-items
# pylint: disable=consider-iterating-dictionary

import sys
from argparse import ArgumentParser
from argparse import Namespace as ParsedArguments
from contextlib import contextmanager
from functools import partial
from signal import ITIMER_REAL, SIGALRM, setitimer, signal
from time import time
from typing import List, Optional, Tuple

import numpy as np
from catalyst_benchmark.types import BenchmarkResult
from numpy.testing import assert_allclose


def catalyst_version() -> str:
    """Determine the catalyst version"""
    # pylint: disable=broad-except,broad-exception-caught,protected-access
    from os.path import dirname
    from subprocess import check_output

    import catalyst._version

    verstring = catalyst._version.__version__
    if "dev" in verstring:
        try:
            commit = (  # nosec
                check_output(["git", "rev-parse", "HEAD"], cwd=dirname(catalyst.__file__))
                .decode()
                .strip()[:7]
            )
            verstring += f"+g{commit}"
        except Exception:
            verstring += "+g?"
    return verstring


@contextmanager
def with_alarm(timeout: float):
    """Set POSIX alarm"""
    prev = None
    try:
        if 0 < timeout < float("inf"):

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
    """Print arguments to the stderr"""
    print(*args, **kwargs, file=sys.stderr)


def parse_implementation(implementation: str) -> Tuple[str, str, Optional[str]]:
    """Parse the implementation parameter, expect the "framework[+jax]/device" syntax."""
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
        raise NotImplementedError(f"Unsupported implementation: {implementation}")
    return framework, device, interface


def measure_compile_catalyst(a: ParsedArguments) -> BenchmarkResult:
    """Catalyst compilation time measurement procedure"""
    import jax
    import jax.numpy as jnp
    import pennylane as qml
    from jax.core import ShapedArray

    from catalyst import qjit

    versions = {
        "pennylane": qml.__version__,
        "jax": jax.__version__,
        "catalyst": catalyst_version(),
    }

    p: Problem  # pylint: disable=used-before-assignment
    if a.problem == "grover":
        from catalyst_benchmark.test_cases.grover_catalyst import ProblemC as Problem
        from catalyst_benchmark.test_cases.grover_catalyst import qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits),
            a.nlayers,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe":
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_catalyst import qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits),
            diff_method=a.vqe_diff_method,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe-hybrid":
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            qcompile_hybrid as qcompile,
        )
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            workflow_hybrid as workflow,
        )

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits),
            diff_method=a.vqe_diff_method,
            expansion_strategy="device",
        )
    elif a.problem == "qft":
        from catalyst_benchmark.test_cases.qft_catalyst import ProblemC as Problem
        from catalyst_benchmark.test_cases.qft_catalyst import qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits), a.nlayers, expansion_strategy="device"
        )
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
    """Catalyst running time measurement procedure"""
    import jax
    import jax.numpy as jnp
    import pennylane as qml
    from jax.core import ShapedArray

    from catalyst import qjit

    versions = {
        "pennylane": qml.__version__,
        "jax": jax.__version__,
        "catalyst": catalyst_version(),
    }

    p: Problem  # pylint: disable=used-before-assignment
    if a.problem == "grover":
        from catalyst_benchmark.test_cases.grover_catalyst import ProblemC as Problem
        from catalyst_benchmark.test_cases.grover_catalyst import qcompile, workflow

        p = Problem(qml.device("lightning.qubit", wires=a.nqubits), a.nlayers)

    elif a.problem == "chemvqe":
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_catalyst import qcompile, workflow

        p = Problem(qml.device("lightning.qubit", wires=a.nqubits), diff_method=a.vqe_diff_method)
    elif a.problem == "chemvqe-hybrid":
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            qcompile_hybrid as qcompile,
        )
        from catalyst_benchmark.test_cases.chemvqe_catalyst import (
            workflow_hybrid as workflow,
        )

        p = Problem(qml.device("lightning.qubit", wires=a.nqubits), diff_method=a.vqe_diff_method)
    elif a.problem == "qft":
        from catalyst_benchmark.test_cases.qft_catalyst import ProblemC as Problem
        from catalyst_benchmark.test_cases.qft_catalyst import qcompile, workflow

        p = Problem(
            qml.device("lightning.qubit", wires=a.nqubits),
            a.nlayers,
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
    """PennyLane/Jax compilation time measurement procedure"""
    import jax
    import pennylane as qml

    versions = {"pennylane": qml.__version__, "jax": jax.__version__}

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    _, device, interface = parse_implementation(a.implementation)

    p: Problem  # pylint: disable=used-before-assignment
    if a.problem == "grover":
        from catalyst_benchmark.test_cases.grover_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.grover_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            a.nlayers,
            interface=interface,
            expansion_strategy="device",
        )

    elif a.problem == "chemvqe":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=jax.grad,
            interface=interface,
            diff_method=a.vqe_diff_method,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe-hybrid":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile_hybrid as qcompile,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import size
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            workflow_hybrid as workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=jax.grad,
            interface=interface,
            diff_method=a.vqe_diff_method,
            expansion_strategy="device",
        )
    elif a.problem == "qft":
        from catalyst_benchmark.test_cases.qft_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.qft_pennylane import qcompile, size, workflow

        p = Problem(
            qml.device(device, wires=a.nqubits),
            interface=interface,
            nlayers=a.nlayers,
            expansion_strategy="device",
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
    """Pennylane/Jax running time measurement procedure"""
    import jax
    import pennylane as qml

    versions = {"pennylane": qml.__version__, "jax": jax.__version__}

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    _, device, interface = parse_implementation(a.implementation)

    p: Problem  # pylint: disable=used-before-assignment
    if a.problem == "grover":
        from catalyst_benchmark.test_cases.grover_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.grover_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(qml.device(device, wires=a.nqubits), a.nlayers, interface=interface)

    elif a.problem == "chemvqe":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=jax.grad,
            interface=interface,
            diff_method=a.vqe_diff_method,
        )
    elif a.problem == "chemvqe-hybrid":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile_hybrid as qcompile,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import size
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            workflow_hybrid as workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=jax.grad,
            interface=interface,
            diff_method=a.vqe_diff_method,
        )
    elif a.problem == "qft":
        from catalyst_benchmark.test_cases.qft_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.qft_pennylane import qcompile, size, workflow

        p = Problem(qml.device(device, wires=a.nqubits), interface=interface, nlayers=a.nlayers)

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
    """PennyLane compilation time measurement procedure"""
    import pennylane as qml

    versions = {"pennylane": qml.__version__}

    _, device, interface = parse_implementation(a.implementation)

    p: Problem  # pylint: disable=used-before-assignment
    if a.problem == "grover":
        from catalyst_benchmark.test_cases.grover_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.grover_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            a.nlayers,
            interface=interface,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=partial(qml.grad, argnum=0),
            diff_method=a.vqe_diff_method,
            expansion_strategy="device",
        )
    elif a.problem == "chemvqe-hybrid":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile_hybrid as qcompile,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import size
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            workflow_hybrid as workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=partial(qml.grad, argnum=0),
            diff_method=a.vqe_diff_method,
            expansion_strategy="device",
        )
    elif a.problem == "qft":
        from catalyst_benchmark.test_cases.qft_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.qft_pennylane import qcompile, size, workflow

        p = Problem(
            qml.device(device, wires=a.nqubits),
            interface=interface,
            nlayers=a.nlayers,
            expansion_strategy="device",
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
    """PennyLanem running time measurement procedure"""
    import pennylane as qml

    versions = {"pennylane": qml.__version__}

    _, device, interface = parse_implementation(a.implementation)

    p: Problem  # pylint: disable=used-before-assignment
    if a.problem == "grover":
        from catalyst_benchmark.test_cases.grover_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.grover_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(qml.device(device, wires=a.nqubits), a.nlayers)
    elif a.problem == "chemvqe":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile,
            size,
            workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=partial(qml.grad, argnum=0),
            diff_method=a.vqe_diff_method,
        )
    elif a.problem == "chemvqe-hybrid":
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            ProblemCVQE as Problem,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            qcompile_hybrid as qcompile,
        )
        from catalyst_benchmark.test_cases.chemvqe_pennylane import size
        from catalyst_benchmark.test_cases.chemvqe_pennylane import (
            workflow_hybrid as workflow,
        )

        p = Problem(
            qml.device(device, wires=a.nqubits),
            grad=partial(qml.grad, argnum=0),
            diff_method=a.vqe_diff_method,
        )
    elif a.problem == "qft":
        from catalyst_benchmark.test_cases.qft_pennylane import ProblemPL as Problem
        from catalyst_benchmark.test_cases.qft_pennylane import qcompile, size, workflow

        p = Problem(
            qml.device(device, wires=a.nqubits),
            interface=interface,
            nlayers=a.nlayers,
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


def parse_args(ap: ArgumentParser, args: List[str]) -> ParsedArguments:
    """Parse arguments and save the original command line"""
    a = ap.parse_args(args)
    setattr(a, "argv", args)
    return a


def selfcheck(ap: ArgumentParser) -> None:
    """Self-check routine"""

    def _runall(cmdline_fn, atol=1e-5):
        r1 = None
        for m_i in REGISTRY.keys():
            a = parse_args(ap, cmdline_fn(*m_i))
            try:
                r = REGISTRY[m_i](a)
                print(f"Checking {(a.problem, m_i)}")
                if r1 is None:
                    r1 = r
                else:
                    assert_allclose(
                        np.array(r1.numeric_result), np.array(r.numeric_result), atol=atol
                    )
            except NotImplementedError as e:
                print(f"Skipping {(a.problem, m_i)} due to: {e}")

    # fmt: off
    _runall(lambda m, i: ["run", "-p", "chemvqe-hybrid", "-m", m, "-i", i, "-n", "1",
                          "-N", "4", "-L", "2", "--numerical-check"])
    _runall(lambda m, i: ["run", "-p", "qft", "-m", m, "-i", i, "-n", "1",
                          "-N", "4", "-L", "2", "--numerical-check"])
    _runall(lambda m, i: ["run", "-p", "chemvqe", "-m", m, "-i", i, "-n", "1",
                          "-N", "4", "--numerical-check"])
    _runall(lambda m, i: ["run", "-p", "grover", "-m", m, "-i", i, "-n", "1",
                          "-N", "9", "--numerical-check"])
    # fmt: on
