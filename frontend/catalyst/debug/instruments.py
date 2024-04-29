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

"""
Instrumentation module to report Catalyst & program performance.
"""

import datetime
import functools
import os
import platform
import sys
import time
from contextlib import contextmanager

# pylint: disable=f-string-without-interpolation
# Tested in LIT test suite.
# pragma: no cover


## API ##
@contextmanager
def instrumentation(session_name, filename=None, detailed=False):
    """Instrumentation session to output information on wall time, CPU time,
    and intermediate program size of a program during compilation and execution.

    A session cannot be tied to the creation of a QJIT object
    nor its lifetime, because it involves both compile time and runtime measurements. It cannot
    be a context-free process either since we need to write results to an existing results file.

    Args:
        session_name (str): identifier to distinguish multiple results, primarily for humans
        filename (str): Desired path to write results to in YAML format. If ``None``, the
            results will instead be printed to the console.
        detailed (bool): whether to instrument finegrained steps in the compiler and runtime.
            If ``False``, only high-level steps such as program capture and
            compilation are reported.

    **Example**

    >>> @qjit
    ... def expensive_function(a, b):
    ...     return a + b
    >>> with debug.instrumentation("session_name", detailed=False):
    >>>     expensive_function(1, 2)
    2024-04-29 18:19:29.349886:
    name: session_name
    system:
    os: Linux-6.1.58+-x86_64-with-glibc2.35
    arch: x86_64
    python: 3.10.12
    results:
    - capture:
        walltime: 6.296216
        cputime: 2.715764
        programsize: 0
    - generate_ir:
        walltime: 8.84289
        cputime: 8.836589
        programsize: 14
    - compile:
        walltime: 199.249725
        cputime: 38.820425
        programsize: 121
    - run:
        walltime: 1.053613
        cputime: 1.019584
    """
    session = InstrumentSession(session_name, filename, detailed)

    try:
        yield None
    finally:
        session.close()


def instrument(fn=None, *, size_from=None, has_finegrained=False):
    """Decorator that marks specific functions as targets for instrumentation.
    nstrumentation is only performed when enabled by a session.

    Args:
        fn (Callable): function to instrument
        size_from (int | None): optional index indicating from which result to measure program size
            by number of newlines in the string representation of the result
        has_finegrained (bool): whether to instrument finegrained steps in the compiler and runtime.
            If ``False``, only high-level steps such as program capture and
            compilation are reported.

    **Example**

    .. code-block:: python

        @instrument
        def expensive_function(a, b):
            return a + b

        @qml.qjit
        def fn(x):
            y = jnp.sin(x) ** 3
            z = expensive_function(x, y)
            return jnp.cos(z)

    >>> with catalyst.debug.instrumentation("session_name"):
    ...     fn(0.43)
    2024-04-29 19:05:00.719805:
      name: session_name
      system:
        os: Linux-6.1.58+-x86_64-with-glibc2.35
        arch: x86_64
        python: 3.10.12
      results:
        - capture:
            walltime: 7.359074
            cputime: 7.324612
            programsize: 5
        - expensive_function:
            walltime: 0.771452
            cputime: 0.765094
        - generate_ir:
            walltime: 11.269458
            cputime: 11.305722
            programsize: 18
        - compile:
            walltime: 164.486692
            cputime: 52.139593
            programsize: 128
        - run:
            walltime: 0.864398
            cputime: 0.859378
    """
    if fn is None:
        return functools.partial(instrument, size_from=size_from, has_finegrained=has_finegrained)

    stage_name = getattr(fn, "__name__", "UNKNOWN")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not InstrumentSession.active:
            return fn(*args, **kwargs)

        with ResultReporter(stage_name, has_finegrained) as reporter:
            fn_results, wall_time, cpu_time = time_function(fn, args, kwargs)
            program_size = measure_program_size(fn_results, size_from)
            reporter.commit_results(wall_time, cpu_time, program_size)

        return fn_results

    return wrapper


## DATA COLLECTION ##
def time_function(fn, args, kwargs):
    """Collect timing information for a function call.

    Args:
        fn (Callable): function to time
        args (Tuple): positional function arguments
        kwargs (Dict): keyword function arguments

    Returns:
        Any: function results
        int: wall time
        int: cpu time
    """

    # Unclear which order is better here since they can't be measured at the same time.
    start_wall = time.perf_counter_ns()
    start_cpu = time.process_time_ns()
    results = fn(*args, **kwargs)
    stop_cpu = time.process_time_ns()
    stop_wall = time.perf_counter_ns()

    return results, stop_wall - start_wall, stop_cpu - start_cpu


def measure_program_size(results, size_from):
    """Collect program size information by counting the number of newlines in the textual form
    of a given program representation. The representation is assumed to be provided in the
    instrumented function results at the provided index.

    Args:
        results (Sequence): instrumented function results
        size_from (int | None): result index to use for size measurement

    Returns:
        int: program size
    """
    if size_from is None:
        return None

    return str(results[size_from]).count("\n")


## REPORTING ##
class ResultReporter:
    """Report the result of a single instrumentation stage. Reporting is done either to the console
    or to a specified file obtained from the instrumentation session. The report is appended to the
    file in that case.

    The functionality is implemented as a stateful context manager since the (high-level)
    instrumentation results are only available after the compiler has reported the low-level
    instrumentation results. We thus need to open the report file first to obtain to correct
    position to insert the results after the function was measured.
    """

    def __init__(self, stage_name, has_finegrained):
        self.stage_name = stage_name
        self.has_finegrained = has_finegrained
        self.enable_finegrained = InstrumentSession.finegrained
        self.insertion_point = None
        self.filename = InstrumentSession.filename

    def __enter__(self):
        """Save the report file insertion point before low-level results are written to it."""
        if self.filename:
            with open(self.filename, mode="a", encoding="UTF-8") as file:
                self.insertion_point = file.tell()
        return self

    def __exit__(self, *_):
        return False

    def commit_results(self, wall_time, cpu_time, program_size):
        """Report the provided results to either the console or file."""
        if self.filename:
            self.dump_results(wall_time, cpu_time, program_size)
        else:
            self.print_results(wall_time, cpu_time, program_size)

    def print_results(self, wall_time, cpu_time, program_size=None):
        """Print the provided results to console with some minor formatting."""
        if self.enable_finegrained:
            print(f"[DIAGNOSTICS] > Total {self.stage_name.ljust(23)}", end="\t", file=sys.stderr)
        else:
            print(f"[DIAGNOSTICS] Running {self.stage_name.ljust(23)}", end="\t", file=sys.stderr)

        formatted_wall_time = (str(wall_time // 1e3 / 1e3) + " ms").ljust(12)
        print(f"walltime: {formatted_wall_time}", end="\t", file=sys.stderr)

        formatted_cpu_time = (str(cpu_time // 1e3 / 1e3) + " ms").ljust(12)
        print(f"cputime: {formatted_cpu_time}", end="\t", file=sys.stderr)

        if program_size is not None:
            print(f"programsize: {program_size} lines", end="", file=sys.stderr)

        print(end="\n", file=sys.stderr)

    def dump_results(self, wall_time, cpu_time, program_size=None):
        """Dump the provided results to file accounting for (potential) low-level instrumentation
        results for the same stage."""
        with open(self.filename, mode="r+", encoding="UTF-8") as file:
            file.seek(self.insertion_point)
            existing_text = file.read()  # from low-level instrumentation
            file.seek(self.insertion_point)

            file.write(f"    - {self.stage_name}:\n")
            file.write(f"        walltime: {wall_time / 1e6}\n")
            file.write(f"        cputime: {cpu_time / 1e6}\n")
            if program_size is not None:
                file.write(f"        programsize: {program_size}\n")
            if self.has_finegrained and self.enable_finegrained:
                file.write(f"        finegrained:\n")

            file.write(existing_text)

    @staticmethod
    def dump_header(session_name):
        """Write the session header to file, including timestamp, session name, and system info."""
        filename = InstrumentSession.filename
        current_time = datetime.datetime.now()

        with open(filename, mode="a", encoding="UTF-8") as file:
            file.write(f"\n{current_time}:\n")
            file.write(f"  name: {session_name}\n")
            file.write(f"  system:\n")
            file.write(f"    os: {platform.platform(terse=True)}\n")
            file.write(f"    arch: {platform.machine()}\n")
            file.write(f"    python: {platform.python_version()}\n")
            file.write(f"  results:\n")


## SESSION ##
class InstrumentSession:
    """Provides access to global state during instrumentation and sets up / tears down
    the environment used by the C++ instrumentation code. To be used as a context manager."""

    active = False
    filename = None
    finegrained = False

    def __init__(self, session_name, filename, detailed):
        self.enable_flag = os.environ.pop("ENABLE_DIAGNOSTICS", None)
        self.path_flag = os.environ.pop("DIAGNOSTICS_RESULTS_PATH", None)

        self.open_session(session_name, filename, detailed)

    def close(self):
        """Terminate the instrumentation session."""
        self.close_session(self.enable_flag, self.path_flag)

    @staticmethod
    def open_session(session_name, filename, detailed):
        """Open an instrumentation session. Sets the global state and env variables."""
        InstrumentSession.active = True
        InstrumentSession.filename = filename
        InstrumentSession.finegrained = detailed

        if detailed:
            os.environ["ENABLE_DIAGNOSTICS"] = "ON"
        if filename:
            os.environ["DIAGNOSTICS_RESULTS_PATH"] = filename
            ResultReporter.dump_header(session_name)

    @staticmethod
    def close_session(enable_flag, path_flag):
        """Close an instrumentation session. Resets the global state and env variables."""
        InstrumentSession.active = False
        InstrumentSession.filename = None
        InstrumentSession.finegrained = False

        if enable_flag is None:
            os.environ.pop("ENABLE_DIAGNOSTICS", None)  # safely delete
        else:
            os.environ["ENABLE_DIAGNOSTICS"] = enable_flag

        if path_flag is None:
            os.environ.pop("ENABLE_DIAGNOSTICS", None)  # safely delete
        else:
            os.environ["DIAGNOSTICS_RESULTS_PATH"] = path_flag
