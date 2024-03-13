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
Instrucmentation functions to report Catalyst & program performance.
"""

import datetime
import functools
import os
import platform
import time
from contextlib import contextmanager


## API ##
@contextmanager
def instrumentation(session_name, filename=None, detailed=False):
    """Start an instrumentation session. A session cannot be tied to the creation of a QJIT object
    nor its lifetime, because it involves both compile time and runtime measurements. It cannot
    be a context-free process either since we need to write results to an existing results file.

    Args:
        session_name (str): identifier to distinguish multiple results, primarily for humans
        filename (str): desired path to write results to in YAML format
        detailed (bool): whether to instrument finegrained steps in the compiler and runtime
    """
    session = InstrumentSession(session_name, filename, detailed)

    try:
        yield None
    finally:
        del session


def instrument(fn=None, *, size_from=None):
    """Decorator that marks functions as targets for instrumentation. Instrumentation is only
    performed when enabled by a session.

    Args:
        fn (Callable): function to instrument
        size_from (int | None): optional index indicating from which result to measure program size
                                by number of newlines in the string representation of the result
    """
    if fn is None:
        return functools.partial(instrument, size_from=size_from)

    stage_name = getattr(fn, "__name__", "UNKNOWN")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not InstrumentSession.active:
            return fn(*args, **kwargs)

        fn_results, wall_time, cpu_time = time_function(fn, args, kwargs)
        program_size = measure_program_size(fn_results, size_from)

        if InstrumentSession.filename:
            dump_result(stage_name, wall_time, cpu_time, program_size)
        else:
            print_result(stage_name, wall_time, cpu_time, program_size)

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


def measure_peak_memory(): ...


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
def print_result(stage, wall_time, cpu_time, program_size=None):
    print(f"[TIMER] Running {stage.ljust(23)}", end="\t")
    print(f"walltime: {wall_time / 1e6}ms", end="\t")
    print(f"cputime: {cpu_time / 1e6}ms", end="\t")
    if program_size is not None:
        print(f"programsize: {program_size} lines", end="")
    print(end="\n")


def dump_header(session_name):
    """Write the session header to file, contains the timestamp, session name, and system info."""
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


def dump_result(stage_name, wall_time, cpu_time, program_size=None):
    """Write single stage results to file."""
    filename = InstrumentSession.filename

    with open(filename, mode="a", encoding="UTF-8") as file:
        file.write(f"    - {stage_name}:\n")
        file.write(f"        walltime: {wall_time / 1e6}\n")
        file.write(f"        cputime: {cpu_time / 1e6}\n")
        if program_size is not None:
            file.write(f"        programsize: {program_size}\n")


def dump_footer():
    """Write the session footer to file."""
    filename = InstrumentSession.filename

    with open(filename, mode="a", encoding="UTF-8") as file:
        file.write("")


## SESSION ##
class InstrumentSession:
    active = False
    filename = None
    finegrained = False

    def __init__(self, session_name, filename, detailed):
        self.enable_timer_flag = os.environ.pop("ENABLE_DEBUG_TIMER", None)
        self.enable_info_flag = os.environ.pop("ENABLE_DEBUG_INFO", None)
        self.path_flag = os.environ.pop("DEBUG_RESULTS_FILE", None)

        InstrumentSession.open(session_name, filename, detailed)

    def __del__(self):
        InstrumentSession.close()

        if self.enable_timer_flag is None:
            os.environ.pop("ENABLE_DEBUG_TIMER", None)  # safely delete
        else:
            os.environ["ENABLE_DEBUG_TIMER"] = self.enable_timer_flag

        if self.enable_info_flag is None:
            os.environ.pop("ENABLE_DEBUG_TIMER", None)  # safely delete
        else:
            os.environ["ENABLE_DEBUG_INFO"] = self.enable_info_flag

        if self.path_flag is None:
            os.environ.pop("ENABLE_DEBUG_TIMER", None)  # safely delete
        else:
            os.environ["DEBUG_RESULTS_FILE"] = self.path_flag

    @staticmethod
    def open(session_name, filename, detailed):
        InstrumentSession.active = True
        InstrumentSession.filename = filename
        InstrumentSession.finegrained = detailed

        if detailed:
            os.environ["ENABLE_DEBUG_TIMER"] = "ON"
            os.environ["ENABLE_DEBUG_INFO"] = "ON"
        if filename:
            os.environ["DEBUG_RESULTS_FILE"] = filename
            dump_header(session_name)

    @staticmethod
    def close():
        if InstrumentSession.filename:
            dump_footer()

        InstrumentSession.active = False
        InstrumentSession.filename = None
        InstrumentSession.finegrained = False
