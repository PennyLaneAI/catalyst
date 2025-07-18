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
"""Benchmarking data type definitions"""

from dataclasses import dataclass
from multiprocessing import cpu_count
from os import uname
from platform import system
from typing import Any, Dict, List, Optional

from cpuinfo import get_cpu_info
from dataclasses_json import dataclass_json
from psutil import virtual_memory


def unpack_complex(l: List[complex]) -> List[List[float]]:
    """Convert a list of complex to a JSON-serializable list of lists of floats."""
    return [[complex(x).real, complex(x).imag] for x in l]


# Cache due to delay in the `get_cpu_info`
CPU_BRAND: Optional[str] = None


@dataclass_json
@dataclass(frozen=True)
class Sysinfo:
    """Basic information about the system which runs the benchmark."""

    systype: str
    hostname: str
    ncpu: int
    ram_gb: int
    cpu_brand: str

    @classmethod
    def fromOS(cls) -> "Sysinfo":
        """Collects the information about the current system"""
        global CPU_BRAND  # pylint: disable=global-statement
        if CPU_BRAND is None:
            CPU_BRAND = get_cpu_info()["brand_raw"]
        return Sysinfo(
            system(),
            uname().nodename,
            cpu_count(),
            virtual_memory().total // 1024 // 1024 // 1024,
            CPU_BRAND,
        )

    def toString(self):
        """Return the string representation"""
        return f"Host: {self.systype} / {self.ram_gb}GB RAM / {self.ncpu} Cores / {self.cpu_brand}"


@dataclass
class Problem:
    """Base class for reference problem."""

    def __init__(self, dev, **qnode_kwargs):
        self.dev = dev
        self.qnode_kwargs = qnode_kwargs
        self.nqubits = len(dev.wires.tolist())

    def trial_params(self, i: int) -> Any:
        """Return problem-specific parameters for trial number `i`. Returns
        typically `pnp.array` or `jnp.array`."""
        raise NotImplementedError()


@dataclass_json
@dataclass
class BenchmarkResultV1:
    """Base class for the measurement results. We use this class for backward compatibility with
    some previously collected data records."""

    sysinfo: Sysinfo
    numeric_result: Optional[List[List[float]]]
    depth_gates: Optional[int]
    argv: List[str]
    prepare_sec: Optional[float]
    measurement_sec: List[float]


@dataclass_json
@dataclass
class BenchmarkResult(BenchmarkResultV1):
    """Measurement results data structure.
    Fields:
        numeric_result: Last numerical result, required primarily for
                        self-check.
        depth_gates: Depth of circuit in gates (if available)
        argv: List of benchmark parameters
        prepare_sec: Time required to prepare the circuit. For runtime
                     measurements this is the compilation time. For compilation
                     time measurements this field is not used.
        measurement_sec: List of main measurement results
        versions: Dictionary specifying versions of important Python packages

    Notes:
    * We do not override the __init__ in order to make automatic JSON
      (de-)serialization work.
    * `dataclass_json` does not provide serialization for complex numbers and
      Python tuples so we convert them to lists of reals with `unpack_complex`
      as a workaround.
    """

    versions: Dict[str, str]
    timeout_sec: Optional[float]

    @classmethod
    def fromMeasurements(
        cls,
        nr,
        argv,
        prep,
        times,
        depth: Optional[int],
        versions: Dict[str, str],
        timeout: Optional[float],
    ):  # pylint: disable=too-many-arguments
        """Format the measurement results"""
        return BenchmarkResult(
            Sysinfo.fromOS(),
            unpack_complex(nr) if nr else None,
            depth,
            argv,
            prep,
            times,
            versions,
            float(timeout),
        )
