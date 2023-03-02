from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from dataclasses_json import dataclass_json

from psutil import virtual_memory
from os import uname
from multiprocessing import cpu_count
from platform import processor, system
from cpuinfo import get_cpu_info


def unpack_complex(l: List[complex]) -> List[List[float]]:
    return [[complex(x).real, complex(x).imag] for x in l]


# Cache due to delay in the `get_cpu_info`
CPU_BRAND: Optional[str] = None


@dataclass_json
@dataclass(frozen=True)
class Sysinfo:
    systype: str
    hostname: str
    ncpu: int
    ram_gb: int
    cpu_brand: str

    @classmethod
    def fromOS(cls) -> "Sysinfo":
        global CPU_BRAND
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
        return f"Host: {self.systype} / {self.ram_gb}GB RAM / {self.ncpu} Cores / {self.cpu_brand}"


class Problem:
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
class BenchmarkResult:
    """Common benchmark result data structure.
    Fields:
        numeric_result: Last numerical result, required primarily for
                        self-check.
        depth_gates: Depth of circuit in gates (if available)
        argv: List of benchmark parameters
        prepare_sec: Time required to prepare the circuit. For runtime
                     measurements this is the compilation time. For compilation
                     time measurements this field is not used.
        measurement_sec: List of main measurement results

    Notes:
    * We do not override the __init__ in order to make automatic JSON
      (de-)serialization work.
    * `dataclass_json` does not provide serialization for complex numbers and
      Python tuples so we convert them to lists of reals with `unpack_complex`
      as a workaround.
    """

    sysinfo: Sysinfo
    numeric_result: Optional[List[List[float]]]
    depth_gates: Optional[int]
    argv: List[str]
    prepare_sec: Optional[float]
    measurement_sec: List[float]

    @classmethod
    def fromMeasurements(cls, nr, argv, prep, times, depth=None):
        return BenchmarkResult(
            Sysinfo.fromOS(), unpack_complex(nr) if nr else None, depth, argv, prep, times
        )
