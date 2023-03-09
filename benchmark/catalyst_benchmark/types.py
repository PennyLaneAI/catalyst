from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from dataclasses_json import dataclass_json

from argparse import Action, SUPPRESS
from psutil import virtual_memory
from os import uname
from multiprocessing import cpu_count
from platform import system
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
class BenchmarkResultV1:
    sysinfo: Sysinfo
    numeric_result: Optional[List[List[float]]]
    depth_gates: Optional[int]
    argv: List[str]
    prepare_sec: Optional[float]
    measurement_sec: List[float]


@dataclass_json
@dataclass
class BenchmarkResult(BenchmarkResultV1):
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
        cls, nr, argv, prep, times, depth: Optional[int], versions: Dict[str, str],
        timeout: Optional[float]
    ):
        return BenchmarkResult(
            Sysinfo.fromOS(), unpack_complex(nr) if nr else None, depth, argv, prep, times,
            versions, float(timeout)
        )


class BooleanOptionalAction(Action):
    """Backported from argparse for Python3.10"""

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith("--"):
                option_string = "--no-" + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None and default is not SUPPRESS:
            help += " (default: %(default)s)"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    def format_usage(self):
        return " | ".join(self.option_strings)
