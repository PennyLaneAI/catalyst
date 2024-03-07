import pathlib
from qcaas_client.client import OQCClient, QPUTask, CompilerConfig
from scc.compiler.config import QuantumResultsFormat

import pennylane as qml
from pennylane.devices import Device
from pennylane.transforms.core import TransformProgram
from pennylane.devices.preprocess import decompose, validate_measurements, validate_observables
from pennylane.devices import ExecutionConfig, DefaultExecutionConfig

from catalyst.compiler import get_lib_path

default_execution_config = ExecutionConfig()

BACKENDS = ["lucy", "toshiko"]
RES_FORMAT = QuantumResultsFormat().binary_count


class OQCDevice(Device):

    config = pathlib.Path(__file__).parent.joinpath("oqc.toml")

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        return "oqc", str(pathlib.Path(__file__).parent.joinpath("src/build/librtd_oqc.so"))

    def __init__(self, wires, backend, credentials, shots=1024, **kwargs):
        self._backend = backend
        _check_backend(backend=backend)
        # self._client = self._authenticate(credentials)
        super().__init__(wires=wires, shots=shots)

    def _authenticate(self, credentials: dict):
        """Function that authenticates a user to the QCaas (OQC cloud)."""
        url = credentials.get("url")
        email = credentials.get("email")
        password = credentials.get("password")
        if url is None or email is None or password is None:
            raise (ValueError, "Wrong credentials format.")
        client = OQCClient(url=url, email=email, password=password)
        client.authenticate()
        return client

    @property
    def backend(self):
        return self._backend

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an updated device configuration."""
        transform_program = TransformProgram()

        # Expand hamiltonian
        # Split non commuting
        # Decompose
        # Validate shots
        # Validate observables
        # Validate measurements

        return transform_program, execution_config

    def execute(self, circuits, execution_config):
        # Check availability
        oqc_tasks = []
        for circuit in circuits:
            oqc_config = CompilerConfig(
                repeats=circuit.shots, results_format=RES_FORMAT, optimizations=None
            )
            oqc_tasks.append(QPUTask(circuit.to_openqasm(), oqc_config))
        results = self._client.execute_tasks(oqc_tasks)
        return results


def _check_backend(backend):
    if backend not in BACKENDS:
        raise (ValueError, "Backend not supported.")
