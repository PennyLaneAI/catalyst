from pennylane.devices import Device
from pennylane import device
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
import pathlib


class OQCDevice(Device):
    
    config = pathlib.Path("/home/romain/Catalyst/catalyst/frontend/catalyst/oqc/oqc.toml")

    def __init__(self, wires, backend, shots=1024, options=None, **kwargs):
        self._backend = backend
        super().__init__(wires=wires, shots=shots)

    def execute(self, circuits, execution_config):
        return super().execute(circuits, execution_config)

    @property
    def backend(self):
        return self._backend
    
    @staticmethod
    def get_c_interface():
        """ Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return "oqc.remote", "/home/romain/Catalyst/catalyst/frontend/catalyst/oqc/oqc.so"
