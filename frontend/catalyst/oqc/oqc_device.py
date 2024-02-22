from pennylane.devices import Device
from pennylane import device
import pathlib
from qcaas_client.client import OQCClient

class OQCDevice(Device):

    config = pathlib.Path("/home/romain/Catalyst/catalyst/frontend/catalyst/oqc/oqc.toml")

    def __init__(self, wires, backend, shots=1024, credentials=None, **kwargs):
        self._backend = backend
        if credentials is not None:
            connect(credentials)
        super().__init__(wires=wires, shots=shots)

    @property
    def backend(self):
        return self._backend

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return "oqc.remote", "/home/romain/Catalyst/catalyst/frontend/catalyst/oqc/oqc.so"

    def execute(self, circuits, execution_config):
        return super().execute(circuits, execution_config)
    
    
def connect(credentials: dict):
    url = credentials.get("url")
    email = credentials.get("email")
    password = credentials.get("password")
    if url is None or email is None or password is None:
        raise(ValueError, "Wrong credentials format.")
    client = OQCClient(url=url, email=email, password=password)
    return client.authenticate()