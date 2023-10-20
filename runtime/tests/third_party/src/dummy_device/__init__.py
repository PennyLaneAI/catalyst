import pennylane as qml

class DummyDevice(qml.QubitDevice):
    name = "Dummy Device"
    short_name = "dummy.device"
    pennylane_requires = "0.32.0"
    version = "0.0.1"
    author = "Dummy"

    # Doesn't matter as at the moment it is dictated by QJITDevice
    operations = []
    observables = []

    def __init__(self, shots=None, wires=None):
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        raise RuntimeError("Only C/C++ interface is defined")

    @staticmethod
    def get_c_interface():
        import os
        return os.path.dirname(os.path.realpath(__file__)) + "/libdummy_device.so"
