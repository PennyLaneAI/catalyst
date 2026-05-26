"""
Catalyst Compatibility Wrapper
==============================

Dynamically injects Catalyst runtime hooks into an existing PennyLane device
without breaking its internal state or inheritance chain.
"""

import os
import json
import base64
import platform

def pycat_device(device, custom_toml_path=None, **extra_kwargs):
    """
    Takes an instantiated PennyLane device and dynamically injects the methods
    required for Catalyst's C++ runtime execution.

    Parameters
    ----------
    device : qml.devices.Device
        The instantiated PennyLane device (e.g., from qml.device).
    custom_toml_path : str, optional
        Path to a custom TOML file defining the device's gate set.
    extra_kwargs : dict
        Any arbitrary kwargs that should be passed through to the C++ backend
        and serialized for the nanobind Python execution layer.
    """

    # 1. Create a dynamic subclass inheriting from the exact class of the provided device.
    # This guarantees that isinstance(device, OriginalClass) remains True and all
    # internal PennyLane methods (like preprocess, _shots, etc.) remain fully intact.
    class CatalystCompatDevice(device.__class__):
        metaname = "pycat_device"

        @staticmethod
        def get_c_interface():
            try:
                # Use Catalyst's internal path resolver if available
                from catalyst.utils.runtime_environment import get_lib_path
                lib_dir = get_lib_path("runtime", "RUNTIME_LIB_DIR")
            except ImportError:
                # Fallback for local development
                import pathlib
                lib_dir = str(pathlib.Path(__file__).parent.parent.parent / "runtime" / "lib")

            sys_platform = platform.system()
            if sys_platform == "Linux":
                lib_path = os.path.join(lib_dir, "librtd_pennylane_python.so")
            elif sys_platform == "Darwin":
                lib_path = os.path.join(lib_dir, "librtd_pennylane_python.dylib")
            else:
                raise NotImplementedError(f"Platform not supported: {sys_platform}")

            return ("PLPythonDevice", lib_path)

        @property
        def device_kwargs(self):
            """Tunnel all device configuration through C++ via Base64 JSON."""
            dev_name = getattr(self, "short_name", getattr(self, "name", "default.qubit"))
            num_wires = len(self.wires) if self.wires else 0

            # Safely extract shots (handling int, None, and new Shots objects)
            shots_obj = getattr(self, "shots", None)
            total_shots = getattr(shots_obj, "total_shots", 0) if shots_obj else (shots_obj if isinstance(shots_obj, int) else 0)

            full_kwargs = {
                "pl_device_name": dev_name,
                "wires": num_wires,
                "shots": total_shots,
            }
            full_kwargs.update(extra_kwargs)

            json_bytes = json.dumps(full_kwargs).encode('utf-8')
            b64_kwargs = base64.b64encode(json_bytes).decode('ascii')

            return {"b64_kwargs": b64_kwargs}

    # 2. Reassign the instance's class to our dynamic subclass
    device.__class__ = CatalystCompatDevice

    # 3. Attach the required TOML configuration path
    if custom_toml_path and os.path.isfile(custom_toml_path):
        device.config_filepath = os.path.abspath(custom_toml_path)
    else:
        try:
            from catalyst.utils.runtime_environment import get_lib_path
            lib_dir = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        except ImportError:
            import pathlib
            lib_dir = str(pathlib.Path(__file__).parent.parent.parent / "runtime" / "lib")

        device.config_filepath = os.path.join(lib_dir, "backend", "pennylane_python.toml")

    return device