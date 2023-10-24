
Custom Devices
##############

Differences between PennyLane and Catalyst
==========================================

PennyLane and Catalyst treat devices a bit differently.
In PennyLane, one is able to `define devices <https://docs.pennylane.ai/en/stable/development/plugins.html>`_ in Python.
Catalyst cannot interface with Python devices yet.
Instead, Catalyst can only interact with devices that implement the `QuantumDevice <../api/file_runtime_include_QuantumDevice.hpp.html>`_ class.

In addition to implementing the ``QuantumDevice`` class, one must implement the following method:

.. code-block:: guess

    extern "C" Catalyst::Runtime::QuantumDevice*
    getCustomDevice() { return new CustomDevice(); }

where ``CustomDevice()`` is a constructor for your custom device.
``CustomDevice``'s destructor will be called by the runtime.

.. note::

    This interface might change quickly in the near future.
    Please check back regularly for updates and to ensure your device is compatible with a specific version of Catalyst.

How to compile custom devices
=============================

There is an example ``CMakeLists.txt`` file and an example third party device in the runtime tests.
Please take a look there.

Integration with Python devices
===============================

If you already have a custom PennyLane device defined in Python and have added a shared object that corresponds to your implementation of the ``QuantumDevice`` class, then all you need to do is to add a ``get_c_interface`` method to your PennyLane device.
The ``get_c_interface`` method should be a static method that takes no parameters and returns the complete path to your shared library with the ``QuantumDevice`` implementation.
After doing so, Catalyst should be able to interface with your custom device with no problem.
