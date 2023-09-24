Debugging and sharp bits
========================

Catalyst is designed to allow you to take the tools and code patterns you are
familiar with when exploring quantum computing (such as Python, NumPy, JAX,
and PennyLane), while unlocking faster execution and the ability to run
hybrid quantum-classical workflows on accelerator devices.

Similar to JAX, Catalyst does this via the ``@qjit`` decorator, which captures
hybrid programs written in Python, PennyLane, and JAX, and compiles them to
native machine code --- preserving important aspects like conditional
branches and classical control.

With Catalyst, we aim to try and support as many idiomatic PennyLane and JAX
hybrid workflow programs as possible, however there will be **various
restrictions and constraints that should be taken into account**.

Here, we aim to provide an overview of the restrictions and constraints
(the 'sharp bits'), as well as debugging tips and common patterns that are
helpful when using Catalyst.

.. note::

	For a more general overview of Catalyst, please see the
	:doc:`quick start guide <quick_start>`.


Debugging functions
-------------------

Catalyst provides 

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~catalyst.debug.print

.. raw:: html

    </div>
    <div class="summary-table">


Compile-time vs. runtime
------------------------

Todo.


JAX support and restrictions
----------------------------

Todo.

Dynamic circuit restrictions
----------------------------

Todo.

Classical control debugging
---------------------------

Todo.

Capturing Python control with AutoGraph
---------------------------------------

Todo.

Common PennyLane patterns for Catalyst
--------------------------------------

Todo.
