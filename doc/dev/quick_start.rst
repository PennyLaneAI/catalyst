Quick Start
###########

Catalyst enables just-in-time (JIT) and ahead-of-time (AOT) compilation of
quantum programs and workflows, while taking into account both classical and quantum code, and
ultimately leverages modern compilation tools to speed up quantum applications.

You can imagine compiling a function once in advance and then benefit from faster
execution on all subsequent calls of the function, similar to the ``jax.jit`` functionality.
However, compared to JAX we are also able to compile the quantum code natively without having
to rely on callbacks to any Python-based PennyLane devices. We can thus compile/execute entire workflows
(such as variational algorithms) as a single program or unit, without having to go back and forth between
device execution and the Python interpreter.

.. raw:: html

    <style>
        .details-header h2 {
            font-size: 20px !important;
        }
    </style>


Importing Catalyst and PennyLane
================================
The first thing we need to do is import :func:`.qjit` and QJIT compatible methods in Catalyst,
as well as `PennyLane <https://pennylane.ai/>`_ and the version of `NumPy <https://jax.readthedocs.io/en/latest/jax.numpy.html>`_
provided by JAX.

.. code-block:: python

    from catalyst import qjit, measure, cond, for_loop, while_loop, grad
    import pennylane as qml
    from jax import numpy as jnp

Constructing the QNode
======================
You should be able to express your quantum functions in the way you are accustomed to using
PennyLane. However, some of PennyLane's features may not be fully supported yet, such as optimizers.

.. warning::

    Not all PennyLane devices currently work with Catalyst. Supported backend devices include
    ``lightning.qubit``, ``lightning.kokkos``, ``lightning.gpu``, and ``braket.aws.qubit``. For
    a full of supported devices, please see :doc:`/dev/devices`.

PennyLane tapes are still used internally by Catalyst and you can express your circuits in the
way you are used to, as long as you ensure that all operations are added to the main tape.

Let's start learning more about Catalyst by running a simple circuit.

.. code-block:: python

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit(theta):
        qml.Hadamard(wires=0)
        qml.RX(theta, wires=1)
        qml.CNOT(wires=[0,1])
        return qml.expval(qml.PauliZ(wires=1))

In PennyLane, the :func:`qml.qnode() <pennylane.qnode>` decorator creates a device specific quantum function. For each quantum
function, we can specify the number of wires.

The :func:`.qjit` decorator can be used to jit a workflow of quantum functions:

.. code-block:: python

    jitted_circuit = qjit(circuit)

>>> jitted_circuit(0.7)
Array(0., dtype=float64)

In Catalyst, dynamic wire values are fully supported for operations, observables and measurements.
For example, the following circuit can be jitted with wires as arguments:

.. code-block:: python

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=5))
    def circuit(arg0, arg1, arg2):
        qml.RX(arg0, wires=[arg1 + 1])
        qml.RY(arg0, wires=[arg2])
        qml.CNOT(wires=[arg1, arg2])
        return qml.probs(wires=[arg1 + 1])

>>> circuit(jnp.pi / 3, 1, 2)
Array([0.625, 0.375], dtype=float64)


Operations
----------
Catalyst allows you to use :doc:`quantum operations <introduction/operations>`
available in PennyLane either via native support by the runtime or PennyLane's decomposition rules.
The :func:`qml.adjoint() <pennylane.adjoint>` and :func:`qml.ctrl() <pennylane.ctrl>` functions in
PennyLane are also supported via the decomposition mechanism in Catalyst. For example,

.. code-block:: python

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit():
        qml.Rot(0.3, 0.4, 0.5, wires=0)
        qml.adjoint(qml.SingleExcitation(jnp.pi / 3, wires=[0, 1]))
        return qml.state()

In addition, you can qjit most :doc:`PennyLane templates <introduction/templates>` to easily construct and evaluate
more complex quantum circuits.

.. important::

   Decomposition will generally happen in accordance with the specification provided by devices,
   which can vary from device to device (e.g., ``default.qubit`` and ``lightning.qubit`` might
   decompose quite differently).
   However, Catalyst's decomposition logic will differ in the following cases:

   1. For devices without native controlled gates support (e.g., ``lightning.kokkos`` and ``lightning.gpu``), all :class:`qml.Controlled <pennylane.ops.op_math.Controlled>` operations will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
   2. The set of operations supported by Catalyst itself can in some instances lead to additional decompositions compared to the device itself.


Observables
-----------
The Catalyst has support for :doc:`PennyLane observables <introduction/operations>`.

For example, the following circuit is a QJIT compatible function that calculates the expectation value of
a tensor product of a :class:`qml.PauliX <pennylane.PauliX>`, :class:`qml.Hadamard <pennylane.Hadamard>` and :class:`qml.Hermitian <pennylane.Hermitian>` observables.

.. code-block:: python

    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit(x, y):
        qml.RX(x, 0)
        qml.RX(y, 1)
        qml.CNOT([0, 2])
        qml.CNOT([1, 2])
        h_matrix = jnp.array(
            [[complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(-1.0, 0.0)]]
        )
        return qml.expval(qml.PauliX(0) @ qml.Hadamard(1) @ qml.Hermitian(h_matrix, 2))

Measurements
------------
Most PennyLane :doc:`measurement processes <introduction/measurements>`
are supported in Catalyst, although not all features are supported for all measurement types.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :func:`qml.expval() <pennylane.expval>`
     - The expectation value of observables is supported analytically as well as with finite-shots.
   * - :func:`qml.var() <pennylane.var>`
     - The variance of observables is supported analytically as well as with finite-shots.
   * - :func:`qml.sample() <pennylane.sample>`
     - Samples in the computational basis only are supported.
   * - :func:`qml.counts() <pennylane.counts>`
     - Sample counts in the computational basis only are supported.
   * - :func:`qml.probs() <pennylane.probs>`
     - The probabilities is supported in the computational basis as well as with finite-shots.
   * - :func:`qml.state() <pennylane.state>`
     - The state in the computational basis only is supported.
   * - :func:`.measure`
     - The projective mid-circuit measurement is supported via its own operation in Catalyst.

For both :func:`qml.sample() <pennylane.sample>` and :func:`qml.counts() <pennylane.counts>` omitting the wires
parameters produces samples on all declared qubits in the same format as in PennyLane.

Counts are returned a bit differently, namely as a pair of arrays representing a dictionary from basis states
to the number of observed samples. We thus have to do a bit of extra work to display them nicely.
Note that the basis states are represented in their equivalent binary integer representation, inside of a
float data type. This way they are compatible with eigenvalue sampling, but this may change in the future.

.. code-block:: python

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
    def counts():
        qml.Rot(0.1, 0.2, 0.3, wires=[0])
        return qml.counts(wires=[0])
    basis_states, counts = counts()

>>> {format(int(state), '01b'): count for state, count in zip(basis_states, counts)}
{'0': 985, '1': 15}

You can specify the number of shots to be used in sample-based measurements when you create a device.
:func:`qml.sample() <pennylane.sample>` and :func:`qml.counts() <pennylane.counts>` will
automatically use the device's ``shots`` parameter when performing measurements.
In the following example, the number of shots is set to :math:`500` in the device instantiation.

.. note::
    You can return any combination of measurement processes as a tuple from quantum functions.
    In addition, Catalyst allows you to return any classical values computed inside quantum functions as well.

.. code-block:: python

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=3, shots=500))
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.RZ(params[2], wires=2)
        return (
            qml.sample(),
            qml.counts(),
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.probs(wires=[0, 1]),
            qml.state(),
        )

>>> circuit([0.3, 0.5, 0.7])
(Array([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]], dtype=int64),
 (Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int64),
  Array([453,   0,  31,   0,  16,   0,   0,   0], dtype=int64)),
 Array(0.936, dtype=float64),
 Array(0.138816, dtype=float64),
 Array([0.926, 0.048, 0.026, 0.   ], dtype=float64),
 Array([ 0.89994966-0.32850727j,  0.        +0.j        ,
        -0.08388168-0.22979488j,  0.        +0.j        ,
        -0.04964902-0.13601409j,  0.        +0.j        ,
        -0.0347301 +0.01267748j,  0.        +0.j        ],      dtype=complex128))

The PennyLane projective mid-circuit measurement is also supported in Catalyst.
:func:`.measure` is a QJIT compatible mid-circuit measurement for Catalyst that only
requires a list of wires that the measurement process acts on.

.. important::

    The :func:`qml.measure() <pennylane.measure>` function is **not** QJIT compatible and :func:`.measure` from Catalyst should be used instead:

    .. code-block:: python

        from catalyst import measure

In the following example, ``m`` will be equal to ``True`` if wire :math:`0` is rotated by :math:`180` degrees.

.. code-block:: python

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit(x):
        qml.RX(x, wires=0)
        m = measure(wires=0)
        return m

>>> circuit(jnp.pi)
Array(True, dtype=bool)
>>> circuit(0.0)
Array(False, dtype=bool)

Compilation Modes
=================

In Catalyst, there are two ways of compiling quantum functions depending on when the compilation
is triggered.

Just-in-time
------------

In just-in-time (JIT), the compilation is triggered at the call site the first time
the quantum function is executed. For example, ``circuit`` is compiled as early as the first call.

.. code-block:: python

    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit(theta):
        qml.Hadamard(wires=0)
        qml.RX(theta, wires=1)
        qml.CNOT(wires=[0,1])
        return qml.expval(qml.PauliZ(wires=1))

>>> circuit(0.5)  # the first call, compilation occurs here
Array(0., dtype=float64)
>>> circuit(0.5)  # the precompiled quantum function is called
Array(0., dtype=float64)

.. _ahead_of_time:

Ahead-of-time
-------------

An alternative is to trigger the compilation without specifying any concrete values for the function
parameters. This works by specifying the argument signature right in the function definition, which
will trigger compilation "ahead-of-time" (AOT) before the program is executed. We can use both builtin
Python scalar types, as well as the special ``ShapedArray`` type that JAX uses to represent the shape
and data type of a tensor:

.. code-block:: python

    from jax.core import ShapedArray

    @qjit  # compilation happens at definition
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
        theta = jnp.abs(x)
        qml.RY(theta, wires=0)
        qml.Rot(z[0], z[1], z[2], wires=0)
        return qml.state()

>>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
Array([0.75634905-0.52801002j, 0. +0.j,
   0.35962678+0.14074839j, 0. +0.j], dtype=complex128)

At this stage the compilation already happened, so the execution of ``circuit`` calls the compiled function directly on
the first call, resulting in faster initial execution. Note that implicit type promotion for most datatypes are allowed
in the compilation as long as it doesn't lead to a loss of data.

Compiling with Control Flow
============================
Catalyst has support for natively compiled control flow as "first-class" components of any quantum
program, providing a much smaller representation and compilation time for large circuits, and also enabling
the compilation of arbitrarily parametrized circuits.


**Catalyst-provided control flow operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~catalyst.cond
    ~catalyst.for_loop
    ~catalyst.while_loop

.. raw:: html

    </div>

.. note::

    Catalyst supports automatic conversion of native Python control
    flow to the Catalyst control flow operations. For more details,
    see the :doc:`AutoGraph guide <autograph>`.

Conditionals
------------
:func:`.cond` is a functional version of the traditional if-else conditional for Catalyst.
This means that each execution path, a ``True`` branch and a ``False`` branch, is provided as a
separate function. Both functions will be traced during compilation, but only one of them the will be
executed at runtime, depending of the value of a Boolean predicate. The JAX equivalent is the
``jax.lax.cond`` function, but this version is optimized to work with quantum programs in PennyLane.

Note that :func:`.cond` can also be used outside of the :func:`.qjit` context for better interoperability with PennyLane.

Values produced inside the scope of a conditional can be returned to the outside context, but
the return type signature of each branch must be identical. If no values are returned, the
``False`` branch is optional. Refer to the example below to learn more about the syntax of this
decorator.

.. code-block:: python

    @cond(predicate: bool)
    def conditional_fn():
        # do something when the predicate is true
        return "optionally return some value"

    @conditional_fn.otherwise
    def conditional_fn():
        # optionally define an alternative execution path
        return "if provided, return types need to be identical in both branches"

    ret_val = conditional_fn()  # must invoke the defined function

.. warning::

    The conditional functions can only return JAX compatible data types.

Loops
-----
:func:`.for_loop` and :func:`.while_loop` are functional versions of the traditional for- and
while-loop for Catalyst. That is, any variables that are modified across iterations need to be
provided as inputs and outputs to the loop body function. Input arguments contain the value of a
variable at the start of an iteration, while output arguments contain the value at the end of the
iteration. The outputs are then fed back as inputs to the next iteration. The final iteration values
are also returned from the transformed function.

:func:`.for_loop` and :func:`.while_loop` can also be interpreted without needing to compile its surrounding context.

**The for-loop statement:**

The :func:`.for_loop` executes a fixed number of iterations as indicated via the values specified
in its header: a ``lower_bound``, an ``upper_bound``, and a ``step`` size.

The loop body function must always have the iteration index (in the below example ``i``) as its
first argument and its value can be used arbitrarily inside the loop body. As the value of the index
across iterations is handled automatically by the provided loop bounds, it must not be returned from
the body function.

.. code-block:: python

    @for_loop(lower_bnd, upper_bnd, step)
    def loop_body(i, *args):
        # code to be executed over index i starting
        # from lower_bnd to upper_bnd - 1 by step
        return args

    final_args = loop_body(init_args)

The semantics of :func:`.for_loop` are given by the following Python implementation:

.. code-block:: python

    for i in range(lower_bnd, upper_bnd, step):
        args = body_fn(i, *args)

**The while-loop statement:**

The :func:`.while_loop`, on the other hand, is able to execute an arbitrary number of iterations,
until the condition function specified in its header returns ``False``.

The loop condition is evaluated every iteration and can be any callable with an identical signature
as the loop body function. The return type of the condition function must be a Boolean.

.. code-block:: python

    @while_loop(lambda *args: "some condition")
    def loop_body(*args):
        # perform some work and update (some of) the arguments
        return args

    final_args = loop_body(init_args)

Calculating Quantum Gradients
=============================

**Catalyst-provided gradient operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~catalyst.grad
    ~catalyst.jacobian
    ~catalyst.value_and_grad
    ~catalyst.vjp
    ~catalyst.jvp

.. raw:: html

    </div>


:func:`.grad` is a QJIT compatible grad decorator in Catalyst that can differentiate a hybrid quantum function
using finite-difference, parameter-shift, or adjoint-jacobian methods. See the documentation for more details.


This decorator accepts the function to differentiate, a differentiation strategy, and the argument indices of the function with which to differentiate:

.. code-block:: python

    @qjit
    def workflow(x):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(jnp.pi * x, wires=0)
            return qml.expval(qml.PauliY(0))

        g = grad(circuit)
        return g(x)

>>> workflow(2.0)
Array(-3.14159265, dtype=float64)

To specify the differentiation strategy, the ``method`` argument can be passed
to the ``grad`` function:

- ``method="auto"``: Quantum components of the hybrid function are
  differentiated according to the corresponding QNode ``diff_method``, while
  the classical computation is differentiated using traditional autodiff.

  With this strategy, Catalyst only currently supports QNodes with
  ``diff_method="parameter-shift"`` and ``diff_method="adjoint"``.

- ``method="fd"``: First-order finite-differences for the entire hybrid
  function. The ``diff_method`` argument for each QNode is ignored.

Currently, higher-order differentiation is only supported by the
finite-difference method. The gradient of circuits with QJIT compatible
control flow is supported for all methods in Catalyst.

You can further provide the step size (``h``-value) of finite-difference in the :func:`.grad` method.
For example, the gradient call to differentiate ``circuit`` with respect to its second argument using
finite-difference and ``h``-value :math:`0.1` should be:

.. code-block:: python

    g_fd = grad(circuit, method="fd", argnums=1, h=0.1)

Gradients of quantum functions can be calculated for a range or tensor of parameters.
For example, ``grad(circuit, argnums=[0, 1])`` would calculate the gradient of
``circuit`` using the finite-difference method for the first and second parameters.
In addition, the gradient of the following circuit with a tensor of parameters is
also feasible.

.. code-block:: python

    @qjit
    def workflow(params):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(params):
            qml.RX(params[0] * params[1], wires=0)
            return qml.expval(qml.PauliY(0))

        return grad(circuit, argnums=0)(params)

>>> workflow(jnp.array([2.0, 3.0]))
Array([-2.88051099, -1.92034063], dtype=float64)

The :func:`.grad` decorator works for functions that return a scalar value. You can also use the :func:`.jacobian`
decorator to compute Jacobian matrices of general hybrid functions with multiple or multivariate results.

.. code-block:: python

    @qjit
    def workflow(x):
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(jnp.pi * x[0], wires=0)
            qml.RY(x[1], wires=0)
            return qml.probs()

        g = jacobian(circuit, method="auto")
        return g(x)

>>> workflow(jnp.array([2.0, 1.0]))
Array([[ 3.48786850e-16 -4.20735492e-01]
       [-8.71967125e-17  4.20735492e-01]], dtype=float64)

This decorator has the same methods and API as ``grad``. See the documentation for more details.

Optimizers
----------

You can develop your own optimization algorithm using the :func:`.grad` method, control-flow operators that are
compatible with QJIT, or by utilizing differentiable optimizers in `Optax <https://optax.readthedocs.io/en/stable/index.html>`_.

.. warning::

    Catalyst currently does not provide any optimization tools and does not support the optimizers offered
    by PennyLane. However, this feature is planned for future implementation.

For example, you can use ``optax.sgd`` in a QJIT workflow to calculate
the gradient descent optimizer. The following example shows a simple use case of this
feature in Catalyst.

The ``optax.sgd`` gets a smooth function of the form ``gd_fun(params, *args, **kwargs)``
and calculates either just the value or both the value and gradient of the function depending on
the value of ``value_and_grad`` argument. To optimize params iteratively, you later need to use
``jax.lax.fori_loop`` to loop over the gradient descent steps.

.. code-block:: python

    import optax
    from jax.lax import fori_loop

    dev = qml.device("lightning.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(param):
        qml.Hadamard(0)
        qml.RY(param, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    def workflow():
        def gd_fun(param):
            diff = grad(circuit, argnums=0)
            return circuit(param), diff(param)

        opt = optax.sgd(learning_rate=0.4)

        def gd_update(i, args):
            param, state = args
            _, gradient = gd_fun(param)
            (updates, state) = opt.update(gradient, state)
            param = optax.apply_updates(param, updates)
            return (param, state)

        param = 0.1
        state = opt.init(param)
        (param, _) = fori_loop(0, 100, gd_update, (param, state))
        return param

>>> workflow()
Array(1.57079633, dtype=float64)

JAX Integration
===============

Catalyst programs can also be used inside of a larger JAX workflow which uses
JIT compilation, automatic differentiation, and other JAX transforms.

.. note::

    Note that, in general, best performance will be seen when the Catalyst
    ``@qjit`` decorator is used to JIT the entire hybrid workflow. However, there
    may be cases where you may want to delegate only the quantum part of your
    workflow to Catalyst, and let JAX handle classical components (for example,
    due to missing a feature or compatibility issue in Catalyst).

For example, call a Catalyst qjit-compiled function from within a JAX jit-compiled
function:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=1)

    @qjit
    @qml.qnode(dev)
    def circuit(x):
      qml.RX(jnp.pi * x[0], wires=0)
      qml.RY(x[1] ** 2, wires=0)
      qml.RX(x[1] * x[2], wires=0)
      return qml.probs(wires=0)

    @jax.jit
    def cost_fn(weights):
      x = jnp.sin(weights)
      return jnp.sum(jnp.cos(circuit(x)) ** 2)

>>> cost_fn(jnp.array([0.1, 0.2, 0.3]))
Array(1.32269195, dtype=float64)

Catalyst-compiled functions can now also be automatically differentiated
via JAX, both in forward and reverse mode to first-order,

>>> jax.grad(cost_fn)(jnp.array([0.1, 0.2, 0.3]))
Array([0.49249037, 0.05197949, 0.02991883], dtype=float64)

as well as vectorized using ``jax.vmap``:

>>> jax.vmap(cost_fn)(jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
Array([1.32269195, 1.53905377], dtype=float64)
