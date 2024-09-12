JAX integration
===============

Catalyst allows you to write hybrid quantum-classical functions in Python that are just-in-time
compiled with the :func:`~.qjit` decorator, and ultimately leverages modern compilation tools to
speed up quantum applications.

To support this, the Catalyst frontend leverages PennyLane for representing quantum instructions,
and utilizes JAX for classical processing and program capture, which means you are able to leverage
the many functions accessible in ``jax`` and ``jax.numpy`` to write code that supports
:func:`@qjit<~.qjit>` and dynamic variables.

Here, we aim to provide an overview of the JAX integration, including the existing support
and limitations.

JAX 'sharp bits'
----------------

While leveraging ``jax.numpy`` makes it easy to port over NumPy-based
PennyLane workflows to Catalyst, we also inherit `various restrictions
and 'gotchas' from JAX
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.
This includes:

* **Pure functions**: Compilation is primarily designed to only work on pure
  functions. That is, functions that do not have any side-effects; the
  output is purely dependent only on function inputs.

* **Lack of stateful random number generators**: In JAX, random number
  generators are stateless, and the key state must be explicitly updated each time you want to compute a random number. For more details, see the `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__.

* **In-place array updates**: Rather than using in-place array updates, the
  syntax ``new_array = jax_array.at[index].set(value)`` should be used. For
  more details, see `jax.numpy.ndarray.at
  <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html>`__.

  .. note::

      Support is being added for automatically capturing native Python in-place array
      update syntax, and automatically converting it to JAX-compatible syntax via our
      :doc:`AutoGraph feature <autograph>`.

For more details, please see the `JAX documentation
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.

JAX control flow
----------------

It is recommended to always use Catalyst control flow functions :func:`~.for_loop`, :func:`~.cond`,
and :func:`~.while_loop` (or our experimental  :doc:`AutoGraph feature <autograph>`).

However, JAX control flow functions, such as ``jax.lax.cond`` and ``jax.lax.fori_loop``, will work
inside qjit-compiled functions **as long as they are not applied directly to quantum instructions**
and only apply outside of QNodes:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=4, shots=10)

    @qml.qnode(dev)
    def circuit(x):
        N = x.shape[0]

        @catalyst.for_loop(0, N, 1)
        def loop_fn(i):
            qml.RX(x[i], wires=i)

        loop_fn()
        return [qml.expval(qml.PauliZ(i)) for i in range(N)]

    @qjit
    def fn(x):

        def cost(j, x):
            return jnp.stack(circuit(x))

        return jax.lax.fori_loop(0, 10, cost, x)

>>> fn(jnp.array([0.1, 0.2, 0.3, 0.5]))
Array([0.6, 0.6, 0.8, 1. ], dtype=float64)

Function support
----------------

Currently, we are aiming to support as many JAX functions as possible, however
there may be cases where there is missing coverage. Known JAX functionality
that doesn't work with Catalyst includes:

- ``jax.numpy.polyfit``
- ``jax.numpy.fft``
- ``jax.numpy.ndarray.at[index]`` when ``index`` corresponds to all array
  indices.

If you come across any other JAX functions that don't work with Catalyst
(and don't already have a Catalyst equivalent), please let us know by opening
a `GitHub issue <https://github.com/PennyLaneAI/catalyst/issues>`__.

Note that there is certain JAX functionality we do not expect to or plan
to support in Catalyst qjit-compiled functions. This includes:

- ``jax.debug``. Please use instead the Catalyst provided :func:`~.print`, :func:`~.callback`,
  and :func:`~.pure_callback` functions.

- JAX device placement. Please use instead the :func:`~.accelerate` decorator.

- Certain functions in the `jax.lax.debug module <https://jax.readthedocs.io/en/latest/jax.lax.html>`__
  which are direct wrappers of XLA functionality with no LLVM/MLIR equivalent.

Dynamically-shaped arrays
-------------------------

One common 'gotcha' of JAX jit-compiled functions is that they cannot create or return arrays with
dynamic shape --- that is, arrays where their shape is determined by a dynamic variable at runtime.
Typically, workarounds involve rewriting the code to utilize ``jnp.where`` where possible.

In Catalyst, however, we have enabled support for dynamically-shaped arrays; qjit-compiled
functions can accept, create, and return arrays of dynamic shape without triggering re-compilation:

>>> @qjit
... def func(size: int):
...     print("Compiling")
...     return jax.numpy.ones([size, size], dtype=float)
>>> func(3)
Compiling
Array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]], dtype=float64)
>>> func(4)
Array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float64)

Dynamic arrays can be created using ``jnp.ones`` and ``jnp.zeros``. Note that ``jnp.arange``
and ``jnp.linspace`` do not currently support generating dynamically-shaped arrays (however, unlike
``jnp.arange``, ``jnp.linspace`` *does* support dynamic variables for its ``start`` and ``stop``
arguments).

For more details, see :ref:`dynamic-arrays`.

JAX transforms on QJIT functions
--------------------------------

Compiled functions remain JAX compatible, and you can call JAX transformations
on them, such as ``jax.grad`` and ``jax.vmap``. You can even call ``jax.jit``
on functions that call qjit-compiled functions:

>>> dev = qml.device("lightning.qubit", wires=2)
>>> @qjit
... @qml.qnode(dev)
... def circuit(x):
...     qml.RX(x, wires=0)
...     return qml.expval(qml.PauliZ(0))
>>> @jax.jit
... def workflow(y):
...     return jax.grad(circuit)(jnp.sin(y))
>>> workflow(0.6)
Array(-0.53511382, dtype=float64, weak_type=True)
>>> jax.vmap(circuit)(jnp.array([0.1, 0.2, 0.3]))
Array([0.99500417, 0.98006658, 0.95533649], dtype=float64)

However, a ``jax.jit`` function calling a ``qjit`` function will always result
in a callback to Python, so will be slower than if the function was purely compiled
using ``jax.jit`` or ``qjit``.

.. note::

    Best performance will be seen when the Catalyst
    ``@qjit`` decorator is used to JIT the entire hybrid workflow. However, there
    may be cases where you may want to delegate only the quantum part of your
    workflow to Catalyst, and let JAX handle classical components.


Internal QJIT JAX transformations
---------------------------------

Inside of a qjit-compiled function, JAX transformations
(``jax.grad``, ``jax.jacobian``, ``jax.vmap``, etc.)
can be used **as long as they are not applied to quantum processing**.

>>> @qjit
... def f(x):
...     def g(y):
...         return -jnp.sin(y) ** 2
...     return jax.grad(g)(x)
>>> f(0.4)
Array(-0.71735609, dtype=float64)

If they are applied to quantum processing, an error will occur:

>>> @qjit
... def f(x):
...     @qml.qnode(dev)
...     def g(y):
...         qml.RX(y, wires=0)
...         return qml.expval(qml.PauliX(0))
...     return jax.grad(lambda y: g(y) ** 2)(x)
>>> f(0.4)
NotImplementedError: must override

Instead, only Catalyst transformations will work when applied to hybrid
quantum-classical processing:

>>> @qjit
... def f(x):
...     @qml.qnode(dev)
...     def g(y):
...         qml.RX(y, wires=0)
...         return qml.expval(qml.PauliZ(0))
...     return grad(lambda y: g(y) ** 2)(x)
>>> f(0.4)
Array(-0.71735609, dtype=float64)

Always use the equivalent Catalyst transformation
(:func:`catalyst.grad`, :func:`catalyst.jacobian`, :func:`catalyst.vjp`, :func:`catalyst.jvp`)
inside of a qjit-compiled function.
