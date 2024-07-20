Callbacks and GPUs
==================

While Catalyst aims to support all classical processing functionality as provided by
JAX, there are sometimes cases where you may need to perform a host callback to execute
arbitrary Python-compatible code. This may include use-cases such as:

- runtime debugging and logging,

- executing classical subroutines on accelerators such as GPUs or TPUs, or

- incorporating non-JAX compatible classical subroutines within a larger QJIT workflow.

Catalyst supports all of these via a collection of callback functions.

Overview
--------

Catalyst provides several callback functions:

- :func:`~.debug.callback` supports callbacks of functions with **no** return values. This makes it
  an easy entry point for debugging, for example via printing or logging at runtime.

- :func:`~.pure_callback` supports callbacks of **pure** functions. That is, functions with no
  side-effects that accept parameters and return values. However, the return type and shape of the
  function must be known in advance, and is provided as a type signature.

  Note that to use :func:`~.pure_callback` within functions that are being differentiated,
  a custom VJP rule **must** be defined so that the Catalyst compiler knows how to
  differentiate the callback. This can be done via the ``pure_callback.fwd`` and
  ``pure_callback.bwd`` methods. See the :func:`~.pure_callback` documentation for
  more details.

- :func:`~.accelerate` is similar to :func:`~.pure_callback` above, but is designed to
  work only with functions that are ``jax.jit`` compatible. As a result of this restriction,
  return types do not have to be provided upfront, and support is provided for executing
  these callbacks directly on classical accelerators such as GPUs and TPUs.

In addition, :func:`~.catalyst.debug.print`, a convenient wrapper around the Python ``print`` function,
is provided for runtime printing support.

Callbacks to arbitrary Python
-----------------------------

When coming across functionality that is not yet supported by Catalyst, such as functions like
``scipy.integrate.simpson``, Python callbacks can be used to call arbitrary Python code within
a qjit-compiled function, as long as the return shape and type is known:

.. code-block:: python

    import scipy as sp

    @pure_callback
    def simpson(x, y) -> float:
        return sp.integrate.simpson(y, x=x)

    @qjit
    def integrate_xsq(a, b):
        x = jnp.linspace(a, b, 100)
        return simpson(x, x ** 2)

>>> integrate_xsq(-1, 1)
Array(0.66666667, dtype=float64)
>>> integrate_xsq(-1, 2)
Array(3., dtype=float64)

Please see the docstring of :func:`~.pure_callback` for more details, including how to define
vector-Jacobian product (VJP) rules for autodifferentiation, and for specifying the return-type
of vector-valued functions.

Callbacks to JIT-compatible code
--------------------------------

If a function is JIT-compatible, then :func:`~.accelerate` can be used, negating the need to manually
provide return shape and dtype information:

.. code-block:: python

    @qjit
    def fn(x):
        x = jnp.sin(x)
        y = catalyst.accelerate(jnp.fft.fft)(x)
        return jnp.sum(y)

>>> x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
>>> fn(x)
Array(4.20735492+0.j, dtype=complex128)

Accelerated functions also fully support autodifferentiation with
:func:`~.grad`, :func:`~.jacobian`, and other Catalyst differentiation functions,
without needing to specify VJP rules manually:

.. code-block:: python

    @qjit
    @grad
    def f(x):
        expm = catalyst.accelerate(jax.scipy.linalg.expm)
        return jnp.sum(expm(jnp.sin(x)) ** 2)

>>> x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
>>> f(x)
Array([[2.80120452, 1.67518663],
       [1.61605839, 4.42856163]], dtype=float64)

Accelerator (GPU and TPU) support
---------------------------------

:func:`~.accelerate` can also be used to execute classical subroutines on
classical accelerators such as GPUs and TPUs:


.. code-block:: python

    @accelerate(dev=jax.devices("gpu")[0])
    def classical_fn(x):
        return jnp.sin(x) ** 2

    @qjit
    def hybrid_fn(x):
        y = classical_fn(jnp.sqrt(x)) # will be executed on a GPU
        return jnp.cos(y)

