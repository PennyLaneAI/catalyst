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
    ~catalyst.autograph_source

.. raw:: html

    </div>
    <div class="summary-table">


Compile-time vs. runtime
------------------------

An important distinction to make in Catalyst, which we typically don't have to
worry about with standard PennyLane, is the concept of **compile time**
vs. **runtime**.

Very roughly, the following three processes occur when using the ``@qjit`` decorator
with just-in-time (JIT) compilation.

#. **Program capture or tracing:** When the ``@qjit`` decorated function is
   first called (or, when the ``@qjit`` is first applied if using function
   type hints and :ref:`ahead-of-time mode <ahead_of_time>`), Catalyst
   will 'capture' the entire hybrid workflow with **placeholder variables of
   unknown value** used as the function arguments
   (the **runtime arguments**). 

   These symbolic tracer objects represent **dynamic variable**, and are used
   to determine how the JIT compiled function transforms its inputs to
   outputs.

#. **Compilation:** The captured program is then compiled to a parametrized
   binary using the Catalyst compiler.

#. **Execution:** Finally, the compiled function is executed with the
   provided numerical function inputs, and the results returned.

Once the function is first compiled, subsequent executions of the function
will simply re-use the previous compiled binary, allowing steps (1) and(2) to
be skipped. (Note: some cases, such as if the function argument types change,
may trigger re-compilation.)

For example, consider the following, where we print out a variable in the middle of
our ``@qjit`` compiled function:

>>> @qjit
... def f(x):
...     print(f"x = {x}")
...     return x ** 2
>>> f(2.)
x = Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
array(4.)
>>> f(3.)
array(9.)

We can see that on the first execution, program capture/tracing occurs, and we
can see the dynamic variable is printed (tracers capture *type*
and *shape*, but not numeric value). This captured program is compiled, and
then the binary is executed directly to return the function value --- the
print statement is never invoked with the numerical value of ``x``.

When we execute the function again, steps (1) and (2) are skipped since we
have already compiled a binary; this is called directly to get the function
result, and again the print statement is never hit.

This allows us to distinguish between computations that happen
at **compile-time** (steps 1 and 2), such as the ``print`` statement above,
and those that happen at **runtime** (step 3).

.. note::

    As a general rule of thumb, for a function that is repeatedly executed
    with different parameters, we want as much evaluation as possible to
    happen at compile time.

    However, computations at compile time cannot depend on the value of
    dynamic variable, since this is not known yet. It can only depend
    on **static variables**, those whose values are known.

.. note::

    A general guideline when working with JIT compilation and Catalyst:

    - Python control flow and third party libraries like NumPy and SciPy will
      be evaluated at compile-time, and can only accept static variables.

    - JAX functions, such as ``jax.numpy``, and Catalyst functions like
      ``catalyst.cond`` and ``catalyst.for_loop`` will be evaluated at
      runtime, and can accept dynamic variables.

    Note that if AutoGraph is enabled, Catalyst will attempt to convert Python
    control flow to its Catalyst equivalent to support dynamic variables.

For example, consider the following:

>>> @qjit
... def f(x):
...     if x > 5:
...       x = x / 2
...     return x ** 2
>>> f(2.)
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]..
The error occurred while tracing the function f at <ipython-input-15-2aa7bf60efbb>:1 for make_jaxpr. This concrete value was not available in Python because it depends on the value of the argument x.
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError

This function will fail, as the Python ``if`` statement cannot accept a dynamic variable (a JAX tracer) as an argument.

Instead, we can use Catalyst control flow :func:`~.cond` here:

>>> @qjit
... def f(x):
... 
...     @cond(x > 5.)
...     def g():
...         return x / 2
... 
...     @g.otherwise
...     def h():
...         return x
...     
...     return g() ** 2
>>> f(2.)
array(4.)
>>> f(6.)
array(9.)

Here, both conditional branches are compiled, and only evaluated at runtime
when the value of ``x`` is known.

.. note::

    AutoGraph is an experimental feature that converts Python control flow
    that depends on dynamic variables to Catalyst control flow behind the
    scenes:


    >>> @qjit(autograph=True)
    ... def f(x):
    ...     if x > 5.:
    ...         print(x)
    ...         x = x / 2
    ...     return x ** 2
    >>> Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
    ... array(4.)
    >>> f(6.)
    ... array(9.)

    For more details, see the AutoGraph guide.

Note that, if the Python ``if`` statement depends only on values that are
static (known at compile time), this is fine --- the ``if`` statement will
simply be evaluated at compile time rather than runtime:

>>> @qjit
... def f(x):
...     for i in range(2):
...         print(i, x)
...         x = x / 2
...     return x ** 2
>>> f(2.)
0 Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
1 Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
array(0.25)

Here, the for loop is evaluated at compile time (notice the multiple tracers
that have been printed out during program capture --- one for each loop!),
rather than runtime.

JAX support and restrictions
----------------------------

Catalyst is utilizes JAX for program capture, which means you are able to
leverage the many functions accessible in ``jax`` and ``jax.numpy`` to write
code that supports ``@qjit`` and dynamic variables.

Currently, we are aiming to support as many JAX functions as possible, however
there may be cases where there is missing coverage. Known JAX functionality
that doesn't work with Catalyst includes:

- ``jax.numpy.polyfit``
- ``jax.debug``
- ``jax.numpy.ndarray.at[index]`` when ``index`` corresponds to all array
  indices.

If you come across any other JAX functions that don't work with Catalyst
(or don't already have a Catalyst equivalents), please let us know by opening
a `GitHub issue <https://github.com/PennyLaneAI/catalyst/issues>`__.

While leveraging ``jax.numpy`` makes it easy to port over NumPy-based
PennyLane workflows to Catalyst, we also inherit `various restrictions
and 'gotchas' from JAX
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.
This includes:

* **Pure functions**: compilation is primarily designed to only work on pure
  functions. That is, functions that do not have any side-effects; the
  output is purely dependent only on function inputs.

* **In-place array updates**: Rather than using in-place array updates, the
  syntax ``new_array = jax_array.at[index].set(value)`` should be used.
  For more details, see `jax.numpy.ndarray.at <https://jax.readthedocs.io/en/latest/_autosummary  /jax.numpy.ndarray.at.html>`__.

* **Lack of stateful random number generators**: In JAX, random number
  generators need to be explicitly created within the ``@qjit`` function
  using ``jax.random.PRNGKey(int)``:
  
  >>> @qjit()
  ... def f():
  ...     key = jax.random.PRNGKey(0)
  ...     a = jax.random.normal(key, shape=(1,))
  ...     return a
  >>> f()
  array([-0.78476578])

* **Dynamic-shaped arrays:** functions that create or return arrays with
  dynamic shape --- that is, arrays where their shape is determined by a
  dynamic variable at runtime -- are currently not supported in JAX nor
  Catalyst. Typically, workarounds involve rewriting the code to utilize
  ``jnp.where`` where possible.

For more details, please see the `JAX documentation
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.

Inspecting and drawing circuits
-------------------------------

A useful tool for debugging quantum algorithms is the ability to draw them. Currently,
``@qjit`` compiled QNodes used as input to ``qml.draw``, with the following caveats:

- ``qml.draw`` must occur outside the ``qjit``

- The ``qjit`` decorator must be placed directly on top of the QNode

- The ``catalyst.measure`` function is not supported in drawn QNodes

- Catalyst conditional functions, such as ``catalyst.cond`` and
  ``catalyst.for_loop``, will be 'unrolled'. That is, the drawn circuit will
  be a straight-line circuit, without any of the control flow represented
  explicitly.

For example,

.. code-block:: python

    @qjit
    @qml.qnode(dev)
    def circuit(x):
        def measurement_loop(i, y):
            qml.RX(y, wires=0)
            qml.RY(y ** 2, wires=1)
            qml.CNOT(wires=[0, 1])

            @cond(y < 0.5)
            def cond_gate():
                qml.CRX(y * jnp.exp(- y ** 2), wires=[0, 1])

            cond_gate()

            return y * 2

        for_loop(0, 3, step=1)(measurement_loop)(x)
        return qml.expval(qml.PauliZ(0))

>>> print(qml.draw(circuit)(0.3))
0: ──RX(0.30)─╭●─╭●─────────RX(0.60)─╭●──RX(1.20)─╭●─┤  <Z>
1: ──RY(0.09)─╰X─╰RX(0.27)──RY(0.36)─╰X──RY(1.44)─╰X─┤     

At the moment, additional PennyLane `circuit inspection functions
<https://docs.pennylane.ai/en/stable/introduction/inspecting_circuits.html>`__
are not supported with Catalyst.

Dynamic circuit restrictions
----------------------------

Todo.

Classical control debugging
---------------------------

Todo.

PennyLane transformations
-------------------------

Todo.

Common PennyLane patterns for Catalyst
--------------------------------------

Todo.
