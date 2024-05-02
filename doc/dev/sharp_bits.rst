Sharp bits and debugging tips
=============================

Catalyst is designed to allow you to take the tools and code patterns you are
familiar with when exploring quantum computing (such as Python, NumPy, JAX,
and PennyLane), while unlocking faster execution and the ability to run
hybrid quantum-classical workflows on accelerator devices.

Similar to JAX, Catalyst does this via the :func:`@qjit <.qjit>` decorator, which captures
hybrid programs written in Python, PennyLane, and JAX, and compiles them to
native machine code --- preserving control flow like conditional branches and loops.

With Catalyst, we aim to support as many idiomatic PennyLane and JAX
hybrid workflow programs as possible, however there will be **various
restrictions and constraints that should be taken into account**.

Here, we aim to provide an overview of the restrictions and constraints
(the 'sharp bits'), as well as debugging tips and common patterns that are
helpful when using Catalyst.

.. note::

    For a more general overview of Catalyst, please see the
    :doc:`quick start guide <quick_start>`.


.. _compile_time:

Compile-time vs. runtime
------------------------

An important distinction to make in Catalyst, which we typically don't have to
worry about with standard PennyLane, is the concept of **compile time**
vs. **runtime**.

Very roughly, the following three processes occur when using the :func:`@qjit <.qjit>` decorator
with just-in-time (JIT) compilation.

#. **Program capture or tracing:** When the :func:`@qjit <.qjit>` decorated function is
   first called (or, when the :func:`@qjit <.qjit>` is first applied if using function
   type hints and :ref:`ahead-of-time mode <ahead_of_time>`), Catalyst
   will 'capture' the entire hybrid workflow with **placeholder variables of
   unknown value** used as the function arguments
   (the **runtime arguments**).

   These symbolic tracer objects represent **dynamic variables**, and are used
   to determine how the JIT compiled function transforms its inputs to
   outputs.

#. **Compilation:** The captured program is then compiled to a parametrized
   binary using the Catalyst compiler.

#. **Execution:** Finally, the compiled function is executed with the
   provided numerical function inputs, and the results returned.

Once the function is first compiled, subsequent executions of the function
will simply re-use the previous compiled binary, allowing steps (1) and (2) to
be skipped. (Note: some cases, such as when the function argument types change,
may trigger re-compilation.)

For example, consider the following, where we print out a variable in the middle of
our :func:`@qjit <~.qjit>` compiled function:

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
have already compiled a binary; the binary is called directly to get the function
result, and again the print statement is never hit.

This allows us to distinguish between computations that happen
at **compile-time** (steps 1 and 2), such as the ``print`` statement above,
and those that happen at **runtime** (step 3).

.. note::

    As a general rule of thumb, things that happen at compile-time
    are slow (or lead to slowdowns), while things that happen at
    runtime are fast (or lead to speadups).

    However, if the same computation is repeated every time the
    compiled function is run (where the results are the same no
    matter the inputs), and it is expensive, then it may be worth
    doing the computation once in Python and use the results
    statically in the program.

    However, computations at compile time cannot depend on the value of
    dynamic variable, since this is not known yet. It can only depend
    on **static variables**, those whose values are known.

.. note::

    A general guideline when working with JIT compilation and Catalyst:

    - Python control flow and third party libraries like NumPy and SciPy will
      be evaluated at compile-time, and can only accept static variables.

    - JAX functions, such as ``jax.numpy``, and Catalyst functions like
      :func:`~.cond` and :func:`~.for_loop` will be evaluated at
      runtime, and can accept dynamic variables.

    Note that if :doc:`AutoGraph <autograph>` is enabled, Catalyst will
    attempt to convert Python control flow to its Catalyst equivalent to
    support dynamic variables.

For example, consider the following:

>>> @qjit
... def f(x):
...     if x > 5:
...       x = x / 2
...     return x ** 2
>>> f(2.)
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]..
The error occurred while tracing the function f at <ipython-input-15-2aa7bf60efbb>:1 for make_jaxpr.
This concrete value was not available in Python because it depends on the value of the argument x.
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError

This function will fail, as the Python ``if`` statement cannot accept a
dynamic variable (a JAX tracer) as an argument.

Instead, we can use Catalyst control flow :func:`~.cond` here:

>>> @qjit
... def f(x):
...     @cond(x > 5.)
...     def g():
...         return x / 2
...     @g.otherwise
...     def h():
...         return x
...     return g() ** 2
>>> f(2.)
array(4.)
>>> f(6.)
array(9.)

Here, both conditional branches are compiled, and only evaluated at runtime
when the value of ``x`` is known.

Note that, if the Python ``if`` statement depends only on values that are
static (known at compile time), this is fine --- the ``if`` statement will
simply be evaluated at compile time rather than runtime:

Let's consider an example where a for loop is evaluated at compile time:

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

    For more details, see the :doc:`AutoGraph guide <autograph>`.

Printing at runtime
-------------------

In the previous section, we saw that the Python ``print`` statement will only
be executed during tracing/compilation, and in particular, will not print
out the value of dynamic variables (since their values are only known at *runtime*).

If we wish to print the value of variables at *runtime*, we can instead use the
:func:`catalyst.debug.print` function:


>>> from catalyst import debug
>>> @qjit
... def g(x):
...     debug.print("Value of x = {x}", x=x)
...     return x ** 2
>>> g(2.)
Value of x = 2.0
array(4.)

Avoiding recompilation
----------------------

In general in Catalyst, recompilation of a QJIT-compiled function will usually
occur when the function is called with different **argument types**
and **shapes**.

For example, consider the following:

>>> @qjit
... def f(x, y):
...     print("Tracing occuring")
...     return x ** 2 + y
>>> f(0.4, 1)
Tracing occuring
array(1.16)
>>> f(0.2, 3)
array(3.04)

However, if we change the argument types in a way where Catalyst can't perform
auto-type promotion before passing the argument to the comppiled function
(e.g., passing a float instead of an integer), recompilation will occur:

>>> f(0.15, 0.65)
Tracing occuring
array(0.6725)

However, changing a float to an integer will not cause recompilation:

>>> f(2, 4.65)
array(8.65)

Similarly, changing the shape of an array will also trigger recompilation:

>>> f(jnp.array([0.2]), jnp.array([0.6]))
Tracing occuring
array([0.64])
>>> f(jnp.array([0.8]), jnp.array([1.6]))
array([2.24])
>>> f(jnp.array([0.8, 0.1]), jnp.array([1.6, -2.0]))
Tracing occuring
array([ 2.24, -1.99])

This is something to be aware of, especially when porting existing PennyLane
code to work with Catalyst. For example, consider the following, where the
size of the input argument determines the number of qubits and gates used:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=4)

    @qjit
    @qml.qnode(dev)
    def circuit(x):
        print("Tracing occurring")

        def loop_fn(i):
            qml.RX(x[i], wires=i)

        for_loop(0, x.shape[0], 1)(loop_fn)()
        return qml.expval(qml.PauliZ(0))

This will run correctly, but tracing and recompilation will occur with every
function execution:

>>> circuit(jnp.array([0.1, 0.2]))
Tracing occurring
array(0.99500417)
>>> circuit(jnp.array([0.1, 0.2, 0.3]))
Tracing occurring
array(0.99500417)

To be explicitly warned about recompilation, you can use ahead-of-time
(AOT) mode, by specifying types and shapes in the function signature
directly:

>>> @qjit
... @qml.qnode(dev)
... def circuit(x: jax.core.ShapedArray((3,), dtype=np.float64)):
...     print("Tracing occurring")
...     def loop_fn(i):
...         qml.RX(x[i], wires=i)
...     for_loop(0, x.shape[0], 1)(loop_fn)()
...     return qml.expval(qml.PauliZ(0))
Tracing occurring

Note that compilation now happens on **function definition**. We can execute
the compiled function as long as the arguments match the specified shapes and
type:

>>> circuit(jnp.array([0.1, 0.2, 0.3]))
array(0.99500417)
>>> circuit(jnp.array([1.4, 1.4, 0.3]))
array(0.16996714)

However, deviating from this will result in recompilation and a warning message:

>>> circuit(jnp.array([1.4, 1.4, 0.3, 0.1]))
UserWarning: Provided arguments did not match declared signature, recompiling...
Tracing occurring
array(0.16996714)

Specifying compile-time constants
---------------------------------

The ``@qjit`` decorator argument ``static_argnums`` allows positional arguments
to be specified which should be treated as compile-time static arguments.

This allows any hashable Python object to be passed to the function during compilation;
the function will only be re-compiled if the hash value of the static arguments change.
Otherwise, re-using previous static argument values will result in no re-compilation:

>>> @qjit(static_argnums=(1,))
... def f(x, y):
...   print(f"Compiling with y={y}")
...   return x + y
>>> f(0.5, 0.3)
Compiling with y=0.3
array(0.8)
>>> f(0.1, 0.3)  # no re-compilation occurs
array(0.4)
>>> f(0.1, 0.4)  # y changes, re-compilation
Compiling with y=0.4
array(0.5)

This functionality can be used to support passing arbitrary Python objects to QJIT-compiled
functions, as long as they are hashable:

.. code-block:: python

    from dataclasses import dataclass

    @dataclass
    class MyClass:
      val: int

      def __hash__(self):
          return hash(str(self))

    @qjit(static_argnums=(1,))
    def f(x: int, y: MyClass):
      return x + y.val

>>> f(1, MyClass(5))
array(6)
>>> f(1, MyClass(6))  # re-compilation
array(7)
>>> f(2, MyClass(5))  # no re-compilation
array(7)

Note that when ``static_argnums`` is used in conjunction with type hinting,
ahead-of-time compilation will not be possible since the static argument values
are not yet available. Instead, compilation will be just-in-time.


Try and compile the full workflow
---------------------------------

When porting your PennyLane code to work with Catalyst and :func:`@qjit <.qjit>`, the
biggest performance advantage you will see is if you compile
your *entire* workflow, not just the QNodes. So think about putting
everything inside your JIT-compiled function, including for loops
(including optimization loops), gradient calls, etc.

Consider the following PennyLane example, where we have a parametrized
circuit, are measuring an expectation value, and are optimizing the result:

.. code-block:: python

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def cost(weights, data):
        qml.AngleEmbedding(data, wires=range(4))

        for x in weights:
            # each trainable layer
            for i in range(4):
                # for each wire
                if x[i] > 0:
                    qml.RX(x[i], wires=i)
                elif x[i] < 0:
                    qml.RY(x[i], wires=i)

            for i in range(4):
                qml.CNOT(wires=[i, (i + 1) % 4])

        return qml.expval(qml.PauliZ(0) + qml.PauliZ(3))

    weights = jnp.array(2 * np.random.random([5, 4]) - 1)
    data = jnp.array(np.random.random([4]))

    opt = jaxopt.GradientDescent(cost, stepsize=0.4, jit=False)

    params = weights
    state = opt.init_state(params)

    for i in range(200):
        (params, _) = tuple(opt.update(params, state, data))

Using PennyLane v0.32 on Google Colab with the Python 3 Google Compute Engine
backend, this optimization takes 3min 28s ± 2.05s to complete.

Let's switch over to `Lightning <https://docs.pennylane.ai/projects/lightning>`__,
our high-performance statevector simulator,
alongside the adjoint differentiation method. To do so, we change the first
two lines of the above code-block to set the device as ``"lightning.qubit"``,
and specify ``diff_method="adjoint"`` in the QNode decorator. With this
change, we have reduced the execution time down to 30.7s ± 1.8s.

We can rewrite this QNode to use Catalyst control flow, and compile
it using Catalyst:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=4)

    @qjit
    @qml.qnode(dev)
    def cost(weights, data):
        qml.AngleEmbedding(data, wires=range(4))

        def layer_loop(i):
            x = weights[i]
            def wire_loop(j):

                @cond(x[j] > 0)
                def trainable_gate():
                    qml.RX(x[j], wires=j)

                @trainable_gate.else_if(x[j] < 0)
                def negative_gate():
                    qml.RY(x[j], wires=j)

                trainable_gate.otherwise(lambda: None)
                trainable_gate()

            def cnot_loop(j):
                qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

            for_loop(0, 4, 1)(wire_loop)()
            for_loop(0, 4, 1)(cnot_loop)()

        for_loop(0, jnp.shape(weights)[0], 1)(layer_loop)()
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(3))

    opt = jaxopt.GradientDescent(cost, stepsize=0.4)

    params = weights
    state = opt.init_state(params)

    for i in range(200):
        (params, _) = tuple(opt.update(params, state, data))

With the quantum function qjit-compiled, the optimization loop
now takes 16.4s ± 1.51s.

However, while the quantum function is now compiled, and the compiled function
is called to compute cost and gradient values, the optimization loop is still
occuring in Python.

Instead, we can write the optimization loop itself as a function and decorate
it with ``@qjit``; this will compile the optimization loop, and allow the full
optimization to take place within Catalyst:

.. code-block:: python

    @qjit
    def optimize(init_weights, data, steps):
        def loss(x):
            dy = grad(cost, argnum=0)(x, data)
            return (cost(x, data), dy)

        opt = jaxopt.GradientDescent(loss, stepsize=0.4, value_and_grad=True)
        update_step = lambda i, *args: tuple(opt.update(*args))

        params = init_weights
        state = opt.init_state(params)

        return for_loop(0, steps, 1)(update_step)(params, state)

The optimization now takes 574ms ± 43.1ms to complete when using 200 steps.
Note that, to compute hybrid quantum-classical gradients within a qjit-compiled function,
the :func:`catalyst.grad` function must be used.

JAX support and restrictions
----------------------------

Catalyst utilizes JAX for program capture, which means you are able to
leverage the many functions accessible in ``jax`` and ``jax.numpy`` to write
code that supports :func:`@qjit <~.qjit>` and dynamic variables.

Currently, we are aiming to support as many JAX functions as possible, however
there may be cases where there is missing coverage. Known JAX functionality
that doesn't work with Catalyst includes:

- ``jax.numpy.polyfit``
- ``jax.numpy.fft``
- ``jax.debug``
- ``jax.scipy.linalg.expm``
- ``jax.numpy.ndarray.at[index]`` when ``index`` corresponds to all array
  indices.

If you come across any other JAX functions that don't work with Catalyst
(and don't already have a Catalyst equivalent), please let us know by opening
a `GitHub issue <https://github.com/PennyLaneAI/catalyst/issues>`__.

While leveraging ``jax.numpy`` makes it easy to port over NumPy-based
PennyLane workflows to Catalyst, we also inherit `various restrictions
and 'gotchas' from JAX
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.
This includes:

* **Pure functions**: Compilation is primarily designed to only work on pure
  functions. That is, functions that do not have any side-effects; the
  output is purely dependent only on function inputs.

* **In-place array updates**: Rather than using in-place array updates, the
  syntax ``new_array = jax_array.at[index].set(value)`` should be used. For
  more details, see `jax.numpy.ndarray.at
  <https://jax.readthedocs.io/en/latest/_autosummary /jax.numpy.ndarray.at.html>`__.

* **Lack of stateful random number generators**: In JAX, random number
  generators are stateless, and the key state must be explicitly updated each time you want to compute a random number. For more details, see the `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__.

* **Dynamic-shaped arrays:** Functions that create or return arrays with
  dynamic shape --- that is, arrays where their shape is determined by a
  dynamic variable at runtime -- are currently not supported in JAX nor
  Catalyst. Typically, workarounds involve rewriting the code to utilize
  ``jnp.where`` where possible.

For more details, please see the `JAX documentation
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__.

Callbacks
---------

When coming across functionality that is not yet supported by Catalyst, such as functions like
``jax.scipy.linalg.expm``, Python callbacks can be used to call arbitrary Python code within
a qjit-compiled function, as long as the return shape and type is known:

.. code-block:: python

    @qjit
    def fn(x):

        A = jnp.sin(x) * jnp.array([0.23, 0.2], [0.43, -0.54.])

        @catalyst.pure_callback
        @jax.jit # since the callback function is pure JAX, we can jit it
        def callback_fn(A) -> jax.ShapeDtypeStruct(A.shape, A.dtype):
            # here we call non-Catalyst compatible code
            return jax.scipy.linalg.expm(A)

        return jnp.cos(callback_fn(A))

>>> fn(0.654)
array([[0.39385058, 0.99369752],
       [0.97097762, 0.74283208]])

Catalyst provides two callback functions:

- :func:`~.pure_callback` supports callbacks of **pure** functions. That is, functions with no
  side-effects that accept parameters and return values. However, the return type and shape of the
  function must be known in advance, and is provided as a type signature.

- :func:`~.debug.callback` supports callbacks of functions with **no** return values. This makes it
   an easy entry point for debugging, for example via printing or logging at runtime.

Note that callbacks do not currently support differentiation, and cannot be used inside
functions that :func:`~.grad` is applied to.

JAX integration
---------------

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

However, a ``jax.jit`` function calling a ``qjit`` function will always result
in a callback to Python, so will be slower than if the function was purely compiled
using ``jax.jit`` or ``qjit``.

If you want to compile some functionality that is not currently Catalyst
compatible, or you want to make use of JAX-supported hardware such as TPUs
for classical processing, mixing ``jax.jit`` and ``qjit`` will allow this.
However, if possible, try to always use ``qjit`` to compile your entire
workflow.

Internal QJIT transformations
-----------------------------

Inside of a qjit-compiled function, JAX transformations
(``jax.grad``, ``jax.jacobian``, ``jax.vmap``, etc.)
can be used **as long as they are not applied to quantum processing**.

>>> @qjit
... def f(x):
...     def g(y):
...         return -jnp.sin(y) ** 2
...     return jax.grad(g)(x)
>>> f(0.4)
array(-0.71735609)

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
array(-0.71735609)

Always use the equivalent Catalyst transformation
(:func:`catalyst.grad`, :func:`catalyst.jacobian`, :func:`catalyst.vjp`, :func:`catalyst.jvp`)
inside of a qjit-compiled function.

Inspecting and drawing circuits
-------------------------------

A useful tool for debugging quantum algorithms is the ability to draw them. Currently,
:func:`@qjit <~.qjit>` compiled QNodes can be used as input to
:func:`qml.draw <pennylane.draw>`, with the following caveats:

- :func:`qml.draw <pennylane.draw>` call must occur outside the :func:`@qjit <.qjit>`

- The ``qml.draw()`` function will only accept plain QNodes as input, *or* QNodes that have been qjit-compiled. It will not accept arbitrary hybrid functions (that may contain QNodes).

- The :func:`catalyst.measure` function is not supported in drawn QNodes

- Catalyst conditional functions, such as :func:`~.cond` and
  :func:`~.for_loop`, will be 'unrolled'. That is, the drawn circuit will
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

Conditional debugging
---------------------

.. note::

    See our :doc:`AutoGraph guide <autograph>` for seamless conversion of
    native Python control flow to QJIT compatible control flow.

There are various constraints and restrictions that should be kept in mind
when working with classical control in Catalyst.

- The return values of all branches of :func:`~.cond` do not have to be the same type;
  Catalyst will perform automatic type promotion (for example, converting integers)
  to floats) where possible.

  >>> @qjit
  ... def f(x: float):
  ...     @cond(x > 1.5)
  ...     def cond_fn():
  ...         return x ** 2  # float
  ...     @cond_fn.otherwise
  ...     def else_branch():
  ...         return 6.  # float
  ...     return cond_fn()
  >>> f(1.5)
  array(6.)

- There may be some cases where automatic type promotion cannot be applied; for example,
  ommitting a return value in one branch (e.g., which by default in Python is equivalent
  to returning ``None``) but not in others. This will result in an error ---
  if other branches do return values, the else branch must be specified.

  >>> @qjit
  ... def f(x: float):
  ...     @cond(x > 1.5)
  ...     def cond_fn():
  ...         return x ** 2
  ...     return cond_fn()
  TypeError: Conditional requires consistent return types across all branches, got:
  - Branch at index 0: [ShapedArray(float64[], weak_type=True)]
  - Branch at index 1: []
  Please specify an else branch if none was specified.

  >>> @qjit
  ... def f(x: float):
  ...     @cond(x > 1.5)
  ...     def cond_fn():
  ...         return x ** 2
  ...     @cond_fn.otherwise
  ...     def else_branch():
  ...         return x
  ...     return cond_fn()
  >>> f(1.6)
  array(2.56)

- Finally, a reminder that conditional functions provided to :func:`~.cond` cannot
  accept any arguments.


Compatibility with PennyLane transforms
---------------------------------------

PennyLane provides a wide variety of
:doc:`transforms <code/qml_transforms>` that
convert a circuit to one or more circuits.

Currently, most PennyLane transforms will work with Catalyst
as long as:

- The circuit does not include any Catalyst-specific features, such
  as Catalyst control flow or measurement,

- The QNode returns only lists of measurement processes,

- AutoGraph is disabled, and

- The transformation does not require or depend on the numeric value of
  dynamic variables.

This includes transforms that generate many circuits,

.. code-block:: python

    @qjit
    @qml.transforms.split_non_commuting
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x,wires=0)
        return [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]

>>> circuit(0.4)
[array(-0.51413599), array(0.85770868)]

as well as transforms that simply map the circuit to another:

.. code-block:: python

    @qjit
    @qml.transforms.merge_rotations()
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RX(x ** 2, wires=0)
        return qml.expval(qml.PauliZ(0))

>>> circuit(0.5)
array(0.73168887)

We can inspect the jaxpr representation of the compiled program, to verify that only
a single RX gate is being applied due to the rotation gate merger:

>>> circuit.jaxpr
{ lambda ; a:f64[]. let
    b:f64[] = func[
      call_jaxpr={ lambda ; c:f64[]. let
          d:f64[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] c
          e:f64[] = integer_pow[y=2] c
          f:f64[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] e
          g:f64[1] = add d f
          h:f64[1] = slice[limit_indices=(1,) start_indices=(0,) strides=(1,)] g
          i:f64[] = squeeze[dimensions=(0,)] h
           = qdevice[
            rtd_kwargs={'shots': 0, 'mcmc': False}
            rtd_lib=/usr/local/lib/python3.10/dist-packages/catalyst/utils/../lib/librtd_lightning.so
            rtd_name=LightningSimulator
          ]
          j:AbstractQreg() = qalloc 2
          k:AbstractQbit() = qextract j 0
          l:AbstractQbit() = qinst[op=RX qubits_len=1] k i
          m:AbstractObs(num_qubits=None,primitive=None) = namedobs[kind=PauliZ] l
          n:f64[] = expval[shots=None] m
          o:AbstractQreg() = qinsert j 0 l
           = qdealloc o
        in (n,) }
      fn=<QNode: wires=2, device='lightning.qubit', interface='auto', diff_method='best'>
    ] a
  in (b,) }

Note that currently PennyLane transforms **cannot** be applied when ``autograph=True``.

Compatibility with PennyLane decompositions
-------------------------------------------

When defining decompositions of PennyLane operations, any control flow depending on dynamic
variables will fail, since decompositions are applied at compile time:

.. code-block:: python

    class RXX(qml.operation.Operation):
        num_params = 1
        num_wires = 2

        def compute_decomposition(self, *params, wires=None):
            theta = params[0]
            ops = []

            if theta == 0.3:
                ops.append(qml.PauliRot(theta / 2 * 2, 'XX', wires=wires))
            else:
                ops.append(qml.PauliRot(theta / 2 * 2, 'XX', wires=wires))

            return ops

    dev = qml.device("lightning.qubit", wires=2)

    @qjit
    @qml.qnode(dev)
    def circuit(theta):
        RXX(theta, wires=[0, 1])
        qml.Hadamard(1)
        return qml.expval(qml.PauliZ(0))

>>> circuit(0.3)
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]..
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError

Instead, Catalyst control flow (such as :func:`~.cond` and :func:`.for_loop`) must be used in order to support control flow on dynamic variables:

.. code-block:: python

    class RXX(qml.operation.Operation):
        num_params = 1
        num_wires = 2

        def compute_decomposition(self, *params, wires=None):
            theta = params[0]

            with qml.tape.QuantumTape() as tape:

                @cond(params[0] == 0.3)
                def branch_fn():
                    qml.PauliRot(theta, 'XX', wires=wires)

                @branch_fn.otherwise
                def branch_fn():
                    qml.PauliRot(theta / 2 * 2, 'XX', wires=wires)

                branch_fn()

            return tape.operations

    @qjit
    @qml.qnode(dev)
    def circuit(theta):
        RXX(theta, wires=[0, 1])
        qml.Hadamard(1)
        return qml.expval(qml.PauliZ(0))

>>> circuit(0.3)
array(0.95533649)

Note that here we make sure to include the Catalyst control flow within a ``QuantumTape`` context.
This is because :func:`~.cond` cannot return operations, only capture queued/instantiated
operations, but the ``Operation.compute_decomposition`` API requires that a list of operations is
returned.

If preferred, AutoGraph can be experimentally enabled on a subset of code within
the decomposition as follows:

.. code-block:: python

    from catalyst.autograph import autograph

    class RXX(qml.operation.Operation):
        num_params = 1
        num_wires = 2

        def compute_decomposition(self, *params, wires=None):
            theta = params[0]

            @autograph
            def f(params):
                if params[0] == 0.3:
                    qml.PauliRot(theta, 'XX', wires=wires)
                else:
                    qml.PauliRot(theta / 2 * 2, 'XX', wires=wires)

            with qml.tape.QuantumTape() as tape:
                f(params)

            return tape.operations

Function argument restrictions
------------------------------

Compiled functions can accept arbitrary function arguments, as long as the
inputs can be represented as `Pytrees
<https://jax.readthedocs.io/en/latest/pytrees.html>`__ --- tree-like
structures built out of Python container objects such as lists, dictionaries,
and tuples --- where the *values* (leaf nodes) are compatible types.

Compatible types includes Booleans, Python numeric types, JAX arrays,
and PennyLane quantum operators.

.. note::

    Non-numeric types, such as strings, are generally not supported as arguments to compiled functions.

For example, consider the following, where we pass arbitrarily nested lists or
dictionaries as input to the compiled function:

>>> f = qjit(lambda *args: args)
>>> x = qml.RX(0.4, wires=0)
>>> y = {"apple": (True, jnp.array([0.1, 0.2, 0.3]))}
>>> f(x, y)
(RX(array(0.4), wires=[0]), {'apple': (array(True), array([0.1, 0.2, 0.3]))})

Arbitrary objects cannot be passed as function arguments, unless they
are registered as Pytrees with compatible data types.

>>> class MyObject:
...     def __init__(self, x, name):
...         self.x = x
...         self.name = name
>>> obj = MyObject(jnp.array(0.4), "test")
>>> f(obj)
TypeError: Unsupported argument type: <class '__main__.MyObject'>

By registring it as a Pytree (that is, specifying to JAX the dynamic and
static compile-time information, we make this object compatible with
Catalyst:

>>> def flatten_fn(my_object):
...     data = (my_object.x,) # Dynamic variables
...     aux = {"name": my_object.name} # static compile-time data
...     return (data, aux)
>>> def unflatten_fn(aux, data):
...     return MyObject(data[0], **aux)
>>> register_pytree_node(MyObject, flatten_fn, unflatten_fn)
>>> f(obj)
<__main__.MyObject at 0x7c061434b820>

Note that the function will only be re-compiled if the custom objects static
compile-time data changes (in this case, ``MyObject.name``); **not** if the
dynamic part of the custom object (``MyObject.x``) changes:

>>> @qjit
... def f(my_object):
...     print("compiling")
...     return my_object.x
>>> f(MyObject(jnp.array(0.1), name="test1"))
Compiling: name=test1
array(0.1)
>>> f(MyObject(jnp.array(0.2), name="test1"))
array(0.2)
>>> f(MyObject(jnp.array(0.2), name="test2"))
Compiling: name=test2
array(0.2)

.. note::

    JAX provides a ``static_argnums`` argument for the ``jax.jit`` function,
    which allows you to specify which arguments to the compile function to treat
    as static compile-time arguments. Changes to these arguments will trigger
    re-compilation.

    The Catalyst ``@qjit`` decorator doesn't yet support this functionality.

Dynamically-shaped arrays
-------------------------

Catalyst provides experimental support for compiling functions that accept
or contain tensors whose dimensions are not know at compile time, without
needing to recompile the function when tensor shapes change.

For example, one might consider a case where a dynamic variable specifies the shape
of a tensor created within (or returned by) the compiled function:

>>> @qjit
... def func(size: int):
...     print("Compiling")
...     return jax.numpy.ones([size, size], dtype=float)
>>> func(3)
Compiling
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
>>> func(4)
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])

We can also pass tensors of variable shape directly as arguments to compiled
functions, however we need to provide the ``abstracted_axes`` argument,
to specify which axes of the tensors should be considered dynamic during compilation.

>>> @qjit(abstracted_axes={0: "n"})
... def sum_fn(x):
...     print("Compiling")
...     return jnp.sum(x)
>>> sum_fn(jnp.array([1., 0.5]))
Compiling
array(1.5)
>>> sum_fn(jnp.array([1., 0.5, 0.6]))
array(2.1)

Note that failure to specify this argument will cause re-compilation each time
input tensor arguments change shape:

>>> @qjit
... def sum_fn(x):
...     print("Compiling")
...     return jnp.sum(x)
>>> sum_fn(jnp.array([1., 0.5]))
Compiling
array(1.5)
>>> sum_fn(jnp.array([1., 0.5, 0.6]))
Compiling
array(2.1)

For more details on using ``abstracted_axes``, please see the :func:`~.qjit` documentation.

Note that using dynamically-shaped arrays within for loops, while loops, and
conditional statements, is not currently supported:

>>> @qjit
... def f(size):
...     a = jnp.ones([size], dtype=float)
...     for i in range(10):
...         a = a
...     @for_loop(0, 10, 2)
...     def loop(_, a):
...         return a
...     return loop(a)
KeyError: 137774138140016

Returning multiple measurements
-------------------------------

A common pattern in PennyLane is to have multiple return statements within
a single QNode, allowing the measurement type to alter based on some condition:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2, shots=10)

    @qml.qnode(dev)
    def circuit(x, sample=False):
        qml.RX(x, wires=0)

        if sample:
            return qml.sample(wires=0)

        return qml.expval(qml.PauliZ(0))

This pattern is currently not supported in Catalyst, and will lead to an error:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=2, shots=10)

    @qjit
    @qml.qnode(dev)
    def circuit(x, sample=False):
        qml.RX(x, wires=0)

        @cond(sample)
        def measure_fn():
            return qml.sample(wires=0)

        @measure_fn.otherwise
        def expval():
            return qml.expval(qml.PauliZ(0))

        return measure_fn()

>>> circuit(3)
TypeError: Value sample(wires=[0]) with type <class 'pennylane.measurements.sample.SampleMP'> is not a valid JAX type

It is recommended for now to create separate QNodes if different measurement statistics need to be
returned, or alternatively using a single return statement with multiple measurements:

>>> @qjit
... @qml.qnode(dev)
... def circuit(x):
...     qml.RX(x, wires=0)
...     return {"samples": qml.sample(), "expval": qml.expval(qml.PauliZ(0))}
>>> circuit(0.3)
{'expval': array(-0.9899925),
 'samples': array([[1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [0., 0.],
        [1., 0.],
        [1., 0.]])}


Recursion
---------

Recursion is not currently supported, and will result in errors. For example,

.. code-block:: python

    @qjit(autograph=True)
    def fibonacci(n: int):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

>>> fibonacci(10)
RecursionError: maximum recursion depth exceeded in comparison


This is due to the fact that during compilation, Catalyst tries to evaluate
both branches of the conditional statement recursively; because there is
``n`` is a dynamic variable, it has no concrete value at compile time, and
tracing can never complete.

Instead, try to write your program without recursion. For example, in this case
we can use a while loop:

.. code-block:: python

    @qjit
    def fibonacci(n):

        @catalyst.while_loop(lambda count, *args: count < n)
        def loop_fn(count, a, b, sum):
            a, b = b, sum
            sum = a + b
            return count + 1, a, b, sum

        _, _, _, result  = loop_fn(1, 0, 1, 1)
        return result

>>> fibonacci(10)
array(89)

Compatibility with broadcasting
-------------------------------

Catalyst does not currently support passing multi-dimensional arrays
as quantum operator parameters ('parameter broadcasting'):

>>> @qml.qnode(dev)
... def circuit(x):
...     qml.RX(x, wires=0)
...     qml.RY(0.1, wires=0)
...     return qml.expval(qml.PauliZ(0))
>>> circuit(jnp.array([0.1, 0.2]))
Array([0.99003329, 0.97517033], dtype=float64)
>>> qjit(circuit)(jnp.array([0.1, 0.2]))
UnboundLocalError: local variable 'baseType' referenced before assignment

While not as flexible as true vectorized quantum operations, as a workaround
``jax.vmap`` can be used to allow for multi-dimensional **function**
arguments:

>>> jax.vmap(qjit(circuit))(jnp.array([0.1, 0.2]))
Array([0.99003329, 0.97517033], dtype=float64)

Note that ``jax.vmap`` cannot be used within a qjit-compiled function:

>>> qjit(jax.vmap(circuit))(jnp.array([0.1, 0.2]))
NotImplementedError: Batching rule for 'qinst' not implemented

Functionality differences from PennyLane
----------------------------------------

The ultimate aim with Catalyst will be the ability to prototype quantum algorithms
in Python with PennyLane, and easily scale up prototypes by simply adding ``@qjit``.
This will require that all PennyLane functionality behaves identically whether or not
the ``@qjit`` decorator is applied.

Currently, however, this is not the case for measurements.

- **Measurement behaviour**. :func:`catalyst.measure` currently behaves
  differently from its PennyLane counterpart :func:`pennylane.measure`.
  In particular:

  - Final measurement statistics occurring after :func:`pennylane.measure`
    will average over all potential measurements, weighted by their
    likelihood.

  - Final measurement statistics occurring after :func:`catalyst.measure` will
    be post-selected on the outcome that was measured. The post-selected
    measurement will change with every execution.
