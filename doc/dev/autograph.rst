AutoGraph guide
===============

.. figure:: ../_static/catalyst-autograph.png
    :width: 70%
    :alt: AutoGraph Illustration
    :align: center

One of the main advantages of Catalyst is that you can represent quantum
programs with **structure**. That is, you can use classical control flow
(such as conditionals and loops) with quantum operations and measurements,
and this structure is captured and preserved during compilation.

Catalyst provides various high-level functions, such as :func:`~.cond`,
:func:`~.for_loop`, and :func:`~.while_loop`, that work with native PennyLane
quantum operations. However, it can sometimes take a bit of work to rewrite
existing Python code using these specific control functions. An experimental
feature of Catalyst, AutoGraph, instead allows Catalyst to work
with **native Python control flow**, such as if statements and for loops.

Here, we'll aim to provide an overview of AutoGraph, as well as various
restrictions and constraints you may discover.

.. note::

    For a more general overview of Catalyst, please see the
    :doc:`quick start guide <quick_start>`.

Using AutoGraph
---------------

AutoGraph currently requires TensorFlow as a dependency; in most cases it can
be installed via

.. code-block:: console

    pip install tensorflow

but please refer to the
`TensorFlow documentation <https://www.tensorflow.org/install>`__
for specific details on installing TensorFlow for your platform.

Once TensorFlow is available, AutoGraph can be enabled by passing
``autograph=True`` to the ``@qjit`` decorator:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=4)

    @qjit(autograph=True)
    @qml.qnode(dev)
    def cost(weights, data):
        qml.AngleEmbedding(data, wires=range(4))

        for x in weights:

            for j, p in enumerate(x):
                if p > 0:
                    qml.RX(p, wires=j)
                elif p < 0:
                    qml.RY(p, wires=j)

            for j in range(4):
                qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

        return qml.expval(qml.PauliZ(0) + qml.PauliZ(3))

>>> weights = jnp.linspace(-1, 1, 20).reshape([5, 4])
>>> data = jnp.ones([4])
>>> cost(weights, data)
array(0.30455313)

This would be equivalent to writing the following program, without using
AutoGraph, but instead using :func:`~.cond` and :func:`~.for_loop`:

.. code-block:: python

    @qjit(autograph=False)
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

>>> cost(weights, data)
array(0.30455313)

Currently, AutoGraph supports converting the following Python control
flow statements:

- ``if`` statements (including ``elif`` and ``else``)
- ``for`` loops

``while`` loops are currently not supported.

Nested functions
----------------

AutoGraph will continue to work even when the qjit-compiled function
itself calls nested functions. All functions called within the
qjit-compiled function will also have Python control flow captured
and converted by AutoGraph.

.. code-block:: python

    def f(x):
        if x > 5:
            y = x ** 2
        else:
            y = x ** 3
        return y

    @qjit(autograph=True)
    def g(x, n):
        for i in range(n):
            x = x + f(x)
        return x

>>> g(0.4, 6)
array(22.14135448)

One way to verify that the control flow is being correctly captured and
converted is to examine the jaxpr representation of the compiled
program:

>>> g.jaxpr
{ lambda ; a:f64[] b:i64[]. let
    c:f64[] = qfor[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; d:i64[] e:f64[]. let
          f:bool[] = gt e 5.0
          g:f64[] = qcond[
            branch_jaxprs=[
              { lambda ; a:f64[] b_:f64[]. let c:f64[] = integer_pow[y=2] a in (c,) },
              { lambda ; a_:f64[] b:f64[]. let c:f64[] = integer_pow[y=3] b in (c,) }
            ]
          ] f e e
          h:f64[] = add e g
        in (h,) }
      body_nconsts=0
    ] 0 b 1 0 a
  in (c,) }

Here, we can see the for loop contained within the ``qcond`` operation, and
the two branches of the ``if`` statement represented by the ``branch_jaxprs``
list.

If statements
-------------

While most ``if`` statements you may write in Python will be automatically
converted, there are some important constraints and restrictions to be aware of.

Return statements
~~~~~~~~~~~~~~~~~

Return statements inside ``if``/``elif``/``else`` statements are not yet
supported. No error will occur, but the resulting function will not have the
expected behaviour.

For example, consider the following pattern, where you return from an ``if``
statement early,

.. code-block:: python

    def f(x):
        if x > 5:
            return x ** 2
        return x ** 3

This will not be correctly captured by AutoGraph, and instead will be
interpreted as

.. code-block:: python

    def f(x):
        if x > 5:
            x = x ** 2
        return x ** 3

Instead of utilizing a return statement, use the following approach instead:

.. code-block:: python

    def f(x):
        if x > 5:
            y = x ** 2
        else:
            y = x ** 3
        return y

Different branches must assign the same type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following will result in an error:

>>> @qjit(autograph=True)
... def f(x):
...     if x > 5:
...         y = 5.0
...     else:
...         y = 4
...     return y
>>> f(0.5)
TypeError: Conditional requires consistent return types across all branches

Instead, make sure that all branches assign the same type to variables:

>>> @qjit(autograph=True)
... def f(x):
...     if x > 5:
...         y = 5.0
...     else:
...         y = 4.0
...     return y
>>> f(0.5)
array(4.)


New variable assignments
~~~~~~~~~~~~~~~~~~~~~~~~

If a new, previously non-existant variable is assigned in one branch, it must
be assigned in **all** branches. This means that you **must** include an
``else`` statement if you are assigning a new variable:

>>> @qjit(autograph=True)
... def f(x):
...     if x > 5:
...         y = 0.4
...     return x
>>> f(0.5)
AutoGraphError: Some branches did not define a value for variable 'y'

If the variable previous exists before the if statement, however, this restriction
does not apply **as long as you don't change the type**:

>>> @qjit(autograph=True)
... def f(x):
...     y = 0.1
...     if x > 5:
...         y = 0.4
...     return y
>>> f(0.5)
array(0.4)

If we change the type of the ``y``, however, we will need to include an
``else`` statement to also change the type:

>>> @qjit(autograph=True)
... def f(x):
...     y = 0.1
...     if x > 5:
...         y = 4
...     else:
...         y = -1
...     return y
>>> f(0.5)
array(-1)

Compatible type assignments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within an if statement, variable assignments must include JAX compatible
types (Booleans, Python numeric types, JAX arrays, and PennyLane quantum
operators). Not compatible types (such as strings) will result in an error:

>>> @qjit(autograph=True)
... def f(x):
...     if x > 5:
...         y = "a"
...     else:
...         y = "b"
...     return y
>>> f(0.5)
TypeError: Value 'a' with type <class 'str'> is not a valid JAX type


For loops
---------

Most ``for`` loop constructs will be properly captured and compiled by AutoGraph.
This includes automatic unpacking and enumeration through JAX arrays:

>>> @qjit(autograph=True)
... def f(weights):
...     z = 0.
...     for i, (x, y) in enumerate(weights):
...         z = i * x + i ** 2 * y
...     return z
>>> weights = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).T
>>> f(weights)
array(8.4)

This also works when looping through Python containers, **as long as the containers
can be converted to a JAX array**:

>>> weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
>>> f(weights)
array(3.4)

If the container cannot be converted to a JAX array (e.g., a list of strings),
then AutoGraph will **not** capture the for loop; instead, AutoGraph will
fallback to Python, and the loop will be unrolled at compile-time:

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=1)

    @qjit(autograph=True)
    @qml.qnode(dev)
    def f():
        params = ["0", "1", "2"]
        for x in params:
            qml.RY(int(x) * jnp.pi / 4, wires=0)
        return qml.expval(qml.PauliZ(0))

>>> f()
array(-0.70710678)

The Python ``range`` function is also fully supported by AutoGraph, even when
its input is a **dynamic variable** (e.g., its numeric value is only known at
compile time):

>>> @qjit(autograph=True)
... def f(n):
...     x = - jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> f(100000)
array(0.57722066)

Indexing within a loop
~~~~~~~~~~~~~~~~~~~~~~

Indexing arrays within a for loop works, but care must be taken.

For example, using static bounds to index a JAX array inside of a for loop:

>>> dev = qml.device("lightning.qubit", wires=3)
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f(x):
...     for i in range(3):
...         qml.RX(x[i], wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> weights = jnp.array([0.1, 0.2, 0.3])
>>> f(weights)
array(0.99500417)

However, indexing within a for loop with AutoGraph requires that the object
indexed is a JAX array or dynamic runtime variable.

If the array you are indexing within the for loop is not a JAX array
or dynamic variable, but an object that can be converted to a JAX array
(such as a NumPy array or a list of floats), then AutoGraph will fail to capture
the for loop, and will fallback to Python to evaluate the loop at compile-time:

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = [0.1, 0.2, 0.3]
...     for i in range(3):
...         qml.RX(x[i], wires=i)
...     return qml.expval(qml.PauliZ(0))
Warning: If you intended for the conversion to happen, make sure that the(now dynamic) loop variable is not used in tracing-incompatible ways,
for instance by indexing a Python list with it. In that case, the list should be wrapped into an array.
To understand different types of JAX tracing errors, please refer to the guide at: https://jax.readthedocs.io/en/latest/errors.html
If you did not intend for the conversion to happen, you may safely ignore this warning.

The compiled function will still execute, but has been compiled without the for
loop (the for loop was unrolled at compilation):

>>> f()
array(0.99500417)

To allow AutoGraph conversion to work in this case, simply convert the list to
a JAX array:

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = jnp.array([0.1, 0.2, 0.3])
...     for i in range(3):
...         qml.RX(x[i], wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> f()
array(0.99500417)


Note that if the object you are indexing **cannot** be converted to a JAX
array? In this case, it is not possible for AutoGraph to capture this for
loop. However, AutoGraph will continue to fallback to Python for interpreting
the for loop:

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = ["0.1", "0.2", "0.3"]
...     for i in range(3):
...         qml.RX(float(x[i]), wires=i)
...     return qml.expval(qml.PauliZ(0))
Warning: If you intended for the conversion to happen, make sure that the(now dynamic) loop variable is not used in tracing-incompatible ways,
for instance by indexing a Python list with it. In that case, the list should be wrapped into an array.
To understand different types of JAX tracing errors, please refer to the guide at: https://jax.readthedocs.io/en/latest/errors.html
If you did not intend for the conversion to happen, you may safely ignore this warning.


.. note::

    If you wish to suppress this warning, or even activate 'strict mode'
    so that AutoGraph warnings are treated as errors, see the :ref:`debugging`
    section.

Dynamic indexing
~~~~~~~~~~~~~~~~

Indexing into arrays where the for loop has **dynamic bounds** (that is, where
the size of the loop is set by a dynamic runtime variable) will also work, as long
as the object indexed is a JAX array:

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f(n):
...     x = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
...     for i in range(n):
...         qml.RY(x[i], wires=0)
...     return qml.expval(qml.PauliZ(0))
>>> f(2)
array(0.70710678)
>>> f(3)
array(-0.70710678)

However AutoGraph conversion will fail if the object being indexed by the
loop with dynamic bounds is **not** a JAX array, because you cannot index
standard Python objects with dyanmic variables.

In this case, AutoGraph will raise a warning, but the compilation of the function
will ultimately fail:

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f(n):
...     x = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
...     for i in range(n):
...         qml.RY(x[i], wires=0)
...     return qml.expval(qml.PauliZ(0))
Warning: If you intended for the conversion to happen, make sure that the(now dynamic) loop variable is not used in tracing-incompatible ways,
for instance by indexing a Python list with it. In that case, the list should be wrapped into an array.
To understand different types of JAX tracing errors, please refer to the guide at: https://jax.readthedocs.io/en/latest/errors.html
If you did not intend for the conversion to happen, you may safely ignore this warning.
TracerIntegerConversionError: The __index__() method was called on traced array with shape int64[].
See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerIntegerConversionError

To resolve this, ensure that all objects that are indexed within dynamic for
loops are JAX arrays.

Break and continue
~~~~~~~~~~~~~~~~~~

Within a for loop, control flow statements ``break`` and ``continue``
are not currently supported. Usage will result in an error:


>>> @qjit(autograph=True)
... def f(x):
...     for i in range(10):
...         x = x + x ** 2
...         if x > 5:
...             break
...     return x
SyntaxError: 'break' outside loop





* for loop which updates a value each iteration is supported.
* temporary local variables can be used inside a loop

.. _debugging:

Debugging
---------

* you can set catalyst.autograph_strict_conversion = True to stop python fallbacks, and instead get exceptions (``AutoGraphError``).

* you can set "catalyst.autograph_ignore_fallbacks", True) to suppress warnings

* autograph source code


Native Python control flow without AutoGraph
--------------------------------------------

mention how native python control works outside of autograph
