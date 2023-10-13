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

No 'early' returns
~~~~~~~~~~~~~~~~~~

The following pattern, where you return from an ``if`` statement early,

.. code-block:: python

    def f(x):
        if x > 5:
            return x ** 2
        return x ** 3

will not be correctly captured by AutoGraph, and instead will be interpreted as

.. code-block:: python

    def f(x):
        if x > 5:
            x = x ** 2
        return x ** 3

Instead of utilizing an early return statement, use the following approach instead:

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

If we change the type of the ``y``, however, we will need to include an ``else`` statement to also change the type:

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

* Do not currently support ``break`` and ``continue``.

* for x, y in params: works where params is a JAX array
* for x in Obj: works, where obj is convertable to an array
  * if ``obj`` is not convertable to an array (e.g., list of strings) it falls back to standard python.
* Nested lists also works
* for in range works
* for i in range:, with static bounds, works to index an array inside the loop (obj[i])
  * what if obj is not a JAX array, but an ndarray or a list? You will get an actionable warning, and will falback to python
  * what if obj is not even convertable to a JAX array? this will fallback to Python with a non-actionable warning.
* for i in range(n): where n is dynamic works, and can also be used to index an array.
  * However, it will fail without an aurograph conversion if obj is not a JAX array (even if a convertable list) because you cannot index a Python list with a dynamic variable. AutoGraph will raises a warning before the main failure. Solution is to convert to a JAX array.

* for in enumerate works
* enumerate with nested unpacking ``for i, (x1, x2) in enumerate(params):`` also works
* same restrictions as range above

* for loop which updates a value each iteration is supported.
* temporary local variables can be used inside a loop


Debugging
---------

* you can set catalyst.autograph_strict_conversion = True to stop python fallbacks, and instead get exceptions (``AutoGraphError``).

* you can set "catalyst.autograph_ignore_fallbacks", True) to suppress warnings

* autograph source code


Native Python control flow without AutoGraph
--------------------------------------------

mention how native python control works outside of autograph
