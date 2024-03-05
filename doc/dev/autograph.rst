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
existing Python code using these specific control flow functions. An experimental
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
                def trainable_gate():
                    qml.RY(x[j], wires=j)

                trainable_gate()

            def cnot_loop(j):
                qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

            for_loop(0, 4, 1)(wire_loop)()
            for_loop(0, 4, 1)(cnot_loop)()

        for_loop(0, jnp.shape(weights)[0], 1)(layer_loop)()
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(3))

>>> cost(weights, data)
array(0.30455313)

Currently, AutoGraph supports converting the following Python statements:

- ``if`` statements (including ``elif`` and ``else``)
- ``for`` loops
- ``while`` loops
- ``and``, ``or`, and ``not`` in certain cases
- Slice assignment in certain cases, such as ``x[i] = 1``

``break`` and ``continue`` statements are currently not supported.

Nested functions
----------------

AutoGraph will continue to work even when the qjit-compiled function
itself calls nested functions. All functions called within the
qjit-compiled function will also have Python control flow captured
and converted by AutoGraph.

In addition, built-in functions from ``jax``, ``pennylane``, and ``catalyst``
are automatically *excluded* from the AutoGraph conversion when called
within the qjit-compiled function.

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
    c:f64[] = for[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; d:i64[] e:f64[]. let
          f:bool[] = gt e 5.0
          g:f64[] = cond[
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

Different branches of an if statement must always assign variables with the same type across branches,
if those variables are used in the outer scope (external variables). The type must be the same in the sense
that the *structure* of the variable should not change across branches. The underlying data type (`dtype`)
may be different, since data type promotion is applied across branches.

In particular, this requires that if an external variable is assigned an array in one
branch, other branches must also assign arrays of the same shape:

>>> @qjit(autograph=True)
... def f(x):
...     if x > 1:
...         y = jnp.array([0.1, 0.2])
...     else:
...         y = jnp.array([0.4, 0.5, -0.1])
...     return jnp.sum(y)
>>> f(0.5)
AssertionError: Expected matching shapes
>>> @qjit(autograph=True)
... def f(x):
...     if x > 1:
...         y = jnp.array([0.1, 0.2, 0.3])
...     else:
...         y = jnp.array([0.4, 0.5, -0.1])
...     return jnp.sum(y)
>>> f(0.5)
array(0.8)

More generally, this also applies to common container classes such as
`dict`, `list`, and `tuple`. If one branch assigns an external variable,
then all other branches must also assign the external variable with the same
type, nested structure, number of elements, element types, and array shapes.

>>> @qjit(autograph=True)
... def f(x):
...     if x > 1:
...         y = {"a": jnp.array([0.1, 0.2, 0.3]), "b": 6}
...     else:
...         y = {"a": jnp.array([0.5, 0., -0.2]), "b": -1}
...     return y
>>> f(1.5)
{'a': array([0.1, 0.2, 0.3]), 'b': array(6)}

Automatic data type promotion in branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different branches of an if statement may assign external variables with different data types (dtypes) ---
Catalyst will automatically perform data type promotion (such as converting integers to floats):

>>> @qjit(autograph=True)
... def f(x):
...     if x > 5:
...         y = 5.0
...     else:
...         y = 4
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

If the variable exists before the if statement, however, this restriction
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
operators). Non-compatible types (such as strings) used
after the if statement will result in an error:

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
its input is a **dynamic variable** (i.e., its numeric value is only known at
runtime):

>>> @qjit(autograph=True)
... def f(n):
...     x = -jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> f(100000)
array(0.57722066)

Indexing within a loop
~~~~~~~~~~~~~~~~~~~~~~

Indexing arrays within a for loop will generally work, but care must be taken.

For example, using a for loop with static bounds to index a JAX array is straightforward:

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

However, for optimal performance, indexing within a for loop with AutoGraph will require
that the object indexed is a JAX array or dynamic runtime variable.

If the array you are indexing within the for loop is not a JAX array
or dynamic variable, but an object that can be converted to a JAX array
(such as a NumPy array or a list of floats), then AutoGraph will raise a warning,
and fallback to Python to evaluate the loop at compile-time:

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


What if the object you are indexing **cannot** be converted to a JAX
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
standard Python objects with dynamic variables.

In this case, AutoGraph will raise a warning, but the compilation of the function
will ultimately fail:

>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f(n):
...     x = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
...     for i in range(n):
...         qml.RY(x[i], wires=0)
...     return qml.expval(qml.PauliZ(0))
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


Updating and assigning variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For loops that update variables can also be converted with AutoGraph:

>>> @qjit(autograph=True)
... def f(x):
...     for y in [0, 4, 5]:
...         x = x + y
...     return x
>>> f(4)
array(13)

However, like with conditionals, a similar restriction applies: variables
which are updated across iterations of the loop must have a JAX compileable
type (Booleans, Python numeric types, and JAX arrays).

You can also utilize temporary variables within a for loop:

>>> @qjit(autograph=True)
... def f(x):
...     for y in [0, 4, 5]:
...         c = 2
...         x = x + y * c
...     return x
>>> f(4)
array(22)

Temporary variables used inside a loop --- and that are **not** passed to a
function within the loop --- do not have any type restrictions.

While loops
-----------

Most ``while`` loop constructs will be properly captured and compiled by
AutoGraph:

>>> @qjit(autograph=True)
... def f(param):
...     n = 0.
...     while param < 0.5:
...         param *= 1.2
...         n += 1
...     return n
>>> f(0.1)
array(9.)

Break and continue
~~~~~~~~~~~~~~~~~~

Within a while loop, control flow statements ``break`` and ``continue``
are not currently supported. Usage will result in an error:


>>> @qjit(autograph=True)
... def f(x):
...     while x < 5:
...         if x < 0:
...             continue
...         x = x + x ** 2
...     return x
SyntaxError: 'continue' not properly in loop


Updating and assigning variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As with for loops, while loops that update variables can also be converted with AutoGraph:

>>> @qjit(autograph=True)
... def f(x):
...     while x < 5:
...         x = x + 2
...     return x
>>> f(4)
array(6.4)

However, like with conditionals, a similar restriction applies: variables
which are updated across iterations of the loop must have a JAX compileable
type (Booleans, Python numeric types, and JAX arrays).

You can also utilize temporary variables within a while loop:

>>> @qjit(autograph=True)
... def f(x):
...     while x < 5:
...         c = "hi"
...         x = x + 2 * len(c)
...     return x
>>> f(4)
array(8.4)

Temporary variables used inside a loop --- and that are **not** passed to a
function within the loop --- do not have any type restrictions.

Logical statements
------------------

AutoGraph has support for capturing logical statements that involve dynamic variables --- that is,
statements involving ``and``, ``not``, and ``or`` that return booleans --- allowing them to be
computed at runtime.

>>> @qjit(autograph=True)
... def f(x: float, y: float):
...     a = x >= 0.0 and x <= 1.0
...     b = not y >= 1.0
...     return a or b
>>> f(0.5, 1.1)
array(True)
>>> f(1.5, 1.6)
array(False)

Internally, logical statements are converted as follows:

- ``x and y`` to ``jnp.logical_and(x, y)``
- ``x or y`` to ``jnp.logical_or(x, y)``
- ``not x`` to ``jnp.logical_not(x)``

This can be useful when building dynamic circuits, with gates dependent on the output
of multiple measurements. For example,

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=2)

    @qjit(autograph=True)
    @qml.qnode(dev)
    def circuit():
        qml.RX(0.1, wires=0)
        qml.RY(0.5, wires=1)

        m1 = measure(0)
        m2 = measure(1)  

        if m1 and not m2:
            qml.Hadamard(wires=1)
        elif m1 and m2:
            qml.RX(0.5, wires=1)
        else:
            qml.RY(0.5, wires=1)

        return qml.expval(qml.PauliZ(1))

>>> circuit()
array(0.87758256)

Note that there are a couple of important constraints and restrictions that must be
considered when working with logical statements.

Slicing
~~~~~~~

Indexing an array by slice is supported (it does not need to be translated).

Slice updates are also allowed, although---somewhat confusingly---autograph
does not support the `Slice` or `Tuple` types.

This means that you can write

>>> @qjit(autograph=True)
... def f(x):
...     first_dim = x.shape[0]
...     x[first_dim - 1] = 0
...     return x
>>> f(jnp.array([1, 2, 3]))
array([1, 2, 0])

and the translation will work fine. But if you want to use a `Slice` or `Tuple`,
you will have to use
` ``s_`` <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.s_.html#jax.numpy.s_> `
or a similar wrapper.

>>> @qjit(autograph=True)
... def f(x):
...     x[jnp.s_[:-1]] = 0
...     return x
>>> f(jnp.array([1, 2, 3]))
array([0, 0, 3])


All arguments must be dynamic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only cases where **all arguments to the logical statement are dynamic** (that is, dependent on
runtime values) are captured and converted by AutoGraph. Cases where one or both of the arguments
are static will result in the logical statement falling back to Python, and
being interpreted at compile time.

For example,

>>> @qjit(autograph=True)
... def f(x):
...     return x and True
TracerBoolConversionError: Attempted boolean conversion of traced array with shape float64[]..
The error occurred while tracing the function f_1 at /tmp/__autograph_generated_file3sgpmu5h.py:6 for make_jaxpr. This concrete value was not available in Python because it depends on the value of the argument x.

Here, ``x`` is dynamic, but the other argument is static. As a result, Python will attempt
to evaluate this expression at compile time and fail.

To avoid this, please use ``jnp.logical_and(x, y)``, ``jnp.logical_or(x, y)``,
and ``jnp.logical_not(x)`` explicitly if one of your arguments is static:

>>> @qjit(autograph=True)
... def f(x):
...     return jnp.logical_and(x, True)
>>> f(False)
array(False)
>>> f(True)
array(True)

Array arguments
~~~~~~~~~~~~~~~

Note that, like with NumPy and JAX, logical operators apply elementwise to array arguments:

>>> @qjit(autograph=True)  
... def f(x, y):  
...     return x and y
>>> f(jnp.array([0, 1]), jnp.array([1, 1]))
array([False,  True])

Care must therefore be taken when using logical operators within conditional branches;
``jnp.all`` and ``jnp.any`` can be used to generate a single boolean for conditionals:

>>> @qjit(autograph=True)  
... def f(x, y):  
...     if jnp.all(x and y):
...         z = 1
...     else:
...         z = -1
...     return z
>>> f(jnp.array([0, 1]), jnp.array([1, 1]))
array(-1)

.. _debugging:

Debugging
---------

Catalyst provides the following functions to help with debugging AutoGraph:

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~catalyst.autograph_strict_conversion
    ~catalyst.autograph_ignore_fallbacks
    ~catalyst.autograph_source

.. raw:: html

    </div>
    <div class="summary-table">

The global variables ``autograph_strict_conversion`` and ``autograph_ignore_fallbacks``
can be useful for changing the behaviour of AutoGraph, to ensure that it is capturing
what is intended.

To avoid Python fallback behaviour, ``autograph_strict_conversion`` can be set
to ``True``, causing conversion failures to be treated as errors, rather than
falling back to interpreting the control flow via Python.

For example, consider the case of indexing a non-JAX array object within a for
loop. By default, AutoGraph will fallback to Python. If we want to instead ensure
that all parts of our program control flow *are* being captured, we can set
``autograph_strict_conversion``:

>>> catalyst.autograph_strict_conversion = True
>>> dev = qml.device("lightning.qubit", wires=1)
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     params = ["0", "1", "2"]
...     for x in params:
...         qml.RY(int(x) * jnp.pi / 4, wires=0)
...     return qml.expval(qml.PauliZ(0))
AutoGraphError: Could not convert the iteration target ['0', '1', '2'] to array while processing the following with AutoGraph:
  File "<ipython-input-44-dbae11e6d745>", line 7, in f
    for x in params:

In other cases, the fallback behaviour might be preferable, and you may want to
silence AutoGraph warnings; this can be done via ``autograph_ignore_fallbacks``:

>>> catalyst.autograph_strict_conversion = False
>>> catalyst.autograph_ignore_fallbacks = True
>>> @qjit(autograph=True)
... @qml.qnode(dev)
... def f():
...     x = ["0.1", "0.2", "0.3"]
...     for i in range(3):
...         qml.RX(float(x[i]), wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> f()
array(0.99500417)

Finally, we've seen examples above where we have used the JAXPR representation
of the compiled function in order to verify that AutoGraph is correctly capturing
the control flow. In addition, the function :func:`~.autograph_source` allows
you to view the converted Python code generated by AutoGraph:

>>> @qjit(autograph=True)
... def f(n):
...     x = - jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> print(catalyst.autograph_source(f))
def f_1(n):
    with ag__.FunctionScope('f', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        x = -ag__.converted_call(jnp.log, (n,), None, fscope)
        def get_state():
            return (x,)
        def set_state(vars_):
            nonlocal x
            (x,) = vars_
        def loop_body(itr):
            nonlocal x
            k = itr
            x = x + 1 / k
        k = ag__.Undefined('k')
        ag__.for_stmt(ag__.converted_call(range, (1, n + 1), None, fscope), None, loop_body, get_state, set_state, ('x',), {'iterate_names': 'k'})
        return x


Native Python control flow without AutoGraph
--------------------------------------------

It's important to note that native Python control flow --- in cases where the
control flow parameters are static --- will continue to work with
Catalyst **without** AutoGraph. However, if AutoGraph is not enabled, such
control flow will be evaluated at compile time, and not preserved in the
compiled program.


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

For more details, see the :ref:`compile-time vs. runtime <compile_time>`
documentation.
