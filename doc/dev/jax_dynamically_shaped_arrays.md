Jax dynamically-shaped arrays
=============================

<!--
``` python
import pennylane as qml
from jax import numpy as jnp
from catalyst import qjit, for_loop, while_loop
from catalyst.jax_extras import DBIdx, expand_args, while_loop_expansion_strategy
import catalyst
print(catalyst.__revision__)
```

``` result
48948061e2d6a9d4ef1fbceb9f0503d2848958fd
```
-->

<!-- vim-markdown-toc GFM -->

* [Overview of the tracing problem](#overview-of-the-tracing-problem)
* [Definitions](#definitions)
  * [Jax](#jax)
  * [Tracer](#tracer)
  * [AbstractValue](#abstractvalue)
  * [DShapedArray](#dshapedarray)
  * [Input Type and Output Type](#input-type-and-output-type)
  * [DBIdx](#dbidx)
  * [InDBIdx/OutDBIdx](#indbidxoutdbidx)
  * [Primitives and binding](#primitives-and-binding)
  * [Explicit/implicit arguments](#explicitimplicit-arguments)
  * [Expanded/collapsed arguments or results](#expandedcollapsed-arguments-or-results)
* [Generic algorithm for binding nested primitives](#generic-algorithm-for-binding-nested-primitives)
  * [The problem](#the-problem)
  * [The essence of the Jax tracing API](#the-essence-of-the-jax-tracing-api)
  * [The algorithm](#the-algorithm)
* [Input type deduction in loops](#input-type-deduction-in-loops)
  * [Reference Python program](#reference-python-program)
  * [Example 1: dimension is an implicit argument, dimension mutations are allowed](#example-1-dimension-is-an-implicit-argument-dimension-mutations-are-allowed)
  * [Example 2: dimension is a constant, mixing arguments with constants is allowed](#example-2-dimension-is-a-constant-mixing-arguments-with-constants-is-allowed)

<!-- vim-markdown-toc -->

Overview of the tracing problem
-------------------------------

Consider the following Python program:

``` python
@qjit
def circuit(sz:int):
    a0 = jnp.ones([sz], dtype=float)

    @for_loop(0, 10, 1)
    def loop(i, a):
        return a + a0

    a2 = loop(a0)
    return a0 + a2
```

`qjit` decorator at the top calls Catalyst *compiler* on this program rather than interpret
it with the actual data. In order to compile the program, Catalyst runs the function with recording
abstract values passed as parameters and then performs the following transformations:

1. Trace what happens with the abstract values in order to obtain the program in
   [Jaxpr language](https://jax.readthedocs.io/en/latest/jaxpr.html)
2. Lower Jaxpr program further into the StableHLO MLIR dialect
3. Apply a number of MLIR passes in order to lower StableHLO into the LLVM MLIR dialect
4. Emit the LLVM code and compile it into the native shared library

This document focuses on the first step of this workflow, with the emphasis on the representation
and tracing of the dynamically-shaped arrays.

In the following snippet we use the debug facility of Catalyst to read and analyze the Jaxpr
equivalent of the above Python program.

``` python
print(circuit.jaxpr)
```

``` result
{ lambda ; a:i64[]. let
    b:f64[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
    c:f64[a] = for_loop[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; d:i64[] e:f64[d] f:i64[] g:f64[d]. let
          h:f64[d] = add g e
        in (h,) }
      body_nconsts=2
      nimplicit=0
      preserve_dimensions=True
    ] a b 0 10 1 0 b
    i:f64[a] = add b c
  in (i,) }
```

The following points are important to note:

* Jaxpr program has the following structure:
  ```
  { lambda CONSTS ; ARGUMENTS . let
    A0 A1 ... An = primA[ ATTRIBUTES ] OPERANDS
    ...
    Z0 Z1 ... Zn = primZ[ ATTRIBUTES ] OPERANDS
  in Z }
  ```
  where primitive attributes might include nested Jaxpr programs.
* Constants, arguments and the left-hand-side parts of assignments are all lists of variables.
* All Jaxpr variables have types.  `x:f64[3,2]` means that `x` is a 2D tensor with dimensions 3 and
  2 consisting of 64-bit floating point elements. `a:i64[]` means that `a` is a scalar integer
  tensor.
* With the Jax dynamic API flag set to True (Catalyst sets it by default) Jaxpr types are allowed to
  contain variables themselves. `f:64[d]` means that the single dimension of `f` is not known at
  compile time.  What is known is that at runtime the actual dimension will be available as variable
  `d`.
* In the above Python program, loop body had only two arguments (`i` and `a`), while `body_jaxpr` of
  the resulting Jaxpr program has four arguments. Arguments `f` and `g` correspond to the ones
  visible in Python. The additional arguments appeared due to the following different reasons:
  - `e:f64[d]` represents the captured `a0` as it is referenced in the loop body. Jaxpr does not
    allow Python-style capturing, so additional body arguments are required.
  - `d:i64[]` represents the dynamic dimension of both captured variable and the argument. Jaxpr
    does not allow nested structures as well, so, again, additional arguments are needed to pass
    the information about dimensions.
* Loop argument `g:f64[d]` and the loop result `h:f64[d]` have the same type. Jax takes special care
  of propagating types across operations. For types with statically-shaped types, same types have
  same shapes. For dynamically-shaped types, dimension variables must be exactly the same.  Binary
  operators like `add`, `mull` would raise shape mismatch error if this is not the case.
* `body_nconsts=2` shows that the Jaxpr program had 2 constants and 2 arguments originally. Catalyst
  transformed it into the no-constant form at some point. We suggest to think of constants as of
  arguments which we do not know in advance. The fact of their presence and values only becomes
  known after the tracing is finished.
* Just to highlight the difference between compilation and regular evaluation: Catalyst executes
  loop body only once, regardless of iteration numbers which might be even unknown.  This is because
  we only want to record the abstract execution path rather then perform the real computation.


Definitions
-----------

In this section we define concepts required to describe the tracing algorithm. Most of the below
terms were defined by the Jax authors.

### Jax

[Jax](https://jax.readthedocs.io) is a library which turns interpreters into compilers by means of
tracing. Jax supports abstract evaluation of programs in two “source” languages:

- Python: by using the regular Python interpreter with a custom Numpy API implementation.
- Jaxpr: by its own interpreter implemented in Python.

### Tracer

Tracers are objects which track abstract arguments of the traced program. By means of tracers, Jax
turns interpretation of a program into its compilation.

Jax tracers
([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L1522))
are Python classes having the following properties:

- Tracers are in 1-to-1 correspondence with Jaxpr variables. It is not true for Pyhon variables
  because If the Jax dynamic API is enabled, tracers are allowed to contain other tracers. Referring
  a Python variable by name might mean referring to more than one tracer.
- Tracers contain `AbstractValue` objects (typically, the `DShapedArrays`) in their `aval` field,
  describing their type.
- Jax manages unique identifiers of tracers in order to distinguish them from each other.
- Tracers has the full set of `+-/*` etc operators mimicking the real arguments.
- Tracers typically belong to a `Trace` object
  ([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L1780))
  representing tracing context. Tracing a program typically requires users to ask its tracing
  context for argument and, optionally, constant tracers.

Examples:

- `Tracer(id=1111, aval=ShapedArray(shape=[3,4], dtype=int))`
  + Tracer describing a statically-shaped two-dimensional array of integers.
- `Tracer(id=2222, aval=DShapedArray(shape=[ Tracer(id=... aval=...) ], dtype=float))`
  + Tracer describing a dynamically-shaped one-dimensional array of floats. The number of dimensions
    might be unknown at the compile time.

### AbstractValue

AbstractValue is a Python class describing arrays. It typically has `shape` (a list of dimensions)
and `dtype`. Jax comes with two notable abstract value implementations:

- *ShapedArray*
  ([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1525))
  is the implementation supporting constant dimensions. With the Jax dynamic API disabled, it is the
  only allowed implementation.
  + Example: `ShapedArray(shape=[3,4],dtype=int)`
- *DShapedArray*
  ([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1685))
  is an implementation where, in addition to numbers, shapes might also contain dimension variable
  tracers. These tracers must describe scalar integers. Dynamically-shaped arrays becomes available
  if the Jax dynamic API is enabled.

### DShapedArray

DShapedArray
([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1685))
are abstract values describing tracers whose dimensions might be unknown at the compile time. Like
for `ShapedArray`, it contains `shape` and `dtype` fields. Unlike ShapedArrays, this class allows
more freedom in contents `shape`:

- Numbers representing static dimensions.
- If used in the `aval` field of a tracer, shape is allowed to contain other tracers. Nested tracers
  must be scalar (of unit shape) and `int` dtype.
- If used in a type signature (see below), shape is allowed to contain de Brujin indices
  *InDBIdx(val)*, *OutDBIdx(val)* and *DBIdx(val)*. They have special meaning described below.

Examples:

- `DShapedArray(shape=[3,4],dtype=int)` - DShapedArrays are mostly backward compatible with
  ShapedArrays
- `DShapedArray(shape=[Tracer(id=23232, aval=ShapedArray(shape=(),dtype=int)),4,1],dtype=float)` -
  shape might contain scalar integer tracers if this object is used as an abstract value of a
  tracer.
- `DShapedArray(shape=[InDBIdx(val=0),InDBIdx(val=1)],dtype=float)` - shape might contain de Bruijn
  indices if this object is used in a type signature.

### Input Type and Output Type

In Jax, `in_type/out_type` objects are list-like tuples of abstract values paired with Booleans.
Types are mainly used to transfer the information about tracers between tracing contexts. Types are
typically deduced from the in the source context and then interpreted in the target context. The
results of this interpretation are new tracers living in the target scope.

- The MyPy type of Jax types is `Tuple[Tuple[AbstractValue,bool],...]`
  [link](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1304)
- The `bool` tuple elements represent “explicitness” in a source Python program.  Explicit arguments
  explicitly appear as a Python variable. Implicit arguments, in contrast, are added by Jax to match
  Jaxpr language requirements.

The important properties of types are the following. Abstract values found in types are allowed to
include indices (called in Jax "de Brjuin indices" in Jax sources, but for some reason, they contain
regular list indices). These indices are represented with `DBIdx`, `InDBIdx` and `OutDBIdx` Jax
classes.

Example:

```
((DShapedArray(shape=(), dtype=int), False),
 (DShapedArray(shape=(), dtype=int), False),
 (DShapedArray(shape=[OutDBIdx(0),OutDBIdx(1),InDBIdx(0)], dtype=float), True)
)
```

The above tuple may represent an output type of a Jaxpr program returning a 3D tensor along with the
two of its three dynamic dimensions. The third dimension is assumed to be taken from the first input
argument of the current program. See below for the description of this semantics.

### DBIdx

`DBIdx(val)` are input type references. They might present in the shape values of `DShapedArray`
objects, found in the *in_type* signature of Python/Jaxpr programs. The integer value of a
reference are interpreted as an index in the same `in_type` signature tuple.

Input type indices are:

- Produced while analyzing abstract arguments to be passed to a nested program
  [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2321)
- Parsed when creating argument tracers of a nested program
  [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2437)

### InDBIdx/OutDBIdx

`InDBIdx(val)/OutDBIdx(val)` are output type references. references allowed in the shape values of `DShapedArray`
objects found in the *out_type* signatures of Python/Jaxpr programs.

- *InDBIdx(val)* refers to the position in the `in_type` signature tuple of the Jaxpr/Python
  program.
- *OutDBIdx(val)* refers to the position in the `out_type` signature tuple of the Jaxpr/Python
  program.

Output type indices are:

- Produced while analyzing the output abstract values of a nested program
  [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2379)
- Parsed when creating output tracers in the scope of an enclosing program
  [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/pjit.py#L1333)

### Primitives and binding

Binding, in Jax terms, is the process of applying a primitive to an interpreter stack. For
primitives which include nested programs, binding would mean:

  - Calculation of the input type from the primitive's input arguments.
  - Creation of an inner tracing context and the corresponding tracers.
  - Calculation of the output type based on the results and constants of the nested program.
  - Creation of the resulting tracers in the caller's tracing context.

### Explicit/implicit arguments

Separating explicit and implicit arguments makes sense when we trace Python program.  **Explicit**
arguments/results are those which were explicitly mentioned in the source Python program. Implicit
arguments are those that were added in order to pass the dimension information.

For example, in the following Python program:

``` python
def f(sz):
  o = jnp.ones((sz+1,), dtype=float)
  return o
```

We map the Python tracer `o` to the two Jaxpr variables `b:i64[], c:f[b]` to get the following
equivalent Jaxpr program.

```
{ lambda ; a:i64[]. let
    b:i64[] = add a 1
    c:f64[b] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 b
  in (b, c) }
```

The first variable `b:i64[]` is the implicit result, while the `c` is the explicit one.

### Expanded/collapsed arguments or results

In Python, tensors tracers hold all the required information about their shapes. In Jaxpr, in
contrast, shape dimension variables must be passes explicitly as implicit arguments or constants. In
order to encode the program transformation, we attribute argument lists as **collapsed** or
**expanded**, depending on whether implicit arguments are contained in shapes or added to the list.

In Jax the expansion is performed simply by adding together the tuple of deduced **implicit** values
with the original tuple of **explicit** values.

In Catalyst sources we name Python lists or tuples as `expanded_` if the implicit arguments are
known to be already prepended to them.


Generic algorithm for binding nested primitives
-----------------------------------------------

In this section we attempt to generalise the problem of nested primitives tracing. Nested primitives
are those that contain one or more Jaxpr programs in its attributes.

Examples of the already-implemented binders are:
- [for_loop](https://github.com/PennyLaneAI/catalyst/blob/4cb6d246315aa0bad6bc143fc9fa1765c7b60e69/frontend/catalyst/api_extensions/control_flow.py#L245)
- [while_loop](https://github.com/PennyLaneAI/catalyst/blob/4cb6d246315aa0bad6bc143fc9fa1765c7b60e69/frontend/catalyst/api_extensions/control_flow.py#L329)
- [cond](https://github.com/PennyLaneAI/catalyst/blob/4cb6d246315aa0bad6bc143fc9fa1765c7b60e69/frontend/catalyst/api_extensions/control_flow.py#L68)

### The problem

Consider the following schematic Python program representing a body of some primitive.

``` python
CONSTANTS = ...

@primitive
def body(*ARGS):
  ... # Calculate RESULTS from ARGS and CONSTANTS
  return RESULTS

```

Later we call such a primitive down the program like this:

``` python
INPUTS = ...
OUTPUTS = body(*INPUTS)
```

What we want is to get a Jax-bind-like handler which would:

1. Resemble the bind interface for regular (non-nested) Jax primitives.
2. Get all the required information from as little attributes as possible. Specifically, we want it
   to trace the `body` function automatically.
3. Produce the Jaxpr primitive instruction of the following form:
   ```
   { lambda ; ... . let
     OUTPUTS = primitive[
       body = { lambda CONSTANT_ARGS ; ARGS . let
         ...  // calculate RESULTS from ARGS and CONSTANT_ARGS
       in RESULTS };
     ] CONSTANTS INPUTS;
     ...
     in ... }
   ```

### The essence of the Jax tracing API

Jax library provides us with roughly the following tracing API, able to handle the tracing of "flat"
programs (names are different, links show the real closest analogs):

- $findTracingContext() \to Context$, see [find_top_trace](https://github.com/google/jax/blob/1c68577dcdf70b4c0b42805d279b6a5c04d144fe/jax/_src/core.py#L1351)
- $newTracingContext(Context) \to Context$, see [new_dynamic_main2](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L192)
  and [extend_jaxpr_stack](https://github.com/google/jax/blob/1c68577dcdf70b4c0b42805d279b6a5c04d144fe/jax/_src/interpreters/partial_eval.py#L2347)
- $newArgTracer(Context) \to Tracer$, see [new_arg](https://github.com/google/jax/blob/1c68577dcdf70b4c0b42805d279b6a5c04d144fe/jax/_src/interpreters/partial_eval.py#L1914)
- $newConstTracer(Context) \to Tracer$, see [new_const](https://github.com/google/jax/blob/1c68577dcdf70b4c0b42805d279b6a5c04d144fe/jax/_src/interpreters/partial_eval.py#L1921)
- $eval(Function, Tracers) \to (Constants, Tracers)$, see
  [eval_jaxpr](https://github.com/google/jax/blob/1c68577dcdf70b4c0b42805d279b6a5c04d144fe/jax/_src/core.py#L494)
  for Jaxpr programs. For Python programs, this method roughly corresponds to just calling the
  Function.
- $emitJaxprLine(Context, Prim, Tracers, Tracers, Attrs)$ - updates the context with a new line
  of Jaxpr program. Our
  [bind_overwrite_classical_tracers](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_tracer.py#L445)
  does approximately this, also `prim.bind` of every Jax primitive do it. Note: real Jax bind must
  create output tracers and return them. In order to match with this API, we had to make our own
  [DynshapePrimitive](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L924)
  primitive subclass hiding the output type interpretation under the hood.
- $toJaxpr(Context, Tracers) \to Jaxpr$, see [to_jaxpr2](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L466)
  and our `trace_to_jaxpr` wrapper.
- A number of $bind_{prim}(Tracers, Attributes) \to Tracers$ for primitives. For example, see how
  Catalyst defines [non-nested primitives](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_primitives.py#L197).

### The algorithm

As users, we want to add primitives supporting nested programs encoded as Python functions. Below we
define how such an algorithm works in Catalyst: `for_loop`, `while_loop` and `cond` functions follow
the same pattern.

* $bindNested_{Prim}(Function, Inputs, S)$:
  1. $ctx \gets findTracingContext()$
  1. $(ExpandedInputs, InputType) \gets deduceInputType(Inputs, S)$
  2. $(Constants, OutputType, Jaxpr) \gets abstractEval(Function, InputType)$ where `abstractEval` is
     defined as:
     1. $ctx2 \gets newTracingContext(ctx)$
     2. $(Constants1, ExpandedInputs) \gets parseInputType(ctx2, InputType, newArgTracer, newConstTracer)$
     3. $(Constants2, ExpandedOutputs) \gets eval(Function, ExpandedInputs)$
     4. $Constants \gets Constants1 + Constants2$
     5. $OutputType \gets deduceOutputType(Constants, ExpandedInputs, ExpandedOutputs, S)$
     6. $Jaxpr \gets toJaxpr(ctx2, ExpandedOutputs)$
     7. $return (Constants, OutputType, Jaxpr)$
  3. $ExpandedOutputs \gets parseOutputType(Constants, ExpandedInputs, OutputType)$
  4. $emitJaxprLine(ctx, Prim, ExpandedInputs, ExpandedOutputs, {Jaxpr})$
  4. $Outputs \gets collapse(ExpandedOutputs)$
  5. $return(Outputs)$

(Remark: the above algorithm describes the pure-qjit tracing. Quantum tracing is more elaborated,
but the complications are not related to Jax. Basically, in quantum tracing we follow the
re-entrable version of the above algorithm twice: first time with Inputs set to classical value
tracers only, second time with the remaining quantum register tracer. The results of the classical
tracing are cached on the quantum tape which allows us to apply PennyLane tape transformations in
the middle of this process)

In this algorithm, we referred the following utility functions which we defined as Jax extensions:
- `deduceInputType`, see [expand_args](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L806)
- `deduceOutputType`, see [expand_results](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L832)
- `parseInputType`, see [input_type_to_tracers](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L546)
- `parseOutputType`, see [output_type_to_tracers](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L572)

Specifically for loop primitives, we need a certain additional information from users in order to
fit into the one-pass tracing requirement. We encode this information as `S`, see the
[ExpansionStrategy](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L633)
data class. More on this is explained the next section.

Input type deduction in loops
-----------------------------

The following two programs illustrate the choice we have to make in the `deduceInputType` and
`parseInputType` functions.

Both example programs below have loops accepting an argument `a` and a constant tensor `x` of the
same shape (and the loop counter which is not important here). According to the jax dynamic API, in
order to preserve the information about the shape equality in the loop body, we must use the same
dimension variable for both tensors.

Unfortunately, this requirement might contradict to the semantic of loop primitives which require
the same number of arguments and results for their bodies.

As a result, for each dynamic dimension variable which we want to pass as a primitive argument, we
must choose between the following two options:

- Pass the dimension variable as an implicit argument
  + Within the loop body, argument shapes will seem different from constant shapes `=>` no mixing
    operations (like add, mull, etc.) are allowed.
  + Loops carries dimension variables between iterations `=>` changing dimensions in loops are
    allowed.
- Pass the dimension variable as constant
  + Within the loop body, same argument and constant dimensions will be represented by a same
    variable, `=>` mixing operations are allowed.
  + Loop carries only tensors  `=>` changing dimensions is not allowed.

In order to keep the user-facing API simple, we use the single boolean flag
[experimental_preserve_dimensions](https://github.com/PennyLaneAI/catalyst/blob/386149bdf580f7f6364e45d5d7138f3a367add7f/frontend/catalyst/jax_extras/tracing.py#L647)
encoding the said selection for all the variables at once.

### Reference Python program

The following snippet shows a Python program illustrating the capturing vs mutable dimensions
problem in loops. Lines 1 and 2 could not be compiled into Jaxpr at once. Setting the
`experimental_preserve_dimensions` flag to True allows line [1] while setting it to False allows
line [2].

``` python
@qjit(abstracted_axes={0: 'n'})
def g(x, y):

  @for_loop(0, 10, 1, experimental_preserve_dimensions=[True/False])
  def loop(_, a):
    c =  a * x                                  # [1]
    c = jnp.ones([a.shape[0]+1], dtype=float)   # [2]
    return c

  return jnp.sum(loop(y))

a = jnp.ones([3], dtype=float)
b = jnp.ones([3], dtype=float)
g(a, b)
```

### Example 1: dimension is an implicit argument, dimension mutations are allowed

Passing dimension variable as an implicit argument

``` jaxpr
{ lambda dim:i64[] . let
  A:f64[dim] = ...
  B:f64[dim] = ...
  C:f64[dim] = add A B
               ^^^^^^^
               OK

  ... = primitive[body =
                       CONSTANT                 ARGUMENT
                       vvvvvvvvvv               vvvvvvvvvv
              { lambda dim1:i64[] a:f64[dim1] ; dim2:i64[] b:f64[dim2] .
                let c:f64[dim1] = add a b
                                  ^^^^^^^
                                  SHAPE MISMATCH: `dim1` is not `dim2`
                    dim2' = dim2 + 1
                    c':f64[dim2'] = ... // jnp.ones([dim2'], float)
                in (dim2', c')
                   ^^^^^^^^^^^
                   MATCHES THE NUMBER OF ARGUMENTS (2)

              }
        ] dim A dim B

  in (...)
}
```

``` python
@qjit(abstracted_axes={0: 'n'})
def g(x, y):

  @for_loop(0, 10, 1, experimental_preserve_dimensions=True)
  def loop(_, a):
    c =  a * x
    # c = jnp.ones([a.shape[0]+1], dtype=float) # Not possible
    return c

  return jnp.sum(loop(y))

a = jnp.ones([3], dtype=float)
b = jnp.ones([3], dtype=float)
g(a, b)
```

``` result
array(3.)
```

``` python
print(g.jaxpr)
```

``` result
{ lambda ; a:i64[] b:f64[a] c:f64[a]. let
    d:f64[a] = for_loop[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; e:i64[] f:f64[e] g:i64[] h:f64[e]. let
          i:f64[e] = mul h f
        in (i,) }
      body_nconsts=2
      nimplicit=0
      preserve_dimensions=True
    ] a b 0 10 1 0 c
    j:f64[] = reduce_sum[axes=(0,)] d
  in (j,) }
```

### Example 2: dimension is a constant, mixing arguments with constants is allowed

Passing dimension variable as a constant

``` jaxpr
{ lambda dim:i64[] . let
  A:f64[dim] = ...
  B:f64[dim] = ...
  C:f64[dim] = add A B
               ^^^^^^^
               OK

  ... = primitive[body =

                    CONSTANT
                    vvvvvvvvvv
           { lambda dim1:i64[] a:f64[dim1] ; b:f64[dim1] .
                let c:f64[dim1] = add a b
                                  ^^^^^^^
                                  OK
                    dim1' = dim1 + 1
                    c':f64[dim1'] = ... // jnp.ones([dim1'], float)
                in (dim1', c')
                   ^^^^^^^^^^^
                   DOES NOT MATCH THE NUMBER OF ARGUMENTS (1 VS 2)
           }
        ] dim A B

  in (...)
}
```


``` python
@qjit(abstracted_axes={0: 'n'})
def g(x, y):

  @for_loop(0, 10, 1, experimental_preserve_dimensions=False)
  def loop(_, a):
    # c =  a * x # Not possible
    c = jnp.ones([a.shape[0]+1], dtype=float)
    return c

  return jnp.sum(loop(y))

a = jnp.ones([3], dtype=float)
b = jnp.ones([3], dtype=float)
g(a, b)
```

``` result
array(13.)
```

``` python
print(g.jaxpr)
```

``` result
{ lambda ; a:i64[] b:f64[a] c:f64[a]. let
    d:i64[] e:f64[d] = for_loop[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; f:i64[] g:i64[] h:f64[f]. let
          i:i64[] = add f 1
          j:f64[i] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0
            i
        in (i, j) }
      body_nconsts=0
      nimplicit=1
      preserve_dimensions=False
    ] a 0 10 1 0 c
    k:f64[] = reduce_sum[axes=(0,)] e
  in (k,) }
```

