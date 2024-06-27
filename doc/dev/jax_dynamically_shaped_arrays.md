Jax dynamically-shaped arrays
=============================

<!--
``` python
import pennylane as qml
from jax import numpy as jnp
from catalyst import qjit, for_loop, while_loop
from catalyst.jax_extras import DBIdx, expand_args, while_loop_expansion_strategy
print("OK")
```

``` result
OK
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
* [Generalisation of the tracing problem](#generalisation-of-the-tracing-problem)
* [TODO](#todo)
* [Catalyst implementation details](#catalyst-implementation-details)
  * [Arguments and results transformations](#arguments-and-results-transformations)
  * [Adding dynamic dimensions](#adding-dynamic-dimensions)
    * [Loop expansion specifics](#loop-expansion-specifics)
    * [For-loop expansion specifics](#for-loop-expansion-specifics)
  * [Adding Jax constants](#adding-jax-constants)
* [Caveats](#caveats)

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


Generalisation of the tracing problem
-------------------------------------

In this section we attempt to generalise the tracing problem. Consider the following schematic
Python program representing a body of some primitive, say, the `for_loop`.

``` python
CONSTANTS = ...

@primitive
def body(*ARGS):
  ... # Calculate RESULTS from ARGS and CONSTANTS
  return RESULTS

```

We call the primitive like this down the program:

``` python
INPUTS = ...
OUTPUTS = body(*INPUTS)
```

What we want is to transform this program into the following Jaxpr:

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

Jax library provides us with roughly the following tracing API, able to handle the tracing of "flat"
programs:

- $newTracingContext() \to Context$
- $newArgTracer(Context) \to Tracer$
- $newConstTracer(Context) \to Tracer$
- $eval(Function, Tracers) \to (Constants, Tracers)$
- A number of $bind_{prim}(Tracers, Attributes) \to Tracers$ for pre-defined primitives.

As users, we want to add primitives supporting nested programs encoded as Python functions. Below we
define how the corresponding $bind$ handler works.

* $bind_{prim}(Function, Inputs, S)$:
  1. $(ExpandedInputs, InputType) \gets deduceInputType(Inputs, S)$
  2. $(Constants, OutputType) \gets abstractEval(Function, InputType, S)$ where `abstractEval` is
     defined as:
     1. $ctx \gets newTracingContext()$
     2. $(Constants1, ExpandedInputs) \gets parseInputType(ctx, InputType, newArgTracer, newConstTracer)$
     3. $(Constants2, ExpandedOutputs) \gets eval(Function, ExpandedInputs)$
     4. $Constants \gets Constants1 + Constants2$
     5. $OutputType \gets deduceOutputType(Constants, ExpandedInputs, ExpandedOutputs, S)$
     6. $return (Constants, OutputType)$
  3. $ExpandedOutputs \gets parseOutputType(Constants, ExpandedInputs, OutputType, S)$
  4. $Outputs \gets collapse(ExpandedOutputs)$
  5. $return(Outputs)$


TODO
----

Below is the old version of the document. TODO: update these descriptions.

The variations of this algorithm are implemented in Catalyst binding functions for `for_loop`,
`while_loop`, `cond`, etc.

- $Inputs$ are input tracers obtained from the user.
- $Outputs_s$ represents output tracers of a Python program obtained using the expansion
  strategy `S`. Any set of tracers might be converted to a Jaxpr program at any time using the core
  Jax IR printer function `to_jaxpr`. Thus, having output tracers is equivalent to having the Jaxpr
  program. The expansion strategy captures specifics of the particular binding function.
- $expandArgs()$ determines the **implicit parameters** using the specified expansion strategy $S$
  and calculates the **input type signature**.
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L800)
- $expandResults()$ calculates **implicit output variables** and obtains the final **output type
  signature**.
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L817)
- $initialize()$  reads the input type information and creates the required tracers in the inner
  tracing context. Note that the function interprets **de Brjuin indices** which might exist in
  inputs.
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L625)
  (inputs)
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L640)
  (outputs)
- $traceNested()$ runs the next recursion step of the tracing. It takes collapsed (not-expanded)
  **list of input tracers** and calculates the **list of output tracers**.


Catalyst implementation details
-------------------------------

### Arguments and results transformations

As illustrated in the overview, in order to transform Python program into a Jaxpr program, arguments
and results of functions needs to be adjusted in the following ways:

1. Dynamic dimensions must be added as implicit arguments to ensure Jaxpr program types are
   correct.
2. Additional "constant" parameters must be added to hoist numeric constants and capture outer-scope
   variables.

### Adding dynamic dimensions

1. We start from the state where the explicit arguments/results mentioned in the source Python
   program are known.
2. The expansion algorithm scans the explicit dimensions and prepends variables to the list of
   explicit arguments.
   - In the basic case, the variables found in the dimensions are added as-is.
   - In case when several tensors use same dimension variables, different decisions are possible. In
     Catalyst, we support the following two cases:
     + (Default) Add only one implicit argument for shared dimension variable. For example:
       `a:f64[d], b:f64[d]` becomes `d:i64[], a:f64[d], b:f64[d]`
     + "Forget" about the sharing and add new variable for every dimension variable separately.
       For example:
       `a:f64[d], b:f64[d]` becomes `d1:i64[], d2:i64[], a:f64[d1], b:f64[d2]`
       This mode is enabled if `experimental_preserve_dimensions` parameter is set to `False`.
3. Produce types (`in_type` for arguments, `out_type` for results), describing the expansion
   results.
   - For arguments, we use `DBIdx` in types to refer to position in the same list. For example:
     + Arguments: `d:i64[], a:f64[d], b:f64[d]`
     + Input type: `[(i64, True), (f64[DBIdx(0)], False), (f64[DBIdx(0)], False)]`
   - For results, we use `InDBIdx` and `OutDBIdx` in type to refer to positions in the argument list
     and the result list (the current one) correspondingly. For example:
     + Arguments: `v:i64[], d:i64[], a:f64[d], b:f64[d]`
     + Results: `e:i64[], a:f64[e], b:f64[d]`
     + Output type: `[(i64, True), (f64[OutDBIdx(0)], False), (f64[InDBIdx(1)], False)]`

In Catalyst, we usually record the number of implicit variables added using
`num_implicit_inputs`/`num_implicit_outputs` attributes.

#### Loop expansion specifics

Loop primitives have notable additional requirements. In order to lower loop bodies, types and
numbers of loop body arguments must match types and numbers of the loop body results.

In the presence of dynamic dimensions, Jax needs determine which dimensions are going to change
during the loop iterations and which one remain the same. Unfortunately, in a single-pass tracer it
is hard to communicate this information to the compiler. We only see iteration-0 arguments and
results and in general we can not extrapolate this information to later iterations.

We developed the following compromise in order to handle this situation:

* By default, we assume that loop results will keep the same dimension sharing pattern as loop
  arguments. For example:

  ``` python
  @for_loop(0, 10, 1)
  def loop(i, a, a_):
      return a, a_
  loop(a0,a0)  # CORRECT: one shared dimension in both inputs and outputs
  ```

  ``` python
  @for_loop(0, 10, 1)
  def loop(i, a, a_):
      b = jnp.ones([sz+1], dtype=float)
      return b, b
  loop(a0,a0)  # CORRECT: still one shared dimension
  ```

  ``` python
  @for_loop(0, 10, 1)
  def loop(i, a, a_):
      b = jnp.ones([sz+1], dtype=float)
      return a, b  # ERROR: dimensions are not the same any more
  loop(a0,a0)
  ```

* With `experimental_preserve_dimensions=False` flag, we treat every same dimension as a 0-iteration
  conincidense. We create separate dimension during the argument/result expansion.

  ``` python
  @for_loop(0, 10, 1, experimental_preserve_dimensions=False)
  def loop(i, a, b, b_):
      return a, b, b_  # CORRECT
      # BUT `b + b_` is not possible, because `b` and `b_` now has different dimensions

  b0 = jnp.ones([sz+1], dtype=float)
  a2, b2, b2_ = loop(a0, b0, b0)
  ```

#### For-loop expansion specifics

A special for for-loops: loop index variable could not be referred using `InDBIdx` index in output
types. For example, in the following program

``` python
@for_loop(0, 10, 1)
def loop(i, a):
    b = jnp.ones([i], dtype=float)
    return b
```

Output type will contain `OuDBIdx` in the dimension of `b`.


### Adding Jax constants

Deduction of Jax constants happens during the final step of the tracing - at the same time with the
Jaxpr program generation. Constants are prepended to argument lists. Thus, the program with
arguments

`d:i64[], a:f64[d], b:f64[d]`

that captures a dinamically-dimensioned tensor `o:f64[od]` from the outside scope might get the
following final list of arguments:

`od:i64, o:f64[od], d:i64[], a:f64[d], b:f64[d]`

In Catalyst, we usually record the number of constants added using `body_nconsts` attribute. This
information is used during the StableHLO lowering.

Caveats
-------

* Dimension variables obtained from constants never matches dimension variables from regular
  parameters. Thus, the following program will raise an error:

  ``` python
  @qjit
  def circuit(sz:int):
      a0 = jnp.ones([sz], dtype=float)

      @for_loop(0, 10, 1)
      def loop(i, a):
          return a + a0  # a0 is a constant, dimension variable is different

      return loop(a0)
  ```




