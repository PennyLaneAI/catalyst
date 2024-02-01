Tracing and the dynamic dimensions
==================================

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

* [Tracing problem overview](#tracing-problem-overview)
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
* [Catalyst implementation details](#catalyst-implementation-details)
  * [Arguments and results transformations](#arguments-and-results-transformations)
  * [Adding dynamic dimensions](#adding-dynamic-dimensions)
    * [Loop expansion specifics](#loop-expansion-specifics)
    * [For-loop expansion specifics](#for-loop-expansion-specifics)
  * [Adding Jax constants](#adding-jax-constants)
* [Caveats](#caveats)

<!-- vim-markdown-toc -->

Tracing problem overview
------------------------

Consider the following Python program:

``` python
@qjit
def circuit(sz:int):
    a0 = jnp.ones([sz], dtype=float)

    @for_loop(0, 10, 1)
    def loop(i, a):
        return a*sz

    a2 = loop(a0)
    return a0 + a2
```

`qjit` decorator at the top means that we are going to *compile* this program rather than interpret
it directly. In order to do it, Catalyst performs a series of transformations:

1. Trace Python program in order to obtain Jaxpr program
2. Lower Jaxpr program fither into the StableHLO MLIR dialect
3. Apply a series of MLIR passes in order to lower the StableHLO into the LLVM MLIR dialect
4. Emit the LLVM code and compile it into the machine's native binary, rendered as a shared library

This document explains the Tracing step of the workflow, with emphasis on the Jax dynamic API
support.

Since we specified argument type in our program, namely `sz:int`, Catalyst has compiled the program
already. To revisit tracing results, lets print an equivalent in the Jaxpr language, the main IR
language of the Jax library:

``` python
print(circuit.jaxpr)
```

``` result
{ lambda ; a:i64[]. let
    b:f64[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
    _:i64[] c:f64[a] = for_loop[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; d:i64[] e:i64[] f:i64[] g:f64[e]. let
          h:f64[] = convert_element_type[new_dtype=float64 weak_type=False] d
          i:f64[e] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] h e
          j:f64[e] = mul g i
        in (e, j) }
      body_nconsts=1
      nimplicit=1
      preserve_dimensions=True
    ] a a 0 10 1 0 b
    k:f64[a] = add b c
  in (k,) }
```

The following points are important to note:

* Jaxpr program has the following structure:
  ```
  { lambda CONSTS ; ARGUMENTS . let
    A0 A1 ... An = opA[ ATTRIBUTES ] OPERANDS
    ...
    Z0 Z1 ... Zn = opZ[ ATTRIBUTES ] OPERANDS
  in Z }
  ```
  where operation attributes might contain nested Jaxpr programs.
* All Jaxpr variables have types. `x:f64[3,2]` means that `x` is a 2D tensor with dimensions 3 and 2
  consisting of 64-bit floating point elements. `a:i64[]` means that `a` is a scalar integer tensor.
* With dynamic API flag set to True (Catalyst sets it by default) Jaxpr types are allowed to
  contain variables. `f:64[d]` means that the single dimension of `f` is not known at compile time.
  What is known is that at runtime the actual dimension will be available as variable `d`.
* Loop body of the source Python program has two arguments, while the `body_jaxpr` of the resulting
  Jaxpr program has four arguments. The additional arguments appeared due to different reasons:
  - `e:i64[]`" Usage of the outer-scope variable in the body loop in the source Python program.
    Jaxpr program does not allow capturing, so we have to pass captured variables as additional
    arguments.
  - `f:i64[]`: Requirement saying that Jaxpr variable must be declared before use. Since we use
    variable `f` in the type of `h`, we pass it an additional argument.
* Loop argument `b:f64[a]` and loop result `c:f64[a]` have the same types. Jax takes special care of
  propagating type variables across primitives where possible. Jax binary operators like `+`, `*`
  requires operand types to be the same.
* In contrast to the regular Python evaluation, loop body is evaluated only once during the tracing.
  This is because we only want to record the execution path rather then perform the real
  computation.


Definitions
-----------

In this section we define concepts required to describe the tracing algorithm.

### Jax

[Jax](https://jax.readthedocs.io) is a library which turns interpreters into compilers by means of
tracing. Jax supports tracing of programs in two “source” languages:

- Python: regular Python interpreter with a custom Numpy API implementation are used for tracing.
- Jaxpr: the IR language with the interpreter implemented in Python.

### Tracer

Tracers are objects which track arguments of the traced program. By means of tracers, Jax transforms
interpretation of a program into compilation.

Jax tracers
([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L1522))
are Python classes having the following properties:

- Tracers are in 1-to-1 correspondence with Python and Jaxpr variables. If the dynamic api is
  enabled, tracers might contain other tracers, so referring a variable by name would mean referring
  to more than one tracer.
- Contain `AbstractValue` objects (typically, the `DShapedArrays`) in their `aval` field.
- Jax tracks unique identifiers of tracers in order to distinguish them from each other.
- Tracers typically belong to a `Trace` object
  ([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L1780))
  representing variable scope. Creating nested scopes requires users to also create tracers.

Examples:

- `Tracer(id=1111, aval=ShapedArray(shape=[3,4], dtype=int))`
- `Tracer(id=2222, aval=DShapedArray(shape=[ Tracer(id=... aval=...) ], dtype=float))`

### AbstractValue

AbstractValue is a Python class describing `shape` (a list of dimensions) and `dtype` (other
features omitted). Jax comes with two notable implementations:

- *ShapedArray*
  ([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1525))
  is the implementation supporting constant dimensions.
  + Example: `ShapedArray(shape=[3,4],dtype=int)`
- *DShapedArray*
  ([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1685))
  is an implementation where, in addition to constants, variable dimensions are allowed in shapes.
  Dynamic dimensions are represented by other scalar tracers which become valid shape values. This
  implementation becomes available if the Dynamic API flag of JAX is enabled.

### DShapedArray

DShapedArray
([source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1685))
are abstract values describing tracers whose dimensions might be unknown at the compile time. Like
for `ShapedArray`, the main fields are `shape` and `dtype`. Unlike ShapedArrays, this class allows
more freedom in the `shape` contents. Possible elements are:

- Numbers representing static dimensions.
- If used in the `aval` field of a tracer, other tracers are allowed in shapes. Nested tracers must
  be scalar (of unit shape) and `int` dtype.
- If used in a type signature (see below), de Brujin indices *InDBIdx(val)*,
  *OutDBIdx(val)* and *DBIdx(val)* are allowed as values and have special meaning described below.

Examples:

- `DShapedArray(shape=[3,4],dtype=int)` - DShapedArrays are mostly backward compatible with
  ShapedArrays
- `DShapedArray(shape=[Tracer(id=23232, aval=ShapedArray(shape=(),dtype=int)),4,1],dtype=float)` -
  shape might contain scalar integer tracers if this object is an abstract value of a tracer.
- `DShapedArray(shape=[InDBIdx(val=0),InDBIdx(val=1)],dtype=float)` - shape might contain de Bruijn
  indices if this object is contained in a type signature.

### Input Type and Output Type

In Jax, `in_type/out_type` objects are list-like tuples of abstract values paired with Booleans.
Types are mainly used to transfer the information about tracers between scopes. Types are typically
deduced from the in the source scope and then interpreted in a target scope. The results of this
interpretation are new tracers living in the target scope.

- The MyPy type of types is `Tuple[Tuple[AbstractValue,bool],...]`
  [link](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1304)
- The `bool` tuple elements represent “explicitness” of argument in a source Python program.
  Explicit arguments explicitly appear in user-defined Python functions. Implicit arguments, in
  contrast, are added by Jax to match Jaxpr typing requirements.

The important properties of types are the following one. Abstract values found in types are allowed
to include indices (called in Jax "de Brjuin indices" for some reason). The indices are described by
objects of `DBIdx`, `InDBIdx` and `OutDBIdx` classes, all having their purpose.

Example:

```
((DShapedArray(shape=(), dtype=int), False),
 (DShapedArray(shape=(), dtype=int), False),
 (DShapedArray(shape=[OutDBIdx(0),OutDBIdx(1),InDBIdx(0)], dtype=float), True)
)
```

The above tuple may represent an output type of a Jaxpr program returning a 3D tensor along with the
two of its three dynamic dimensions. The last dimension is assumed to be taken from the first input
argument of the current program.

### DBIdx

`DBIdx(val)` are input type references. They are allowed in the shape values of `DShapedArray`
objects, found in the *in_type* signature of Python/Jaxpr programs. The integer value of a
reference are interpreted as an index in the current `in_type` signature tuple.

Input type indices are:

- Produced when calculating the input type of a nested program [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2321)
- Consumed when creating argument tracers of a nested program [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2437)

### InDBIdx/OutDBIdx

`InDBIdx(val)/OutDBIdx(val)` are output type references. references allowed in the shape values of `DShapedArray`
objects found in the *out_type* signatures of Python/Jaxpr programs.

- *InDBIdx(val)* refers to the position in the `in_type` signature tuple of the Jaxpr/Python
  program.
- *OutDBIdx(val)* refers to the position in the `out_type` signature tuple of the Jaxpr/Python
  program.

Output type indices are:

- Produced when calculating the output type of a nested program
  [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2379)
- Consumed when creating output tracers in the scope of an enclosing program
  [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/pjit.py#L1333)

### Primitives and binding

Primitives are like functions in Jaxpr. If the primitive has a body, then binding is associated with
tracing of the program representing primitive’s body. The binding of such a primitive includes:
  - Calculation of the input type based on the input arguments
  - Creation of the inner tracing scope and the corresponding tracers.
  - Calculation of the output type based on the results of the nested program.
  - Creation of the resulting tracers in the caller's tracing scope.

### Explicit/implicit arguments

Separating explicit and implicit arguments makes sense when we trace Python program.  **Explicit**
arguments/results are those which were explicitly mentioned in the source Python program. Implicit
arguments are those that have to be added in order to fit into Jaxpr typing requirements.

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

The first variable `b:i64[]` is the implicit result, while the second one is the explicit one.

### Expanded/collapsed arguments or results

Python and Jaxpr programs represent function arguments and results in a different level of details.
In Python tensors are objects holding all the required information about there shapes. In Jaxpr, in
contrast, shape dimension variables must be passes explicitly as additional arguments. In order to
encode the program transformation, we attribute argument lists as **collapsed** or
**expanded**, depending on whether implicit arguments were added or not.

In Jax the expansion is performed simply by adding together the tuple of deduced **implicit** values
with another tuple of known **explicit** values.

We name lists or tuples as `expanded_` if the implicit arguments are known to be already prepended
to them.


Generalisation of the tracing problem
-------------------------------------

In this section we attempt to generalise the tracing problem. Consider the following schematic
Python program:

``` python
def nested_function(*ARGS):
  ... # Calculate RESULTS from ARGS
  return RESULTS

INPUTS = ... # Obtain INPUTS somehow
OUTPUTS = bind(nested_function, *INPUTS)
```

Notes:
- In this example, `nested_function` plays the role of `for_loop` body, `cond` branch or `adjoint`
  region.
- `bind` represents the binding API function, e.g. `for_loop`.

We want to transform this program into the following Jaxpr program (also schematic):

```
{ lambda ; INPUTS . let
    OUTPUTS = bind[
      nested_function = { lambda ; ARGS . let
        ...  // calculate RESULTS
      in RESULTS };
    ] INPUTS;
  in OUTPUTS }
```

In order to do so, Jax evaluates the source Python program by passing **tracers** objects as INPUTS.
All operations applied to the tracers, including the nested function call, are recorded into the
internal Jax equation list, which is then used to print the final Jaxpr program.

In the above example, **bind** is the most important operation, joining the outer and inner scope
tracing into a single recursive tracing algorithm.

Below we describe one step of this algorithm:

* $bind(Function, Inputs, S) -> Outputs_s$, where:
  1. $(ExpandedInputs_s, InputType_s) \gets expandArgs(Inputs, strategy = S)$
  2. $OutputType_s \gets AbstractEvaluation(InputType_s)$ where $AbstractEvaluation$ is defined as follows:
     1. $ExpandedArguments_s \gets initialize(InputType_s)$
     2. $Arguments \gets collapse(ExpandedArguments_s)$
     3. $Results \gets traceNested(Function, Arguments)$
     4. $OutputType_s \gets expandResults(ExpandedArguments_s, Results)$
     5. $return(OutputType_s)$
  3. $ExpandedOutputs_s \gets initialize(OutputType_s, ExpandedInputs_s)$
  4. $Outputs_s \gets collapse(ExpandedOutputs_s)$
  5. $return(Outputs_s)$

The variations of this algorithm are implemented in every Catalyst binding function such as
`for_loop`, `while_loop`, `cond`, etc. The common properties, however, are the same.

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



