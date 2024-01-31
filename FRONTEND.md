Catalyst frontend architecture
==============================

<!--
``` python
import pennylane as qml
import jax.numpy as jnp
from catalyst import qjit, for_loop
print("OK")
```

``` result
OK
```
-->

<!-- vim-markdown-toc GFM -->

* [Tracing problem overview](#tracing-problem-overview)
* [Tracing generalization](#tracing-generalization)
* [Terms and definitions](#terms-and-definitions)
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
* [Catalyst implementation details](#catalyst-implementation-details)

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
    return a2
```

`qjit` decorator at the top means that we are going to *compile* this program rather than interpret
it directly. In order to do this, Catalyst performs a series of transformations, the highlevel
overview of the whole chain is below:

1. **Trace** the Python program in order to obtain Jaxpr program
2. Lower Jaxpr program into the StableHLO MLIR dialect
3. Apply a series of MLIR transformations in order to lower the StableHLO into the LLVM dialect
4. Emit the LLVM code and compile it into the machine's native binary, rendered as a shared library

This document explains the Tracing step of this workflow in details. Below we print the target
program written in Jaxpr language, the main IR language used within Jax:

``` python
print(circuit.jaxpr)
```

``` result
{ lambda ; a:i64[]. let
    b:f64[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
    c:i64[] d:f64[c] = for_loop[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; e:i64[] f:i64[] g:i64[] h:f64[f]. let
          i:f64[] = convert_element_type[new_dtype=float64 weak_type=False] e
          j:f64[f] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] i f
          k:f64[f] = mul h j
        in (f, k) }
      body_nconsts=1
      nimplicit=1
      preserve_dimensions=True
    ] a a 0 10 1 0 b
  in (c, d) }
```

The following points are important to note:

* All Jaxpr variables have types. `x:f64[3,2]` means that `x` is a 2D tensor with dimensions 3 and 2
  consisting of 64-bit floating point elements.
* Jaxpr types are allowed to contain variables. `f:64[d]` means that the single dimension of `f` is
  not known at compile time. What is known is that at runtime the actual dimension will be available
  as variable `d`.
* Loop body of the source Python program has two arguments, while the `body_jaxpr` of the resulting
  Jaxpr program has four arguments. The additional two arguments appeared because:
  - `e:i64[]`" Usage of the outer-scope variable in the body loop in the source Python program.
    Jaxpr program does not allow capturing, so we have to pass captured variables as additional
    arguments.
  - `f:i64[]`: Requirement saying that Jaxpr variable must be declared before use. Since we use
    variable `f` in the type of `h`, we pass it an additional argument.
* In contrast to the regular Python evaluation, loop body is evaluated only once during the tracing.
  This is because we only want to record the execution path rather then perform the real
  computation.


Definitions
-----------

In this section we define terms and concepts required to describe the tracing algorithm.

### Jax

Jax is a library which turns interpreters into compilers by means of tracing. Jax supports tracing
of programs in two “source” languages:

- Python: regular Python interpreter with a custom Numpy API implementation are used for tracing.
- Jaxpr: the IR language with the interpreter implemented in Python.

### Tracer

Tracers are the objects which track arguments of input programs. By means of tracers, Jax turns
Python interpretation process into a process of program transformation.

Jax tracers are Python classes having the following properties:

- Are in 1-to-1 correspondence with Python and Jaxpr variables. If the dynamic api
  is enabled, tracers might contain other tracers, so referring a variable by name would mean
  referring to more than one tracer.
- Contain AbstractValue subclass in the `aval` field
- Jax tracks unique identifiers of tracers in order to distinguish them from each other.

Example representation:

- `Tracer(id=232323, aval=DShapedArray(shape=[3,4],dtype=int))`

### AbstractValue

AbstractValue is a Python class describing `shape` (a list of dimensions) and `dtype` (other
features omitted). Jax comes with two notable implementations:

- *ShapedArray*: the implementation supporting constant dimensions.
  + Example: `ShapedArray(shape=[3,4],dtype=int)`

- *DShapedArray*: in addition to constants, variable dimensions are allowed. They are represented by
  scalar tracers which become valid shape values. This implementation becomes available if one
  enables the Dynamic API of JAX.

### DShapedArray

DShapedArray are abstract values describing tracers whose dimensions might be unknown at the compile
time. Like in `ShapedArray`, the main fields are `shape` and `dtype`. Unlike ShapedArrays, this
class allows more freedom among the `shape` contents:

- As in `ShapedArray`, numbers represent static dimensions.
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

`in_type/out_type` are lists of tuples of abstract values paired with Booleans. They encode input
and output parts of the signature of a Jaxpr program.

- The MyPy type is `tuple[tuple[AbstractValue,bool],...]` [link](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/core.py#L1304)
- The `bool` tuple elements represent “explicitness” of an argument in a Python program. Explicit
  arguments appear in user-defined Python functions. Implicit arguments, in contrast, are added by
  Jax to pass the dimension variables.

Example:

```
((DShapedArray(shape=(), dtype=int), False),
 (DShapedArray(shape=(), dtype=int), False),
 (DShapedArray(shape=[OutDBIdx(0),OutDBIdx(1),InDBIdx(0)], dtype=float), True)
)
```

The above tuple may represent an output type of a Jaxpr program returning a 3D tensor along with the
two of its three dimensions. The last dimension is to be taken from the first input argument of the
program.

### DBIdx

`DBIdx(val)` is a reference allowed in the shape values of `DShapedArray` objects, found in
the *in_type* signature of Python/Jaxpr programs. The integer value of the reference are interpreted
as an index in the current `in_type` signature tuple.

Properties:

- Produced when calculating the input type of a nested program [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2321)
- Consumed when creating argument tracers of a nested program [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2437)

### InDBIdx/OutDBIdx

`InDBIdx(val)/OutDBIdx(val)` are relative references allowed in the shape values of `DShapedArray`
objects found in the *out_type* signatures of Python/Jaxpr programs.

- *InDBIdx(val)* refers to the position in the `in_type` signature tuple of a Jaxpr/Python program.
- *OutDBIdx(val)* refers to the position in the current `out_type` signature tuple of a Jaxpr/Python
  program.

Source locations (related to the pjit primitive binding):

- Produced when calculating the output type of a nested program [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/interpreters/partial_eval.py#L2379)
- Consumed when creating output tracers in the scope of an enclosing program [source](https://github.com/google/jax/blob/88a60b808c1f91260cc9e75b9aa2508aae5bc9f9/jax/_src/pjit.py#L1333)

### Primitives and binding

Primitives are like functions in Jaxpr. Binding a primitive is associated with tracing of the
subprogram representing primitive’s body. The binding involves the calculation of the output type
signature from the known input type signature of the subprogram.


### Explicit/implicit arguments

Separating explicit and implicit arguments makes sense when we argue about the Python tracing. In
Jaxpr, all arguments are explicit.

Jaxpr variables holding array dimensions must always present in the same scope with their array
variables. For Python, this requirement is relaxed so functions may take and return array variables
alone. Jax does the dimension variable tracking automatically by linking tracers. The consequence of
this - Jaxpr programs might have more arguments and results then their source Python programs. The
automatically added dimension arguments are called *implicit*.

For example, when tracing the following Python program:

``` python
def f(sz):
  o = jnp.ones((sz+1,), dtype=float)
  return o
```

We might need to map the Python tracer `o` to the two variables `b:i64[], c:f[b]` of the following
equivalent Jaxpr program

```
{ lambda ; a:i64[]. let
    b:i64[] = add a 1
    c:f64[b] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 b
  in (b, c) }
```

In Jax this is done by adding together the tuple of **explicit** values with another
tuple of calculated **implicit** values.

Note that we also link corresponding type signature using the appropriate `*DBIdx` indices.


### Expanded/collapsed arguments or results

Python and Jaxpr programs represent function arguments and results in a different level of details.
In Python tensors are objects holding all the required information about there shapes. In Jaxpr, in
contrast, shape dimension variables must be passes explicitly as additional arguments. In order to
encode the program transformation, we attribute argument lists as **collapsed** or
**expanded**, depending on whether implicit arguments were added or not.

We use `expanded_` prefix in Python list name if the implicit arguments are known to be already
prepended to the list.


Tracing problem generalization
------------------------------

In this section we attempt to generalize the tracing problem. Consider the following schematic
Python program:

``` python
def nested_function(*ARGS):
  ... # calculate RESULTS from ARGS
  return RESULTS

INPUTS = ... # Obtain INPUTS from the context
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

In order to do so, Jax evaluates the source Python program passing **tracers** objects as INPUTS.
All operations applied to the tracers, including the nested function call, are recorded into the
internal Jax equation list, which is then used to print the final Jaxpr program.

In the above example, **bind** in Jax's terms, is the most important operation, joining the outer
and inner tracing processes into a single recursive tracing algorithm.

Below we give the description of one recursion step of the tracing algorithm:

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

The variations of this algorithm is implemented in the Catalyst repository for every binding
function, examples are `for_loop`, `while_loop`, `cond`, etc. Below we describe its steps in more
details and give source references.

- $read()$ obtains input **Tracers** from the context.
- $Outputs_s$ represents output tracers of a Python program obtained using the expansion
  strategy `S`. Any set of tracers might be converted to a Jaxpr program at any time using the core
  Jax IR printer function `to_jaxpr`. Thus, having output tracers is equivalent to having the Jaxpr
  program.
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

TODO

