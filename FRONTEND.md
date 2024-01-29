Catalyst frontend architecture
==============================

<!-- vim-markdown-toc GFM -->

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
* [Tracing problem](#tracing-problem)

<!-- vim-markdown-toc -->

Terms and definitions
---------------------

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

Separating explicit and implicit arguments makes sense when we speak about the Python tracing. In
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

In Jax this is done by expanding the one-element result tuple of **explicit** values by adding a
tuple of calculated **implicit** values and by linking them using the de Bruijn indices in type
signatures.

If we want to use this code as a body of some primitive, we need the binding program to collapse the
expanded results of the Jaxpr subprogram back into the structured tracer representing the single
output Python variable to continue the Python-tracing of the outermost program.


Tracing problem
---------------

To understand the tracing problem, consider the following schematic Python program:

``` python
def nested_function(*ARGS):
  ... # calculate RESULTS from ARGS
  return RESULTS

INPUTS = ... # Obtain INPUTS
OUTPUTS = bind(nested_function, *INPUTS)
```

We want to transform this program into another program in (schematic) Jaxpr language:

```
{ lambda ; INPUTS . let
    OUTPUTS = call[
      nested_function = { lambda ; ARGS . let
        ...  // calculate RESULTS
      in RESULTS };
    ] INPUTS;
  in OUTPUTS }
```

In order to do so, Jax evaluates the source Python program passing **tracers** objects as INPUTS.
All operations applied to the tracers, including the nested function call, are recorded into the
internal Jax equation list, which is then used to print the final Jaxpr program.

In the above example, nested function call, or **bind** in Jax's terms, is shown because it is
indeed the most important operation, joining the outer and inner tracing processes into a single
recursive tracing algorithm.

Consider the formal description of the single recursion step of the tracing algorithm.

1. $(Inputs, S) \gets read()$ (obtain from the context)
2. $(ExpandedInputs_s, InputType_s) \gets expandArgs(Inputs, strategy = S)$
3. $OutputType_s \gets AbstractEvaluation(InputType_s)$ where $AbstractEvaluation$ is defined as follows:
    1. $ExpandedArguments_s \gets initialize(InputType_s)$
    2. $Arguments \gets collapse(ExpandedArguments_s)$
    3. $Results \gets traceNested(Arguments)$
    4. $OutputType_s \gets expandResults(ExpandedArguments_s, Results)$
    5. $return(OutputType_s)$
4. $ExpandedOutputs_s \gets initialize(OutputType_s, ExpandedInputs_s)$
5. $return(ExpandedOutputs_s, OutputType_s)$

The above algorithm is implemented in the Catalyst repository. It differs from the similar algorithm
of the upstream Jax by the extended support of **Dynamic shapes**. Below we describe its steps in a
more details and give source code references.

- $read()$ obtains input **Tracers** from the context.
- $expandArgs()$ determines the **implicit parameters** using the specified expansion strategy $S$ and
  calculates the **input type signature**.
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L800)
- $expandResults()$ calculates **implicit output variables** and obtains the final **output type
  signature**.
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L817)
- $initialize()$  reads the input type information and creates the required tracers in the inner
  tracing context. Note that the function needs an access to input type in order to interpret **de
  Brjuin indices** which might be contained in inputs.
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L625)
  (inputs)
  [Source](https://github.com/PennyLaneAI/catalyst/blob/7349a7e05868289142a237f7c62aa6ddc60563ea/frontend/catalyst/utils/jax_extras.py#L640)
  (outputs)
- $traceNested()$ runs the next recursion step of the tracing. It takes collapsed (not-expanded)
  **list of input tracers** and calculates the **list of output tracers**.



