# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains classes and decorators for just-in-time and ahead-of-time
compilation of hybrid quantum-classical functions using Catalyst.
"""

import copy
import functools
import inspect
import logging
import os
import warnings

import jax
import jax.numpy as jnp
import pennylane as qml
from jax.interpreters import mlir
from jax.tree_util import tree_flatten, tree_unflatten
from malt.core import config as ag_config

import catalyst
from catalyst.autograph import ag_primitives, run_autograph
from catalyst.compiled_functions import CompilationCache, CompiledFunction
from catalyst.compiler import CompileOptions, Compiler
from catalyst.debug.instruments import instrument
from catalyst.from_plxpr import trace_from_pennylane
from catalyst.jax_tracer import lower_jaxpr_to_mlir, trace_to_jaxpr
from catalyst.logging import debug_logger, debug_logger_init
from catalyst.passes import PipelineNameUniquer, _inject_transform_named_sequence
from catalyst.qfunc import QFunc
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import (
    filter_static_args,
    get_abstract_signature,
    get_type_annotations,
    merge_static_argname_into_argnum,
    merge_static_args,
    promote_arguments,
    verify_static_argnums,
)
from catalyst.utils.c_template import mlir_type_to_numpy_type
from catalyst.utils.callables import CatalystCallable
from catalyst.utils.exceptions import CompileError
from catalyst.utils.filesystem import WorkspaceManager
from catalyst.utils.gen_mlir import inject_functions
from catalyst.utils.patching import Patcher

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Required for JAX tracer objects as PennyLane wires.
# pylint: disable=unnecessary-lambda
setattr(jax.interpreters.partial_eval.DynamicJaxprTracer, "__hash__", lambda x: id(x))

# This flag cannot be set in ``QJIT.get_mlir()`` because values created before
# that function is called must be consistent with the JAX configuration value.
jax.config.update("jax_enable_x64", True)


## API ##
@debug_logger
def qjit(
    fn=None,
    *,
    autograph=False,
    autograph_include=(),
    async_qnodes=False,
    target="binary",
    keep_intermediate=False,
    verbose=False,
    logfile=None,
    pipelines=None,
    static_argnums=None,
    static_argnames=None,
    abstracted_axes=None,
    disable_assertions=False,
    seed=None,
    experimental_capture=False,
    circuit_transform_pipeline=None,
):  # pylint: disable=too-many-arguments,unused-argument
    """A just-in-time decorator for PennyLane and JAX programs using Catalyst.

    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on whether function argument type hints are provided.

    .. note::

        Not all PennyLane devices currently work with Catalyst. Supported backend devices include
        ``lightning.qubit``, ``lightning.kokkos``, ``lightning.gpu``, and ``braket.aws.qubit``. For
        a full of supported devices, please see :doc:`/dev/devices`.

    Args:
        fn (Callable): the quantum or classical function
        autograph (bool): Experimental support for automatically converting Python control
            flow statements to Catalyst-compatible control flow. Currently supports Python ``if``,
            ``elif``, ``else``, and ``for`` statements. Note that this feature requires an
            available TensorFlow installation. For more details, see the
            :doc:`AutoGraph guide </dev/autograph>`.
        autograph_include: A list of (sub)modules to be allow-listed for autograph conversion.
        async_qnodes (bool): Experimental support for automatically executing
            QNodes asynchronously, if supported by the device runtime.
        target (str): the compilation target
        keep_intermediate (bool): Whether or not to store the intermediate files throughout the
            compilation. If ``True``, intermediate representations are available via the
            :attr:`~.QJIT.mlir`, :attr:`~.QJIT.jaxpr`, and :attr:`~.QJIT.qir`, representing
            different stages in the optimization process.
        verbosity (bool): If ``True``, the tools and flags used by Catalyst behind the scenes are
            printed out.
        logfile (Optional[TextIOWrapper]): File object to write verbose messages to (default -
            ``sys.stderr``).
        pipelines (Optional(List[Tuple[str,List[str]]])): A list of pipelines to be executed. The
            elements of this list are named sequences of MLIR passes to be executed. A ``None``
            value (the default) results in the execution of the default pipeline. This option is
            considered to be used by advanced users for low-level debugging purposes.
        static_argnums(int or Seqence[Int]): an index or a sequence of indices that specifies the
            positions of static arguments.
        static_argnames(str or Seqence[str]): a string or a sequence of strings that specifies the
            names of static arguments.
        abstracted_axes (Sequence[Sequence[str]] or Dict[int, str] or Sequence[Dict[int, str]]):
            An experimental option to specify dynamic tensor shapes.
            This option affects the compilation of the annotated function.
            Function arguments with ``abstracted_axes`` specified will be compiled to ranked tensors
            with dynamic shapes. For more details, please see the Dynamically-shaped Arrays section
            below.
        disable_assertions (bool): If set to ``True``, runtime assertions included in
            ``fn`` via :func:`~.debug_assert` will be disabled during compilation.
        seed (Optional[Int]):
            The seed for circuit readout results when the qjit-compiled function is executed
            on simulator devices including ``lightning.qubit``, ``lightning.kokkos``, and
            ``lightning.gpu``. The default value is None, which means no seeding is performed,
            and all processes are random. A seed is expected to be an unsigned 32-bit integer.
            Currently, the following measurement processes are seeded: :func:`~.measure`,
            :func:`qml.sample() <pennylane.sample>`, :func:`qml.counts() <pennylane.counts>`.
        experimental_capture (bool): If set to ``True``, the qjit decorator
            will use PennyLane's experimental program capture capabilities
            to capture the decorated function for compilation.
        circuit_transform_pipeline (Optional[dict[str, dict[str, str]]]):
            A dictionary that specifies the quantum circuit transformation pass pipeline order,
            and optionally arguments for each pass in the pipeline. Keys of this dictionary
            should correspond to names of passes found in the `catalyst.passes <https://docs.
            pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes>`_
            module, values should either be empty dictionaries (for default pass options) or
            dictionaries of valid keyword arguments and values for the specific pass.
            The order of keys in this dictionary will determine the pass pipeline.
            If not specified, the default pass pipeline will be applied.

    Returns:
        QJIT object.

    Raises:
        FileExistsError: Unable to create temporary directory
        PermissionError: Problems creating temporary directory
        OSError: Problems while creating folder for intermediate files
        AutoGraphError: Raised if there was an issue converting the given the function(s).
        ImportError: Raised if AutoGraph is turned on and TensorFlow could not be found.

    **Example**

    In just-in-time (JIT) mode, the compilation is triggered at the call site the
    first time the quantum function is executed. For example, ``circuit`` is
    compiled as early as the first call.

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(wires=1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    Array(0., dtype=float64)
    >>> circuit(0.5)  # the precompiled quantum function is called
    Array(0., dtype=float64)

    Alternatively, if argument type hints are provided, compilation
    can occur 'ahead of time' when the function is decorated.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def circuit(x: complex, z: jax.ShapeDtypeStruct((3,), jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

    >>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
    Array([0.75634905-0.52801002j, 0.        +0.j        ,
           0.35962678+0.14074839j, 0.        +0.j        ], dtype=complex128)

    For more details on compilation and debugging, please see :doc:`/dev/sharp_bits`.

    .. details::
        :title: AutoGraph and Python control flow

        Catalyst also supports capturing imperative Python control flow in compiled programs. You
        can enable this feature via the ``autograph=True`` parameter. Note that it does come with
        some restrictions, in particular whenever global state is involved. Refer to the
        :doc:`AutoGraph guide </dev/autograph>` for a complete discussion of the
        supported and unsupported use-cases.

        .. code-block:: python

            @qjit(autograph=True)
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circuit(x: int):

                if x < 5:
                    qml.Hadamard(wires=0)
                else:
                    qml.T(wires=0)

                return qml.expval(qml.PauliZ(0))

        >>> circuit(3)
        Array(0., dtype=float64)

        >>> circuit(5)
        Array(1., dtype=float64)

        Note that imperative control flow will still work in Catalyst even when the AutoGraph
        feature is turned off, it just won't be captured in the compiled program and cannot involve
        traced values. The example above would then raise a tracing error, as there is no value for
        ``x`` yet than can be compared in the if statement. A loop like ``for i in range(5)`` would
        be unrolled during tracing, "copy-pasting" the body 5 times into the program rather than
        appearing as is.

    .. details::
        :title: In-place JAX array updates with Autograph

        To update array values when using JAX, the JAX syntax for array modification
        (which uses methods like ``at``, ``set``, ``multiply``, etc) must be used:

        .. code-block:: python

            @qjit(autograph=True)
            def f(x):
                first_dim = x.shape[0]
                result = jnp.empty((first_dim,), dtype=x.dtype)
                for i in range(first_dim):
                    result = result.at[i].set(x[i])
                    result = result.at[i].multiply(10)
                    result = result.at[i].add(5)

                return result

        However, if updating a single index or slice of the array, Autograph supports conversion of
        Python's standard arithmatic array assignment operators to the equivalent in-place
        expressions listed in the JAX documentation for ``jax.numpy.ndarray.at``:

        .. code-block:: python

            @qjit(autograph=True)
            def f(x):
                first_dim = x.shape[0]
                result = jnp.empty((first_dim,), dtype=x.dtype)
                for i in range(first_dim):
                    result[i] = x[i]
                    result[i] *= 10
                    result[i] += 5

                return result

        Under the hood, Catalyst converts anything coming in the latter notation into the
        former one.

        The list of supported operators includes: ``=``, ``+=``, ``-=``, ``*=``, ``/=``, and ``**=``.

    .. details::
        :title: Static arguments

        - ``static_argnums`` defines which positional arguments should be treated as static. If it takes an
        integer, it means the argument whose index is equal to the integer is static. If it takes
        an iterable of integers, arguments whose index is contained in the iterable are static.
        Changing static arguments will introduce re-compilation.

        - ``static_argnames`` defines which named function arguments should be treated as static.

        A valid static argument must be hashable and its ``__hash__`` method must be able to
        reflect any changes of its attributes.

        .. code-block:: python

            @dataclass
            class MyClass:
                val: int

                def __hash__(self):
                    return hash(str(self))

            @qjit(static_argnums=1)
            def f(
                x: int,
                y: MyClass,
            ):
                return x + y.val

            f(1, MyClass(5))
            f(1, MyClass(6)) # re-compilation
            f(2, MyClass(5)) # no re-compilation

        In the example above, ``y`` is static. Note that the second function call triggers
        re-compilation since the input object is different from the previous one. However,
        the third function call direcly uses the previous compiled one and does not introduce
        re-compilation.

        .. code-block:: python

            @dataclass
            class MyClass:
                val: int

                def __hash__(self):
                    return hash(str(self))

            @qjit(static_argnums=(1, 2))
            def f(
                x: int,
                y: MyClass,
                z: MyClass,
            ):
                return x + y.val + z.val

            my_obj_1 = MyClass(5)
            my_obj_2 = MyClass(6)
            f(1, my_obj_1, my_obj_2)
            my_obj_1.val = 7
            f(1, my_obj_1, my_obj_2) # re-compilation

        In the example above, ``y`` and ``z`` are static. The second function should make
        function ``f`` be re-compiled because ``my_obj_1`` is changed. This requires that
        the mutation is properly reflected in the hash value.

        Note that even when ``static_argnums`` is used in conjunction with type hinting,
        ahead-of-time compilation will not be possible since the static argument values
        are not yet available. Instead, compilation will be just-in-time.


    .. details::
        :title: Dynamically-shaped arrays

        There are three ways to use ``abstracted_axes``; by passing a sequence of tuples, a
        dictionary, or a sequence of dictionaries. Passing a sequence of tuples:

        .. code-block:: python

            abstracted_axes=((), ('n',), ('m', 'n'))

        Each tuple in the sequence corresponds to one of the arguments in the annotated
        function. Empty tuples can
        be used and correspond to parameters with statically known shapes.
        Non-empty tuples correspond to parameters with dynamically known shapes.

        In this example above,

        - the first argument will have a statically known shape,

        - the second argument has its zeroth axis have dynamic
          shape ``n``, and

        - the third argument will have its zeroth axis with dynamic shape
          ``m`` and first axis with dynamic shape ``n``.

        Passing a dictionary:

        .. code-block:: python

            abstracted_axes={0: 'n'}

        This approach allows a concise expression of the relationships
        between axes for different function arguments. In this example,
        it specifies that for all function arguments, the zeroth axis will
        have dynamic shape ``n``.

        Passing a sequence of dictionaries:

        .. code-block:: python

            abstracted_axes=({}, {0: 'n'}, {1: 'm', 0: 'n'})

        The example here is a more verbose version of the tuple example. This convention
        allows axes to be omitted from the list of abstracted axes.

        Using ``abstracted_axes`` can help avoid the cost of recompilation.
        By using ``abstracted_axes``, a more general version of the compiled function will be
        generated. This more general version is parametrized over the abstracted axes and
        allows results to be computed over tensors independently of their axes lengths.

        For example:

        .. code-block:: python

            @qjit
            def sum(arr):
                return jnp.sum(arr)

            sum(jnp.array([1]))     # Compilation happens here.
            sum(jnp.array([1, 1]))  # And here!

        The ``sum`` function would recompile each time an array of different size is passed
        as an argument.

        .. code-block:: python

            @qjit(abstracted_axes={0: "n"})
            def sum_abstracted(arr):
                return jnp.sum(arr)

            sum(jnp.array([1]))     # Compilation happens here.
            sum(jnp.array([1, 1]))  # No need to recompile.

        the ``sum_abstracted`` function would only compile once and its definition would be
        reused for subsequent function calls.
    """
    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(qjit, **kwargs)

    return QJIT(fn, CompileOptions(**kwargs))


## IMPL ##


# pylint: disable=too-many-instance-attributes
class QJIT(CatalystCallable):
    """Class representing a just-in-time compiled hybrid quantum-classical function.

    .. note::

        ``QJIT`` objects are created by the :func:`~.qjit` decorator. Please see
        the :func:`~.qjit` documentation for more details.

    Args:
        fn (Callable): the quantum or classical function to compile
        compile_options (CompileOptions): compilation options to use

    :ivar original_function: This attribute stores `fn`, the quantum or classical function
                             object to compile, as is, without any modifications
    :ivar jaxpr: This attribute stores the Jaxpr compiled from the function as a string.
    :ivar mlir: This attribute stores the MLIR compiled from the function as a string.
    :ivar qir: This attribute stores the QIR in LLVM IR form compiled from the function as a string.

    """

    @debug_logger_init
    def __init__(self, fn, compile_options):
        functools.update_wrapper(self, fn)
        self.original_function = fn
        self.compile_options = compile_options
        self.compiler = Compiler(compile_options)
        self.fn_cache = CompilationCache(
            compile_options.static_argnums, compile_options.abstracted_axes
        )
        # Active state of the compiler.
        # TODO: rework ownership of workspace, possibly CompiledFunction
        self.workspace = None
        self.c_sig = None
        self.out_treedef = None
        self.compiled_function = None
        self.jaxed_function = None
        # IRs are only available for the most recently traced function.
        self.jaxpr = None
        self.mlir = None  # string form (historic presence)
        self.mlir_module = None
        self.qir = None
        self.out_type = None
        self.overwrite_ir = None

        self.user_sig = get_type_annotations(fn)
        self._validate_configuration()

        # If static_argnames are present, convert them to static_argnums
        if compile_options.static_argnames is not None:
            compile_options.static_argnums = merge_static_argname_into_argnum(
                fn, compile_options.static_argnames, compile_options.static_argnums
            )

        # Patch the conversion rules by adding the included modules before the block list
        include_convertlist = tuple(
            ag_config.Convert(rule) for rule in self.compile_options.autograph_include
        )
        self.patched_module_allowlist = include_convertlist + ag_primitives.module_allowlist

        # Pre-compile with the patched conversion rules
        with Patcher(
            (ag_primitives, "module_allowlist", self.patched_module_allowlist),
        ):
            self.user_function = self.pre_compilation()

        # Static arguments require values, so we cannot AOT compile.
        if self.user_sig is not None and not self.compile_options.static_argnums:
            self.aot_compile()

        super().__init__("user_function")

    @debug_logger
    def __call__(self, *args, **kwargs):
        # Transparantly call Python function in case of nested QJIT calls.
        if EvaluationContext.is_tracing():
            isQNode = isinstance(self.user_function, qml.QNode)
            if isQNode and self.compile_options.static_argnums:
                kwargs = {"static_argnums": self.compile_options.static_argnums, **kwargs}

            return self.user_function(*args, **kwargs)

        requires_promotion = self.jit_compile(args, **kwargs)

        # If we receive tracers as input, dispatch to the JAX integration.
        if any(isinstance(arg, jax.core.Tracer) for arg in tree_flatten(args)[0]):
            if self.jaxed_function is None:
                self.jaxed_function = JAX_QJIT(self)  # lazy gradient compilation
            return self.jaxed_function(*args, **kwargs)

        elif requires_promotion:
            dynamic_args = filter_static_args(args, self.compile_options.static_argnums)
            args = promote_arguments(self.c_sig, dynamic_args)

        return self.run(args, kwargs)

    @debug_logger
    def aot_compile(self):
        """Compile Python function on initialization using the type hint signature."""

        self.workspace = self._get_workspace()

        # TODO: awkward, refactor or redesign the target feature
        if self.compile_options.target in ("jaxpr", "mlir", "binary"):
            # Capture with the patched conversion rules
            with Patcher(
                (ag_primitives, "module_allowlist", self.patched_module_allowlist),
            ):
                self.jaxpr, self.out_type, self.out_treedef, self.c_sig = self.capture(
                    self.user_sig or ()
                )

        if self.compile_options.target in ("mlir", "binary"):
            self.mlir_module, self.mlir = self.generate_ir()

        if self.compile_options.target in ("binary",):
            self.compiled_function, self.qir = self.compile()
            self.fn_cache.insert(
                self.compiled_function, self.user_sig, self.out_treedef, self.workspace
            )

    @debug_logger
    def jit_compile(self, args, **kwargs):
        """Compile Python function on invocation using the provided arguments.

        Args:
            args (Iterable): arguments to use for program capture

        Returns:
            bool: whether the provided arguments will require promotion to be used with the compiled
                  function
        """

        cached_fn, requires_promotion = self.fn_cache.lookup(args)

        if cached_fn is None:
            if self.user_sig and not self.compile_options.static_argnums:
                msg = "Provided arguments did not match declared signature, recompiling..."
                warnings.warn(msg, UserWarning)

            # Cleanup before recompilation:
            #  - recompilation should always happen in new workspace
            #  - compiled functions for jax integration are not yet cached
            #  - close existing shared library
            self.workspace = self._get_workspace()
            self.jaxed_function = None
            if self.compiled_function and self.compiled_function.shared_object:
                self.compiled_function.shared_object.close()

            # Capture with the patched conversion rules
            with Patcher(
                (ag_primitives, "module_allowlist", self.patched_module_allowlist),
            ):
                self.jaxpr, self.out_type, self.out_treedef, self.c_sig = self.capture(
                    args, **kwargs
                )

            self.mlir_module, self.mlir = self.generate_ir()
            self.compiled_function, self.qir = self.compile()

            self.fn_cache.insert(self.compiled_function, args, self.out_treedef, self.workspace)

        elif self.compiled_function is not cached_fn.compiled_fn:
            # Restore active state from cache.
            self.workspace = cached_fn.workspace
            self.compiled_function = cached_fn.compiled_fn
            self.out_treedef = cached_fn.out_treedef
            self.c_sig = cached_fn.signature
            self.jaxed_function = None

            self.compiled_function.shared_object.open()

        return requires_promotion

    # Processing Stages #

    @instrument
    @debug_logger
    def pre_compilation(self):
        """Perform pre-processing tasks on the Python function, such as AST transformations."""
        processed_fn = self.original_function

        if self.compile_options.autograph:
            processed_fn = run_autograph(self.original_function)

        return processed_fn

    @instrument(size_from=0)
    @debug_logger
    def capture(self, args, **kwargs):
        """Capture the JAX program representation (JAXPR) of the wrapped function.

        Args:
            args (Iterable): arguments to use for program capture

        Returns:
            ClosedJaxpr: captured JAXPR
            PyTreeDef: PyTree metadata of the function output
            Tuple[Any]: the dynamic argument signature
        """
        verify_static_argnums(args, self.compile_options.static_argnums)
        static_argnums = self.compile_options.static_argnums
        abstracted_axes = self.compile_options.abstracted_axes

        dynamic_args = filter_static_args(args, static_argnums)
        dynamic_sig = get_abstract_signature(dynamic_args)
        full_sig = merge_static_args(dynamic_sig, args, static_argnums)

        def fn_with_transform_named_sequence(*args, **kwargs):
            """
            This function behaves exactly like the user function being jitted,
            taking in the same arguments and producing the same results, except
            it injects a transform_named_sequence jax primitive at the beginning
            of the jaxpr when being traced.

            Note that we do not overwrite self.original_function and self.user_function;
            this fn_with_transform_named_sequence is ONLY used here to produce tracing
            results with a transform_named_sequence primitive at the beginning of the
            jaxpr. It is never executed or used anywhere, except being traced here.
            """
            _inject_transform_named_sequence()
            return self.user_function(*args, **kwargs)

        if self.compile_options.experimental_capture:
            return trace_from_pennylane(
                fn_with_transform_named_sequence, static_argnums, abstracted_axes, full_sig, kwargs
            )

        def closure(qnode, *args, **kwargs):
            params = {}
            params["static_argnums"] = kwargs.pop("static_argnums", static_argnums)
            params["_out_tree_expected"] = []
            return QFunc.__call__(
                qnode,
                pass_pipeline=self.compile_options.circuit_transform_pipeline,
                *args,
                **dict(params, **kwargs),
            )

        with Patcher(
            (qml.QNode, "__call__", closure),
        ):
            # TODO: improve PyTree handling
            jaxpr, out_type, treedef = trace_to_jaxpr(
                fn_with_transform_named_sequence,
                static_argnums,
                abstracted_axes,
                full_sig,
                kwargs,
            )

        PipelineNameUniquer.reset()
        return jaxpr, out_type, treedef, dynamic_sig

    @instrument(size_from=0, has_finegrained=True)
    @debug_logger
    def generate_ir(self):
        """Generate Catalyst's intermediate representation (IR) as an MLIR module.

        Returns:
            Tuple[ir.Module, str]: the in-memory MLIR module and its string representation
        """

        mlir_module, ctx = lower_jaxpr_to_mlir(self.jaxpr, self.__name__)

        # Inject Runtime Library-specific functions (e.g. setup/teardown).
        inject_functions(mlir_module, ctx, self.compile_options.seed)

        # Canonicalize the MLIR since there can be a lot of redundancy coming from JAX.
        options = copy.deepcopy(self.compile_options)
        options.pipelines = [("0_canonicalize", ["canonicalize"])]
        options.lower_to_llvm = False
        canonicalizer = Compiler(options)

        # TODO: the in-memory and textual form are different after this, consider unification
        _, mlir_string = canonicalizer.run(mlir_module, self.workspace)

        return mlir_module, mlir_string

    @instrument(size_from=1, has_finegrained=True)
    @debug_logger
    def compile(self):
        """Compile an MLIR module to LLVMIR and shared library code.

        Returns:
            Tuple[CompiledFunction, str]: the compilation result and LLVMIR
        """

        # WARNING: assumption is that the first function is the entry point to the compiled program.
        entry_point_func = self.mlir_module.body.operations[0]
        restype = entry_point_func.type.results

        for res in restype:
            baseType = mlir.ir.RankedTensorType(res).element_type
            # This will make a check before sending it to the compiler that the return type
            # is actually available in most systems. f16 needs a special symbol and linking
            # will fail if it is not available.
            mlir_type_to_numpy_type(baseType)

        # The function name out of MLIR has quotes around it, which we need to remove.
        # The MLIR function name is actually a derived type from string which has no
        # `replace` method, so we need to get a regular Python string out of it.
        func_name = str(self.mlir_module.body.operations[0].name).replace('"', "")
        if self.overwrite_ir:
            shared_object, llvm_ir = self.compiler.run_from_ir(
                self.overwrite_ir,
                str(self.mlir_module.operation.attributes["sym_name"]).replace('"', ""),
                self.workspace,
            )
        else:
            shared_object, llvm_ir = self.compiler.run(self.mlir_module, self.workspace)

        compiled_fn = CompiledFunction(
            shared_object, func_name, restype, self.out_type, self.compile_options
        )

        return compiled_fn, llvm_ir

    @instrument(has_finegrained=True)
    @debug_logger
    def run(self, args, kwargs):
        """Invoke a previously compiled function with the supplied arguments.

        Args:
            args (Iterable): the positional arguments to the compiled function
            kwargs: the keyword arguments to the compiled function

        Returns:
            Any: results of the execution arranged into the original function's output PyTrees
        """

        results = self.compiled_function(*args, **kwargs)

        # TODO: Move this to the compiled function object.
        return tree_unflatten(self.out_treedef, results)

    # Helper Methods #

    def _validate_configuration(self):
        """Run validations on the supplied options and parameters."""
        if not hasattr(self.original_function, "__name__"):
            self.__name__ = "unknown"  # allow these cases anyways?

        if not self.compile_options.autograph and len(self.compile_options.autograph_include) > 0:
            raise CompileError(
                "In order for 'autograph_include' to work, 'autograph' must be set to True"
            )

    def _get_workspace(self):
        """Get or create a workspace to use for compilation."""

        workspace_name = self.__name__
        preferred_workspace_dir = os.getcwd() if self.compile_options.keep_intermediate else None

        return WorkspaceManager.get_or_create_workspace(workspace_name, preferred_workspace_dir)


class JAX_QJIT:
    """Wrapper class around :class:`~.QJIT` that enables compatibility with JAX transformations.

    The primary mechanism through which this is effected is by wrapping the invocation of the QJIT
    object inside a JAX ``pure_callback``. Additionally, a custom JVP is defined in order to support
    JAX-based differentiation, which is itself a ``pure_callback`` around a second QJIT object which
    invokes :func:`~.grad` on the original function. Using this class thus incurs additional
    compilation time.

    Args:
        qjit_function (QJIT): the compiled quantum function object to wrap
    """

    @debug_logger_init
    def __init__(self, qjit_function):
        @jax.custom_jvp
        def jaxed_function(*args, **kwargs):
            return self.wrap_callback(qjit_function, *args, **kwargs)

        self.qjit_function = qjit_function
        self.derivative_functions = {}
        self.jaxed_function = jaxed_function
        jaxed_function.defjvp(self.compute_jvp, symbolic_zeros=True)

    @staticmethod
    @debug_logger
    def wrap_callback(qjit_function, *args, **kwargs):
        """Wrap a QJIT function inside a jax host callback."""
        data = jax.pure_callback(
            qjit_function, qjit_function.jaxpr.out_avals, *args, vectorized=False, **kwargs
        )

        # Unflatten the return value w.r.t. the original PyTree definition if available
        assert qjit_function.out_treedef is not None, "PyTree shape must not be none."
        return tree_unflatten(qjit_function.out_treedef, data)

    @debug_logger
    def get_derivative_qjit(self, argnums):
        """Compile a function computing the derivative of the wrapped QJIT for the given argnums."""

        argnum_key = "".join(str(idx) for idx in argnums)
        if argnum_key in self.derivative_functions:
            return self.derivative_functions[argnum_key]

        # Here we define the signature for the new QJIT object explicitly, rather than relying on
        # functools.wrap, in order to guarantee compilation is triggered on instantiation.
        # The signature of the original QJIT object is guaranteed to be defined by now, located
        # in QJIT.c_sig, however we don't update the original function with these annotations.
        annotations = {}
        updated_params = []
        signature = inspect.signature(self.qjit_function)
        for idx, (arg_name, param) in enumerate(signature.parameters.items()):
            annotations[arg_name] = self.qjit_function.c_sig[idx]
            updated_params.append(param.replace(annotation=annotations[arg_name]))

        def deriv_wrapper(*args, **kwargs):
            return catalyst.jacobian(self.qjit_function, argnums=argnums)(*args, **kwargs)

        deriv_wrapper.__name__ = "deriv_" + self.qjit_function.__name__
        deriv_wrapper.__annotations__ = annotations
        deriv_wrapper.__signature__ = signature.replace(parameters=updated_params)

        self.derivative_functions[argnum_key] = QJIT(
            deriv_wrapper, self.qjit_function.compile_options
        )
        return self.derivative_functions[argnum_key]

    @debug_logger
    def compute_jvp(self, primals, tangents):
        """Compute the set of results and JVPs for a QJIT function."""
        # Assume we have primals of shape `[a,b]` and results of shape `[c,d]`. Derivatives [2]
        # would get the shape `[c,d,a,b]` and tangents [1] would have the same shape as primals.
        # Now, In this function we apply tensordot using the pattern `[c,d,a,b]*[a,b] -> [c,d]`.

        # Optimization: Do not compute Jacobians for arguments which do not participate in
        #               differentiation.
        argnums = []
        for idx, tangent in enumerate(tangents):
            if not isinstance(tangent, jax.custom_derivatives.SymbolicZero):
                argnums.append(idx)

        results = self.wrap_callback(self.qjit_function, *primals)
        results_data, _results_shape = tree_flatten(results)
        derivatives = self.wrap_callback(self.get_derivative_qjit(argnums), *primals)
        derivatives_data, _derivatives_shape = tree_flatten(derivatives)

        jvps = [jnp.zeros_like(results_data[res_idx]) for res_idx in range(len(results_data))]
        for diff_arg_idx, arg_idx in enumerate(argnums):
            tangent = tangents[arg_idx]  # [1]
            taxis = list(range(tangent.ndim))
            for res_idx in range(len(results_data)):
                deriv_idx = diff_arg_idx + res_idx * len(argnums)
                deriv = derivatives_data[deriv_idx]  # [2]
                daxis = list(range(deriv.ndim - tangent.ndim, deriv.ndim))
                jvp = jnp.tensordot(deriv, tangent, axes=(daxis, taxis))
                jvps[res_idx] = jvps[res_idx] + jvp

        # jvps must match the type of primals
        # due to pytrees, primals are a tuple
        primal_type = type(primals)
        jvps = primal_type(jvps)
        if len(jvps) == 1:
            jvps = jvps[0]

        return results, jvps

    @debug_logger
    def __call__(self, *args, **kwargs):
        return self.jaxed_function(*args, **kwargs)
