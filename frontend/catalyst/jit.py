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
import os
import warnings

import jax
import jax.numpy as jnp
import pennylane as qml
from jax.interpreters import mlir
from jax.tree_util import tree_flatten, tree_unflatten

import catalyst
from catalyst.autograph import run_autograph
from catalyst.compiled_functions import CompilationCache, CompiledFunction
from catalyst.compiler import CompileOptions, Compiler
from catalyst.jax_tracer import lower_jaxpr_to_mlir, trace_to_jaxpr
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import (
    filter_static_args,
    get_abstract_signature,
    get_type_annotations,
    merge_static_args,
    promote_arguments,
)
from catalyst.utils.c_template import mlir_type_to_numpy_type
from catalyst.utils.exceptions import CompileError
from catalyst.utils.filesystem import WorkspaceManager
from catalyst.utils.gen_mlir import inject_functions
from catalyst.utils.patching import Patcher

# Required for JAX tracer objects as PennyLane wires.
# pylint: disable=unnecessary-lambda
setattr(jax.interpreters.partial_eval.DynamicJaxprTracer, "__hash__", lambda x: id(x))

# This flag cannot be set in ``QJIT.get_mlir()`` because values created before
# that function is called must be consistent with the JAX configuration value.
jax.config.update("jax_enable_x64", True)


# pylint: disable=too-many-instance-attributes
class QJIT:
    """Class representing a just-in-time compiled hybrid quantum-classical function.

    .. note::

        ``QJIT`` objects are created by the :func:`~.qjit` decorator. Please see
        the :func:`~.qjit` documentation for more details.

    Args:
        fn (Callable): the quantum or classical function to compile
        compile_options (CompileOptions): compilation options to use
    """

    def __init__(self, fn, compile_options):
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

        functools.update_wrapper(self, fn)
        self.user_sig = get_type_annotations(fn)
        self._validate_configuration()

        self.user_function = self.pre_compilation()

        # Static arguments require values, so we cannot AOT compile.
        if self.user_sig is not None and not self.compile_options.static_argnums:
            self.aot_compile()

    def __call__(self, *args, **kwargs):
        # Transparantly call Python function in case of nested QJIT calls.
        if EvaluationContext.is_tracing():
            return self.user_function(*args, **kwargs)

        requires_promotion = self.jit_compile(args)

        # If we receive tracers as input, dispatch to the JAX integration.
        if any(isinstance(arg, jax.core.Tracer) for arg in tree_flatten(args)[0]):
            if self.jaxed_function is None:
                self.jaxed_function = JAX_QJIT(self)  # lazy gradient compilation
            return self.jaxed_function(*args, **kwargs)

        elif requires_promotion:
            dynamic_args = filter_static_args(args, self.compile_options.static_argnums)
            args = promote_arguments(self.c_sig, dynamic_args)

        return self.run(args, kwargs)

    def aot_compile(self):
        """Compile Python function on initialization using the type hint signature."""

        self.workspace = self._get_workspace()

        # TODO: awkward, refactor or redesign the target feature
        if self.compile_options.target in ("jaxpr", "mlir", "binary"):
            self.jaxpr, self.out_treedef, self.c_sig = self.capture(self.user_sig or ())

        if self.compile_options.target in ("mlir", "binary"):
            self.mlir_module, self.mlir = self.generate_ir()

        if self.compile_options.target in ("binary",):
            self.compiled_function, self.qir = self.compile()
            self.fn_cache.insert(
                self.compiled_function, self.user_sig, self.out_treedef, self.workspace
            )

    def jit_compile(self, args):
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

            self.jaxpr, self.out_treedef, self.c_sig = self.capture(args)
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

    def pre_compilation(self):
        """Perform pre-processing tasks on the Python function, such as AST transformations."""
        processed_fn = self.original_function

        if self.compile_options.autograph:
            processed_fn = run_autograph(self.original_function)

        return processed_fn

    def capture(self, args):
        """Capture the JAX program representation (JAXPR) of the wrapped function.

        Args:
            args (Iterable): arguments to use for program capture

        Returns:
            ClosedJaxpr: captured JAXPR
            PyTreeDef: PyTree metadata of the function output
            Tuple[Any]: the dynamic argument signature
        """

        self._verify_static_argnums(args)
        static_argnums = self.compile_options.static_argnums
        abstracted_axes = self.compile_options.abstracted_axes

        dynamic_args = filter_static_args(args, static_argnums)
        dynamic_sig = get_abstract_signature(dynamic_args)
        full_sig = merge_static_args(dynamic_sig, args, static_argnums)

        with Patcher(
            (qml.QNode, "__call__", catalyst.pennylane_extensions.QFunc.__call__),
        ):
            # TODO: improve PyTree handling
            jaxpr, treedef = trace_to_jaxpr(
                self.user_function, static_argnums, abstracted_axes, full_sig, {}
            )

        return jaxpr, treedef, dynamic_sig

    def generate_ir(self):
        """Generate Catalyst's intermediate representation (IR) as an MLIR module.

        Returns:
            Tuple[ir.Module, str]: the in-memory MLIR module and its string representation
        """

        mlir_module, ctx = lower_jaxpr_to_mlir(self.jaxpr, self.__name__)

        # Inject Runtime Library-specific functions (e.g. setup/teardown).
        inject_functions(mlir_module, ctx)

        # Canonicalize the MLIR since there can be a lot of redundancy coming from JAX.
        options = copy.deepcopy(self.compile_options)
        options.pipelines = [("0_canonicalize", ["canonicalize"])]
        options.lower_to_llvm = False
        canonicalizer = Compiler(options)

        # TODO: the in-memory and textual form are different after this, consider unification
        _, mlir_string, _ = canonicalizer.run(mlir_module, self.workspace)

        return mlir_module, mlir_string

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
        shared_object, llvm_ir, _ = self.compiler.run(self.mlir_module, self.workspace)
        compiled_fn = CompiledFunction(shared_object, func_name, restype, self.compile_options)

        return compiled_fn, llvm_ir

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

    def _verify_static_argnums(self, args):
        for argnum in self.compile_options.static_argnums:
            if argnum < 0 or argnum >= len(args):
                msg = f"argnum {argnum} is beyond the valid range of [0, {len(args)})."
                raise CompileError(msg)

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

    def __init__(self, qjit_function):
        @jax.custom_jvp
        def jaxed_function(*args, **kwargs):
            return self.wrap_callback(qjit_function, *args, **kwargs)

        self.qjit_function = qjit_function
        self.derivative_functions = {}
        self.jaxed_function = jaxed_function
        jaxed_function.defjvp(self.compute_jvp, symbolic_zeros=True)

    @staticmethod
    def wrap_callback(qjit_function, *args, **kwargs):
        """Wrap a QJIT function inside a jax host callback."""
        data = jax.pure_callback(
            qjit_function, qjit_function.jaxpr.out_avals, *args, vectorized=False, **kwargs
        )

        # Unflatten the return value w.r.t. the original PyTree definition if available
        assert qjit_function.out_treedef is not None, "PyTree shape must not be none."
        return tree_unflatten(qjit_function.out_treedef, data)

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
            return catalyst.jacobian(self.qjit_function, argnum=argnums)(*args, **kwargs)

        deriv_wrapper.__name__ = "deriv_" + self.qjit_function.__name__
        deriv_wrapper.__annotations__ = annotations
        deriv_wrapper.__signature__ = signature.replace(parameters=updated_params)

        self.derivative_functions[argnum_key] = QJIT(
            deriv_wrapper, self.qjit_function.compile_options
        )
        return self.derivative_functions[argnum_key]

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

    def __call__(self, *args, **kwargs):
        return self.jaxed_function(*args, **kwargs)


def qjit(
    fn=None,
    *,
    autograph=False,
    async_qnodes=False,
    target="binary",
    keep_intermediate=False,
    verbose=False,
    logfile=None,
    pipelines=None,
    static_argnums=None,
    abstracted_axes=None,
):  # pylint: disable=too-many-arguments
    """A just-in-time decorator for PennyLane and JAX programs using Catalyst.

    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on whether function argument type hints are provided.

    .. note::

        Currently, ``lightning.qubit`` is the only supported backend device
        for Catalyst compilation. For a list of supported operations, observables,
        and measurements, please see the :doc:`/dev/quick_start`.

    Args:
        fn (Callable): the quantum or classical function
        autograph (bool): Experimental support for automatically converting Python control
            flow statements to Catalyst-compatible control flow. Currently supports Python ``if``,
            ``elif``, ``else``, and ``for`` statements. Note that this feature requires an
            available TensorFlow installation. For more details, see the
            :doc:`AutoGraph guide </dev/autograph>`.
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
        abstracted_axes (Sequence[Sequence[str]] or Dict[int, str] or Sequence[Dict[int, str]]):
            An experimental option to specify dynamic tensor shapes.
            This option affects the compilation of the annotated function.
            Function arguments with ``abstracted_axes`` specified will be compiled to ranked tensors
            with dynamic shapes. For more details, please see the Dynamically-shaped Arrays section
            below.

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
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    Alternatively, if argument type hints are provided, compilation
    can occur 'ahead of time' when the function is decorated.

    .. code-block:: python

        from jax.core import ShapedArray

        @qjit  # compilation happens at definition
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

    >>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
    array([0.75634905-0.52801002j, 0. +0.j,
           0.35962678+0.14074839j, 0. +0.j])

    For more details on compilation and debugging, please see :doc:`/dev/sharp_bits`.

    .. important::

        Most decomposition logic will be equivalent to PennyLane's decomposition.
        However, decomposition logic will differ in the following cases:

        1. All :class:`qml.Controlled <pennylane.ops.op_math.Controlled>` operations will decompose
            to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.

        2. :class:`qml.ControlledQubitUnitary <pennylane.ControlledQubitUnitary>` operations will
            decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.

        3. The list of device-supported gates employed by Catalyst is currently different than that
            of the ``lightning.qubit`` device, as defined by the
            :class:`~.pennylane_extensions.QJITDevice`.

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
        array(0.)

        >>> circuit(5)
        array(1.)

        Note that imperative control flow will still work in Catalyst even when the AutoGraph
        feature is turned off, it just won't be captured in the compiled program and cannot involve
        traced values. The example above would then raise a tracing error, as there is no value for
        ``x`` yet than can be compared in the if statement. A loop like ``for i in range(5)`` would
        be unrolled during tracing, "copy-pasting" the body 5 times into the program rather than
        appearing as is.

    .. details::
        :title: Static arguments

        ``static_argnums`` defines which elements should be treated as static. If it takes an
        integer, it means the argument whose index is equal to the integer is static. If it takes
        an iterable of integers, arguments whose index is contained in the iterable are static.
        Changing static arguments will introduce re-compilation.

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

    argnums = static_argnums
    axes = abstracted_axes
    if fn is not None:
        return QJIT(
            fn,
            CompileOptions(
                verbose,
                logfile,
                target,
                keep_intermediate,
                pipelines,
                autograph,
                async_qnodes,
                static_argnums=argnums,
                abstracted_axes=axes,
            ),
        )

    def wrap_fn(fn):
        return QJIT(
            fn,
            CompileOptions(
                verbose,
                logfile,
                target,
                keep_intermediate,
                pipelines,
                autograph,
                async_qnodes,
                static_argnums=argnums,
                abstracted_axes=axes,
            ),
        )

    return wrap_fn
