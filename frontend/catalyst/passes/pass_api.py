# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
from importlib.metadata import entry_points
from pathlib import Path
from typing import TypeAlias

import pennylane as qml

from catalyst.tracing.contexts import EvaluationContext

PipelineDict: TypeAlias = dict[str, dict[str, str]]


## API ##
def pipeline(pass_pipeline: PipelineDict):
    """Configures the Catalyst MLIR pass pipeline for quantum circuit transformations for a QNode
    within a qjit-compiled program.

    Args:
        pass_pipeline (dict[str, dict[str, str]]): A dictionary that specifies the pass pipeline
            order, and optionally arguments for each pass in the pipeline. Keys of this dictionary
            should correspond to names of passes found in the
            `catalyst.passes <https://docs.pennylane.ai/projects/catalyst/en/stable/code/__init__.html#module-catalyst.passes>`_
            module, values should either be empty dictionaries (for default pass options) or
            dictionaries of valid keyword arguments and values for the specific pass.
            The order of keys in this dictionary will determine the pass pipeline.
            If not specified, the default pass pipeline will be applied.

    Returns:
        callable : A decorator that can be applied to a qnode.

    For a list of available passes, please see the :doc:`catalyst.passes module <code/passes>`.

    The default pass pipeline when used with Catalyst is currently empty.

    **Example**

    ``pipeline`` can be used to configure the pass pipeline order and options
    of a QNode within a qjit-compiled function.

    Configuration options are passed to specific passes via dictionaries:

    .. code-block:: python

        my_pass_pipeline = {
            "cancel_inverses": {},
            "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
        }

        @pipeline(my_pass_pipeline)
        @qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit
        def fn(x):
            return jnp.sin(circuit(x ** 2))

    ``pipeline`` can also be used to specify different pass pipelines for different parts of the
    same qjit-compiled workflow:

    .. code-block:: python

        my_pipeline = {
            "cancel_inverses": {},
            "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
        }

        my_other_pipeline = {"cancel_inverses": {}}

        @qjit
        def fn(x):
            circuit_pipeline = pipeline(my_pipeline)(circuit)
            circuit_other = pipeline(my_other_pipeline)(circuit)
            return jnp.abs(circuit_pipeline(x) - circuit_other(x))

    .. note::

        As of Python 3.7, the CPython dictionary implementation orders dictionaries based on
        insertion order. However, for an API guarantee of dictionary order,
        ``collections.OrderedDict`` may also be used.

    Note that the pass pipeline order and options can be configured *globally* for a
    qjit-compiled function, by using the ``circuit_transform_pipeline`` argument of
    the :func:`~.qjit` decorator.

    .. code-block:: python

        my_pass_pipeline = {
            "cancel_inverses": {},
            "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
        }

        @qjit(circuit_transform_pipeline=my_pass_pipeline)
        def fn(x):
            return jnp.sin(circuit(x ** 2))

    Global and local (via ``@pipeline``) configurations can coexist, however local pass pipelines
    will always take precedence over global pass pipelines.
    """

    def _decorator(qnode=None):
        if not isinstance(qnode, qml.QNode):
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        clone = copy.copy(qnode)
        clone.__name__ += "_transformed"

        @functools.wraps(clone)
        def wrapper(*args, **kwargs):
            if EvaluationContext.is_tracing():
                passes = kwargs.pop("pass_pipeline", [])
                passes += dictionary_to_list_of_passes(pass_pipeline)
                kwargs["pass_pipeline"] = passes
            return clone(*args, **kwargs)

        return wrapper

    return _decorator


def apply_pass(pass_name: str, *flags, **valued_options):
    """Applies a single pass to the QNode, where the pass is from Catalyst or a third-party
    if `entry_points` has been implemented. See
    :doc:`the compiler plugin documentation <dev/plugins>` for more details.

    Args:
        pass_name (str): Name of the pass
        *flags: Pass options
        **valued_options: options with values

    Returns:
        Function that can be used as a decorator to a QNode.
        E.g.,

        .. code-block:: python

            @passes.apply_pass("merge-rotations")
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def qnode():
                return qml.state()

            @qml.qjit(target="mlir")
            def module():
                return qnode()
    """

    def decorator(qnode):

        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        @functools.wraps(qnode)
        def qnode_call(*args, **kwargs):
            pass_pipeline = kwargs.get("pass_pipeline", [])
            pass_pipeline.append(Pass(pass_name, *flags, **valued_options))
            kwargs["pass_pipeline"] = pass_pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator


def apply_pass_plugin(path_to_plugin: str | Path, pass_name: str, *flags, **valued_options):
    """Applies a pass plugin to the QNode. See
    :doc:`the compiler plugin documentation <dev/plugins>` for more details.

    Args:
        path_to_plugin (str | Path): full path to plugin
        pass_name (str): Name of the pass
        *flags: Pass options
        **valued_options: options with values

    Returns:
        Function that can be used as a decorator to a QNode.
        E.g.,

        .. code-block:: python

            from standalone import getStandalonePluginAbsolutePath

            @passes.apply_pass_plugin(getStandalonePluginAbsolutePath(), "standalone-switch-bar-foo")
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def qnode():
                return qml.state()

            @qml.qjit(target="mlir")
            def module():
                return qnode()
    """

    if not isinstance(path_to_plugin, Path):
        path_to_plugin = Path(path_to_plugin)

    if not path_to_plugin.exists():
        raise FileNotFoundError(f"File '{path_to_plugin}' does not exist.")

    def decorator(qnode):
        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        @functools.wraps(qnode)
        def qnode_call(*args, **kwargs):
            pass_pipeline = kwargs.get("pass_pipeline", [])
            pass_pipeline.append(PassPlugin(path_to_plugin, pass_name, *flags, **valued_options))
            kwargs["pass_pipeline"] = pass_pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator


class Pass:
    """Class intended to hold options for passes.

    :class:`Pass` will be used when generating `ApplyRegisteredPassOp`s.
    The attribute `pass_name` corresponds to the field `name`.
    The attribute `options` is generated by the `get_options` method.

    People working on MLIR plugins may use this or :class:`PassPlugin` to
    schedule their compilation pass. E.g.,

    .. code-block:: python

        def an_optimization(qnode):
            @functools.wraps(qnode)
            def wrapper(*args, **kwargs):
                pass_pipeline = kwargs.pop("pass_pipeline", [])
                pass_pipeline.append(Pass("my_library.my_optimization", *args, **kwargs))
                kwargs["pass_pipeline"] = pass_pipeline
                return qnode(*args, **kwargs)
        return wrapper
    """

    def __init__(self, name: str, *options: list[str], **valued_options: dict[str, str]):
        self.options = options
        self.valued_options = valued_options
        if "." in name:
            resolution_functions = entry_points(group="catalyst.passes_resolution")
            key, passname = name.split(".")
            resolution_function = resolution_functions[key + ".passes"]
            module = resolution_function.load()
            path, name = module.name2pass(passname)
            assert EvaluationContext.is_tracing()
            EvaluationContext.add_plugin(path)

        self.name = name

    def get_options(self):
        """
        Stringify options according to what mlir-opt expects.

          ApplyRegisteredPassOp expects options to be a single StringAttr
          which follows the same format as the one used with mlir-opt.

        https://mlir.llvm.org/docs/Dialects/Transform/#transformapply_registered_pass-transformapplyregisteredpassop

          Options passed to a pass are specified via the syntax {option1=value1 option2=value2 ...},
          i.e., use space-separated key=value pairs for each option.

        https://mlir.llvm.org/docs/Tutorials/MlirOpt/#running-a-pass-with-options

        Experimentally we found that single-options also work without values.
        """
        retval = " ".join(f"{str(option)}" for option in self.options)
        retval2 = " ".join(f"{str(key)}={str(value)}" for key, value in self.valued_options.items())
        return " ".join([retval, retval2]).strip()

    def __repr__(self):
        return (
            self.name
            + " ".join(f"--{str(option)}" for option in self.options)
            + " ".join(
                [f"--{str(option)}={str(value)}" for option, value in self.valued_options.items()]
            )
        )


class PassPlugin(Pass):
    """Similar to :class:`Pass` but takes into account the plugin.

    The plugin is used during the creation of the compilation command.
    E.g.,

      --pass-plugin=path/to/plugin --dialect-plugin=path/to/plugin

    People working on MLIR plugins may use this or :class:`Pass` to
    schedule their compilation pass. E.g.,

    .. code-block:: python

        def an_optimization(qnode):
            @functools.wraps(qnode)
            def wrapper(*args, **kwargs):
                pass_pipeline = kwargs.pop("pass_pipeline", [])
                pass_pipeline.append(PassPlugin(path_to_plugin, "my_optimization", *args, **kwargs))
                kwargs["pass_pipeline"] = pass_pipeline
                return qnode(*args, **kwargs)
        return wrapper
    """

    def __init__(
        self, path: Path, name: str, *options: list[str], **valued_options: dict[str, str]
    ):
        assert EvaluationContext.is_tracing()
        EvaluationContext.add_plugin(path)
        self.path = path
        super().__init__(name, *options, **valued_options)


## PRIVATE ##
def dictionary_to_list_of_passes(pass_pipeline: PipelineDict):
    """Convert dictionary of passes into list of passes."""

    if pass_pipeline == None:
        return []

    if type(pass_pipeline) != dict:
        return pass_pipeline

    passes = []
    pass_names = _API_name_to_pass_name()
    for API_name, pass_options in pass_pipeline.items():
        name = pass_names.get(API_name, API_name)
        passes.append(Pass(name, **pass_options))
    return passes


def _API_name_to_pass_name():
    return {
        "cancel_inverses": "remove-chained-self-inverse",
        "merge_rotations": "merge-rotations",
        "ions_decomposition": "ions-decomposition",
    }
