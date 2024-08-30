# Copyright 2023 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import re
import shutil
import subprocess

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from jax.tree_util import register_pytree_node_class

from catalyst import debug, for_loop, qjit, value_and_grad
from catalyst.compiler import CompileOptions, Compiler
from catalyst.debug import (
    compile_executable,
    compile_from_mlir,
    get_cmain,
    get_compilation_stage,
    replace_ir,
)
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import get_lib_path


class TestDebugPrint:
    """Test suite for the runtime print functionality."""

    @pytest.mark.parametrize(
        ("arg"),
        [
            True,
            3,
            3.5,
            3 + 4j,
            np.array(3),
            jnp.array(3),
            jnp.array(3.000001),
            jnp.array(3.1 + 4j),
            jnp.array([3]),
            jnp.array([3, 4, 5]),
            jnp.array([[3, 4], [5, 6], [7, 8]]),
        ],
    )
    def test_function_arguments(self, capfd, arg):
        """Test printing of arbitrary JAX tracer values."""

        @qjit
        def test(x):
            debug.print(x)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        out, err = capfd.readouterr()
        assert err == ""
        expected = str(arg)
        assert expected == out.strip()

    def test_optional_descriptor(self, capfd):
        """Test the optional memref descriptor functionality."""

        @qjit
        def test(x):
            debug.print_memref(x)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(jnp.array([[1, 2, 3], [4, 5, 6]]))

        memref = (
            r"MemRef: base\@ = [0-9a-fx]+ rank = 2 offset = 0 "
            r"sizes = \[2, 3\] strides = \[3, 1\] data ="
            "\n"
        ) + re.escape("[[1,   2,   3], \n [4,   5,   6]]\n")

        regex = re.compile("^" + memref + "$")  # match exactly: ^ - start, $ - end

        out, err = capfd.readouterr()
        assert err == ""
        assert regex.match(out)

    def test_bad_argument(self):
        """Test bad argument."""

        @qjit
        def test(_x):
            debug.print_memref("foo")

        msg = "Arguments to print_memref must be of type jax.core.Tracer"
        with pytest.raises(TypeError, match=msg):
            test(3.14)

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (0, ""),
            (1, "0\n"),
            (6, "0\n1\n2\n3\n4\n5\n"),
        ],
    )
    def test_intermediate_values(self, capfd, arg, expected):
        """Test printing of arbitrary JAX tracer values."""

        @qjit
        def test(n):
            @for_loop(0, n, 1)
            def loop(i):
                debug.print(i)

            loop()

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert expected == out

    @register_pytree_node_class
    class MyObject:
        def __init__(self, string):
            self.string = string

        def __str__(self):
            return f"MyObject({self.string})"

        def tree_flatten(self):
            """tree flatten"""
            return ([], [self.string])

        @classmethod
        def tree_unflatten(cls, aux_data, _children):
            """unflatten"""
            return cls(*aux_data)

    @pytest.mark.parametrize(("arg"), [3, "hi", MyObject("hello")])
    def test_compile_time_values(self, capfd, arg):
        """Test printing of arbitrary Python objects, including strings."""

        expected = str(arg)

        @qjit
        def test():
            debug.print(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test()

        out, err = capfd.readouterr()
        assert err == ""
        assert out.strip() == expected

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (True, "True\n"),
            (3, "3\n"),
            (3.5, "3.5\n"),
            (3 + 4j, "(3+4j)\n"),
            (np.array(3), "3\n"),
            (np.array([3]), "[3]\n"),
            (jnp.array(3), "3\n"),
            (jnp.array([3]), "[3]\n"),
            (jnp.array([[3, 4], [5, 6], [7, 8]]), "[[3 4]\n [5 6]\n [7 8]]\n"),
            ("hi", "hi\n"),
            (MyObject("hello"), "MyObject(hello)\n"),
        ],
    )
    def test_no_qjit(self, capfd, arg, expected):
        """Test printing in interpreted mode."""

        debug.print(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == expected

    def test_multiple_prints(self, capfd):
        "Test printing strings in multiple prints"

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def func1():
            debug.print("hello")
            return qml.state()

        @qjit
        def func2():
            func1()
            debug.print("goodbye")
            return

        func2()
        out, err = capfd.readouterr()
        assert err == ""
        assert out == "hello\ngoodbye\n"

    def test_fstring_print(self, capsys):
        """Test fstring like function."""

        @qjit
        def cir(a, b, c):
            debug.print("{c} {b} {a}", a=a, b=b, c=c)

        cir(1, 2, 3)
        out, err = capsys.readouterr()
        expected = "3 2 1"
        assert expected == out.strip()


class TestPrintStage:
    """Test that compilation pipeline results can be printed."""

    def test_hlo_lowering_stage(self, capsys):
        """Test that the IR can be printed after the HLO lowering pipeline."""

        @qjit(keep_intermediate=True)
        def func():
            return 0

        print(get_compilation_stage(func, "HLOLoweringPass"))

        out, _ = capsys.readouterr()
        assert "@jit_func() -> tensor<i64>" in out
        assert "stablehlo.constant" not in out

        func.workspace.cleanup()

    def test_invalid_object(self):
        """Test the function on a non-QJIT object."""

        def func():
            return 0

        with pytest.raises(TypeError, match="needs to be a 'QJIT' object"):
            print(get_compilation_stage(func, "HLOLoweringPass"))


class TestCompileFromIR:
    """Test the debug feature that compiles from a string representation of the IR."""

    def test_compiler_from_textual_ir(self):
        """Test the textual IR compilation."""
        full_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        extension = ".so" if platform.system() == "Linux" else ".dylib"

        # pylint: disable=line-too-long
        ir = (
            r"""
module @workflow {
  func.func public @catalyst.entry_point(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @workflow(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @workflow(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "finite-diff", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device ["""
            + r'"'
            + full_path
            + r"""/librtd_lightning"""
            + extension
            + """", "LightningSimulator", "{'shots': 0}"]
    %0 = stablehlo.constant dense<4> : tensor<i64>
    %1 = quantum.alloc( 4) : !quantum.reg
    %2 = stablehlo.constant dense<0> : tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %3 = quantum.extract %1[%extracted] : !quantum.reg -> !quantum.bit
    %4 = quantum.custom "PauliX"() %3 : !quantum.bit
    %5 = stablehlo.constant dense<1> : tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %1[%extracted_0] : !quantum.reg -> !quantum.bit
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %7 = quantum.custom "RX"(%extracted_1) %6 : !quantum.bit
    %8 = quantum.namedobs %4[ PauliZ] : !quantum.obs
    %9 = quantum.expval %8 : f64
    %from_elements = tensor.from_elements %9 : tensor<f64>
    quantum.dealloc %1 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">){
      transform.yield
    }
  }
}
"""
        )
        compiled_function = compile_from_mlir(ir)
        assert compiled_function(0.1) == [-1]

    def test_compile_from_ir_with_compiler(self):
        """Supply a custom compiler instance to the textual compilation function."""

        options = CompileOptions(static_argnums=[1])
        compiler = Compiler(options)

        ir = r"""
module @workflow {
  func.func public @catalyst.entry_point(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    return %arg0 : tensor<f64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">){
      transform.yield
    }
  }
}
"""

        compiled_function = compile_from_mlir(ir, compiler=compiler)
        assert compiled_function(0.1, 0.2) == [0.1]  # allow call with one extra argument

    def test_parsing_errors(self):
        """Test parsing error handling."""

        ir = r"""
module @workflow {
  func.func public @catalyst.entry_point(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %c = stablehlo.constant dense<4.0> : tensor<i64>
    return %c : tensor<f64> // Invalid type
  }
}
"""
        with pytest.raises(CompileError) as e:
            compile_from_mlir(ir)(0.1)

        assert "Failed to parse module as MLIR source" in e.value.args[0]
        assert "Failed to parse module as LLVM source" in e.value.args[0]


class TestCProgramGeneration:
    """Test C Program generation"""

    def test_program_generation(self):
        """Test C Program generation"""
        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def f(x: float):
            """Returns two states."""
            qml.RX(x, wires=1)
            return qml.state(), qml.state()

        template = get_cmain(f, 4.0)
        assert "main" in template
        assert "struct result_t result_val;" in template
        assert "buff_0 = 4.0" in template
        assert "arg_0 = { &buff_0, &buff_0, 0 }" in template
        assert "_catalyst_ciface_jit_f(&result_val, &arg_0);" in template

    def test_program_without_return_nor_arguments(self):
        """Test program without return value nor arguments."""

        @qjit
        def f():
            """No-op function."""
            return None

        template = get_cmain(f)
        assert "struct result_t result_val;" not in template
        assert "buff_0" not in template
        assert "arg_0" not in template

    def test_generation_with_promotion(self):
        """Test that C program generation works on QJIT objects and args that require promotion."""

        @qjit
        def f(x: float):
            """Identity function."""
            return x

        template = get_cmain(f, 1)

        assert "main" in template
        assert "buff_0 = 1.0" in template  # argument was automaatically promoted

    def test_raises_error_if_tracing(self):
        """Test errors if c program generation requested during tracing."""

        @qjit
        def f(x: float):
            """Identity function."""
            return x

        with pytest.raises(CompileError, match="C interface cannot be generated"):

            @qjit
            def error_fn(x: float):
                """Should raise an error as we try to generate the C template during tracing."""
                return get_cmain(f, x)

    def test_error_non_qjit_object(self):
        """An error should be raised if the object supplied to the debug function is not a QJIT."""

        def f(x: float):
            """Identity function."""
            return x

        with pytest.raises(TypeError, match="First argument needs to be a 'QJIT' object"):
            get_cmain(f, 0.5)

    @pytest.mark.parametrize(
        ("pass_name", "target", "replacement"),
        [
            (
                "mlir",
                "%0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>\n",
                "%x = stablehlo.multiply %arg0, %arg0 : tensor<f64>\n"
                + "    %0 = stablehlo.multiply %x, %arg0 : tensor<f64>\n",
            ),
            (
                "HLOLoweringPass",
                "%0 = arith.mulf %extracted, %extracted : f64\n",
                "%t = arith.mulf %extracted, %extracted : f64\n"
                + "    %0 = arith.mulf %t, %extracted : f64\n",
            ),
            (
                "QuantumCompilationPass",
                "%0 = arith.mulf %extracted, %extracted : f64\n",
                "%t = arith.mulf %extracted, %extracted : f64\n"
                + "    %0 = arith.mulf %t, %extracted : f64\n",
            ),
            (
                "BufferizationPass",
                "%2 = arith.mulf %1, %1 : f64",
                "%t = arith.mulf %1, %1 : f64\n" + "    %2 = arith.mulf %t, %1 : f64\n",
            ),
            (
                "MLIRToLLVMDialect",
                "%5 = llvm.fmul %4, %4  : f64\n",
                "%t = llvm.fmul %4, %4  : f64\n" + "    %5 = llvm.fmul %t, %4  : f64\n",
            ),
            (
                "llvm_ir",
                "%5 = fmul double %4, %4\n",
                "%t = fmul double %4, %4\n" + "%5 = fmul double %t, %4\n",
            ),
            (
                "last",
                "%5 = fmul double %4, %4\n",
                "%t = fmul double %4, %4\n" + "%5 = fmul double %t, %4\n",
            ),
        ],
    )
    def test_modify_ir(self, pass_name, target, replacement):
        """Turn a square function in IRs into a cubic one."""

        def f(x):
            """Square function."""
            return x**2

        f.__name__ = f.__name__ + pass_name

        jit_f = qjit(f, keep_intermediate=True)
        data = 2.0
        old_result = jit_f(data)
        old_ir = get_compilation_stage(jit_f, pass_name)
        old_workspace = str(jit_f.workspace)

        new_ir = old_ir.replace(target, replacement)
        replace_ir(jit_f, pass_name, new_ir)
        new_result = jit_f(data)

        shutil.rmtree(old_workspace, ignore_errors=True)
        shutil.rmtree(str(jit_f.workspace), ignore_errors=True)
        assert old_result * data == new_result

    @pytest.mark.parametrize("pass_name", ["HLOLoweringPass", "O2Opt", "Enzyme"])
    def test_modify_ir_file_generation(self, pass_name):
        """Test if recompilation rerun the same pass."""

        def f(x: float):
            """Square function."""
            return x**2

        f.__name__ = f.__name__ + pass_name

        jit_f = qjit(f)
        jit_grad_f = qjit(value_and_grad(jit_f), keep_intermediate=True)
        jit_grad_f(3.0)
        ir = get_compilation_stage(jit_grad_f, pass_name)
        old_workspace = str(jit_grad_f.workspace)

        replace_ir(jit_grad_f, pass_name, ir)
        jit_grad_f(3.0)
        file_list = os.listdir(str(jit_grad_f.workspace))
        res = [i for i in file_list if pass_name in i]

        shutil.rmtree(old_workspace, ignore_errors=True)
        shutil.rmtree(str(jit_grad_f.workspace), ignore_errors=True)
        assert len(res) == 0

    def test_get_compilation_stage_without_keep_intermediate(self):
        """Test if error is raised when using get_pipeline_output without keep_intermediate."""

        @qjit
        def f(x: float):
            """Square function."""
            return x**2

        f(2.0)

        with pytest.raises(
            CompileError,
            match="Attempting to get output for pipeline: mlir, "
            "but no file was found.\nAre you sure the file exists?",
        ):
            get_compilation_stage(f, "mlir")

    @pytest.mark.parametrize(
        "arg",
        [
            5,
            np.ones(5, dtype=int),
            np.ones((5, 2), dtype=int),
        ],
    )
    def test_executable_generation(self, arg):
        """Test if generated C Program produces correct results."""

        @qjit
        def f(x):
            """Square function with debugging print."""
            y = x * x
            debug.print_memref(y)
            return y

        ans = str(f(arg).tolist()).replace(" ", "")

        binary = compile_executable(f, arg)
        result = subprocess.run(binary, capture_output=True, text=True, check=True)

        # Clean up generated files.
        directory_path = os.path.dirname(binary)
        os.remove(binary)
        os.remove(directory_path + "/" + f.__name__ + ".so")
        os.remove(directory_path + "/main.c")

        assert ans in result.stdout.replace(" ", "").replace("\n", "")

    def test_executable_generation_without_precompiled_function(self):
        """Test if generated C Program produces correct results."""

        @qjit
        def f(x):
            """identity function with debugging print."""
            debug.print_memref(x)
            return x

        arg = 5
        binary = compile_executable(f, arg)
        result = subprocess.run(binary, capture_output=True, text=True, check=True)

        # Clean up generated files.
        directory_path = os.path.dirname(binary)
        os.remove(binary)
        os.remove(directory_path + "/" + f.__name__ + ".so")
        os.remove(directory_path + "/main.c")

        assert str(arg) in result.stdout.replace(" ", "").replace("\n", "")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
