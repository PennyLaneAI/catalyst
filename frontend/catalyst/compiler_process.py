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

"""This module contains the functions necessary for calling into the
python bindings of the compiler driver. This module should be isolated
as much as possible from regular python environment as we want to avoid
polluting both the child process and the parent process.

The parent process is protected from the symbols in the child process
just by virtue of multiprocessing using the spawn strategy.

However, the child process will load this module, so it
is best to avoid loading unnecessary modules. E.g., tensorflow, jax, etc.

What kind of errors are we protecting against?

From dlopen(3):

    Any  global  symbols in the executable that were placed into its dynamic symbol table by ld(1)
    can also be used to resolve references in a dynamically loaded shared object.

    Symbols may be placed in the dynamic symbol table either because:
    * the executable was linked with the flag "-rdynamic" (or, synonymously, "--export-dynamic"),
      which causes all of the executable's global symbols to  be  placed in the dynamic symbol table, or
    * because ld(1) noted a dependency on a symbol in another object during static linking.

This means that if I have a shared object that is being loaded into the current process by dlopen,
and that it will call a function through the procedure linkage table (PLT) then it is possible
that the function being called is not the one we expected (that may be included in the same
translation unit) but the one in the parent process.

From https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes:

    As to why -fvisibility=hidden is necessary, because pybind modules could have been compiled
    under different versions of pybind itself, it is also important that the symbols defined in
    one module do not clash with the potentially-incompatible symbols defined in another. 

    While Python extension modules are usually loaded with localized symbols (under POSIX systems
    typically using dlopen with the RTLD_LOCAL flag), this Python default can be changed, but even
    if it isn’t it is not always enough to guarantee complete independence of the symbols involved
    when not using -fvisibility=hidden.

This also tells us that normally we expect all symbols in pybind11 modules to have a hidden visibility
(with the exception of the PyInit_{ModuleName} function). However, do note that for MLIR plugins
this is not possible. In order for MLIR plugins to work, we need visibility=default.

Since we need visibility=default then we need to isolate the compiler as much as possible.

It is possible to isolate the compiler driver even further by using a subprocess
and calling quantum-opt / catalyst-cli directly. However, at the moment, having this
python wrapper around the binary and keeping the python bindings feels more convenient.

The spawning strategy is documented to be slow. It is possible to optimize
this to use a forkserver strategy. That way, the python process is only
spawned once (which is slow) and then every time a new request is made
the spawned process is forked, which is a bit faster. This would
provide the necessary isolation to prevent 
"""

import multiprocessing as mp



def request_handler(conn, ir, workspace, module_name, options, lower_to_llvm):
    """Call the compiler driver from a newly spawned python interpreter.

    This spawned python interpreter communicates with the parent python
    interpreter by sending messages through the `conn`ected Pipe.
    """
    # We need this here to avoid importing the compiler_driver to
    # the parent process.
    from mlir_quantum.compiler_driver import run_compiler_driver

    try:

        compiler_output = run_compiler_driver(
            ir,
            str(workspace),
            module_name,
            keep_intermediate=options.keep_intermediate,
            async_qnodes=options.async_qnodes,
            verbose=options.verbose,
            pipelines=options.get_pipelines(),
            lower_to_llvm=lower_to_llvm,
            checkpoint_stage=options.checkpoint_stage,
        )

        filename = compiler_output.get_object_filename()
        out_IR = compiler_output.get_output_ir()
        diagnostic_messages = compiler_output.get_diagnostic_messages()

        conn.send((filename, out_IR, diagnostic_messages, None))
    except Exception as error:
        conn.send((None, None, None, error))
    finally:
        conn.close()


def request(ir, workspace, module_name, options, lower_to_llvm):
    """Handles the necessary logic for spawning a child interpreter"""
    ctx = mp.get_context("spawn")
    conn1, conn2 = mp.Pipe(duplex=False)
    process_args = (conn2, ir, workspace, module_name, options, lower_to_llvm)
    process = mp.Process(target=request_handler, args=process_args)
    process.start()
    if not conn1.poll(timeout=5):
        raise CompileError("Connection timed out")
    filename, out_IR, diagnostic_messages, exception = conn1.recv()
    process.join()
    return filename, out_IR, diagnostic_messages, exception
