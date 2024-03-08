#!/usr/bin/python3
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

# This program checks the syntax of quantum device configuration files. It is a strict parser of
# TOML format, narrowed down to match our requirements.  For the Lark's EBNF dialect syntax, see the
# Lark grammar reference:
#
# * https://lark-parser.readthedocs.io/en/latest/grammar.html

import sys
from argparse import ArgumentParser
from textwrap import dedent

try:
    from lark import Lark, LarkError, UnexpectedInput
except ImportError as e:
    raise RuntimeError("toml-check.py requires `lark` library. Consider using `pip install lark`") \
    from e

parser=Lark(dedent('''
    start: schema_body \
           gates_native_section \
           gates_decomp_section \
           gates_matrix_section \
           gates_observables_section \
           measurement_processes_section \
           compilation_section
    schema_body: schema_decl -> schema
    gates_native_section: "[operators.gates.native]" gate_decls -> native
    gates_decomp_section: "[operators.gates.decomp]" gate_decls -> decomp
    gates_matrix_section: "[operators.gates.matrix]" gate_decls -> matrix
    gates_observables_section: "[operators.observables]" gate_decls -> observables
    measurement_processes_section: "[measurement_processes]" gate_decls -> measurement
    compilation_section: "[compilation]" flag_decl* -> compilation
    schema_decl: "schema" "=" "2"
    gate_decls: (gate_decl)*
    gate_decl: /[a-zA-Z0-9_]+/ "=" "{" (gate_trait ("," gate_trait)*)? "}"
    gate_trait: gate_condition | gate_properties
    gate_condition: "condition" "=" "[" ( "\\"finiteshots\\"" | "\\"analytic\\"" ) "]"
    gate_properties: "properties" "=" "[" gate_property ("," gate_property)* "]"
    gate_property: "\\"controllable\\"" | "\\"invertible\\"" | "\\"differentiable\\""
    flag_decl: ( "qjit_compatible" | "runtime_code_generation" | \
                 "mid_circuit_measurement" | "dynamic_qubit_management" ) "=" boolean
    boolean: "true" | "false"
    COMMENT: "#" /./*
    %import common.WS
    %ignore WS
    %ignore COMMENT
    '''))


if __name__ == '__main__':
    ap = ArgumentParser(prog='toml-check.py')
    ap.add_argument('filenames', metavar='TOML', type=str, nargs='+',
                    help='One or more *toml files to check')
    ap.add_argument('--verbose', action='store_true', help='Be verbose')
    fname = None
    try:
        arguments = ap.parse_args(sys.argv[1:])
        for fname in arguments.filenames:
            with open(fname, 'r', encoding='utf-8') as f:
                contents = f.read()
            tree = parser.parse(contents)
            if arguments.verbose:
                print(tree.pretty())
    except UnexpectedInput as e:
        print(f"toml-check: error in {fname}:{e.line}:{e.column}", file=sys.stderr)
        raise e
    except LarkError as e:
        print(f"toml-check: error in {fname}", file=sys.stderr)
        raise e


