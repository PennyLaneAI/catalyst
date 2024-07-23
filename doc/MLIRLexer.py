# Pygments lexer for MLIR.
# Authors: Karl F. A. Friebel (@KFAFSP), Clément Fournier (@oowekyala)
# Usage: pygmentize -x -l ./MLIRLexer.py:MLIRLexer file.mlir
#
#  MIT License
#
# Copyright (c) 2024 Clément Fournier, Karl F. A. Friebel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pygments.lexer import RegexLexer, bygroups, include, default, using, this
from pygments.token import Name, Keyword, Operator, Comment, Text, Punctuation, Literal, Whitespace

comment_rule = (r"//.*?\n", Comment)
ssa_value_rule = (r"%[^[ )]*]*", Name.Variable)
symbol_rule = (r"@[^({]*", Name.Function)
basic_block_rule = (r"\^[^(:\]]*", Name.Label)
operation_rule = (
    r"(=)( +)([a-z_]+)(\.)([a-z_]+)",
    bygroups(Operator, Text, Name.Namespace, Text, Keyword.Function),
)
opInRegion_rule = (r"(=)( +)([a-z_]+)", bygroups(Operator, Text, Keyword.Function))
opNoResults_rule = (r"([a-z_]+)( +)(%[^[ )]*]*)", bygroups(Keyword.Function, Text, Name.Variable))
non_assign_operation_rule = (
    r"([a-z_]+)(\.)([a-z_]+)",
    bygroups(Name.Namespace, Text, Keyword.Function),
)
type_rule = (
    r"(!)([a-z_]+)(\.)([a-z0-9_]+)(<([^>]*)>)?",
    bygroups(Operator, Name.Namespace, Text, Keyword.Type, Keyword.Type),
)
complex_type_rule = (
    r"(complex)(<([^>]*)>)?",
    bygroups(Operator, Keyword.Type),
)
int_float_rule = (r"(i|f)([0-9]+)", bygroups(Text, Keyword.Type))
abbrev_type_tule = (r"(!)([a-z0-9]+)", bygroups(Operator, Keyword.Type))
first_attribute_rule = (
    r'([{\[])([a-z_A-Z]+)( = +)([@a-z0-9">=]+)',
    bygroups(Text, Name.Attribute, Text, Name.Tag),
)
following_attribute_rule = (
    r'(, +)([a-z_]+)( = +)([a-z0-9">=@]+)',
    bygroups(Text, Name.Attribute, Text, Name.Tag),
)
abbrev_following_attribute_rule = (r"(, +)([a-z_]+)( = +)", bygroups(Text, Name.Attribute, Text))

digit = r"[0-9]"
hex_digit = r"[0-9a-fA-F]"
letter = r"[a-zA-Z]"
id_punct = r"[$._\-]"

decimal_literal = f"{digit}+"
hexadecimal_literal = f"0x{hex_digit}+"
float_literal = r"[-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?"
string_literal = r'"[^"\n\f\v\r]*"'

suffix_id = f"(?:{digit}+|(?:{letter}|{id_punct})(?:{letter}|{id_punct}|{digit})*)"
value_id = f"%{suffix_id}"

symbol_ref_id = f"@(?:{suffix_id}|{string_literal})"

bare_id = f"(?:{letter}|_)(?:{letter}|{digit}|[_$.])*"
bare_id_without_ns = f"(?:{letter}|_)(?:{letter}|{digit}|[_$])*"
bare_id_with_ns = fr"((?:{letter}|_)(?:{letter}|{digit}|[_$])*)(\.)((?:{letter}|{digit}|[_$.])+)"

integer_type = f"[su]?i{digit}+"
float_type = (
    r"(?:f(?:16|32|64|80|128)|bf16|tf32|f8E5M2|f8E4M3FN|f8E5M2FNUZ|f8E5M3FNUZ|f8E4M3B11FNUZ)"
)

op_result_1 = f"({value_id})(:)({decimal_literal})"


class MLIRLexer(RegexLexer):
    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]

    tokens = {
        "comments": [(r"//.*?\n", Comment), (r"\.\.\.", Comment)],  # pretend ellipsis is comment
        "literals": [
            (float_literal, Literal.Number),
            (hexadecimal_literal, Literal.Number),
            (decimal_literal, Literal.Number),
            (string_literal, Literal.String),
            (r"[^\S\r\n]+", Whitespace),
            default("#pop"),
        ],
        "any-literal-no-default": [
            (float_literal, Literal.Number),
            (hexadecimal_literal, Literal.Number),
            (decimal_literal, Literal.Number),
            (string_literal, Literal.String),
            (r"[^\S\r\n]+", Whitespace),
            include("tensor-literal"),
        ],
        "tensor-literal": [
            (r"\[", Punctuation, "tensor-literal-1"),
            (
                f"({hexadecimal_literal}|{decimal_literal}|true|false)(,)",
                bygroups(Literal.Number, Punctuation),
            ),
            (f"{hexadecimal_literal}|{decimal_literal}|true|false", Literal.Number, "#pop"),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "tensor-literal-1": [
            (r"\[", Punctuation, "#push"),
            (r"\]", Punctuation, "#pop"),
            (
                f"({hexadecimal_literal}|{decimal_literal}|true|false)(,)",
                bygroups(Literal.Number, Punctuation),
            ),
            (f"{hexadecimal_literal}|{decimal_literal}|true|false", Literal.Number),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "id-list": [
            (f"({bare_id})(,)", bygroups(Name.Variable, Punctuation)),
            (f"{bare_id}", Name.Variable, "#pop"),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "rparen": [(r"\)", Punctuation, "#pop"), (r"[^\S\r\n]+", Whitespace)],
        "rsquare": [(r"\]", Punctuation, "#pop"), (r"[^\S\r\n]+", Whitespace)],
        "rangle": [(r">", Punctuation, "#pop"), (r"[^\S\r\n]+", Whitespace)],
        "int-attr": [
            (
                fr"({decimal_literal}|{hexadecimal_literal})\s*(:)\s*({integer_type})",
                bygroups(Literal.Number, Punctuation, Keyword.Type),
            ),
            (f"true|false", Keyword),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "float-attr": [
            (
                fr"({float_literal})\s*(:)\s*({float_type})",
                bygroups(Literal.Number, Punctuation, Keyword.Type),
            ),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "affine-map": [
            (r"\(", Punctuation, ("affine-map-1", "rparen", "id-list")),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "affine-map-1": [
            (r"\[", Punctuation, ("affine-map-2", "rsquare", "id-list")),
            include("affine-map-2"),
        ],
        "affine-map-2": [
            (r"\s*(->)\s*(\()", bygroups(Operator, Punctuation), "affine-expr"),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "affine-expr": [
            (r"\(", Punctuation, "#push"),
            (r"\)", Punctuation, "#pop"),
            (r",", Punctuation),
            (r"[+\-*]|mod|floordiv|ceildiv", Operator),
            (f"{decimal_literal}", Literal.Number),
            (f"{bare_id}", Name.Variable),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "affine-set": [
            (r"\(", Punctuation, ("affine-set-1", "rparen", "id-list")),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "affine-set-1": [
            (r"\[", Punctuation, ("affine-set-2", "rsquare", "id-list")),
            include("affine-set-2"),
        ],
        "affine-set-2": [
            (r"\s*(:)\s*(\()", bygroups(Operator, Punctuation), "affine-cstr"),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "affine-cstr": [
            include("affine-expr"),
            (r"==", Operator),
            (r">", Operator),
            (r"<", Operator),
            (r"<=", Operator),
            (r">=", Operator),
        ],
        "array-attr": [(r",", Punctuation), (r"\]", Punctuation, "#pop"), include("attr")],
        "dict-attr": [
            (r",", Punctuation),
            (r"\}", Punctuation, "#pop"),
            (fr"({string_literal})\s*,(,)", bygroups(Literal.String, Punctuation)),
            (fr"({bare_id})\s*(,)", bygroups(Text, Punctuation)),
            (fr"({string_literal})\s*(=)\s*", bygroups(Literal.String, Operator), "attr"),
            (fr"({bare_id})\s*(=)\s*", bygroups(Text, Operator), "attr"),
            (r"\s+", Whitespace),
            include("comments"),
        ],
        "attr": [
            include("int-attr"),
            include("float-attr"),
            (r"unit", Keyword.Attribute, "#pop"),
            (r"(affine_map)(<)", bygroups(Keyword.Type, Punctuation), ("rangle", "affine-map")),
            (r"(affine_set)(<)", bygroups(Keyword.Type, Punctuation), ("rangle", "affine-set")),
            (r"\[", Punctuation, "array-attr"),
            (r"\{", Punctuation, "dict-attr"),
            (
                r"(array)(<)\s*({integer_type}|{float_type})\s*(:)",
                bygroups(Keyword.Type, Punctuation, Keyword.Type, Punctuation),
                ("rangle", "tensor-literal"),
            ),
            (
                f"(#){bare_id_with_ns}(<)",
                bygroups(Punctuation, Name.Namespace, Punctuation, Keyword.Type, Punctuation),
                "attr-params",
            ),
            (
                f"(#){bare_id_with_ns}",
                bygroups(Punctuation, Name.Namespace, Punctuation, Keyword.Type),
                "#pop",
            ),
            (
                f"(#)({bare_id_without_ns})(<)({string_literal})(>)",
                bygroups(Punctuation, Keyword.Attribute, Punctuation, Literal.String, Punctuation),
                "#pop",
            ),
            (f"(#)({bare_id_without_ns})", bygroups(Punctuation, Name.Variable), "#pop"),
            # Best effort: parse some literals.
            include("literals"),
        ],
        "attr-content": [include("attr"), (r"=", Operator), (r",", Punctuation), (r".+", Text)],
        "attr-params": [
            (r"<", Punctuation, "#push"),
            (r">", Punctuation, "#pop"),
            (r"\(", Punctuation, "attr-params-1"),
            (r"\[", Punctuation, "attr-params-2"),
            (r"\{", Punctuation, "attr-params-3"),
            include("attr-content"),
        ],
        "attr-params-1": [
            (r"\(", Punctuation, "#push"),
            (r"\)", Punctuation, "#pop"),
            (r"\[", Punctuation, "attr-params-2"),
            (r"\{", Punctuation, "attr-params-3"),
            include("attr-content"),
        ],
        "attr-params-2": [
            (r"\[", Punctuation, "#push"),
            (r"\]", Punctuation, "#pop"),
            (r"\(", Punctuation, "attr-params-1"),
            (r"\{", Punctuation, "attr-params-3"),
            include("attr-content"),
        ],
        "attr-params-3": [
            (r"\{", Punctuation, "#push"),
            (r"\}", Punctuation, "#pop"),
            (r"\[", Punctuation, "attr-params-2"),
            (r"\(", Punctuation, "attr-params-1"),
            include("attr-content"),
        ],
        "type": [
            (f"{integer_type}", Keyword.Type, "#pop"),
            (f"{float_type}", Keyword.Type, "#pop"),
            (r"index|none", Keyword.Type, "#pop"),
            (
                r"(!)(closure)(.)(box)(<)",
                bygroups(Punctuation, Name.Namespace, Punctuation, Keyword.Type, Punctuation),
                ("#pop", "rangle", "func-type"),
            ),
            (r"(memref)(<)", bygroups(Keyword.Type, Punctuation), ("attr-params", "dim-list")),
            (r"(tensor)(<)", bygroups(Keyword.Type, Punctuation), ("attr-params", "dim-list")),
            (r"(complex)(<)", bygroups(Keyword.Type, Punctuation), ("inside_type")),
            (r"(vector)(<)", bygroups(Keyword.Type, Punctuation), "vector-params"),
            (
                f"(!){bare_id_with_ns}(<)",
                bygroups(Punctuation, Name.Namespace, Punctuation, Keyword.Type, Punctuation),
                "attr-params",
            ),
            (
                f"(!){bare_id_with_ns}",
                bygroups(Punctuation, Name.Namespace, Punctuation, Keyword.Type),
                "#pop",
            ),
            (f"(!)({bare_id_without_ns})(<)", bygroups(Keyword.Type, Punctuation), "attr-params"),
            (f"(!)({bare_id_without_ns})", bygroups(Punctuation, Keyword.Type), "#pop"),
            (r"\s+", Whitespace),
        ],
        "func-type": [
            (r"\(", Punctuation, ("arrow-type", "type-list-tail", "type")),
            (r"[^\S\r\n]+", Whitespace),
            default("#pop"),
        ],
        "inside_type": [
            (r"([?x0-9\[\]]+)", bygroups(using(this, state="number"))),
            (r"->|>=", Punctuation),
            (r"\b([\w.$\-]+)\b", Keyword),
            (r">", Punctuation, "#pop"),
        ],
        "type-list-tail": [
            (r"\)", Punctuation, "#pop"),
            (r",", Punctuation, "type"),
            (r"[^\S\r\n]+", Whitespace),
            default("#pop"),
        ],
        "arrow-type": [
            (r"->", Punctuation, "type-or-list"),
            (r"[^\S\r\n]+", Whitespace),
            default("#pop"),
        ],
        "type-or-list": [(r"\(", Punctuation, "type-list-tail", "type"), include("type")],
        "dim-list": [
            (r">", Punctuation, "#pop"),
            (r"\*x", Keyword),
            (r"(\?)(x)", bygroups(Keyword, Punctuation)),
            (f"({decimal_literal})(x)", bygroups(Literal.Number, Punctuation)),
            include("type"),
        ],
        "vector-params": [
            (r">", Punctuation, "#pop"),
            (f"({decimal_literal})(x)", bygroups(Literal.Number, Punctuation)),
            (f"{integer_type}", Keyword.Type),
            (f"{float_type}", Keyword.Type),
            (r"index", Keyword.Type),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "op-results": [
            (r"(=)", bygroups(Operator), "op"),
            (f"(,)(\s*)({value_id})", bygroups(Punctuation, Whitespace, Name.Variable)),
            (f"(:)({decimal_literal})", bygroups(Punctuation, Literal.Number), "op-results-1"),
            (r"[^\S\r\n]+", Whitespace),
            default("#pop"),
        ],
        "op-results-1": [
            (r"(=)", Operator, "op"),
            (
                f"(,)(\s*)({value_id})",
                bygroups(Punctuation, Whitespace, Name.Variable),
                "op-results",
            ),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "ssa-operand": [
            (
                f"({value_id})(#)({decimal_literal})",
                bygroups(Name.Variable, Punctuation, Literal.Number),
            ),
            (f"{value_id}", Name.Variable),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "operand-list": [
            (r"\)", Punctuation, "#pop"),
            (r",", Punctuation),
            include("ssa-operand"),
            include("comments"),
        ],
        "successor-list": [
            (r"\]", Punctuation, "#pop"),
            (r",", Punctuation),
            (f"\^{bare_id}", Name.Label),
            (r"[^\S\r\n]+", Whitespace),
            include("comments"),
        ],
        "block-args": [
            (r"(\))(\s*)(:)", bygroups(Punctuation, Whitespace, Punctuation), "#pop"),
            (f"({value_id})(\s*)(:)", bygroups(Name.Variable, Whitespace, Punctuation), ("type")),
            (r",", Punctuation),
            (r"[^\S\r\n]+", Whitespace),
        ],
        "region-list": [
            (r"\)", Punctuation, "#pop"),
            (r"\{", Punctuation, "region"),
            (r",", Punctuation),
            (r"[^\S]+", Whitespace),
            include("comments"),
        ],
        "region": [
            include("comments"),
            (r"(\n|\s)+", Whitespace),
            (r"\}", Punctuation, "#pop"),
            (
                fr"(\^{bare_id})(\s*)(\()",
                bygroups(Name.Label, Whitespace, Punctuation),
                "block-args",
            ),
            (fr"(\^{bare_id})(\s*)(:)", bygroups(Name.Label, Whitespace, Punctuation)),
            (f"{value_id}", Name.Variable, "op-results"),
            include("op"),
        ],
        "dict-or-region": [include("dict-attr"), include("region")],
        "op-no-default": [
            (f"{string_literal}", Literal.String, "op-generic"),
            (
                f"{bare_id_with_ns}",
                bygroups(Name.Namespace, Punctuation, Name.Function),
                "op-pretty",
            ),
            (f"{bare_id_without_ns}", Name.Function, "op-pretty"),
            (r"[^\S]+", Whitespace),
            include("comments"),
        ],
        "op": [
            include("op-no-default"),
            default("#pop"),
        ],
        "op-generic": [
            (r"\(", Punctuation, ("op-generic-1", "operand-list")),
            (r"[^\S]+", Whitespace),
            include("comments"),
        ],
        "op-generic-1": [
            (r"\[", Punctuation, ("op-generic-2", "successor-list")),
            include("op-generic-2"),
        ],
        "op-generic-2": [
            (
                r"(<)(\s*)(\{)",
                bygroups(Punctuation, Whitespace, Punctuation),
                ("op-generic-3", "rangle", "dict-attr"),
            ),
            include("op-generic-3"),
        ],
        "op-generic-3": [
            (r"\(", Punctuation, ("op-generic-4", "region-list")),
            include("op-generic-4"),
        ],
        "op-generic-4": [
            (r"\{", Punctuation, ("op-generic-5", "dict-attr")),
            include("op-generic-5"),
        ],
        "op-generic-5": [
            (r":", Punctuation, ("func-type")),
            (r"[^\S]+", Whitespace),
            include("comments"),
        ],
        "op-pretty": [
            include("ssa-operand"),
            include("comments"),
            include("any-literal-no-default"),
            (fr"\^{bare_id}", Name.Label),
            (r"[\(\)\[\],=]", Punctuation),
            (f"{symbol_ref_id}", Name.Function),
            (f"(::)({symbol_ref_id})", bygroups(Punctuation, Name.Function)),
            (r":", Punctuation, ("type")),
            (r"->", Punctuation, ("type-or-list")),
            (r"\{", Punctuation, "dict-or-region"),
            (r"\S+", Text),
            (
                r"\n",
                Whitespace,
                "#pop",
            ),  # assume pretty ops fit on one line (regions can be multiline)
        ],
        "root": [
            include("comments"),
            (r"(\n|\s)+", Whitespace),
            (
                fr"(#)({bare_id})(\s*)(=)",
                bygroups(Punctuation, Name.Variable, Whitespace, Operator),
                "attr",
            ),
            (
                fr"(!)({bare_id})(\s*)(=)",
                bygroups(Punctuation, Keyword.Type, Whitespace, Operator),
                "type",
            ),
            (r"\}", Punctuation, "op-pretty"),
            (f"(#)({bare_id})", bygroups(Punctuation, Name.Variable)),
            (f"(!)({bare_id})", bygroups(Punctuation, Keyword.Type)),
            (f"{value_id}", Name.Variable, "op-results"),
            include("op-no-default"),  # cannot include default match bc infinite loop
        ],
    }
