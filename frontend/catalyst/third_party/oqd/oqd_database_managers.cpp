// Copyright 2024 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/eval.h>
#include <nanobind/stl/string.h>

const std::string program = R"(
def py_hi():
   print("hi!")
)";

extern "C" NB_EXPORT void lib_oqd_hi()
{
    namespace nb = nanobind;
    nb::gil_scoped_acquire lock;

    // Evaluate in scope of main module
    nb::object scope = nb::module_::import_("__main__").attr("__dict__");
    nb::exec(nb::str(program.c_str()), scope);
    scope["py_hi"]();
}
