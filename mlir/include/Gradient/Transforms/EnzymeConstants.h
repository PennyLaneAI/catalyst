// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

namespace catalyst {
namespace gradient {

static constexpr const char *enzyme_autodiff_func_name = "__enzyme_autodiff";
static constexpr const char *enzyme_allocation_key = "__enzyme_allocation_like";
static constexpr const char *enzyme_custom_gradient_key = "__enzyme_register_gradient_";
static constexpr const char *enzyme_like_free_key = "__enzyme_function_like_free";
static constexpr const char *enzyme_const_key = "enzyme_const";
static constexpr const char *enzyme_dupnoneed_key = "enzyme_dupnoneed";
static constexpr const char *enzyme_inactivefn_key = "__enzyme_inactivefn";

}
}
