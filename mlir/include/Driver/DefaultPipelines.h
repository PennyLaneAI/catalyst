// Copyright 2025 Xanadu Quantum Technologies Inc.

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

#include <numeric>
#include <string>
#include <vector>

namespace catalyst {
namespace driver {

using PassNames = std::vector<std::string>;
struct PipelineInfo {
    std::string name;
    PassNames passNames;
};
using PipelineNames = std::vector<std::string>;
using PipelineList = std::vector<PipelineInfo>;

const PipelineList pipelineList
{
    {
        "enforce-runtime-invariants-pipeline",
        {
            // We want the invariant that transforms that generate multiple
            // tapes will generate multiple qnodes. One for each tape.
            // Split multiple tapes enforces that invariant.
            "split-multiple-tapes",
            // Run the transform sequence defined in the MLIR module
            "builtin.module(apply-transform-sequence)",
            // Nested modules are something that will be used in the future
            // for making device specific transformations.
            // Since at the moment, nothing in the runtime is using them
            // and there is no lowering for them,
            // we inline them to preserve the semantics. We may choose to
            // keep inlining modules targeting the Catalyst runtime.
            // But qnodes targeting other backends may choose to lower
            // this into something else.
            "inline-nested-module"
        }
    }
};

PipelineNames getPipelineNames()
{
    static std::vector<std::string> names =
        std::accumulate(driver::pipelineList.begin(), driver::pipelineList.end(),
            std::vector<std::string>{},
            [](auto acc, const auto &pipelineInfo) {
                acc.emplace_back(pipelineInfo.name);
                return acc;
            }
        );
    return names;
}

const PassNames& getEnforceRuntimeInvariantsPipeline()
{
    return pipelineList[0].passNames;
}

} // namespace driver
} // namespace catalyst
