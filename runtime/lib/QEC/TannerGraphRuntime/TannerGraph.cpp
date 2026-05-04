// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "TannerGraph.h"

#include "Types.h"

extern "C" {
void __catalyst__qecp__tanner_graph_int32(MemRefT_int32_1d *row_idx_tanner,
                                          MemRefT_int32_1d *col_ptr_tanner,
                                          TannerGraph_CSC_int32 *tanner_graph)
{
    tanner_graph->row_idx = row_idx_tanner;
    tanner_graph->col_ptr = col_ptr_tanner;
}

} // extern "C"
