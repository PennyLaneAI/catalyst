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

#include <stdint.h>
#include <stdio.h>

struct EncodedMemref {
    int64_t  rank;
    void    *data_aligned;
    int8_t   dtype;
    int64_t *sizes;
};

void xor_reduce(void **args, void **results)
{
    struct EncodedMemref *in  = (struct EncodedMemref *)args[0];
    struct EncodedMemref *out = (struct EncodedMemref *)results[0];

    int8_t  *in_data = (int8_t  *)in->data_aligned;
    int32_t *out_data = (int32_t *)out->data_aligned;
    int64_t  n   = in->sizes[0];
    int32_t  acc = 0;
    for (int64_t i = 0; i < n; ++i)
        acc ^= (int32_t)in_data[i];
    out_data[0] = acc;
}
