#include <stdint.h>
#include <stdio.h>

struct EncodedMemref {
    int64_t rank;
    void *data_aligned;
    int8_t dtype;
    int64_t *sizes;
};

void xor_reduce(void **args, void **results)
{
    struct EncodedMemref *in = (struct EncodedMemref *)args[0];
    struct EncodedMemref *out = (struct EncodedMemref *)results[0];

    int8_t *in_data = (int8_t *)in->data_aligned;
    int32_t *out_data = (int32_t *)out->data_aligned;
    int64_t n = in->sizes[0];
    int32_t acc = 0;
    for (int64_t i = 0; i < n; ++i)
        acc ^= (int32_t)in_data[i];
    out_data[0] = acc;
}
