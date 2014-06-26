#include <Random123/philox.h>
inline ulong random(ulong2 r123, ulong tid)
{
    union {philox2x32_ctr_t c; ulong ul;} ctr, res;
    ctr.ul = r123.s0 + tid;
    res.c = philox2x32(ctr.c, (philox2x32_key_t){{(uint)r123.s1}});
    return res.ul;
} 
