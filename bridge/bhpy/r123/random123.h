#include <stdint.h>
#include <Random123/philox.h>

uint64_t r123_ph2x32(uint64_t c, uint64_t k)
{
    union {philox2x32_ctr_t c; uint64_t ul;} ctr, res;
    ctr.ul = c;
    res.c = philox2x32(ctr.c, (philox2x32_key_t){{(uint32_t)k}});
    return res.ul;
} 
