//
// RANDOM, unclassified operation
{
    uint64_t nelements  = a{{NR_OUTPUT}}_nelem;
    uint64_t r_start    = a{{NR_FINPUT}}_first->first;
    uint64_t r_key      = a{{NR_FINPUT}}_first->second;

    union {philox2x32_ctr_t c; uint64_t ul;} ctr, res;
    ctr.ul = r_start;

    for(int64_t i=0; i<nelements; i++) {
        res.c = philox2x32(ctr.c, (philox2x32_key_t) { { (uint32_t)r_key } });
        a{{NR_OUTPUT}}_first[i] = res.ul;
        ctr.ul++;
    }
    // TODO: scalar-management..
}
