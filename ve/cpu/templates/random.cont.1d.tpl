// RANDOM, unclassified operation
{
    uint64_t nelements  = a{{NR_OUTPUT}}_nelem;
    uint64_t r_start    = *(uint64_t*)args->data[{{NR_OUTPUT}}+1];
    uint64_t r_key      = *(uint64_t*)args->data[{{NR_OUTPUT}}+2];

    threefry2x64_ctr_t ctr123;
    ctr123.v[0] = r_start;
    ctr123.v[1] = 0;          // index

    threefry2x64_key_t key123;
    key123.v[0] = r_key;
    key123.v[1] = 0xdeadbeef;   // seed

    for(int64_t i=0; i<nelements; i++) {
        threefry2x64_ctr_t c = threefry2x64(ctr123, key123);
        a{{NR_OUTPUT}}_first[i] = c.v[0];
        ctr123.v[0]++;
    }
}

