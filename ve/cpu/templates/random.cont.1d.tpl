//
// RANDOM, unclassified operation
{
    uint64_t nelements  = iterspace->nelem;
    uint64_t r_start    = a{{NR_FINPUT}}_first->first;  // Counter "offset"
    uint64_t r_key      = a{{NR_FINPUT}}_first->second; // Key / Seed

    philox2x32_key_t key = { {r_key} };                 // Assign the key/seed

    #pragma omp parallel for
    for(uint64_t i=0; i<nelements; i++) {
        philox2x32_as_1x64_t ctr, rand;
        ctr.combined = r_start+i;

        rand.orig = philox2x32(ctr.orig, key);
        a{{NR_OUTPUT}}_first[i] = rand.combined;
    }
}
