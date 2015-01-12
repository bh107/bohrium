//
// RANDOM, unclassified operation
{
    uint64_t nelements  = iterspace->nelem;

    #pragma omp parallel for
    for(uint64_t i=0; i<nelements; i++) {
        philox2x32_as_1x64_t ctr{{NR_FINPUT}};
        philox2x32_as_1x64_t rand{{NR_OUTPUT}};
        ctr{{NR_FINPUT}}.combined = *a{{NR_FINPUT}}_first + i;
        rand{{NR_OUTPUT}}.orig = philox2x32(
            ctr{{NR_FINPUT}}.orig,
            (philox2x32_key_t){ { *a{{NR_SINPUT}}_first } }
        );
        a{{NR_OUTPUT}}_first[i] = rand{{NR_OUTPUT}}.combined;
    }
}
