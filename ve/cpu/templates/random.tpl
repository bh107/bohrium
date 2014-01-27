{
    va_list list;
    va_start(list,tool);
    uint64_t nelements  = va_arg(list, uint64_t);
    uint64_t r_start    = va_arg(list, uint64_t);
    uint64_t r_key      = va_arg(list, uint64_t);

    {{#OPERAND}}
    {{TYPE}} *a{{NR}}_data = va_arg(list, {{TYPE}}*);
    {{/OPERAND}}
    va_end(list);

    threefry2x64_ctr_t ctr123;
    ctr123.v[0] = r_start;
    ctr123.v[1] = 0;          // index

    threefry2x64_key_t key123;
    key123.v[0] = r_key;
    key123.v[1] = 0xdeadbeef;   // seed

    for(int64_t i=0; i<nelements; i++) {
        threefry2x64_ctr_t c = threefry2x64(ctr123, key123);
        {{#OPERAND}}
        a{{NR}}_data[i] = c.v[0];
        {{/OPERAND}}
        ctr123.v[0]++;
    }
}

