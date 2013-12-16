void {{SYMBOL}}(int tool, ...)
{
    va_list list;
    va_start(list,tool);
    {{TYPE_A0}} *a0_data = va_arg(list, {{TYPE_A0}}*);
    uint64_t nelements  = va_arg(list, uint64_t);
    uint64_t start      = va_arg(list, uint64_t);
    uint64_t key        = va_arg(list, uint64_t);
    va_end(list);

    threefry2x64_ctr_t ctr123;
    ctr123.v[0] = start;
    ctr123.v[1] = 0;          // index

    threefry2x64_key_t key123;
    key123.v[0] = key;
    key123.v[1] = 0xdeadbeef;   // seed

    for(int64_t i=0; i<nelements; i++) {
        threefry2x64_ctr_t c = threefry2x64(ctr123, key123);
        a0_data[i] = c.v[0];
        ctr123.v[0]++;
    }
}

