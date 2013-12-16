void {{SYMBOL}}(int tool, ...)
{
    va_list list;
    va_start(list,tool);
    {{TYPE_A0}} *a0_data = va_arg(list, {{TYPE_A0}}*);
    uint64_t nelements  = va_arg(list, uint64_t);
    uint64_t start      = va_arg(list, uint64_t);
    uint64_t key        = va_arg(list, uint64_t);
    va_end(list);

    for(int64_t i=0; i<nelements; i++) {
       a0_data[i] = 1; 
    }
}

