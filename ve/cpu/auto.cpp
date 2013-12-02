const char* bhtype_to_ctype(bh_type type)
{
    switch(type) {
        case BH_BOOL: return "unsigned char";
        case BH_INT8: return "int8_t";
        case BH_INT16: return "int16_t";
        case BH_INT32: return "int32_t";
        case BH_INT64: return "int64_t";
        case BH_UINT8: return "uint8_t";
        case BH_UINT16: return "uint16_t";
        case BH_UINT32: return "uint32_t";
        case BH_UINT64: return "uint64_t";
        case BH_FLOAT16: return "uint16_t";
        case BH_FLOAT32: return "float";
        case BH_FLOAT64: return "double";
        case BH_COMPLEX64: return "complex<float>";
        case BH_COMPLEX128: return "complex<double>";
        case BH_UNKNOWN: return "<UNKNOWN>";

        default:
            return "{{UNKNOWN}}";
    }
}



const char* bhtype_to_shorthand(bh_type type)
{
    switch(type) {
        case BH_BOOL: return "z";
        case BH_INT8: return "b";
        case BH_INT16: return "s";
        case BH_INT32: return "i";
        case BH_INT64: return "l";
        case BH_UINT8: return "B";
        case BH_UINT16: return "S";
        case BH_UINT32: return "I";
        case BH_UINT64: return "L";
        case BH_FLOAT16: return "h";
        case BH_FLOAT32: return "f";
        case BH_FLOAT64: return "d";
        case BH_COMPLEX64: return "c";
        case BH_COMPLEX128: return "C";
        case BH_UNKNOWN: return "U";

        default:
            return "{{UNKNOWN}}";
    }
}



const char* enumstr_to_ctypestr(const char* enumstr)
{
    if (false) {}
    else if (strcmp("BH_BOOL", enumstr)==0) { return "unsigned char"; }
    else if (strcmp("BH_INT8", enumstr)==0) { return "int8_t"; }
    else if (strcmp("BH_INT16", enumstr)==0) { return "int16_t"; }
    else if (strcmp("BH_INT32", enumstr)==0) { return "int32_t"; }
    else if (strcmp("BH_INT64", enumstr)==0) { return "int64_t"; }
    else if (strcmp("BH_UINT8", enumstr)==0) { return "uint8_t"; }
    else if (strcmp("BH_UINT16", enumstr)==0) { return "uint16_t"; }
    else if (strcmp("BH_UINT32", enumstr)==0) { return "uint32_t"; }
    else if (strcmp("BH_UINT64", enumstr)==0) { return "uint64_t"; }
    else if (strcmp("BH_FLOAT16", enumstr)==0) { return "uint16_t"; }
    else if (strcmp("BH_FLOAT32", enumstr)==0) { return "float"; }
    else if (strcmp("BH_FLOAT64", enumstr)==0) { return "double"; }
    else if (strcmp("BH_COMPLEX64", enumstr)==0) { return "float complex"; }
    else if (strcmp("BH_COMPLEX128", enumstr)==0) { return "double complex"; }
    else if (strcmp("BH_UNKNOWN", enumstr)==0) { return "<UNKNOWN>"; }
    else { return "{{UNKNOWN}}"; }
}



const char* enumstr_to_shorthand(const char* enumstr)
{
    if (false) {}
    else if (strcmp("BH_BOOL", enumstr)==0) { return "unsigned char"; }
    else if (strcmp("BH_INT8", enumstr)==0) { return "int8_t"; }
    else if (strcmp("BH_INT16", enumstr)==0) { return "int16_t"; }
    else if (strcmp("BH_INT32", enumstr)==0) { return "int32_t"; }
    else if (strcmp("BH_INT64", enumstr)==0) { return "int64_t"; }
    else if (strcmp("BH_UINT8", enumstr)==0) { return "uint8_t"; }
    else if (strcmp("BH_UINT16", enumstr)==0) { return "uint16_t"; }
    else if (strcmp("BH_UINT32", enumstr)==0) { return "uint32_t"; }
    else if (strcmp("BH_UINT64", enumstr)==0) { return "uint64_t"; }
    else if (strcmp("BH_FLOAT16", enumstr)==0) { return "uint16_t"; }
    else if (strcmp("BH_FLOAT32", enumstr)==0) { return "float"; }
    else if (strcmp("BH_FLOAT64", enumstr)==0) { return "double"; }
    else if (strcmp("BH_COMPLEX64", enumstr)==0) { return "float complex"; }
    else if (strcmp("BH_COMPLEX128", enumstr)==0) { return "double complex"; }
    else if (strcmp("BH_UNKNOWN", enumstr)==0) { return "<UNKNOWN>"; }
    else { return "{{UNKNOWN}}"; }
}


