const char* enumstr_to_shorthand(const char* enumstr)
{
    if (false) {}
    else if (strcmp("BH_BOOL", enumstr)==0) { return "z"; }
    else if (strcmp("BH_INT8", enumstr)==0) { return "b"; }
    else if (strcmp("BH_INT16", enumstr)==0) { return "s"; }
    else if (strcmp("BH_INT32", enumstr)==0) { return "i"; }
    else if (strcmp("BH_INT64", enumstr)==0) { return "l"; }
    else if (strcmp("BH_UINT8", enumstr)==0) { return "B"; }
    else if (strcmp("BH_UINT16", enumstr)==0) { return "S"; }
    else if (strcmp("BH_UINT32", enumstr)==0) { return "I"; }
    else if (strcmp("BH_UINT64", enumstr)==0) { return "L"; }
    else if (strcmp("BH_FLOAT16", enumstr)==0) { return "h"; }
    else if (strcmp("BH_FLOAT32", enumstr)==0) { return "f"; }
    else if (strcmp("BH_FLOAT64", enumstr)==0) { return "d"; }
    else if (strcmp("BH_COMPLEX64", enumstr)==0) { return "c"; }
    else if (strcmp("BH_COMPLEX128", enumstr)==0) { return "C"; }
    else if (strcmp("BH_UNKNOWN", enumstr)==0) { return "U"; }
    else { return "{{UNKNOWN}}"; }
}


const char* bh_layoutmask_to_shorthand(const int mask)
{
    switch(mask) {
        case 1: return "C"; 
        case 2: return "D"; 
        case 4: return "S"; 
        case 8: return "P"; 
        case 17: return "CC"; 
        case 18: return "DC"; 
        case 20: return "SC"; 
        case 24: return "PC"; 
        case 33: return "CD"; 
        case 34: return "DD"; 
        case 36: return "SD"; 
        case 40: return "PD"; 
        case 65: return "CS"; 
        case 66: return "DS"; 
        case 68: return "SS"; 
        case 72: return "PS"; 
        case 129: return "CP"; 
        case 130: return "DP"; 
        case 132: return "SP"; 
        case 136: return "PP"; 
        case 273: return "CCC"; 
        case 274: return "DCC"; 
        case 276: return "SCC"; 
        case 280: return "PCC"; 
        case 289: return "CDC"; 
        case 290: return "DDC"; 
        case 292: return "SDC"; 
        case 296: return "PDC"; 
        case 321: return "CSC"; 
        case 322: return "DSC"; 
        case 324: return "SSC"; 
        case 328: return "PSC"; 
        case 385: return "CPC"; 
        case 386: return "DPC"; 
        case 388: return "SPC"; 
        case 392: return "PPC"; 
        case 529: return "CCD"; 
        case 530: return "DCD"; 
        case 532: return "SCD"; 
        case 536: return "PCD"; 
        case 545: return "CDD"; 
        case 546: return "DDD"; 
        case 548: return "SDD"; 
        case 552: return "PDD"; 
        case 577: return "CSD"; 
        case 578: return "DSD"; 
        case 580: return "SSD"; 
        case 584: return "PSD"; 
        case 641: return "CPD"; 
        case 642: return "DPD"; 
        case 644: return "SPD"; 
        case 648: return "PPD"; 
        case 1041: return "CCS"; 
        case 1042: return "DCS"; 
        case 1044: return "SCS"; 
        case 1048: return "PCS"; 
        case 1057: return "CDS"; 
        case 1058: return "DDS"; 
        case 1060: return "SDS"; 
        case 1064: return "PDS"; 
        case 1089: return "CSS"; 
        case 1090: return "DSS"; 
        case 1092: return "SSS"; 
        case 1096: return "PSS"; 
        case 1153: return "CPS"; 
        case 1154: return "DPS"; 
        case 1156: return "SPS"; 
        case 1160: return "PPS"; 
        case 2065: return "CCP"; 
        case 2066: return "DCP"; 
        case 2068: return "SCP"; 
        case 2072: return "PCP"; 
        case 2081: return "CDP"; 
        case 2082: return "DDP"; 
        case 2084: return "SDP"; 
        case 2088: return "PDP"; 
        case 2113: return "CSP"; 
        case 2114: return "DSP"; 
        case 2116: return "SSP"; 
        case 2120: return "PSP"; 
        case 2177: return "CPP"; 
        case 2178: return "DPP"; 
        case 2180: return "SPP"; 
        case 2184: return "PPP"; 

        default:
            printf("Err: Unsupported layoutmask [%d]\n", mask);
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
    else if (strcmp("BH_COMPLEX64", enumstr)==0) { return "struct { float real, imag; }"; }
    else if (strcmp("BH_COMPLEX128", enumstr)==0) { return "struct { double real, imag; }"; }
    else if (strcmp("BH_UNKNOWN", enumstr)==0) { return "<UNKNOWN>"; }
    else { return "{{UNKNOWN}}"; }
}


const char* enum_to_shorthand(bh_type type)
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


const char* enum_to_ctypestr(bh_type type)
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


