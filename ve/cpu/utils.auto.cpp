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
    else if (strcmp("BH_FLOAT32", enumstr)==0) { return "f"; }
    else if (strcmp("BH_FLOAT64", enumstr)==0) { return "d"; }
    else if (strcmp("BH_COMPLEX64", enumstr)==0) { return "c"; }
    else if (strcmp("BH_COMPLEX128", enumstr)==0) { return "C"; }
    else if (strcmp("BH_R123", enumstr)==0) { return "R"; }
    else if (strcmp("BH_UNKNOWN", enumstr)==0) { return "U"; }
    else { return "{{UNKNOWN}}"; }
}

const char* bh_layoutmask_to_shorthand(const int mask)
{
    switch(mask) {
        case 1: return "K"; 
        case 2: return "C"; 
        case 4: return "S"; 
        case 8: return "P"; 
        case 17: return "KK"; 
        case 18: return "CK"; 
        case 20: return "SK"; 
        case 24: return "PK"; 
        case 33: return "KC"; 
        case 34: return "CC"; 
        case 36: return "SC"; 
        case 40: return "PC"; 
        case 65: return "KS"; 
        case 66: return "CS"; 
        case 68: return "SS"; 
        case 72: return "PS"; 
        case 129: return "KP"; 
        case 130: return "CP"; 
        case 132: return "SP"; 
        case 136: return "PP"; 
        case 273: return "KKK"; 
        case 274: return "CKK"; 
        case 276: return "SKK"; 
        case 280: return "PKK"; 
        case 289: return "KCK"; 
        case 290: return "CCK"; 
        case 292: return "SCK"; 
        case 296: return "PCK"; 
        case 321: return "KSK"; 
        case 322: return "CSK"; 
        case 324: return "SSK"; 
        case 328: return "PSK"; 
        case 385: return "KPK"; 
        case 386: return "CPK"; 
        case 388: return "SPK"; 
        case 392: return "PPK"; 
        case 529: return "KKC"; 
        case 530: return "CKC"; 
        case 532: return "SKC"; 
        case 536: return "PKC"; 
        case 545: return "KCC"; 
        case 546: return "CCC"; 
        case 548: return "SCC"; 
        case 552: return "PCC"; 
        case 577: return "KSC"; 
        case 578: return "CSC"; 
        case 580: return "SSC"; 
        case 584: return "PSC"; 
        case 641: return "KPC"; 
        case 642: return "CPC"; 
        case 644: return "SPC"; 
        case 648: return "PPC"; 
        case 1041: return "KKS"; 
        case 1042: return "CKS"; 
        case 1044: return "SKS"; 
        case 1048: return "PKS"; 
        case 1057: return "KCS"; 
        case 1058: return "CCS"; 
        case 1060: return "SCS"; 
        case 1064: return "PCS"; 
        case 1089: return "KSS"; 
        case 1090: return "CSS"; 
        case 1092: return "SSS"; 
        case 1096: return "PSS"; 
        case 1153: return "KPS"; 
        case 1154: return "CPS"; 
        case 1156: return "SPS"; 
        case 1160: return "PPS"; 
        case 2065: return "KKP"; 
        case 2066: return "CKP"; 
        case 2068: return "SKP"; 
        case 2072: return "PKP"; 
        case 2081: return "KCP"; 
        case 2082: return "CCP"; 
        case 2084: return "SCP"; 
        case 2088: return "PCP"; 
        case 2113: return "KSP"; 
        case 2114: return "CSP"; 
        case 2116: return "SSP"; 
        case 2120: return "PSP"; 
        case 2177: return "KPP"; 
        case 2178: return "CPP"; 
        case 2180: return "SPP"; 
        case 2184: return "PPP"; 

        default:
            return "{{UNSUPPORTED}}";
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
    else if (strcmp("BH_FLOAT32", enumstr)==0) { return "float"; }
    else if (strcmp("BH_FLOAT64", enumstr)==0) { return "double"; }
    else if (strcmp("BH_COMPLEX64", enumstr)==0) { return "float complex"; }
    else if (strcmp("BH_COMPLEX128", enumstr)==0) { return "double complex"; }
    else if (strcmp("BH_R123", enumstr)==0) { return "bh_r123"; }
    else if (strcmp("BH_UNKNOWN", enumstr)==0) { return "<UNKNOWN>"; }
    else { return "{{UNKNOWN}}"; }
}


bool bh_typesig_check(int typesig)
{
    switch(typesig) {
        case 273: return true; // zzz: BH_BOOL + (BH_BOOL << 4) + (BH_BOOL << 8)
        case 3549: return true; // CCC: BH_COMPLEX128 + (BH_COMPLEX128 << 4) + (BH_COMPLEX128 << 8)
        case 3276: return true; // ccc: BH_COMPLEX64 + (BH_COMPLEX64 << 4) + (BH_COMPLEX64 << 8)
        case 2730: return true; // fff: BH_FLOAT32 + (BH_FLOAT32 << 4) + (BH_FLOAT32 << 8)
        case 3003: return true; // ddd: BH_FLOAT64 + (BH_FLOAT64 << 4) + (BH_FLOAT64 << 8)
        case 819: return true; // sss: BH_INT16 + (BH_INT16 << 4) + (BH_INT16 << 8)
        case 1092: return true; // iii: BH_INT32 + (BH_INT32 << 4) + (BH_INT32 << 8)
        case 1365: return true; // lll: BH_INT64 + (BH_INT64 << 4) + (BH_INT64 << 8)
        case 546: return true; // bbb: BH_INT8 + (BH_INT8 << 4) + (BH_INT8 << 8)
        case 1911: return true; // SSS: BH_UINT16 + (BH_UINT16 << 4) + (BH_UINT16 << 8)
        case 2184: return true; // III: BH_UINT32 + (BH_UINT32 << 4) + (BH_UINT32 << 8)
        case 2457: return true; // LLL: BH_UINT64 + (BH_UINT64 << 4) + (BH_UINT64 << 8)
        case 1638: return true; // BBB: BH_UINT8 + (BH_UINT8 << 4) + (BH_UINT8 << 8)
        case 2721: return true; // zff: BH_BOOL + (BH_FLOAT32 << 4) + (BH_FLOAT32 << 8)
        case 2993: return true; // zdd: BH_BOOL + (BH_FLOAT64 << 4) + (BH_FLOAT64 << 8)
        case 817: return true; // zss: BH_BOOL + (BH_INT16 << 4) + (BH_INT16 << 8)
        case 1089: return true; // zii: BH_BOOL + (BH_INT32 << 4) + (BH_INT32 << 8)
        case 1361: return true; // zll: BH_BOOL + (BH_INT64 << 4) + (BH_INT64 << 8)
        case 545: return true; // zbb: BH_BOOL + (BH_INT8 << 4) + (BH_INT8 << 8)
        case 1905: return true; // zSS: BH_BOOL + (BH_UINT16 << 4) + (BH_UINT16 << 8)
        case 2177: return true; // zII: BH_BOOL + (BH_UINT32 << 4) + (BH_UINT32 << 8)
        case 2449: return true; // zLL: BH_BOOL + (BH_UINT64 << 4) + (BH_UINT64 << 8)
        case 1633: return true; // zBB: BH_BOOL + (BH_UINT8 << 4) + (BH_UINT8 << 8)
        case 3537: return true; // zCC: BH_BOOL + (BH_COMPLEX128 << 4) + (BH_COMPLEX128 << 8)
        case 3265: return true; // zcc: BH_BOOL + (BH_COMPLEX64 << 4) + (BH_COMPLEX64 << 8)
        case 1297: return true; // zzl: BH_BOOL + (BH_BOOL << 4) + (BH_INT64 << 8)
        case 1501: return true; // CCl: BH_COMPLEX128 + (BH_COMPLEX128 << 4) + (BH_INT64 << 8)
        case 1484: return true; // ccl: BH_COMPLEX64 + (BH_COMPLEX64 << 4) + (BH_INT64 << 8)
        case 1450: return true; // ffl: BH_FLOAT32 + (BH_FLOAT32 << 4) + (BH_INT64 << 8)
        case 1467: return true; // ddl: BH_FLOAT64 + (BH_FLOAT64 << 4) + (BH_INT64 << 8)
        case 1331: return true; // ssl: BH_INT16 + (BH_INT16 << 4) + (BH_INT64 << 8)
        case 1348: return true; // iil: BH_INT32 + (BH_INT32 << 4) + (BH_INT64 << 8)
        case 1314: return true; // bbl: BH_INT8 + (BH_INT8 << 4) + (BH_INT64 << 8)
        case 1399: return true; // SSl: BH_UINT16 + (BH_UINT16 << 4) + (BH_INT64 << 8)
        case 1416: return true; // IIl: BH_UINT32 + (BH_UINT32 << 4) + (BH_INT64 << 8)
        case 1433: return true; // LLl: BH_UINT64 + (BH_UINT64 << 4) + (BH_INT64 << 8)
        case 1382: return true; // BBl: BH_UINT8 + (BH_UINT8 << 4) + (BH_INT64 << 8)
        case 17: return true; // zz: BH_BOOL + (BH_BOOL << 4)
        case 170: return true; // ff: BH_FLOAT32 + (BH_FLOAT32 << 4)
        case 187: return true; // dd: BH_FLOAT64 + (BH_FLOAT64 << 4)
        case 51: return true; // ss: BH_INT16 + (BH_INT16 << 4)
        case 68: return true; // ii: BH_INT32 + (BH_INT32 << 4)
        case 85: return true; // ll: BH_INT64 + (BH_INT64 << 4)
        case 34: return true; // bb: BH_INT8 + (BH_INT8 << 4)
        case 119: return true; // SS: BH_UINT16 + (BH_UINT16 << 4)
        case 136: return true; // II: BH_UINT32 + (BH_UINT32 << 4)
        case 153: return true; // LL: BH_UINT64 + (BH_UINT64 << 4)
        case 102: return true; // BB: BH_UINT8 + (BH_UINT8 << 4)
        case 221: return true; // CC: BH_COMPLEX128 + (BH_COMPLEX128 << 4)
        case 204: return true; // cc: BH_COMPLEX64 + (BH_COMPLEX64 << 4)
        case 161: return true; // zf: BH_BOOL + (BH_FLOAT32 << 4)
        case 177: return true; // zd: BH_BOOL + (BH_FLOAT64 << 4)
        case 49: return true; // zs: BH_BOOL + (BH_INT16 << 4)
        case 65: return true; // zi: BH_BOOL + (BH_INT32 << 4)
        case 81: return true; // zl: BH_BOOL + (BH_INT64 << 4)
        case 33: return true; // zb: BH_BOOL + (BH_INT8 << 4)
        case 113: return true; // zS: BH_BOOL + (BH_UINT16 << 4)
        case 129: return true; // zI: BH_BOOL + (BH_UINT32 << 4)
        case 145: return true; // zL: BH_BOOL + (BH_UINT64 << 4)
        case 97: return true; // zB: BH_BOOL + (BH_UINT8 << 4)
        case 29: return true; // Cz: BH_COMPLEX128 + (BH_BOOL << 4)
        case 205: return true; // Cc: BH_COMPLEX128 + (BH_COMPLEX64 << 4)
        case 173: return true; // Cf: BH_COMPLEX128 + (BH_FLOAT32 << 4)
        case 189: return true; // Cd: BH_COMPLEX128 + (BH_FLOAT64 << 4)
        case 61: return true; // Cs: BH_COMPLEX128 + (BH_INT16 << 4)
        case 77: return true; // Ci: BH_COMPLEX128 + (BH_INT32 << 4)
        case 93: return true; // Cl: BH_COMPLEX128 + (BH_INT64 << 4)
        case 45: return true; // Cb: BH_COMPLEX128 + (BH_INT8 << 4)
        case 125: return true; // CS: BH_COMPLEX128 + (BH_UINT16 << 4)
        case 141: return true; // CI: BH_COMPLEX128 + (BH_UINT32 << 4)
        case 157: return true; // CL: BH_COMPLEX128 + (BH_UINT64 << 4)
        case 109: return true; // CB: BH_COMPLEX128 + (BH_UINT8 << 4)
        case 28: return true; // cz: BH_COMPLEX64 + (BH_BOOL << 4)
        case 220: return true; // cC: BH_COMPLEX64 + (BH_COMPLEX128 << 4)
        case 172: return true; // cf: BH_COMPLEX64 + (BH_FLOAT32 << 4)
        case 188: return true; // cd: BH_COMPLEX64 + (BH_FLOAT64 << 4)
        case 60: return true; // cs: BH_COMPLEX64 + (BH_INT16 << 4)
        case 76: return true; // ci: BH_COMPLEX64 + (BH_INT32 << 4)
        case 92: return true; // cl: BH_COMPLEX64 + (BH_INT64 << 4)
        case 44: return true; // cb: BH_COMPLEX64 + (BH_INT8 << 4)
        case 124: return true; // cS: BH_COMPLEX64 + (BH_UINT16 << 4)
        case 140: return true; // cI: BH_COMPLEX64 + (BH_UINT32 << 4)
        case 156: return true; // cL: BH_COMPLEX64 + (BH_UINT64 << 4)
        case 108: return true; // cB: BH_COMPLEX64 + (BH_UINT8 << 4)
        case 26: return true; // fz: BH_FLOAT32 + (BH_BOOL << 4)
        case 186: return true; // fd: BH_FLOAT32 + (BH_FLOAT64 << 4)
        case 58: return true; // fs: BH_FLOAT32 + (BH_INT16 << 4)
        case 74: return true; // fi: BH_FLOAT32 + (BH_INT32 << 4)
        case 90: return true; // fl: BH_FLOAT32 + (BH_INT64 << 4)
        case 42: return true; // fb: BH_FLOAT32 + (BH_INT8 << 4)
        case 122: return true; // fS: BH_FLOAT32 + (BH_UINT16 << 4)
        case 138: return true; // fI: BH_FLOAT32 + (BH_UINT32 << 4)
        case 154: return true; // fL: BH_FLOAT32 + (BH_UINT64 << 4)
        case 106: return true; // fB: BH_FLOAT32 + (BH_UINT8 << 4)
        case 27: return true; // dz: BH_FLOAT64 + (BH_BOOL << 4)
        case 171: return true; // df: BH_FLOAT64 + (BH_FLOAT32 << 4)
        case 59: return true; // ds: BH_FLOAT64 + (BH_INT16 << 4)
        case 75: return true; // di: BH_FLOAT64 + (BH_INT32 << 4)
        case 91: return true; // dl: BH_FLOAT64 + (BH_INT64 << 4)
        case 43: return true; // db: BH_FLOAT64 + (BH_INT8 << 4)
        case 123: return true; // dS: BH_FLOAT64 + (BH_UINT16 << 4)
        case 139: return true; // dI: BH_FLOAT64 + (BH_UINT32 << 4)
        case 155: return true; // dL: BH_FLOAT64 + (BH_UINT64 << 4)
        case 107: return true; // dB: BH_FLOAT64 + (BH_UINT8 << 4)
        case 19: return true; // sz: BH_INT16 + (BH_BOOL << 4)
        case 163: return true; // sf: BH_INT16 + (BH_FLOAT32 << 4)
        case 179: return true; // sd: BH_INT16 + (BH_FLOAT64 << 4)
        case 67: return true; // si: BH_INT16 + (BH_INT32 << 4)
        case 83: return true; // sl: BH_INT16 + (BH_INT64 << 4)
        case 35: return true; // sb: BH_INT16 + (BH_INT8 << 4)
        case 115: return true; // sS: BH_INT16 + (BH_UINT16 << 4)
        case 131: return true; // sI: BH_INT16 + (BH_UINT32 << 4)
        case 147: return true; // sL: BH_INT16 + (BH_UINT64 << 4)
        case 99: return true; // sB: BH_INT16 + (BH_UINT8 << 4)
        case 20: return true; // iz: BH_INT32 + (BH_BOOL << 4)
        case 164: return true; // if: BH_INT32 + (BH_FLOAT32 << 4)
        case 180: return true; // id: BH_INT32 + (BH_FLOAT64 << 4)
        case 52: return true; // is: BH_INT32 + (BH_INT16 << 4)
        case 84: return true; // il: BH_INT32 + (BH_INT64 << 4)
        case 36: return true; // ib: BH_INT32 + (BH_INT8 << 4)
        case 116: return true; // iS: BH_INT32 + (BH_UINT16 << 4)
        case 132: return true; // iI: BH_INT32 + (BH_UINT32 << 4)
        case 148: return true; // iL: BH_INT32 + (BH_UINT64 << 4)
        case 100: return true; // iB: BH_INT32 + (BH_UINT8 << 4)
        case 21: return true; // lz: BH_INT64 + (BH_BOOL << 4)
        case 165: return true; // lf: BH_INT64 + (BH_FLOAT32 << 4)
        case 181: return true; // ld: BH_INT64 + (BH_FLOAT64 << 4)
        case 53: return true; // ls: BH_INT64 + (BH_INT16 << 4)
        case 69: return true; // li: BH_INT64 + (BH_INT32 << 4)
        case 37: return true; // lb: BH_INT64 + (BH_INT8 << 4)
        case 117: return true; // lS: BH_INT64 + (BH_UINT16 << 4)
        case 133: return true; // lI: BH_INT64 + (BH_UINT32 << 4)
        case 149: return true; // lL: BH_INT64 + (BH_UINT64 << 4)
        case 101: return true; // lB: BH_INT64 + (BH_UINT8 << 4)
        case 18: return true; // bz: BH_INT8 + (BH_BOOL << 4)
        case 162: return true; // bf: BH_INT8 + (BH_FLOAT32 << 4)
        case 178: return true; // bd: BH_INT8 + (BH_FLOAT64 << 4)
        case 50: return true; // bs: BH_INT8 + (BH_INT16 << 4)
        case 66: return true; // bi: BH_INT8 + (BH_INT32 << 4)
        case 82: return true; // bl: BH_INT8 + (BH_INT64 << 4)
        case 114: return true; // bS: BH_INT8 + (BH_UINT16 << 4)
        case 130: return true; // bI: BH_INT8 + (BH_UINT32 << 4)
        case 146: return true; // bL: BH_INT8 + (BH_UINT64 << 4)
        case 98: return true; // bB: BH_INT8 + (BH_UINT8 << 4)
        case 23: return true; // Sz: BH_UINT16 + (BH_BOOL << 4)
        case 167: return true; // Sf: BH_UINT16 + (BH_FLOAT32 << 4)
        case 183: return true; // Sd: BH_UINT16 + (BH_FLOAT64 << 4)
        case 55: return true; // Ss: BH_UINT16 + (BH_INT16 << 4)
        case 71: return true; // Si: BH_UINT16 + (BH_INT32 << 4)
        case 87: return true; // Sl: BH_UINT16 + (BH_INT64 << 4)
        case 39: return true; // Sb: BH_UINT16 + (BH_INT8 << 4)
        case 135: return true; // SI: BH_UINT16 + (BH_UINT32 << 4)
        case 151: return true; // SL: BH_UINT16 + (BH_UINT64 << 4)
        case 103: return true; // SB: BH_UINT16 + (BH_UINT8 << 4)
        case 24: return true; // Iz: BH_UINT32 + (BH_BOOL << 4)
        case 168: return true; // If: BH_UINT32 + (BH_FLOAT32 << 4)
        case 184: return true; // Id: BH_UINT32 + (BH_FLOAT64 << 4)
        case 56: return true; // Is: BH_UINT32 + (BH_INT16 << 4)
        case 72: return true; // Ii: BH_UINT32 + (BH_INT32 << 4)
        case 88: return true; // Il: BH_UINT32 + (BH_INT64 << 4)
        case 40: return true; // Ib: BH_UINT32 + (BH_INT8 << 4)
        case 120: return true; // IS: BH_UINT32 + (BH_UINT16 << 4)
        case 152: return true; // IL: BH_UINT32 + (BH_UINT64 << 4)
        case 104: return true; // IB: BH_UINT32 + (BH_UINT8 << 4)
        case 25: return true; // Lz: BH_UINT64 + (BH_BOOL << 4)
        case 169: return true; // Lf: BH_UINT64 + (BH_FLOAT32 << 4)
        case 185: return true; // Ld: BH_UINT64 + (BH_FLOAT64 << 4)
        case 57: return true; // Ls: BH_UINT64 + (BH_INT16 << 4)
        case 73: return true; // Li: BH_UINT64 + (BH_INT32 << 4)
        case 89: return true; // Ll: BH_UINT64 + (BH_INT64 << 4)
        case 41: return true; // Lb: BH_UINT64 + (BH_INT8 << 4)
        case 121: return true; // LS: BH_UINT64 + (BH_UINT16 << 4)
        case 137: return true; // LI: BH_UINT64 + (BH_UINT32 << 4)
        case 105: return true; // LB: BH_UINT64 + (BH_UINT8 << 4)
        case 22: return true; // Bz: BH_UINT8 + (BH_BOOL << 4)
        case 166: return true; // Bf: BH_UINT8 + (BH_FLOAT32 << 4)
        case 182: return true; // Bd: BH_UINT8 + (BH_FLOAT64 << 4)
        case 54: return true; // Bs: BH_UINT8 + (BH_INT16 << 4)
        case 70: return true; // Bi: BH_UINT8 + (BH_INT32 << 4)
        case 86: return true; // Bl: BH_UINT8 + (BH_INT64 << 4)
        case 38: return true; // Bb: BH_UINT8 + (BH_INT8 << 4)
        case 118: return true; // BS: BH_UINT8 + (BH_UINT16 << 4)
        case 134: return true; // BI: BH_UINT8 + (BH_UINT32 << 4)
        case 150: return true; // BL: BH_UINT8 + (BH_UINT64 << 4)
        case 232: return true; // IR: BH_UINT32 + (BH_R123 << 4)
        case 233: return true; // LR: BH_UINT64 + (BH_R123 << 4)
        case 219: return true; // dC: BH_FLOAT64 + (BH_COMPLEX128 << 4)
        case 202: return true; // fc: BH_FLOAT32 + (BH_COMPLEX64 << 4)
        case 8: return true; // I: BH_UINT32
        case 9: return true; // L: BH_UINT64

        default:
            return false;
    }
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
        case BH_FLOAT32: return "f";
        case BH_FLOAT64: return "d";
        case BH_COMPLEX64: return "c";
        case BH_COMPLEX128: return "C";
        case BH_R123: return "R";
        case BH_UNKNOWN: return "U";

        default:
            return "{{UNKNOWN}}";
    }
}


const char* bh_typesig_to_shorthand(int typesig)
{
    switch(typesig) {
        case 273: return "zzz"; // BH_BOOL + (BH_BOOL << 4) + (BH_BOOL << 8)
        case 3549: return "CCC"; // BH_COMPLEX128 + (BH_COMPLEX128 << 4) + (BH_COMPLEX128 << 8)
        case 3276: return "ccc"; // BH_COMPLEX64 + (BH_COMPLEX64 << 4) + (BH_COMPLEX64 << 8)
        case 2730: return "fff"; // BH_FLOAT32 + (BH_FLOAT32 << 4) + (BH_FLOAT32 << 8)
        case 3003: return "ddd"; // BH_FLOAT64 + (BH_FLOAT64 << 4) + (BH_FLOAT64 << 8)
        case 819: return "sss"; // BH_INT16 + (BH_INT16 << 4) + (BH_INT16 << 8)
        case 1092: return "iii"; // BH_INT32 + (BH_INT32 << 4) + (BH_INT32 << 8)
        case 1365: return "lll"; // BH_INT64 + (BH_INT64 << 4) + (BH_INT64 << 8)
        case 546: return "bbb"; // BH_INT8 + (BH_INT8 << 4) + (BH_INT8 << 8)
        case 1911: return "SSS"; // BH_UINT16 + (BH_UINT16 << 4) + (BH_UINT16 << 8)
        case 2184: return "III"; // BH_UINT32 + (BH_UINT32 << 4) + (BH_UINT32 << 8)
        case 2457: return "LLL"; // BH_UINT64 + (BH_UINT64 << 4) + (BH_UINT64 << 8)
        case 1638: return "BBB"; // BH_UINT8 + (BH_UINT8 << 4) + (BH_UINT8 << 8)
        case 2721: return "zff"; // BH_BOOL + (BH_FLOAT32 << 4) + (BH_FLOAT32 << 8)
        case 2993: return "zdd"; // BH_BOOL + (BH_FLOAT64 << 4) + (BH_FLOAT64 << 8)
        case 817: return "zss"; // BH_BOOL + (BH_INT16 << 4) + (BH_INT16 << 8)
        case 1089: return "zii"; // BH_BOOL + (BH_INT32 << 4) + (BH_INT32 << 8)
        case 1361: return "zll"; // BH_BOOL + (BH_INT64 << 4) + (BH_INT64 << 8)
        case 545: return "zbb"; // BH_BOOL + (BH_INT8 << 4) + (BH_INT8 << 8)
        case 1905: return "zSS"; // BH_BOOL + (BH_UINT16 << 4) + (BH_UINT16 << 8)
        case 2177: return "zII"; // BH_BOOL + (BH_UINT32 << 4) + (BH_UINT32 << 8)
        case 2449: return "zLL"; // BH_BOOL + (BH_UINT64 << 4) + (BH_UINT64 << 8)
        case 1633: return "zBB"; // BH_BOOL + (BH_UINT8 << 4) + (BH_UINT8 << 8)
        case 3537: return "zCC"; // BH_BOOL + (BH_COMPLEX128 << 4) + (BH_COMPLEX128 << 8)
        case 3265: return "zcc"; // BH_BOOL + (BH_COMPLEX64 << 4) + (BH_COMPLEX64 << 8)
        case 1297: return "zzl"; // BH_BOOL + (BH_BOOL << 4) + (BH_INT64 << 8)
        case 1501: return "CCl"; // BH_COMPLEX128 + (BH_COMPLEX128 << 4) + (BH_INT64 << 8)
        case 1484: return "ccl"; // BH_COMPLEX64 + (BH_COMPLEX64 << 4) + (BH_INT64 << 8)
        case 1450: return "ffl"; // BH_FLOAT32 + (BH_FLOAT32 << 4) + (BH_INT64 << 8)
        case 1467: return "ddl"; // BH_FLOAT64 + (BH_FLOAT64 << 4) + (BH_INT64 << 8)
        case 1331: return "ssl"; // BH_INT16 + (BH_INT16 << 4) + (BH_INT64 << 8)
        case 1348: return "iil"; // BH_INT32 + (BH_INT32 << 4) + (BH_INT64 << 8)
        case 1314: return "bbl"; // BH_INT8 + (BH_INT8 << 4) + (BH_INT64 << 8)
        case 1399: return "SSl"; // BH_UINT16 + (BH_UINT16 << 4) + (BH_INT64 << 8)
        case 1416: return "IIl"; // BH_UINT32 + (BH_UINT32 << 4) + (BH_INT64 << 8)
        case 1433: return "LLl"; // BH_UINT64 + (BH_UINT64 << 4) + (BH_INT64 << 8)
        case 1382: return "BBl"; // BH_UINT8 + (BH_UINT8 << 4) + (BH_INT64 << 8)
        case 17: return "zz"; // BH_BOOL + (BH_BOOL << 4)
        case 170: return "ff"; // BH_FLOAT32 + (BH_FLOAT32 << 4)
        case 187: return "dd"; // BH_FLOAT64 + (BH_FLOAT64 << 4)
        case 51: return "ss"; // BH_INT16 + (BH_INT16 << 4)
        case 68: return "ii"; // BH_INT32 + (BH_INT32 << 4)
        case 85: return "ll"; // BH_INT64 + (BH_INT64 << 4)
        case 34: return "bb"; // BH_INT8 + (BH_INT8 << 4)
        case 119: return "SS"; // BH_UINT16 + (BH_UINT16 << 4)
        case 136: return "II"; // BH_UINT32 + (BH_UINT32 << 4)
        case 153: return "LL"; // BH_UINT64 + (BH_UINT64 << 4)
        case 102: return "BB"; // BH_UINT8 + (BH_UINT8 << 4)
        case 221: return "CC"; // BH_COMPLEX128 + (BH_COMPLEX128 << 4)
        case 204: return "cc"; // BH_COMPLEX64 + (BH_COMPLEX64 << 4)
        case 161: return "zf"; // BH_BOOL + (BH_FLOAT32 << 4)
        case 177: return "zd"; // BH_BOOL + (BH_FLOAT64 << 4)
        case 49: return "zs"; // BH_BOOL + (BH_INT16 << 4)
        case 65: return "zi"; // BH_BOOL + (BH_INT32 << 4)
        case 81: return "zl"; // BH_BOOL + (BH_INT64 << 4)
        case 33: return "zb"; // BH_BOOL + (BH_INT8 << 4)
        case 113: return "zS"; // BH_BOOL + (BH_UINT16 << 4)
        case 129: return "zI"; // BH_BOOL + (BH_UINT32 << 4)
        case 145: return "zL"; // BH_BOOL + (BH_UINT64 << 4)
        case 97: return "zB"; // BH_BOOL + (BH_UINT8 << 4)
        case 29: return "Cz"; // BH_COMPLEX128 + (BH_BOOL << 4)
        case 205: return "Cc"; // BH_COMPLEX128 + (BH_COMPLEX64 << 4)
        case 173: return "Cf"; // BH_COMPLEX128 + (BH_FLOAT32 << 4)
        case 189: return "Cd"; // BH_COMPLEX128 + (BH_FLOAT64 << 4)
        case 61: return "Cs"; // BH_COMPLEX128 + (BH_INT16 << 4)
        case 77: return "Ci"; // BH_COMPLEX128 + (BH_INT32 << 4)
        case 93: return "Cl"; // BH_COMPLEX128 + (BH_INT64 << 4)
        case 45: return "Cb"; // BH_COMPLEX128 + (BH_INT8 << 4)
        case 125: return "CS"; // BH_COMPLEX128 + (BH_UINT16 << 4)
        case 141: return "CI"; // BH_COMPLEX128 + (BH_UINT32 << 4)
        case 157: return "CL"; // BH_COMPLEX128 + (BH_UINT64 << 4)
        case 109: return "CB"; // BH_COMPLEX128 + (BH_UINT8 << 4)
        case 28: return "cz"; // BH_COMPLEX64 + (BH_BOOL << 4)
        case 220: return "cC"; // BH_COMPLEX64 + (BH_COMPLEX128 << 4)
        case 172: return "cf"; // BH_COMPLEX64 + (BH_FLOAT32 << 4)
        case 188: return "cd"; // BH_COMPLEX64 + (BH_FLOAT64 << 4)
        case 60: return "cs"; // BH_COMPLEX64 + (BH_INT16 << 4)
        case 76: return "ci"; // BH_COMPLEX64 + (BH_INT32 << 4)
        case 92: return "cl"; // BH_COMPLEX64 + (BH_INT64 << 4)
        case 44: return "cb"; // BH_COMPLEX64 + (BH_INT8 << 4)
        case 124: return "cS"; // BH_COMPLEX64 + (BH_UINT16 << 4)
        case 140: return "cI"; // BH_COMPLEX64 + (BH_UINT32 << 4)
        case 156: return "cL"; // BH_COMPLEX64 + (BH_UINT64 << 4)
        case 108: return "cB"; // BH_COMPLEX64 + (BH_UINT8 << 4)
        case 26: return "fz"; // BH_FLOAT32 + (BH_BOOL << 4)
        case 186: return "fd"; // BH_FLOAT32 + (BH_FLOAT64 << 4)
        case 58: return "fs"; // BH_FLOAT32 + (BH_INT16 << 4)
        case 74: return "fi"; // BH_FLOAT32 + (BH_INT32 << 4)
        case 90: return "fl"; // BH_FLOAT32 + (BH_INT64 << 4)
        case 42: return "fb"; // BH_FLOAT32 + (BH_INT8 << 4)
        case 122: return "fS"; // BH_FLOAT32 + (BH_UINT16 << 4)
        case 138: return "fI"; // BH_FLOAT32 + (BH_UINT32 << 4)
        case 154: return "fL"; // BH_FLOAT32 + (BH_UINT64 << 4)
        case 106: return "fB"; // BH_FLOAT32 + (BH_UINT8 << 4)
        case 27: return "dz"; // BH_FLOAT64 + (BH_BOOL << 4)
        case 171: return "df"; // BH_FLOAT64 + (BH_FLOAT32 << 4)
        case 59: return "ds"; // BH_FLOAT64 + (BH_INT16 << 4)
        case 75: return "di"; // BH_FLOAT64 + (BH_INT32 << 4)
        case 91: return "dl"; // BH_FLOAT64 + (BH_INT64 << 4)
        case 43: return "db"; // BH_FLOAT64 + (BH_INT8 << 4)
        case 123: return "dS"; // BH_FLOAT64 + (BH_UINT16 << 4)
        case 139: return "dI"; // BH_FLOAT64 + (BH_UINT32 << 4)
        case 155: return "dL"; // BH_FLOAT64 + (BH_UINT64 << 4)
        case 107: return "dB"; // BH_FLOAT64 + (BH_UINT8 << 4)
        case 19: return "sz"; // BH_INT16 + (BH_BOOL << 4)
        case 163: return "sf"; // BH_INT16 + (BH_FLOAT32 << 4)
        case 179: return "sd"; // BH_INT16 + (BH_FLOAT64 << 4)
        case 67: return "si"; // BH_INT16 + (BH_INT32 << 4)
        case 83: return "sl"; // BH_INT16 + (BH_INT64 << 4)
        case 35: return "sb"; // BH_INT16 + (BH_INT8 << 4)
        case 115: return "sS"; // BH_INT16 + (BH_UINT16 << 4)
        case 131: return "sI"; // BH_INT16 + (BH_UINT32 << 4)
        case 147: return "sL"; // BH_INT16 + (BH_UINT64 << 4)
        case 99: return "sB"; // BH_INT16 + (BH_UINT8 << 4)
        case 20: return "iz"; // BH_INT32 + (BH_BOOL << 4)
        case 164: return "if"; // BH_INT32 + (BH_FLOAT32 << 4)
        case 180: return "id"; // BH_INT32 + (BH_FLOAT64 << 4)
        case 52: return "is"; // BH_INT32 + (BH_INT16 << 4)
        case 84: return "il"; // BH_INT32 + (BH_INT64 << 4)
        case 36: return "ib"; // BH_INT32 + (BH_INT8 << 4)
        case 116: return "iS"; // BH_INT32 + (BH_UINT16 << 4)
        case 132: return "iI"; // BH_INT32 + (BH_UINT32 << 4)
        case 148: return "iL"; // BH_INT32 + (BH_UINT64 << 4)
        case 100: return "iB"; // BH_INT32 + (BH_UINT8 << 4)
        case 21: return "lz"; // BH_INT64 + (BH_BOOL << 4)
        case 165: return "lf"; // BH_INT64 + (BH_FLOAT32 << 4)
        case 181: return "ld"; // BH_INT64 + (BH_FLOAT64 << 4)
        case 53: return "ls"; // BH_INT64 + (BH_INT16 << 4)
        case 69: return "li"; // BH_INT64 + (BH_INT32 << 4)
        case 37: return "lb"; // BH_INT64 + (BH_INT8 << 4)
        case 117: return "lS"; // BH_INT64 + (BH_UINT16 << 4)
        case 133: return "lI"; // BH_INT64 + (BH_UINT32 << 4)
        case 149: return "lL"; // BH_INT64 + (BH_UINT64 << 4)
        case 101: return "lB"; // BH_INT64 + (BH_UINT8 << 4)
        case 18: return "bz"; // BH_INT8 + (BH_BOOL << 4)
        case 162: return "bf"; // BH_INT8 + (BH_FLOAT32 << 4)
        case 178: return "bd"; // BH_INT8 + (BH_FLOAT64 << 4)
        case 50: return "bs"; // BH_INT8 + (BH_INT16 << 4)
        case 66: return "bi"; // BH_INT8 + (BH_INT32 << 4)
        case 82: return "bl"; // BH_INT8 + (BH_INT64 << 4)
        case 114: return "bS"; // BH_INT8 + (BH_UINT16 << 4)
        case 130: return "bI"; // BH_INT8 + (BH_UINT32 << 4)
        case 146: return "bL"; // BH_INT8 + (BH_UINT64 << 4)
        case 98: return "bB"; // BH_INT8 + (BH_UINT8 << 4)
        case 23: return "Sz"; // BH_UINT16 + (BH_BOOL << 4)
        case 167: return "Sf"; // BH_UINT16 + (BH_FLOAT32 << 4)
        case 183: return "Sd"; // BH_UINT16 + (BH_FLOAT64 << 4)
        case 55: return "Ss"; // BH_UINT16 + (BH_INT16 << 4)
        case 71: return "Si"; // BH_UINT16 + (BH_INT32 << 4)
        case 87: return "Sl"; // BH_UINT16 + (BH_INT64 << 4)
        case 39: return "Sb"; // BH_UINT16 + (BH_INT8 << 4)
        case 135: return "SI"; // BH_UINT16 + (BH_UINT32 << 4)
        case 151: return "SL"; // BH_UINT16 + (BH_UINT64 << 4)
        case 103: return "SB"; // BH_UINT16 + (BH_UINT8 << 4)
        case 24: return "Iz"; // BH_UINT32 + (BH_BOOL << 4)
        case 168: return "If"; // BH_UINT32 + (BH_FLOAT32 << 4)
        case 184: return "Id"; // BH_UINT32 + (BH_FLOAT64 << 4)
        case 56: return "Is"; // BH_UINT32 + (BH_INT16 << 4)
        case 72: return "Ii"; // BH_UINT32 + (BH_INT32 << 4)
        case 88: return "Il"; // BH_UINT32 + (BH_INT64 << 4)
        case 40: return "Ib"; // BH_UINT32 + (BH_INT8 << 4)
        case 120: return "IS"; // BH_UINT32 + (BH_UINT16 << 4)
        case 152: return "IL"; // BH_UINT32 + (BH_UINT64 << 4)
        case 104: return "IB"; // BH_UINT32 + (BH_UINT8 << 4)
        case 25: return "Lz"; // BH_UINT64 + (BH_BOOL << 4)
        case 169: return "Lf"; // BH_UINT64 + (BH_FLOAT32 << 4)
        case 185: return "Ld"; // BH_UINT64 + (BH_FLOAT64 << 4)
        case 57: return "Ls"; // BH_UINT64 + (BH_INT16 << 4)
        case 73: return "Li"; // BH_UINT64 + (BH_INT32 << 4)
        case 89: return "Ll"; // BH_UINT64 + (BH_INT64 << 4)
        case 41: return "Lb"; // BH_UINT64 + (BH_INT8 << 4)
        case 121: return "LS"; // BH_UINT64 + (BH_UINT16 << 4)
        case 137: return "LI"; // BH_UINT64 + (BH_UINT32 << 4)
        case 105: return "LB"; // BH_UINT64 + (BH_UINT8 << 4)
        case 22: return "Bz"; // BH_UINT8 + (BH_BOOL << 4)
        case 166: return "Bf"; // BH_UINT8 + (BH_FLOAT32 << 4)
        case 182: return "Bd"; // BH_UINT8 + (BH_FLOAT64 << 4)
        case 54: return "Bs"; // BH_UINT8 + (BH_INT16 << 4)
        case 70: return "Bi"; // BH_UINT8 + (BH_INT32 << 4)
        case 86: return "Bl"; // BH_UINT8 + (BH_INT64 << 4)
        case 38: return "Bb"; // BH_UINT8 + (BH_INT8 << 4)
        case 118: return "BS"; // BH_UINT8 + (BH_UINT16 << 4)
        case 134: return "BI"; // BH_UINT8 + (BH_UINT32 << 4)
        case 150: return "BL"; // BH_UINT8 + (BH_UINT64 << 4)
        case 232: return "IR"; // BH_UINT32 + (BH_R123 << 4)
        case 233: return "LR"; // BH_UINT64 + (BH_R123 << 4)
        case 219: return "dC"; // BH_FLOAT64 + (BH_COMPLEX128 << 4)
        case 202: return "fc"; // BH_FLOAT32 + (BH_COMPLEX64 << 4)
        case 8: return "I"; // BH_UINT32
        case 9: return "L"; // BH_UINT64

        default:
            //printf( "cpu(bh_typesig_to_shorthand): "
            //        "Unsupported type signature %d.\n", typesig);
            return "{{UNSUPPORTED}}";
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
        case BH_FLOAT32: return "float";
        case BH_FLOAT64: return "double";
        case BH_COMPLEX64: return "float complex";
        case BH_COMPLEX128: return "double complex";
        case BH_R123: return "bh_r123";
        case BH_UNKNOWN: return "<UNKNOWN>";

        default:
            return "{{UNKNOWN}}";
    }
}

const char* bh_opcode_to_cstr_short(bh_opcode const opcode)
{
    switch(opcode) {
        case BH_ADD: return "ADD";
        case BH_SUBTRACT: return "SUBTRACT";
        case BH_MULTIPLY: return "MULTIPLY";
        case BH_DIVIDE: return "DIVIDE";
        case BH_POWER: return "POWER";
        case BH_ABSOLUTE: return "ABSOLUTE";
        case BH_GREATER: return "GREATER";
        case BH_GREATER_EQUAL: return "GREATER_EQUAL";
        case BH_LESS: return "LESS";
        case BH_LESS_EQUAL: return "LESS_EQUAL";
        case BH_EQUAL: return "EQUAL";
        case BH_NOT_EQUAL: return "NOT_EQUAL";
        case BH_LOGICAL_AND: return "LOGICAL_AND";
        case BH_LOGICAL_OR: return "LOGICAL_OR";
        case BH_LOGICAL_XOR: return "LOGICAL_XOR";
        case BH_LOGICAL_NOT: return "LOGICAL_NOT";
        case BH_MAXIMUM: return "MAXIMUM";
        case BH_MINIMUM: return "MINIMUM";
        case BH_BITWISE_AND: return "BITWISE_AND";
        case BH_BITWISE_OR: return "BITWISE_OR";
        case BH_BITWISE_XOR: return "BITWISE_XOR";
        case BH_INVERT: return "INVERT";
        case BH_LEFT_SHIFT: return "LEFT_SHIFT";
        case BH_RIGHT_SHIFT: return "RIGHT_SHIFT";
        case BH_COS: return "COS";
        case BH_SIN: return "SIN";
        case BH_TAN: return "TAN";
        case BH_COSH: return "COSH";
        case BH_SINH: return "SINH";
        case BH_TANH: return "TANH";
        case BH_ARCSIN: return "ARCSIN";
        case BH_ARCCOS: return "ARCCOS";
        case BH_ARCTAN: return "ARCTAN";
        case BH_ARCSINH: return "ARCSINH";
        case BH_ARCCOSH: return "ARCCOSH";
        case BH_ARCTANH: return "ARCTANH";
        case BH_ARCTAN2: return "ARCTAN2";
        case BH_EXP: return "EXP";
        case BH_EXP2: return "EXP2";
        case BH_EXPM1: return "EXPM1";
        case BH_LOG: return "LOG";
        case BH_LOG2: return "LOG2";
        case BH_LOG10: return "LOG10";
        case BH_LOG1P: return "LOG1P";
        case BH_SQRT: return "SQRT";
        case BH_CEIL: return "CEIL";
        case BH_TRUNC: return "TRUNC";
        case BH_FLOOR: return "FLOOR";
        case BH_RINT: return "RINT";
        case BH_MOD: return "MOD";
        case BH_ISNAN: return "ISNAN";
        case BH_ISINF: return "ISINF";
        case BH_IDENTITY: return "IDENTITY";
        case BH_DISCARD: return "DISCARD";
        case BH_FREE: return "FREE";
        case BH_SYNC: return "SYNC";
        case BH_NONE: return "NONE";
        case BH_ADD_REDUCE: return "ADD_REDUCE";
        case BH_MULTIPLY_REDUCE: return "MULTIPLY_REDUCE";
        case BH_MINIMUM_REDUCE: return "MINIMUM_REDUCE";
        case BH_MAXIMUM_REDUCE: return "MAXIMUM_REDUCE";
        case BH_LOGICAL_AND_REDUCE: return "LOGICAL_AND_REDUCE";
        case BH_BITWISE_AND_REDUCE: return "BITWISE_AND_REDUCE";
        case BH_LOGICAL_OR_REDUCE: return "LOGICAL_OR_REDUCE";
        case BH_BITWISE_OR_REDUCE: return "BITWISE_OR_REDUCE";
        case BH_LOGICAL_XOR_REDUCE: return "LOGICAL_XOR_REDUCE";
        case BH_BITWISE_XOR_REDUCE: return "BITWISE_XOR_REDUCE";
        case BH_RANDOM: return "RANDOM";
        case BH_RANGE: return "RANGE";
        case BH_REAL: return "REAL";
        case BH_IMAG: return "IMAG";
        case BH_ADD_ACCUMULATE: return "ADD_ACCUMULATE";
        case BH_MULTIPLY_ACCUMULATE: return "MULTIPLY_ACCUMULATE";

        default:
            return "{{UNKNOWN}}";
    }
}

