/*
std::string const_as_string(bh_constant constant)
{
    std::ostringstream buff;
    switch(constant.type) {
        case BH_BOOL:
            buff << constant.value.bool8;
            break;
        case BH_INT8:
            buff << constant.value.int8;
            break;
        case BH_INT16:
            buff << constant.value.int16;
            break;
        case BH_INT32:
            buff << constant.value.int32;
            break;
        case BH_INT64:
            buff << constant.value.int64;
            break;
        case BH_UINT8:
            buff << constant.value.uint8;
            break;
        case BH_UINT16:
            buff << constant.value.uint16;
            break;
        case BH_UINT32:
            buff << constant.value.uint32;
            break;
        case BH_UINT64:
            buff << constant.value.uint64;
            break;
        case BH_FLOAT16:
            buff << constant.value.float16;
            break;
        case BH_FLOAT32:
            buff << constant.value.float32;
            break;
        case BH_FLOAT64:
            buff << constant.value.float64;
            break;
        case BH_COMPLEX64:
            buff << constant.value.complex64.real << constant.value.complex64.imag;
            break;
        case BH_COMPLEX128:
            buff << constant.value.complex128.real << constant.value.complex128.imag;
            break;

        case BH_UNKNOWN:
        default:
            buff << "__ERROR__";
    }

    return buff.str();
}
*/

