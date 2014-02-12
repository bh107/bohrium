const char* operator_cexpr(OPERATION const op, OPERATOR const oper, const bh_type type)
{
    switch(oper) {
        case BH_MAXIMUM_REDUCE:
            return "rvar = rvar < *tmp_current ? *tmp_current : rvar";
        case BH_LOGICAL_AND_REDUCE:
            return "rvar = rvar && *tmp_current";
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
            return "rvar = rvar || *tmp_current";
        case BH_BITWISE_OR_REDUCE:
            return "rvar |= *tmp_current";

        case BH_LOGICAL_XOR_REDUCE:
            return "rvar = !rvar != !*tmp_current";
        case BH_BITWISE_XOR_REDUCE:
            return "rvar = rvar ^ *tmp_current";

        case ADD:
            switch(op) {
                case EWISE_B:
                    return "*a0_current = *a1_current + *a2_current";
                case SCAN:
                    return "cvar += *a1_current; *a0_current = cvar;";
                case REDUCE:
                    return "rvar += *tmp_current";
                default:
                    return "__UNS_OP_FOR_OPER__";
            }
        case SUBTRACT:
            return "*a0_current = *a1_current - *a2_current";
        case MULTIPLY:
            switch(op) {
                case EWISE_B:
                    return "*a0_current = *a1_current * *a2_current";
                case SCAN:
                    return "cvar *= *a1_current; *a0_current = cvar;";
                case REDUCE:
                    return "rvar *= *tmp_current";
                default:
                    return "__UNS_OP_FOR_OPER__";
            }
        case DIVIDE:
            return "*a0_current = *a1_current / *a2_current";
        case POWER:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = cpowf( *a1_current, *a2_current )";
                case BH_COMPLEX128:
                    return "*a0_current = cpow( *a1_current, *a2_current )";
                default:
                    return "*a0_current = pow( *a1_current, *a2_current )";
            }
        case GREATER:
            return "*a0_current = *a1_current > *a2_current";
        case GREATER_EQUAL:
            return "*a0_current = *a1_current >= *a2_current";
        case LESS:
            return "*a0_current = *a1_current < *a2_current";
        case LESS_EQUAL:
            return "*a0_current = *a1_current <= *a2_current";
        case EQUAL:
            return "*a0_current = *a1_current == *a2_current";
        case NOT_EQUAL:
            return "*a0_current = *a1_current != *a2_current";
        case LOGICAL_AND:
            return "*a0_current = *a1_current && *a2_current";
        case LOGICAL_OR:
            return "*a0_current = *a1_current || *a2_current";
        case LOGICAL_XOR:
            return "*a0_current = (!*a1_current != !*a2_current)";
        case MAXIMUM:
            return "*a0_current = *a1_current < *a2_current ? *a2_current : *a1_current";
        case MINIMUM:
            switch(op) {
                case EWISE_B:
                    return "*a0_current = *a1_current < *a2_current ? *a1_current : *a2_current";
                case REDUCE:
                    return "rvar = rvar < *tmp_current ? rvar : *tmp_current";
                default:
                    return "__OPER_MIN_UNS__";
            }
        case BITWISE_AND:
            switch(op) {
                case EWISE_B:
                    return "*a0_current = *a1_current & *a2_current";
                case REDUCE:
                    return "rvar &= *tmp_current";
                default:
                    "__UNS__";
            }
        case BITWISE_OR:
            return "*a0_current = *a1_current | *a2_current";
        case BITWISE_XOR:
            return "*a0_current = *a1_current ^ *a2_current";
        case LEFT_SHIFT:
            return "*a0_current = (*a1_current) << (*a2_current)";
        case RIGHT_SHIFT:
            return "*a0_current = (*a1_current) >> (*a2_current)";
        case ARCTAN2:
            return "*a0_current = atan2( *a1_current, *a2_current )";
        case MOD:
            return "*a0_current = *a1_current - floor(*a1_current / *a2_current) * *a2_current";

        //
        // Unary Elementwise: SQRT, SIN...
        case ABSOLUTE:
            return "*a0_current = *a1_current < 0.0 ? -*a1_current: *a1_current";
        case LOGICAL_NOT:
            return "*a0_current = !*a1_current";
        case INVERT:
            return "*a0_current = ~*a1_current";
        case COS:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = ccosf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = ccos( *a1_current )";
                default:
                    return "*a0_current = cos( *a1_current )";
            }
        case SIN:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = csinf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = csin( *a1_current )";
                default:
                    return "*a0_current = sin( *a1_current )";
            }
        case TAN:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = ctanf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = ctan( *a1_current )";
                default:
                    return "*a0_current = tan( *a1_current )";
            }
        case COSH:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = ccoshf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = ccosh( *a1_current )";
                default:
                    return "*a0_current = cosh( *a1_current )";
            }
        case SINH:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = csinhf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = csinh( *a1_current )";
                default:
                    return "*a0_current = sinh( *a1_current )";
            }
        case TANH:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = ctanhf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = ctanh( *a1_current )";
                default:
                    return "*a0_current = tanh( *a1_current )";
            }
        case ARCSIN:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = casinf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = casin( *a1_current )";
                default:
                    return "*a0_current = asin( *a1_current )";
            }
        case ARCCOS:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = cacosf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = cacos( *a1_current )";
                default:
                    return "*a0_current = acos( *a1_current )";
            }
        case ARCTAN:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = catanf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = catan( *a1_current )";
                default:
                    return "*a0_current = atan( *a1_current )";
            }
        case ARCSINH:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = casinhf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = casinh( *a1_current )";
                default:
                    return "*a0_current = asinh( *a1_current )";
            }
        case ARCCOSH:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = cacoshf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = cacosh( *a1_current )";
                default:
                    return "*a0_current = acosh( *a1_current )";
            }
        case ARCTANH:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = catanhf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = catanh( *a1_current )";
                default:
                    return "*a0_current = atanh( *a1_current )";
            }
        case EXP:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = cexpf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = cexp( *a1_current )";
                default:
                    return "*a0_current = exp( *a1_current )";
            }
        case EXP2:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = cpowf( 2, *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = cpow( 2, *a1_current )";
                default:
                    return "*a0_current = pow( 2, *a1_current )";
            }
        case EXPM1:
            return "*a0_current = expm1( *a1_current )";
        case LOG:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = clogf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = clog( *a1_current )";
                default:
                    return "*a0_current = log( *a1_current )";
            }
        case LOG2:
            return "*a0_current = log2( *a1_current )";
        case LOG10:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = clogf( *a1_current )/log(10)";
                case BH_COMPLEX128:
                    return "*a0_current = clog( *a1_current )/log(10)";
                default:
                    return "*a0_current = log( *a1_current )/log(10)";
            }
        case LOG1P:
            return "*a0_current = log1p( *a1_current )";
        case SQRT:
            switch(type) {
                case BH_COMPLEX64:
                    return "*a0_current = csqrtf( *a1_current )";
                case BH_COMPLEX128:
                    return "*a0_current = csqrt( *a1_current )";
                default:
                    return "*a0_current = sqrt( *a1_current )";
            }
        case CEIL:
            return "*a0_current = ceil( *a1_current )";
        case TRUNC:
            return "*a0_current = trunc( *a1_current )";
        case FLOOR:
            return "*a0_current = floor( *a1_current )";
            
        case RINT:
            return "*a0_current = (*a1_current > 0.0) ? floor(*a1_current + 0.5) : ceil(*a1_current - 0.5)";
        case ISNAN:
            return "*a0_current = isnan(*a1_current)";
        case ISINF:
            return "*a0_current = isinf(*a1_current)";
        case IDENTITY:
            return "*a0_current = *a1_current";
        case REAL:
            return (type==BH_FLOAT32) ? "*a0_current = crealf(*a1_current)": "*a0_current = creal(*a1_current)";
        case IMAG:
            return (type==BH_FLOAT32) ? "*a0_current = cimagf(*a1_current)": "*a0_current = cimagf(*a1_current)";

        default:
            return "__UNKNOWN__";
    }
}