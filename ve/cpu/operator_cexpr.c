const char* operator_cexpr(OPERATION const op, OPERATOR const oper, const bh_type type)
{
    switch(oper) {
        case REAL:
            return "out = realf(in1)";
        case COS:
            return "out = cos(in1)";
        case EXP:
            return "out = exp(in1)";
        case LESS:
            return "out = in1 < in2";
        case RINT:
            return "out = (in1 > 0.0) ? floor(in1 + 0.5) : ceil(in1 - 0.5)";
        case LEFT_SHIFT:
            return "out = in1 << in2";
        case EQUAL:
            return "out = in1 == in2";
        case COSH:
            return "out = cosh(in1)";
        case LOGICAL_AND:
            return "out = in1 || in2";
        case SINH:
            return "out = sinh(in1)";
        case MULTIPLY:
            return "out = in1 * in2";
        case SUBTRACT:
            return "out = in1 - in2";
        case SIN:
            return "out = sin(in1)";
        case ABSOLUTE:
            return "out = in1 < 0.0 ? -in1 : in1";
        case EXPM1:
            return "out = expm1(in1)";
        case NOT_EQUAL:
            return "out = in1 != in2";
        case LOG2:
            return "out = log2(in1)";
        case ARCSINH:
            return "out = arcsinh(in1)";
        case GREATER_EQUAL:
            return "out = in1 >= in2";
        case TANH:
            return "out = tanh(in1)";
        case LOG10:
            return "out = log(in1)/log(10)";
        case ISNAN:
            return "out = isnan(in1)";
        case MINIMUM:
            return "out = in1 < in2 ? in1 : in2";
        case ARCTANH:
            return "out = arctanh(in1)";
        case ISINF:
            return "out = isinf(in1)";
        case TAN:
            return "out = tan(in1)";
        case IMAG:
            return "out = imagf(in1)";
        case TRUNC:
            return "out = trunc(in1)";
        case BITWISE_XOR:
            return "out = in1 ^ in2";
        case DIVIDE:
            return "out = in1 / in2";
        case FLOOR:
            return "out = floor(in1)";
        case ARCSIN:
            return "out = arcsin(in1)";
        case EXP2:
            return "out = exp2(in1)";
        case LOG:
            return "out = log(in1)";
        case MAXIMUM:
            return "out = in1 < in2 ? in2 : in1";
        case ARCTAN2:
            return "out = atan2(in1, in2)";
        case ADD:
            return "out = in1 + in2";
        case LESS_EQUAL:
            return "out = in1 <= in2";
        case CEIL:
            return "out = ceil(in1)";
        case ARCCOSH:
            return "out = arccosh(in1)";
        case IDENTITY:
            return "out = in1";
        case GREATER:
            return "out = in1 > in2";
        case LOGICAL_XOR:
            return "out = (!in1 != !in2)";
        case LOGICAL_NOT:
            return "out = !in1";
        case POWER:
            return "out = pow(in1, in2)";
        case INVERT:
            return "out = ~in1";
        case SQRT:
            return "out = sqrt(in1)";
        case ARCTAN:
            return "out = arctan(in1)";
        case BITWISE_OR:
            return "out = in1 | in2";
        case RIGHT_SHIFT:
            return "out = in1 >> in2";
        case ARCCOS:
            return "out = arccos(in1)";
        case BITWISE_AND:
            return "out = in1 & in2";
        case LOG1P:
            return "out = log1p(in1)";
        case MOD:
            return "out = in1 - floor(in1/in2) * in2";

        default:
            return "__UNK_OPER__";
    }
}
