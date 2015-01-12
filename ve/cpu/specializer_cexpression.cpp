#include "specializer.hpp"
//
//  NOTE: This file is autogenerated based on the tac-definition.
//        You should therefore not edit it manually.
//
using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

//
// NOTE: This function relies on the posix entension for positional arguments
// to print format string.
//
string Specializer::cexpression(SymbolTable& symbol_table, const Block& block, size_t tac_idx)
{
    tac_t& tac  = block.array_tac(tac_idx);
    ETYPE etype = symbol_table[tac.out].etype;

    string expr_text;

    char out_c = ' ';
    char in1_c = ' ';
    char in2_c = ' ';

    switch(core::tac_noperands(tac)) {
        case 3:
            if ((symbol_table[tac.in2].layout & ARRAY_LAYOUT) >0) {
                in2_c = '*';
            }
        case 2:
            if ((symbol_table[tac.in1].layout & ARRAY_LAYOUT) >0) {
                in1_c = '*';
            }
        case 1:
            if ((symbol_table[tac.out].layout & ARRAY_LAYOUT) >0) {
                out_c = '*';
            }
            break;
    }

    switch(tac.oper) {
        case ABSOLUTE:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current < 0.0 ? -%3$ca%4$d_current: %3$ca%4$d_current"; break;
            break;
        case ADD:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current + %5$ca%6$d_current"; break;
                case SCAN:
                    expr_text = "state += %3$ca%4$d_current; %1$ca%2$d_current = state"; break;
                case REDUCE:
                    expr_text = "state += *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case ARCCOS:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = cacos( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = cacosf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = acos( %3$ca%4$d_current )"; break;
            }
            break;
        case ARCCOSH:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = cacosh( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = cacoshf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = acosh( %3$ca%4$d_current )"; break;
            }
            break;
        case ARCSIN:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = casin( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = casinf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = asin( %3$ca%4$d_current )"; break;
            }
            break;
        case ARCSINH:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = casinh( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = casinhf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = asinh( %3$ca%4$d_current )"; break;
            }
            break;
        case ARCTAN:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = catan( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = catanf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = atan( %3$ca%4$d_current )"; break;
            }
            break;
        case ARCTAN2:            
            
            expr_text = "%1$ca%2$d_current = atan2( %3$ca%4$d_current, %5$ca%6$d_current )"; break;
            break;
        case ARCTANH:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = catanh( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = catanhf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = atanh( %3$ca%4$d_current )"; break;
            }
            break;
        case BITWISE_AND:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current & %5$ca%6$d_current"; break;
                case REDUCE:
                    expr_text = "state &= *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case BITWISE_OR:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current | %5$ca%6$d_current"; break;
                case REDUCE:
                    expr_text = "state = state | *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case BITWISE_XOR:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current ^ %5$ca%6$d_current"; break;
                case REDUCE:
                    expr_text = "state = state ^ *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case CEIL:            
            
            expr_text = "%1$ca%2$d_current = ceil( %3$ca%4$d_current )"; break;
            break;
        case COS:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = ccos( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = ccosf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = cos( %3$ca%4$d_current )"; break;
            }
            break;
        case COSH:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = ccosh( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = ccoshf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = cosh( %3$ca%4$d_current )"; break;
            }
            break;
        case DISCARD:            
            
            expr_text = "__ERROR__SYSTEM_DISCARD_SHOULD_NOT_BE_HERE__"; break;
            break;
        case DIVIDE:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current / %5$ca%6$d_current"; break;
            break;
        case EQUAL:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current == %5$ca%6$d_current"; break;
            break;
        case EXP:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = cexp( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = cexpf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = exp( %3$ca%4$d_current )"; break;
            }
            break;
        case EXP2:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = cpow( 2, %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = cpowf( 2, %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = pow( 2, %3$ca%4$d_current )"; break;
            }
            break;
        case EXPM1:            
            
            expr_text = "%1$ca%2$d_current = expm1( %3$ca%4$d_current )"; break;
            break;
        case EXTENSION_OPERATOR:            
            
            expr_text = "__ERROR__EXTENSION_EXTENSION_OPERATOR_SHOULD_NOT_BE_HERE__"; break;
            break;
        case FLOOD:            
            
            expr_text = "__ERROR__GENERATOR_FLOOD_NOT_IMLEMENTED__"; break;
            break;
        case FLOOR:            
            
            expr_text = "%1$ca%2$d_current = floor( %3$ca%4$d_current )"; break;
            break;
        case FREE:            
            
            expr_text = "__ERROR__SYSTEM_FREE_SHOULD_NOT_BE_HERE__"; break;
            break;
        case GREATER:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current > %5$ca%6$d_current"; break;
            break;
        case GREATER_EQUAL:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current >= %5$ca%6$d_current"; break;
            break;
        case IDENTITY:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current"; break;
            break;
        case IMAG:            
            
            switch(etype) {
                case FLOAT32: expr_text = "%1$ca%2$d_current = cimagf(%3$ca%4$d_current)"; break;
                default: expr_text = "%1$ca%2$d_current = cimag(%3$ca%4$d_current)"; break;
            }
            break;
        case INVERT:            
            
            switch(etype) {
                case BOOL: expr_text = "%1$ca%2$d_current = !%3$ca%4$d_current"; break;
                default: expr_text = "%1$ca%2$d_current = ~%3$ca%4$d_current"; break;
            }
            break;
        case ISINF:            
            
            expr_text = "%1$ca%2$d_current = isinf(%3$ca%4$d_current)"; break;
            break;
        case ISNAN:            
            
            expr_text = "%1$ca%2$d_current = isnan(%3$ca%4$d_current)"; break;
            break;
        case LEFT_SHIFT:            
            
            expr_text = "%1$ca%2$d_current = (%3$ca%4$d_current) << (%5$ca%6$d_current)"; break;
            break;
        case LESS:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current < %5$ca%6$d_current"; break;
            break;
        case LESS_EQUAL:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current <= %5$ca%6$d_current"; break;
            break;
        case LOG:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = clog( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = clogf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = log( %3$ca%4$d_current )"; break;
            }
            break;
        case LOG10:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = clog( %3$ca%4$d_current )/log(10)"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = clogf( %3$ca%4$d_current )/log(10)"; break;
                default: expr_text = "%1$ca%2$d_current = log( %3$ca%4$d_current )/log(10)"; break;
            }
            break;
        case LOG1P:            
            
            expr_text = "%1$ca%2$d_current = log1p( %3$ca%4$d_current )"; break;
            break;
        case LOG2:            
            
            expr_text = "%1$ca%2$d_current = log2( %3$ca%4$d_current )"; break;
            break;
        case LOGICAL_AND:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current && %5$ca%6$d_current"; break;
                case REDUCE:
                    expr_text = "state = state && *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case LOGICAL_NOT:            
            
            expr_text = "%1$ca%2$d_current = !%3$ca%4$d_current"; break;
            break;
        case LOGICAL_OR:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current || %5$ca%6$d_current"; break;
                case REDUCE:
                    expr_text = "state = state || *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case LOGICAL_XOR:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = (!%3$ca%4$d_current != !%5$ca%6$d_current)"; break;
                case REDUCE:
                    expr_text = "state = !state != !*tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case MAXIMUM:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current < %5$ca%6$d_current ? %5$ca%6$d_current : %3$ca%4$d_current"; break;
                case REDUCE:
                    expr_text = "state = state < *tmp_current ? *tmp_current : state"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case MINIMUM:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current < %5$ca%6$d_current ? %3$ca%4$d_current : %5$ca%6$d_current"; break;
                case REDUCE:
                    expr_text = "state = state < *tmp_current ? state : *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case MOD:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current - floor(%3$ca%4$d_current / %5$ca%6$d_current) * %5$ca%6$d_current"; break;
            break;
        case MULTIPLY:            
            switch (tac.op) {
                case ZIP:
                    expr_text = "%1$ca%2$d_current = %3$ca%4$d_current * %5$ca%6$d_current"; break;
                case SCAN:
                    expr_text = "state *= %3$ca%4$d_current; %1$ca%2$d_current = state"; break;
                case REDUCE:
                    expr_text = "state *= *tmp_current"; break;
                default:
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            break;
        case NONE:            
            
            expr_text = "__ERROR__SYSTEM_NONE_SHOULD_NOT_BE_HERE__"; break;
            break;
        case NOT_EQUAL:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current != %5$ca%6$d_current"; break;
            break;
        case POWER:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = cpow( %3$ca%4$d_current, %5$ca%6$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = cpowf( %3$ca%4$d_current, %5$ca%6$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = pow( %3$ca%4$d_current, %5$ca%6$d_current )"; break;
            }
            break;
        case RANDOM:            
            
            expr_text = "philox2x32_as_1x64_t ctr_%4$d; philox2x32_as_1x64_t rand_%2$d; ctr_%4$d.combined = %3$ca%4$d_current + i; rand_%2$d.orig = philox2x32(ctr_%4$d.orig, (philox2x32_key_t){ { %5$ca%6$d_current } } ); %1$ca%2$d_current = rand_%2$d.combined;"; break;
            break;
        case RANGE:            
            
            expr_text = "%1$ca%2$d_current = i"; break;
            break;
        case REAL:            
            
            switch(etype) {
                case FLOAT32: expr_text = "%1$ca%2$d_current = crealf(%3$ca%4$d_current)"; break;
                default: expr_text = "%1$ca%2$d_current = creal(%3$ca%4$d_current)"; break;
            }
            break;
        case RIGHT_SHIFT:            
            
            expr_text = "%1$ca%2$d_current = (%3$ca%4$d_current) >> (%5$ca%6$d_current)"; break;
            break;
        case RINT:            
            
            expr_text = "%1$ca%2$d_current = (%3$ca%4$d_current > 0.0) ? floor(%3$ca%4$d_current + 0.5) : ceil(%3$ca%4$d_current - 0.5)"; break;
            break;
        case SIN:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = csin( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = csinf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = sin( %3$ca%4$d_current )"; break;
            }
            break;
        case SINH:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = csinh( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = csinhf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = sinh( %3$ca%4$d_current )"; break;
            }
            break;
        case SQRT:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = csqrt( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = csqrtf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = sqrt( %3$ca%4$d_current )"; break;
            }
            break;
        case SUBTRACT:            
            
            expr_text = "%1$ca%2$d_current = %3$ca%4$d_current - %5$ca%6$d_current"; break;
            break;
        case SYNC:            
            
            expr_text = "__ERROR__SYSTEM_SYNC_SHOULD_NOT_BE_HERE__"; break;
            break;
        case TAN:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = ctan( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = ctanf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = tan( %3$ca%4$d_current )"; break;
            }
            break;
        case TANH:            
            
            switch(etype) {
                case COMPLEX128: expr_text = "%1$ca%2$d_current = ctanh( %3$ca%4$d_current )"; break;
                case COMPLEX64: expr_text = "%1$ca%2$d_current = ctanhf( %3$ca%4$d_current )"; break;
                default: expr_text = "%1$ca%2$d_current = tanh( %3$ca%4$d_current )"; break;
            }
            break;
        case TRUNC:            
            
            expr_text = "%1$ca%2$d_current = trunc( %3$ca%4$d_current )"; break;
            break;
    }


    switch(core::tac_noperands(tac)) {
        case 3:
            return core::string_format(
                expr_text,
                out_c, block.global_to_local(tac.out), 
                in1_c, block.global_to_local(tac.in1),
                in2_c, block.global_to_local(tac.in2)
            );
        case 2:
            return core::string_format(
                expr_text, 
                out_c, block.global_to_local(tac.out),
                in1_c, block.global_to_local(tac.in1)
            );
        case 1:
            return core::string_format(
                expr_text,
                out_c, block.global_to_local(tac.out)
            );
        default:
            return expr_text;
    }
}    

}}}
