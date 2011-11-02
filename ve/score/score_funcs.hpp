#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846  
#endif
 
#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)

template <typename T>
cphvb_error score_add( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 + *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_subtract( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 - *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_multiply( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 * *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_divide( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 / *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_logaddexp( T *op1, T *op2, T *op3 ) {
    *op1 = log( exp(*op2) + exp(*op3) );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_logaddexp2( T *op1, T *op2, T *op3 ) {
    *op1 = log2( pow(2, *op2) + pow(2, *op3) );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_true_divide( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 / *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_floor_divide( T *op1, T *op2, T *op3 ) {
    *op1 = floor( *op2 / *op3 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_power( T *op1, T *op2, T *op3 ) {
    *op1 = pow( *op2, *op3 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_remainder( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 - floor(*op2 / *op3) * *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_mod( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 - floor(*op2 / *op3) * *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_fmod( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 - floor(*op2 / *op3) * *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_bitwise_and( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 & *op3;
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_bitwise_and( cphvb_float32 *op1, cphvb_float32 *op2, cphvb_float32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_bitwise_and( cphvb_float64 *op1, cphvb_float64 *op2, cphvb_float64 *op3 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_bitwise_or( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 | *op3;
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_bitwise_or( cphvb_float32 *op1, cphvb_float32 *op2, cphvb_float32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_bitwise_or( cphvb_float64 *op1, cphvb_float64 *op2, cphvb_float64 *op3 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_bitwise_xor( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 ^ *op3;
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_bitwise_xor( cphvb_float32 *op1, cphvb_float32 *op2, cphvb_float32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_bitwise_xor( cphvb_float64 *op1, cphvb_float64 *op2, cphvb_float64 *op3 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_logical_not( T *op1, T *op2 ) {
    *op1 = !*op2;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_logical_and( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 && *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_logical_or( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 || *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_logical_xor( T *op1, T *op2, T *op3 ) {
    *op1 = (*op2 != *op3) ? 0 : 1;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_left_shift( T *op1, T *op2, T *op3 ) {
    *op1 = (*op2) << (*op3);
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_left_shift( cphvb_float32 *op1, cphvb_float32 *op2, cphvb_float32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_left_shift( cphvb_float64 *op1, cphvb_float64 *op2, cphvb_float64 *op3 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_right_shift( T *op1, T *op2, T *op3 ) {
    *op1 = (*op2) >> (*op3);
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_right_shift( cphvb_float32 *op1, cphvb_float32 *op2, cphvb_float32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_right_shift( cphvb_float64 *op1, cphvb_float64 *op2, cphvb_float64 *op3 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_greater( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 > *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_greater_equal( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 >= *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_less( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 < *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_less_equal( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 <= *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_not_equal( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 != *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_equal( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 == *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_maximum( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 < *op3 ? *op3 : *op2;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_minimum( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 < *op3 ? *op2 : *op3;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_ldexp( T *op1, T *op2, T *op3 ) {
    *op1 = *op2 * pow(2, *op3);
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_negative( T *op1, T *op2 ) {
    *op1 = -*op2;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_absolute( T *op1, T *op2 ) {
    *op1 = *op2 < 0.0 ? -*op2: *op2; 
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_rint( T *op1, T *op2 ) {
    *op1 = (*op2 > 0.0) ? floor(*op2 + 0.5) : ceil(*op2 - 0.5);
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_sign( T *op1, T *op2 ) {
    *op1 = *op2 > 0.0 ? 1.0 : (*op2 == 0 ? 0 : -1);
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_exp( T *op1, T *op2 ) {
    *op1 = exp( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_exp2( T *op1, T *op2 ) {
    *op1 = pow( 2, *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_log( T *op1, T *op2 ) {
    *op1 = log( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_log2( T *op1, T *op2 ) {
    *op1 = log2( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_log10( T *op1, T *op2 ) {
    *op1 = log10( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_expm1( T *op1, T *op2 ) {
    *op1 = expm1( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_log1p( T *op1, T *op2 ) {
    *op1 = log1p( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_sqrt( T *op1, T *op2 ) {
    *op1 = sqrt( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_square( T *op1, T *op2 ) {
    *op1 = *op2 * *op2;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_reciprocal( T *op1, T *op2 ) {
    *op1 = 1.0 / *op2;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_sin( T *op1, T *op2 ) {
    *op1 = sin( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_cos( T *op1, T *op2 ) {
    *op1 = cos( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_tan( T *op1, T *op2 ) {
    *op1 = tan( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arcsin( T *op1, T *op2 ) {
    *op1 = asin( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arccos( T *op1, T *op2 ) {
    *op1 = acos( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arctan( T *op1, T *op2 ) {
    *op1 = atan( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_hypot( T *op1, T *op2, T *op3 ) {
    *op1 = sqrt( pow(*op2, 2) + pow(*op3, 2) );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_sinh( T *op1, T *op2 ) {
    *op1 = sinh( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_cosh( T *op1, T *op2 ) {
    *op1 = cosh( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_tanh( T *op1, T *op2 ) {
    *op1 = tanh( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arcsinh( T *op1, T *op2 ) {
    *op1 = asinh( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arccosh( T *op1, T *op2 ) {
    *op1 = acosh( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arctanh( T *op1, T *op2 ) {
    *op1 = atanh( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_deg2rad( T *op1, T *op2 ) {
    *op1 = *op2 * DEG_RAD;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_rad2deg( T *op1, T *op2 ) {
    *op1 = *op2 * RAD_DEG;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_invert( T *op1, T *op2 ) {
    *op1 = ~*op2;
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_invert( cphvb_float32 *op1, cphvb_float32 *op2 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_invert( cphvb_float64 *op1, cphvb_float64 *op2 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_signbit( T *op1, T *op2 ) {
    *op1 = *op2 < 0.0;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_floor( T *op1, T *op2 ) {
    *op1 = floor( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_ceil( T *op1, T *op2 ) {
    *op1 = ceil( *op2 );
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_trunc( T *op1, T *op2 ) {
    *op1 = trunc( *op2 );
    return CPHVB_SUCCESS;
}


template <typename T>
cphvb_error score_modf( T *op1, T *op2, T *op3 ) {
    *op1 = *op2;
    //*op1 = modf( *op3, &op2 );
    //*op3 = modf( *op2, &op1 );
    return CPHVB_SUCCESS;
}
template <>
cphvb_error score_modf( cphvb_float32 *op1, cphvb_float32 *op2, cphvb_float32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_float64 *op1, cphvb_float64 *op2, cphvb_float64 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_int8 *op1, cphvb_int8 *op2, cphvb_int8 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_int16 *op1, cphvb_int16 *op2, cphvb_int16 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_int32 *op1, cphvb_int32 *op2, cphvb_int32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_int64 *op1, cphvb_int64 *op2, cphvb_int64 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_uint8 *op1, cphvb_uint8 *op2, cphvb_uint8 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_uint16 *op1, cphvb_uint16 *op2, cphvb_uint16 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_uint32 *op1, cphvb_uint32 *op2, cphvb_uint32 *op3 ) {
    return CPHVB_ERROR;
}
template <>
cphvb_error score_modf( cphvb_uint64 *op1, cphvb_uint64 *op2, cphvb_uint64 *op3 ) {
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_frexp( T *op1, T *op2, T *op3 ) {

    int exponent;
    *op1 = frexp( *op3, &exponent );
    *op2 = (T)exponent;

    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_random( T *op1 ) {
    *op1 = (T)std::rand();
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_arange( T *op1 ) {
    // TODO: implement
    *op1 = 0.0;
    return CPHVB_SUCCESS;
}

template <typename T>
cphvb_error score_conj( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_ones_like( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_arctan2( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_isfinite( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_isinf( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_isnan( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}


template <typename T>
cphvb_error score_isreal( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}

template <typename T>
cphvb_error score_iscomplex( T *op1, T *op2 ) {
    // TODO: implement    
    *op1 = 100;
    return CPHVB_ERROR;
}
