#include <cmath>
#include <cstdlib>
#include <cphvb_win_math.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)

template <typename T1, typename T2, typename T3>
struct add_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 + *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct subtract_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 - *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct multiply_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 * *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct divide_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 / *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct logaddexp_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = log( exp(*op2) + exp(*op3) );
    }
};

template <typename T1, typename T2, typename T3>
struct logaddexp2_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = log2( pow(2, *op2) + pow(2, *op3) );
    }
};

template <typename T1, typename T2, typename T3>
struct true_divide_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 / *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct floor_divide_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = floor( *op2 / *op3 );
    }
};

template <typename T1, typename T2, typename T3>
struct power_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = pow( *op2, *op3 );
    }
};

template <typename T1, typename T2, typename T3>
struct remainder_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
#ifdef _WIN32
		//We need a typecast here so VC can pick the right version
        *op1 = *op2 - floor((T1)(*op2 / *op3)) * *op3;
#else
        *op1 = *op2 - floor(*op2 / *op3) * *op3;
#endif
    }
};

template <typename T1, typename T2, typename T3>
struct mod_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 - floor(*op2 / *op3) * *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct fmod_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 - floor(*op2 / *op3) * *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct bitwise_and_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 & *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct bitwise_or_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 | *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct bitwise_xor_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 ^ *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct logical_and_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 && *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct logical_or_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 || *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct logical_xor_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = (!*op2 != !*op3);
    }
};

template <typename T1, typename T2, typename T3>
struct left_shift_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = (*op2) << (*op3);
    }
};

template <typename T1, typename T2, typename T3>
struct right_shift_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = (*op2) >> (*op3);
    }
};

template <typename T1, typename T2, typename T3>
struct greater_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 > *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct greater_equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 >= *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct less_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 < *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct less_equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 <= *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct not_equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 != *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 == *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct maximum_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 < *op3 ? *op3 : *op2;
    }
};

template <typename T1, typename T2, typename T3>
struct minimum_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 < *op3 ? *op2 : *op3;
    }
};

template <typename T1, typename T2, typename T3>
struct ldexp_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = *op2 * pow(2, *op3);
    }
};

template <typename T1, typename T2>
struct negative_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = -*op2;
    }
};

template <typename T1, typename T2>
struct absolute_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2 < 0.0 ? -*op2: *op2;
    }
};

template <typename T1, typename T2>
struct rint_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = (*op2 > 0.0) ? floor(*op2 + 0.5) : ceil(*op2 - 0.5);
    }
};

template <typename T1, typename T2>
struct sign_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2 > 0.0 ? 1.0 : (*op2 == 0 ? 0 : -1);
    }
};



template <typename T1, typename T2>
struct exp_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = exp( *op2 );
    }
};

template <typename T1, typename T2>
struct exp2_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = pow( 2, *op2 );
    }
};

template <typename T1, typename T2>
struct log2_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = log2( *op2 );
    }
};

template <typename T1, typename T2>
struct log_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = log( *op2 );
    }
};

template <typename T1, typename T2>
struct log10_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = log10( *op2 );
    }
};

template <typename T1, typename T2>
struct expm1_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = expm1( *op2 );
    }
};

template <typename T1, typename T2>
struct log1p_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = log1p( *op2 );
    }
};

template <typename T1, typename T2>
struct sqrt_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = sqrt( *op2 );
    }
};

template <typename T1, typename T2>
struct square_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2 * *op2;
    }
};

template <typename T1, typename T2>
struct reciprocal_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 1.0 / *op2;
    }
};


template <typename T1, typename T2>
struct sin_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = sin( *op2 );
    }
};

template <typename T1, typename T2>
struct cos_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = cos( *op2 );
    }
};

template <typename T1, typename T2>
struct tan_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = tan( *op2 );
    }
};

template <typename T1, typename T2>
struct arcsin_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = asin( *op2 );
    }
};

template <typename T1, typename T2>
struct arccos_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = acos( *op2 );
    }
};

template <typename T1, typename T2>
struct arctan_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = atan( *op2 );
    }
};

template <typename T1, typename T2, typename T3>
struct arctan2_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = atan2( *op2, *op3 );
    }
};

template <typename T1, typename T2, typename T3>
struct hypot_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = sqrt( pow(*op2, 2) + pow(*op3, 2) );
    }
};

template <typename T1, typename T2>
struct sinh_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = sinh( *op2 );
    }
};

template <typename T1, typename T2>
struct cosh_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = cosh( *op2 );
    }
};

template <typename T1, typename T2>
struct tanh_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = tanh( *op2 );
    }
};

template <typename T1, typename T2>
struct arcsinh_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = asinh( *op2 );
    }
};

template <typename T1, typename T2>
struct arccosh_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = acosh( *op2 );
    }
};

template <typename T1, typename T2>
struct arctanh_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = atanh( *op2 );
    }
};

template <typename T1, typename T2>
struct deg2rad_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2 * DEG_RAD;
    }
};

template <typename T1, typename T2>
struct rad2deg_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2 * RAD_DEG;
    }
};

template <typename T1, typename T2>
struct logical_not_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = !*op2;
    }
};

template <typename T1, typename T2>
struct invert_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = ~*op2;
    }
};

// TODO: implement
template <typename T1, typename T2>
struct isinf_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 42; // TODO: implement
    }
};

template <typename T1, typename T2>
struct isnan_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 42; // TODO: implement
    }
};

template <typename T1, typename T2>
struct signbit_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2 < 0;
    }
};

template <typename T1, typename T2>
struct floor_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = floor( *op2 );
    }
};

template <typename T1, typename T2>
struct ceil_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = ceil( *op2 );
    }
};

template <typename T1, typename T2>
struct trunc_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = trunc( *op2 );
    }
};

template <typename T1, typename T2>
struct identity_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = *op2;
    }
};

/* IMPLEMENT TOGETHER WITH COMPLEX NUMBERS
template <typename T1, typename T2>
struct conj_functor {
    void operator()(T1 *op1, T2 *op2) {
        // TODO: implement
        *op1 = 100;
    }
};

template <typename T1, typename T2>
struct isreal_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 100; // TODO: implement
    }
};

template <typename T1, typename T2>
struct iscomplex_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 100; // TODO: implement
    }
};

*/

/*  THESE MIGHT NEVER BE IMPLEMENTED.

template <typename T1, typename T2>
struct ones_like_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 100; // TODO: implement
    }
};

template <typename T1, typename T2>
struct isfinite_functor {
    void operator()(T1 *op1, T2 *op2) {
        *op1 = 100; // TODO: implement
    }
};

template <typename T1, typename T2, typename T3>
struct modf_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        *op1 = 100; // TODO: implement
    }
};

template <typename T1, typename T2, typename T3>
struct frexp_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) {
        int exponent;
        *op1 = frexp( *op3, &exponent );
        *op2 = (T2)exponent;
    }
};

*/
