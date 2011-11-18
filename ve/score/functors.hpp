
#include <cmath>
#include <cstdlib>

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
    }
};  

template <typename T1, typename T2, typename T3> 
struct logaddexp2_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct true_divide_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct floor_divide_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct power_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct remainder_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct mod_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct fmod_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct bitwise_and_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct bitwise_or_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct bitwise_xor_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct logical_and_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct logical_or_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct logical_xor_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct left_shift_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct right_shift_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct greater_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct greater_equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct less_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct less_equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct not_equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct equal_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct maximum_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct minimum_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct ldexp_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2> 
struct negative_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct absolute_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct rint_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct sign_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct conj_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct exp_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct exp2_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct log2_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct log_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct log10_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct expm1_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct log1p_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct sqrt_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct square_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct reciprocal_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct ones_like_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct sin_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct cos_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct tan_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arcsin_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arccos_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arctan_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arctan2_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct hypot_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct sinh_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct cosh_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct tanh_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arcsinh_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arccosh_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct arctanh_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct deg2rad_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct rad2deg_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct logical_not_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct invert_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct isfinite_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct isinf_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct isnan_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct signbit_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct floor_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct ceil_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct trunc_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct isreal_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2> 
struct iscomplex_functor {
    void operator()(T1 *op1, T2 *op2) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct modf_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1, typename T2, typename T3> 
struct frexp_functor {
    void operator()(T1 *op1, T2 *op2, T3 *op3) { 
    }
};  

template <typename T1> 
struct random_functor {
    void operator()(T1 *op1) { 
    }
};  

template <typename T1> 
struct arange_functor {
    void operator()(T1 *op1) { 
    }
};  
