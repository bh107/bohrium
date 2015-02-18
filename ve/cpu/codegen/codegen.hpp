#ifndef __BH_VE_CPU_CODEGEN
#define __BH_VE_CPU_CODEGEN

#include <string>
#include <map>
#include <tac.h>
#include <block.hpp>
#include <plaid.hpp>

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

// Primitive types
std::string _int8(void);
std::string _int16(void);
std::string _int32(void);
std::string _int64(void);
std::string _uint8(void);
std::string _uint16(void);
std::string _uint32(void);
std::string _uint64(void);
std::string _float(void);
std::string _double(void);
std::string _float_complex(void);
std::string _double_complex(void);

// Variable declaration, access and modifiers
std::string _ref(std::string object);
std::string _deref(std::string object);
std::string _index(std::string object, int64_t idx);

std::string _access(std::string object, std::string member);
std::string _access_ptr(std::string object, std::string member);

std::string _ptr(std::string object);
std::string _ptr_const(std::string object);
std::string _const(std::string object);
std::string _const_ptr(std::string object);

std::string _assert_not_null(std::string object);
std::string _assign(std::string lvalue, std::string rvalue);
std::string _end(void);
std::string _end(std::string comment);
std::string _line(std::string object);

std::string _cast(std::string type, std::string object);

std::string _declare(std::string type, std::string variable);
std::string _declare_init(std::string type, std::string variable, std::string expr);

// Operators
std::string _inc(std::string object);
std::string _dec(std::string object);

std::string _add_assign(std::string left, std::string right);
std::string _sub_assign(std::string left, std::string right);

std::string _add(std::string left, std::string right);
std::string _sub(std::string left, std::string right);
std::string _mul(std::string left, std::string right);
std::string _div(std::string left, std::string right);
std::string _mod(std::string left, std::string right);
std::string _pow(std::string left, std::string right);
std::string _cpow(std::string left, std::string right);
std::string _cpowf(std::string left, std::string right);
std::string _abs(std::string left, std::string right);

std::string _max(std::string left, std::string right);
std::string _min(std::string left, std::string right);

std::string _gt(std::string left, std::string right);
std::string _gteq(std::string left, std::string right);
std::string _lt(std::string left, std::string right);
std::string _lteq(std::string left, std::string right);
std::string _eq(std::string left, std::string right);
std::string _neq(std::string left, std::string right);

std::string _logic_and(std::string left, std::string right);
std::string _logic_or(std::string left, std::string right);
std::string _logic_xor(std::string left, std::string right);
std::string _logic_not(std::string right);

std::string _bitw_and(std::string left, std::string right);
std::string _bitw_or(std::string left, std::string right);
std::string _bitw_xor(std::string left, std::string right);

std::string _invert(std::string right);
std::string _invertb(std::string right);

std::string _bitw_leftshift(std::string left, std::string right);
std::string _bitw_rightshift(std::string left, std::string right);

std::string _sin(std::string right);
std::string _csinf(std::string right);
std::string _csin(std::string right);

std::string _sinh(std::string right);
std::string _csinhf(std::string right);
std::string _csinh(std::string right);

std::string _asin(std::string right);
std::string _casinf(std::string right);
std::string _casin(std::string right);

std::string _asinh(std::string right);
std::string _casinhf(std::string right);
std::string _casinh(std::string right);

std::string _cos(std::string right);
std::string _ccosf(std::string right);
std::string _ccos(std::string right);

std::string _cosh(std::string right);
std::string _ccoshf(std::string right);
std::string _ccosh(std::string right);

std::string _acos(std::string right);
std::string _cacos(std::string right);
std::string _cacosf(std::string right);

std::string _acosh(std::string right);
std::string _cacoshf(std::string right);
std::string _cacosh(std::string right);

std::string _tan(std::string right);
std::string _ctanf(std::string right);
std::string _ctan(std::string right);

std::string _tanh(std::string right);
std::string _ctanhf(std::string right);
std::string _ctanh(std::string right);

std::string _atan(std::string right);
std::string _catanf(std::string right);
std::string _catan(std::string right);

std::string _atanh(std::string right);
std::string _catanhf(std::string right);
std::string _catanh(std::string right);

std::string _atan2(std::string left, std::string right);

std::string _exp(std::string right);
std::string _cexpf(std::string right);
std::string _cexp(std::string right);

std::string _exp2(std::string right);
std::string _cexp2f(std::string right);
std::string _cexp2(std::string right);

std::string _expm1(std::string right);

std::string _log(std::string right);
std::string _clog(std::string right);
std::string _clogf(std::string right);

std::string _log10(std::string right);
std::string _clog10(std::string right);
std::string _clog10f(std::string right);

std::string _log2(std::string right);
std::string _log1p(std::string right);

std::string _sqrt(std::string right);
std::string _csqrt(std::string right);
std::string _csqrtf(std::string right);
std::string _ceil(std::string right);
std::string _trunc(std::string right);
std::string _floor(std::string right);
std::string _rint(std::string right);
std::string _range(void);
std::string _random(std::string left, std::string right);
std::string _isnan(std::string right);
std::string _isinf(std::string right);

std::string _creal(std::string right);
std::string _crealf(std::string right);

std::string _cimagf(std::string right);
std::string _cimag(std::string right);

class Operand
{
public:
    Operand(void);
    Operand(operand_t* operand, uint32_t id);

    std::string name(void);
    
    std::string walker(void);
    std::string walker_val(void);

    std::string first(void);
    std::string layout(void);
    std::string etype(void);
    std::string ndim(void);
    std::string shape(void);
    std::string stride(void);

    operand_t* operand_;

private:
    uint32_t id_;
};

class Walker
{
public:
    Walker(Plaid& plaid, bohrium::core::Block& block);

    std::string generate_source(void);
    
private:
    std::string declare_operands(void);
    std::string declare_operand(uint32_t oidx);

    std::string oper(tac_t tac);

    //
    //  map / zip / flood / generate
    //
    std::string ewise_operations(void);

    // Offsets
    std::string ewise_cont_offset(uint32_t oidx);
    std::string ewise_cont_offset(void);

    std::string ewise_strided_1d_offset(uint32_t oidx);
    std::string ewise_strided_1d_offset(void);

    std::string ewise_strided_2d_offset(uint32_t oidx);
    std::string ewise_strided_2d_offset(void);

    std::string ewise_strided_3d_offset(uint32_t oidx);
    std::string ewise_strided_3d_offset(void);

    std::string ewise_strided_nd_offset(uint32_t oidx);
    std::string ewise_strided_nd_offset(void);

    // Steps
    std::string ewise_cont_step(uint32_t oidx);
    std::string ewise_cont_step(void);

    std::string ewise_strided_1d_step(unsigned int oidx);
    std::string ewise_strided_1d_step(void);

    std::string ewise_strided_nd_step(unsigned int oidx);
    std::string ewise_strided_nd_step(void);

    Plaid& plaid_;
    bohrium::core::Block& block_;
};

class Kernel
{
public:
    Kernel(Plaid& plaid, bohrium::core::Block& block);
    
    std::string generate_source(void);

private:
        
    std::string unpack_arguments(void);
    std::string unpack_argument(uint32_t id);
    
    std::string args(void);
    std::string iterspace(void);

    Plaid& plaid_;
    bohrium::core::Block& block_;
};

}}}}

#endif
