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
std::string _index(std::string object, std::string idx);

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
std::string _abs(std::string right);

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
    Operand(operand_t* operand, uint32_t local_id);

    std::string name(void);
    
    std::string walker(void);
    std::string walker_val(void);

    std::string first(void);
    std::string layout(void);
    std::string etype(void);
    std::string ndim(void);
    std::string shape(void);
    std::string stride(void);
    std::string stridevar(uint32_t dim);
    std::string stepsize(uint32_t dim);

    operand_t& meta(void);
    uint64_t local_id(void);

private:
    operand_t* operand_;
    uint64_t local_id_;
};

typedef std::map<uint64_t, Operand> kernel_operands;
typedef kernel_operands::iterator kernel_operand_iter;

typedef std::vector<tac_t*> kernel_tacs;
typedef kernel_tacs::iterator kernel_tac_iter;

class Iterspace
{
public:
    Iterspace(iterspace_t& iterspace);

    std::string name(void);
    std::string ndim(void);
    std::string shape(uint32_t dim);

    iterspace_t& meta(void);
private:
    iterspace_t& iterspace_;
};

class Kernel
{
public:
    Kernel(Plaid& plaid, bohrium::core::Block& block);
    
    std::string generate_source(void);

    uint64_t noperands(void);
    Operand& operand_glb(uint64_t gidx);
    Operand& operand_lcl(uint64_t lidx);

    kernel_operand_iter operands_begin(void);
    kernel_operand_iter operands_end(void);

    uint32_t omask(void);

    uint64_t ntacs(void);
    tac_t& tac(uint64_t tidx);
    kernel_tac_iter tacs_begin(void);
    kernel_tac_iter tacs_end(void);

    Iterspace& iterspace(void);

private:
        
    std::string unpack_arguments(void);
    
    std::string args(void);

    void add_operand(uint64_t global_idx);

    Plaid& plaid_;
    bohrium::core::Block& block_;
    kernel_operands operands_;
    kernel_tacs tacs_;
    Iterspace iterspace_;
    
};

class Walker
{
public:
    Walker(Plaid& plaid, Kernel& kernel);

    std::string generate_source(void);
    std::string oper_neutral_element(OPERATOR oper);
    
private:
    std::string declare_operands(void);
    std::string declare_operand(uint32_t oidx);

    // Construct the operator source for the tac.oper
    std::string oper(OPERATOR oper, ETYPE etype, std::string in1, std::string in2);

    /**
     *  Generate a comment describing the tac-operation.
     */
    std::string oper_description(tac_t tac);

    //
    //  map / zip / flood / generate
    //
    std::string ewise_operations(void);

    std::string declare_stridesize(uint64_t oidx);
    std::string declare_stridesizes(void);

    // Ewise walker -- innards
    std::string ewise_declare_stepsizes(uint32_t rank);
    std::string ewise_assign_offset(uint32_t rank);
    std::string ewise_assign_offset(uint32_t rank, uint64_t oidx);
    std::string step_fwd(uint32_t dim, uint64_t oidx);
    std::string step_fwd(uint32_t dim);

    std::string reduce_par_operations(void);
    std::string reduce_seq_operations(void);
    
    Plaid& plaid_;
    Kernel& kernel_;
};



}}}}

#endif
