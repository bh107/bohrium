#ifndef __KP_ENGINE_CODEGEN_HPP
#define __KP_ENGINE_CODEGEN_HPP 1

#include <sstream>
#include <string>
#include <map>
#include "kp.h"
#include "block.hpp"
#include "plaid.hpp"

namespace kp{
namespace engine{
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
std::string _restrict(std::string object);
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
std::string _comment(std::string comment);
std::string _end(void);
std::string _end(std::string comment);
std::string _line(std::string object);

std::string _parens(std::string expr);
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
std::string _cabs(std::string right);
std::string _cabsf(std::string right);

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

// OpenMP stuff

std::string _omp_reduction_oper(KP_OPERATOR oper);

// Anonymous critical section
std::string _omp_critical(std::string expr);
// Named critical section
std::string _omp_critical(std::string expr, std::string name);
std::string _omp_atomic(std::string expr);

std::string _beef(std::string info);

class Codeblock
{
public:
    Codeblock(Plaid& plaid, std::string template_fn);
    
    virtual void prolog(std::string source);

    virtual void epilog(std::string source);

    virtual void pragma(std::string source);

    virtual void head(std::string source);

    virtual void body(std::string source);

    virtual void foot(std::string source);

    virtual std::string prolog(void);

    virtual std::string epilog(void);

    virtual std::string pragma(void);

    virtual std::string head(void);

    virtual std::string body(void);

    virtual std::string foot(void);

    virtual std::string emit(void);

protected:
    Plaid& plaid_;

    std::string template_fn_;

    std::stringstream prolog_;

    std::stringstream epilog_;

    std::stringstream pragma_;

    std::stringstream head_;

    std::stringstream body_;

    std::stringstream foot_;

private:
    Codeblock(void);

};

class Loop : public Codeblock
{
public:
    Loop(Plaid& plaid, std::string template_fn);
    
    void init(std::string source);

    void cond(std::string source);

    void incr(std::string source);

    std::string init(void);

    std::string cond(void);

    std::string incr(void);

    std::string emit(void);

protected:
    std::stringstream init_;
    std::stringstream cond_;
    std::stringstream incr_;

private:
    Loop(void);

};

class Buffer
{
public:
    Buffer(void);
    Buffer(kp_buffer* buffer, int64_t buffer_id);

    std::string name(void);

    std::string data(void);
    std::string nelem(void);
    std::string etype(void);

    kp_buffer & meta(void);
    int64_t id(void);

private:
    kp_buffer * buffer_;
    int64_t id_;
};

class Operand
{
public:
    Operand(void);
    Operand(kp_operand * operand, uint32_t local_id, Buffer* buffer);

    std::string name(void);

    std::string data(void);
    std::string nelem(void);
    std::string start(void);
    std::string layout(void);
    std::string etype(void);
    std::string ndim(void);
    std::string shape(void);

    std::string strides(void);
    std::string stride_inner(void);
    std::string stride_outer(void);
    std::string stride_axis(void);

    std::string accu_shared(void);
    std::string accu_private(void);
    std::string walker(void);
    std::string walker_val(void);
    std::string walker_subscript_val(void);

    kp_operand & meta(void);

    std::string buffer_name(void);
    std::string buffer_data(void);
    std::string buffer_nelem(void);
    std::string buffer_etype(void);
    kp_buffer * buffer_meta(void);

    uint64_t local_id(void);

private:
    uint64_t local_id_;
    kp_operand * operand_;
    Buffer* buffer_;
};

typedef std::map<uint64_t, Operand> kernel_operands;    // Operand
typedef kernel_operands::iterator kernel_operand_iter;

typedef std::map<size_t, Buffer> kernel_buffers;
typedef kernel_buffers::iterator kernel_buffer_iter;    // Buffer

typedef std::vector<kp_tac *> kernel_tacs;
typedef kernel_tacs::iterator kernel_tac_iter;

class Iterspace
{
public:
    Iterspace(kp_iterspace & iterspace);

    std::string name(void);
    std::string layout(void);
    std::string ndim(void);
    std::string shape(void);
    std::string shape(uint32_t dim);
    std::string nelem(void);
    
    kp_iterspace & meta(void);
private:
    kp_iterspace & iterspace_;
};

class Kernel
{
public:
    Kernel(Plaid& plaid, kp::core::Block& block);
    
    std::string generate_source(bool offload);

    uint64_t noperands(void);
    Operand& operand_glb(uint64_t gidx);
    Operand& operand_lcl(uint64_t lidx);

    kernel_operand_iter operands_begin(void);
    kernel_operand_iter operands_end(void);

    kernel_buffer_iter buffers_begin(void);
    kernel_buffer_iter buffers_end(void);

    Iterspace& iterspace(void);

    uint64_t base_refcount(uint64_t gidx);

    uint32_t omask(void);
    std::string text(void);

    uint64_t ntacs(void);
    kp_tac & tac(uint64_t tidx);
    kernel_tac_iter tacs_begin(void);
    kernel_tac_iter tacs_end(void);

private:
    std::string unpack_buffers(void);
    std::string unpack_arguments(void);
    std::string unpack_iterspace(void);
    
    std::string buffers(void);
    std::string args(void);

    void add_operand(uint64_t global_idx);

    Plaid& plaid_;
    kp::core::Block& block_;

    kernel_buffers buffers_;
    kernel_operands operands_;
    Iterspace iterspace_;

    kernel_tacs tacs_;
};

class Walker
{
public:
    Walker(Plaid& plaid, Kernel& kernel);

    std::string generate_source(bool offload);
    std::string oper_neutral_element(KP_OPERATOR oper, KP_ETYPE etype);
    
private:
    std::string declare_operands(void);
    std::string declare_operand(uint32_t oidx);

    std::string offload_block_leo(void);
    //std::string offload_loop_openacc(void);
    std::string offload_loop_openacc(kp_tac* tac_reduction, Operand* out);
    std::string offload_loop_sequel_openacc(kp_tac* tac_reduction, Operand* out);

    // Construct the operator source for the tac.oper
    std::string oper(KP_OPERATOR oper, KP_ETYPE etype, std::string in1, std::string in2);
    std::string synced_oper(KP_OPERATOR oper, KP_ETYPE etype, std::string out, std::string in1, std::string in2);

    /**
     *  Generate a comment describing the tac-operation.
     */
    std::string oper_description(kp_tac tac);

    //
    //  map / zip / flood / generate
    //

    /**
     * Construct an ordered sequence of applications of operators.
     * also note use of operands for outer/inner.
     */
    std::string operations(void);
    std::string write_expanded_scalars(void);

    std::string declare_stride_inner(uint64_t oidx);
    std::string declare_stride_inner(void);

    std::string declare_stride_outer_2D(uint64_t oidx);
    std::string declare_stride_outer_2D(void);

    std::string declare_stride_axis(uint64_t oidx);
    std::string declare_stride_axis(void);
    
    std::string assign_collapsed_offset(uint32_t rank);
    std::string assign_collapsed_offset(uint32_t rank, uint64_t oidx);

    std::string assign_offset_outer_2D();
    std::string assign_offset_outer_2D(uint64_t oidx);

    std::string step_fwd_outer_2D(uint64_t glb_idx);
    std::string step_fwd_outer_2D(void);

    std::string step_fwd_outer(uint64_t glb_idx);
    std::string step_fwd_outer(void);

    std::string step_fwd_inner(uint64_t glb_idx);
    std::string step_fwd_inner(void);

    std::string step_fwd_other(uint64_t glb_idx, std::string dimvar);
    std::string step_fwd_other(void);

    std::string step_fwd_axis(uint64_t glb_idx);
    std::string step_fwd_axis(void);

    Plaid& plaid_;
    Kernel& kernel_;

    std::set<uint64_t> inner_opds_; // Set of global ids for inner operands
    std::set<uint64_t> outer_opds_; // Set of global ids for outer operands

};

}}}

#endif
