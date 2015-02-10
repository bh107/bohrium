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
std::string _line(std::string object);

std::string _cast(std::string type, std::string object);

std::string _declare(std::string type, std::string variable);
std::string _declare(std::string type, std::string variable, std::string expr);

// Operators
std::string _add(std::string left, std::string right);
std::string _sub(std::string left, std::string right);
std::string _mul(std::string left, std::string right);
std::string _div(std::string left, std::string right);
std::string _mod(std::string left, std::string right);
std::string _inc(std::string object);

class Operand
{
public:
    Operand(operand_t& operand, uint32_t id);

    std::string name(void);
    
    std::string current(void);

    std::string first(void);
    std::string layout(void);
    std::string etype(void);
    std::string nelem(void);
    std::string ndim(void);
    std::string shape(void);
    std::string stride(void);
    std::string start(void);

    operand_t& operand_;

private:
    uint32_t id_;
};

class Kernel
{
public:
    Kernel(Plaid::Plaid& plaid, bohrium::core::Block& block);
    
    std::string generate_source(void);

private:
    std::string unpack_operands(void);
    std::string unpack_operand(uint32_t id);

    std::string head(void);
    std::string body(void);
    std::string foot(void);

    std::string args(void);
    std::string iterspace(void);

    Plaid::Plaid& plaid_;
    bohrium::core::Block& block_;
};

class Expr
{

};

class Loop
{
    Loop();
};

}}}}

#endif
