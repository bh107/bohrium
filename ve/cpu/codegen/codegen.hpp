#ifndef __BH_VE_CPU_CODEGEN
#define __BH_VE_CPU_CODEGEN

#include <string>
#include <map>
#include <tac.h>
#include <block.hpp>

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

std::string ref(std::string object);
std::string deref(std::string object);
std::string ptr_type(std::string object);
std::string const_type(std::string object);
std::string assert_not_null(std::string object);
std::string declare_var(std::string object);
std::string declare_ptr_var(std::string object);
std::string index(std::string object, int64_t idx);
std::string end(void);

class Operand
{
public:
    Operand(operand_t operand, uint32_t id);

    std::string name(void);
    std::string first(void);
    std::string current(void);
    
    std::string layout(void);
    std::string etype(void);

    std::string nelem(void);
    std::string ndim(void);
    std::string shape(void);
    std::string stride(void);

    operand_t operand_;

private:
    uint32_t id_;
};

class Kernel
{
public:
    Kernel();
    std::string unpack_operands(void);
    std::string unpack_operand(uint32_t id);

    void add_operand(Operand& operand, uint32_t id);
    
private:
    std::map<uint32_t, Operand*> operands_;
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
