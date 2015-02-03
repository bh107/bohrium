#ifndef __BH_VE_CPU_CODEGEN
#define __BH_VE_CPU_CODEGEN

#include <string>
#include <map>
#include <tac.h>

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

class Operand
{
public:
    Operand(operand_t operand, uint32_t id);

    std::string name(void);
    std::string first(void);
    std::string current(void);
    
    // Accessors
    LAYOUT layout(void);
    ETYPE etype(void);
private:
    operand_t operand_;
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
