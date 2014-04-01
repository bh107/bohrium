#ifndef __BH_VE_CPU_SYMBOLTABLE
#define __BH_VE_CPU_SYMBOLTABLE
#include "bh.h"
#include "tac.h"
#include <string>

namespace bohrium {
namespace engine {
namespace cpu {

class SymbolTable {
public:
    SymbolTable(void);
    SymbolTable(size_t capacity);

    ~SymbolTable(void);

    std::string text(void);
    std::string text(std::string prefix);

    size_t map_operand(bh_instruction& instr, size_t operand_idx);

private:
    void init(size_t capacity);

    size_t nsymbols;
    operand_t* table;

};

}}}

#endif
