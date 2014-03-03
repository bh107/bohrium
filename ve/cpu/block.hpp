#ifndef __BH_VE_CPU_BLOCK
#define __BH_VE_CPU_BLOCK
#include <string>

#include "bh.h"
#include "tac.h"
#include "utils.hpp"

namespace bohrium{
namespace engine{
namespace cpu{

class Block {
public:
    Block(bh_ir& ir, bh_dag& dag);
    ~Block();

    std::string text(std::string prefix);
    std::string text();

    std::string scope_text(std::string prefix);
    std::string scope_text();

    bool compose();
    bool symbolize(const bool optimized);    

    bh_instruction** instr;     // Pointers to instructions

    tac_t* program;             // Ordered list of TACs
    operand_t* scope;           // Array of block arguments

    size_t noperands;           // Number of arguments to the block
    size_t length;                 // Number of tacs in program
    uint32_t omask;             // Mask of the OPERATIONS in the block

    std::string symbol_text;    // Textual representation of the block
    std::string symbol;         // Hash of textual representation

private:
    size_t add_operand(bh_instruction& instr, size_t operand_idx);
    bh_ir& ir;
    bh_dag& dag;
};

}}}
#endif
