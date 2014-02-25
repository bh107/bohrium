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

    bh_instruction** instr;     // Pointers to instructions

    tac_t* program;             // Ordered list of TACs
    operand_t* scope;           // Array of block arguments

    uint32_t noperands;         // Number of arguments to the block
    int length;                 // Number of tacs in program
    uint32_t omask;             // Mask of the OPERATIONS in the block
    std::string symbol;         // Textual representation of the block

    Block(bh_ir& ir, bh_dag& dag);
    ~Block();

    std::string text();

private:
    bool symbolize(int64_t const optimized);
    uint32_t add_operand(bh_instruction& instr, int operand_idx);
};

}}}
#endif