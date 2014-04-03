#ifndef __BH_VE_CPU_BLOCK
#define __BH_VE_CPU_BLOCK
#include <string>
#include <map>

#include "bh.h"
#include "tac.h"
#include "symbol_table.hpp"
#include "utils.hpp"

namespace bohrium{
namespace engine{
namespace cpu{

class Block {
public:
    Block(SymbolTable& symbol_table, const bh_ir& ir, size_t dag_idx);
    ~Block();

    std::string text(std::string prefix);
    std::string text(void);

    std::string scope_text(std::string prefix);
    std::string scope_text();

    bool compose();
    bool compose(bh_intp node_start, bh_intp node_end);

    /**
     *  Create a symbol for the block.
     *
     *  The textual version of the  symbol looks something like::
     *  
     *  symbol_text = ZIP-ADD-2D~1~2~3_~1Cf~2Cf~3Cf
     *
     *  Which will be hashed to some uint32_t value::
     *
     *  symbol = 2111321312412321432424
     *
     *  NOTE: System and extension operations are ignored.
     *        If a block consists of nothing but system and/or extension
     *        opcodes then the symbol will be the empty string "".
     */
    bool symbolize();    
    bool symbolize(size_t tac_start, size_t tac_end);

    bh_instruction** instr;     // Pointers to instructions

    tac_t* program;             // Ordered list of TACs
    operand_t** scope;          // Array of pointers to block operands

    size_t noperands;           // Number of arguments to the block
    size_t length;              // Number of tacs in program
    uint32_t omask;             // Mask of the OPERATIONS in the block

    std::string symbol_text;    // Textual representation of the block
    std::string symbol;         // Hash of textual representation

    const bh_dag& get_dag(void);

    std::map<size_t, size_t> operand_map; // Mapping of tac-operands to block-scope

    SymbolTable& symbol_table;

private:

    /**
     *  Add instruction operand as argument to block.
     *
     *  Reuses operands of equivalent meta-data.
     *
     *  @param instr        The instruction whos operand should be converted.
     *  @param operand_idx  Index of the operand to represent as arg_t
     *  @param block        The block in which scope the argument will exist.
     *
     *  @returns The symbol for the operand
     */
    size_t add_operand(bh_instruction& instr, size_t operand_idx);
    
    const bh_ir& ir;
    const bh_dag& dag;

    static const char TAG[];
};

}}}
#endif
