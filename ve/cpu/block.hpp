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

    std::string text(std::string prefix) const;
    std::string text(void) const;

    std::string scope_text(std::string prefix) const;
    std::string scope_text() const;

    /**
     *  Returns the operand with opr_idx in block-scope.
     *
     *  @param opr_idx Index / name in the block-scope of the operand.
     *  @param A reference to the requested operand.
     */
    const operand_t& scope(size_t operand_idx) const;

    /**
     *  Return the operand_idx in block-scope correspnding to the given symbol index.
     *
     *  @param symbol_idx The symbol_idx to resolve
     *  @return Operand index in block scope
     */
    size_t resolve(size_t symbol_idx) const;

    /**
     *  Return the operand correponding to the given symbol_idx.
     *  The operand is fetched from the symbol_table.
     */
    operand_t& operand(size_t symbol_idx) const;

    tac_t& program(size_t pc) const;

    size_t size(void) const;

    /**
     *  Return the operation mask of the tacs in the block.
     */
    uint32_t omask(void) const;

    bh_instruction& instr(size_t instr_idx) const;

    operand_t** operands(void) const;
    size_t noperands(void) const;

    std::string symbol(void) const;
    std::string symbol_text(void) const;

    /**
     *  Return the dag on which the block is based.
     */
    const bh_dag& get_dag(void);

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
    bool symbolize(void);
    bool symbolize(size_t tac_start, size_t tac_end);

private:



    bh_instruction** instr_;     // Pointers to instructions
    operand_t** operands_;       // Array of pointers to block operands
    size_t noperands_;           // Number of arguments to the block

    std::string symbol_text_;               // Textual representation of the block
    std::string symbol_;                    // Hash of textual representation

    uint32_t omask_;                        // Mask of the OPERATIONS in the block

    tac_t* tacs;                            // Ordered list of TACs
    size_t ntacs_;                          // Number of tacs in program
    std::map<size_t, size_t> operand_map;   // Mapping of tac operands to block-scope

    const bh_ir& ir;
    const bh_dag& dag;
    SymbolTable& symbol_table;

    static const char TAG[];
};

}}}
#endif
