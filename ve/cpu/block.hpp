#ifndef __BH_VE_CPU_BLOCK
#define __BH_VE_CPU_BLOCK
#include <string>
#include <map>

#include "bh.h"
#include "tac.h"
#include "symbol_table.hpp"
#include "utils.hpp"
#include "dag.hpp"

namespace bohrium{
namespace core{

class Block {
public:
    Block(SymbolTable& globals, std::vector<tac_t>& program);
    ~Block();

    /**
     *  Compose a block of tacs in a legal execution order, with a
     *  block-scoped symbol table and construct a symbol representing the block.
     *
     *  NOTE: This will reset the current state of the block.
     */
    bool compose(Graph& subgraph);

    /**
     *  Return the block-local operand-index corresponding 
     *  to the given global operand-index.
     *
     *  @param global_idx The global operand-index to resolve
     *  @return The block-local operand-index
     */
    size_t global_to_local(size_t global_idx) const;

    /**
     *  Create an operand with block scope based on the operand in global scope.
     *
     *  Reuses operands of equivalent meta-data.
     *
     *  @param global_idx Global index of the operand to add to block scope.
     *
     *  @returns The block-scoped index.
     */
    size_t localize(size_t global_idx);

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

    //
    // Various getters
    //

    /**
     *  Returns the operand with local in block-scope.
     *
     *  @param local_idx Index / name in the block-scope of the operand.
     *  @param A reference to the requested operand.
     */
    operand_t& operand(size_t local_idx);

    operand_t* operands(void);
    size_t noperands(void);

    /**
     *  Return the operation mask of the tacs in the block.
     */
    uint32_t omask(void) const;

    /**
     * Return the tac-instance with the given index.
     */
    tac_t& tac(size_t idx) const;

    size_t size(void) const;

    std::string symbol(void) const;
    std::string symbol_text(void) const;

private:

    Block();

    SymbolTable& globals_;          // A reference to the global symbol table
    std::vector<tac_t>& program_;   // A reference to the entire bytecode program

    std::vector<tac_t*> tacs_;      // A subset of the tac-program reprensting the block.

    SymbolTable locals_;             // A symbol table with block-scope
    std::map<size_t, size_t> global_to_local_;  // Mapping from global to block-local scope.

    std::string symbol_text_;       // Textual representation of the block
    std::string symbol_;            // Hash of textual representation

    uint32_t omask_;                // Mask of the OPERATIONS in the block

    static const char TAG[];
};

}}
#endif
