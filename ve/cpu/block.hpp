#ifndef __BH_VE_CPU_BLOCK
#define __BH_VE_CPU_BLOCK
#include <string>
#include <map>

#include "bh.h"
#include "tac.h"
#include "symbol_table.hpp"
#include "utils.hpp"

namespace bohrium{
namespace core{

class Block {
public:
    Block(SymbolTable& globals, std::vector<tac_t>& program);
    ~Block();

    /**
     *  Clear the composed block, does not de-allocate or remove
     *  the associated globals, program and memory allocated for the
     *  block.
     *  It simply clears it for reuse.
     */
    void clear(void);

    /**
     *  Compose the block based based on a subset of program instructions.
     *
     *  @param prg_begin Start of the subset.
     *  @param prg_end End of the subset, to and including prg_end.
     */
    void compose(size_t prg_begin, size_t prg_end);

    /**
     *  Compose the block based based on a subset of program instructions.
     *
     *  TODO: Document
     *
     *  @param prg_begin Start of the subset.
     *  @param prg_end End of the subset, to and including prg_end.
     */
    void compose(bh_ir_kernel& krnl);

    /**
     *  Return the block-local operand-index corresponding 
     *  to the given global operand-index.
     *
     *  @param global_idx The global operand-index to resolve
     *  @return The block-local operand-index
     */
    size_t global_to_local(size_t global_idx) const;

    /**
     *  Return the global operand-index corresponding 
     *  to the given local operand-index.
     *
     *  @param local_idx The local operand-index to resolve
     *  @return The global operand-index
     */
    size_t local_to_global(size_t local_idx) const;

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
    // Getters
    //

    /**
     *  Returns the operand with local in block-scope.
     *
     *  @param local_idx Index / name in the block-scope of the operand.
     *  @param A reference to the requested operand.
     */
    operand_t& operand(size_t local_idx);

    /**
     *  Return the array of pointer-operands.
     */
    operand_t** operands(void);

    /**
     * Count of operands in the block.
     */
    size_t noperands(void) const;

    /**
     * Return the tac-instance with the given index.
     */
    tac_t& tac(size_t idx) const;
    tac_t& array_tac(size_t idx) const;

    /**
     *  Count of tacs in the block.
     */
    size_t ntacs(void) const;

    /**
     *  Count of array-tacs in the block.
     */
    size_t narray_tacs(void) const;

    std::string symbol(void) const;
    std::string symbol_text(void) const;

    /**
     *  Returns the iteration space of the block.
     */
    iterspace_t& iterspace(void);

    /**
     *  Update the iteration space of the block.
     *
     *  This means determing the "dominating" LAYOUT, ndim, shape,
     *  and number of elements of an operation within the block.
     *
     *  That is choosing the "worst" LAYOUT, highest ndim, and then
     *  choosing the shape of the operand with chose characteristics.
     *
     *  Since this is what will be the upper-bounds used in when
     *  generating / specializing code, primarily for fusion / contraction.
     *
     *  NOTE: This should be done after applying array contraction or 
     *  any other changes to tacs and operands.
     */
    void update_iterspace(void);

    /**
     * Returns a textual representation of the block in dot-format.
     */
    std::string dot(void) const;

    /**
     * Returns a plaintext representation of the block.
     */
    std::string text(void);

    uint32_t omask(void);    

private:

    Block();

    SymbolTable& globals_;          // A reference to the global symbol table
    std::vector<tac_t>& program_;   // A reference to the entire bytecode program

    iterspace_t iterspace_;           // The iteration-space of the block

    std::vector<tac_t*> tacs_;      // A subset of the tac-program reprensting the block.
    std::vector<tac_t*> array_tacs_;// A subset of the tac-program containing only array ops.

    //SymbolTable locals_;             // A symbol table with block-scope
    operand_t** operands_;
    size_t noperands_;
    std::map<size_t, size_t> global_to_local_;  // Mapping from global to block-local scope.
    std::map<size_t, size_t> local_to_global_;  // Mapping from global to block-local scope.

    std::string symbol_text_;       // Textual representation of the block
    std::string symbol_;            // Hash of textual representation

    uint32_t omask_ = 0;

    static const char TAG[];
};

}}
#endif
