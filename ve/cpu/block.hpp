#ifndef __KP_CORE_BLOCK_HPP
#define __KP_CORE_BLOCK_HPP 1
#include <string>
#include <map>

#include "bh.h"
#include "kp_utils.h"
#include "program.hpp"
#include "symbol_table.hpp"
#include "utils.hpp"

namespace kp{
namespace core{

class Block {
public:
    Block(SymbolTable& globals, Program& tac_program);
    ~Block();

    /**
     *  Clear the composed block, does not de-allocate or remove
     *  the associated globals, program and memory allocated for the
     *  block.
     *  It simply clears it for reuse.
     */
    void clear(void);

    /**
     *  Compose the block based on a single program instruction.
     *
     *  This method is intended for SIJ-mode only.
     */
    void compose(bh_ir_kernel& krnl, size_t prg_idx);

    /**
     *  Compose the block base on every program instruction
     *  in the given kernel.
     *
     *  @param krnl Bhir kernel
     */
    void compose(bh_ir_kernel& krnl, bool array_contraction);

    /**
     *  Return the block-local kp_operand-index corresponding
     *  to the given global kp_operand-index.
     *
     *  @param global_idx The global kp_operand-index to resolve
     *  @return The block-local kp_operand-index
     */
    size_t global_to_local(size_t global_idx) const;

    /**
     *  Return the global kp_operand-index corresponding
     *  to the given local kp_operand-index.
     *
     *  @param local_idx The local kp_operand-index to resolve
     *  @return The global kp_operand-index
     */
    size_t local_to_global(size_t local_idx) const;

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
     *  Returns the kp_operand with local idx in block-scope.
     *
     *  @param local_idx Index / name in the block-scope of the kp_operand.
     *  @param A reference to the requested kp_operand.
     */
    kp_operand& operand(size_t local_idx);

    /**
     *  Return the array of pointer-operands.
     */
    kp_operand** operands(void);

    /**
     * Count of operands in the block.
     */
    int64_t noperands(void) const;

    /**
     *  Grab buffer with the given id.
     */
    kp_buffer& buffer(size_t buffer_id);

    /**
     *  Returns the buffer id for the provided buffer pointer.
     */
    size_t resolve_buffer(kp_buffer* buffer);

    /**
     *  Return the array of buffers
     */
    kp_buffer** buffers(void);

    /**
     *  Count of buffers in the block.
     */
    int64_t nbuffers();

    /**
     *  How many operands use this buffer.
     */
    size_t buffer_refcount(kp_buffer *buffer);

    /**
     * Return the tac-instance with the given index.
     */
    kp_tac & tac(size_t idx) const;
    kp_tac & array_tac(size_t idx) const;

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
    kp_iterspace& iterspace(void);

    /**
     * Returns a textual representation of the block in dot-format.
     */
    std::string dot(void) const;

    /**
     * Returns a plaintext representation of the block.
     */
    std::string text(void);

    /**
     * Returns a compact plaintext representation of the block.
     */
    std::string text_compact(void);

    uint32_t omask(void);    

private:

    Block();

    /**
     *  This is a helper for the two public compose methods.
     *
     *  Performing what both composition methods needs.
     *
     */
    void _compose(bh_ir_kernel& krnl, bool array_contraction, size_t prg_idx);

    /**
     *  Create an kp_operand with block scope based on the kp_operand in global scope.
     *
     *  Reuses operands of equivalent meta-data.
     *
     *  @param global_idx Global index of the kp_operand to add to block scope.
     *
     *  @returns The block-scoped index.
     */
    size_t _localize_scope(size_t global_idx);

    /**
     *  Determine use of buffer usage within block.
     *
     */
    void _bufferize(size_t local_idx);

    /**
     *  Update the iteration space of the block.
     *
     *  This means determing the "dominating" KP_LAYOUT, ndim, shape,
     *  and number of elements of an operation within the block.
     *
     *  That is choosing the "worst" KP_LAYOUT, highest ndim, and then
     *  choosing the shape of the kp_operand with chose characteristics.
     *
     *  Since this is what will be the upper-bounds used in when
     *  generating / specializing code, primarily for fusion / contraction.
     *
     *  NOTE: This should be done after applying array contraction or 
     *  any other changes to tacs and operands.
     */
    void _update_iterspace(void);

    kp_block block_;                            // Buffers, operands, iterspace,  omask,
                                                // tacs, ntacs, array_tacs, and narray_tacs.

    SymbolTable& globals_;                      // A reference to the global symbol table
    Program& tac_program_;                      // A reference to the entire bytecode program

    std::map<kp_buffer*, size_t> buffer_ids_;
    std::set<kp_buffer*> input_buffers_;
    std::set<kp_buffer*> output_buffers_;
    std::map<kp_buffer*, std::set<uint64_t>> buffer_refs_;

    std::map<size_t, size_t> global_to_local_;  // Mapping from global to block-local scope.
    std::map<size_t, size_t> local_to_global_;  // Mapping from block-local to global scope.

    std::string symbol_text_;                   // Textual representation of the block
    std::string symbol_;                        // Hash of textual representation
       
    static const char TAG[];
};

}}

#endif
