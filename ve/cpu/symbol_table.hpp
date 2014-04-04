#ifndef __BH_VE_CPU_SYMBOLTABLE
#define __BH_VE_CPU_SYMBOLTABLE
#include "bh.h"
#include "tac.h"
#include <string>
#include <set>

namespace bohrium {
namespace engine {
namespace cpu {

/**
 *  Maintains a symbol table for tac_t operands (operand_t).
 *
 *  The symbol table relies on and uses bh_instructions, this is
 *  to ensure compability with the rest of Bohrium.
 *  The symbols and the operand_t provides identification of operands
 *  as well as auxilary information.
 *
 *  Populating the symbol table is therefore done by "mapping"
 *  a bh_instruction operand TO a symbol represented as an operand_t.
 *
 *  In the future this could be changed to actually be self-contained
 *  by replacing bh_instruction with tac_t. And instead of "mapping"
 *  one could "add" operands to the table.
 *
 */
class SymbolTable {
public:

    /**
     *  Construct a symbol table with a capacity for 100 symbols.
     */
    SymbolTable(void);
    
    /**
     *  Construct a symbol table a capacity of n elements.
     *
     *  @param n Upper bound on the amount of symbols in the table.
     */
    SymbolTable(size_t n);

    /**
     *  Deconstructor.
     */
    ~SymbolTable(void);

    /**
     *  Return the number of operands which can be stored in the symbol table.
     */
    size_t capacity(void);

    /**
     *  Return the current amount of operands stored in the symbol table.
     */
    size_t size(void);

    /**
     * Create a textual representation of the table.
     */
    std::string text(void);
    
    /**
     *  Return a textual representation of meta-data; size, capacity, etc.
     */
    std::string text_meta(void);

    /**
     * Create a textual representation of the table, using a prefix.
     */
    std::string text(std::string prefix);

    operand_t& operator[](size_t operand_idx);

    /**
     *  Add instruction operand as argument to block.
     *
     *  Reuses operands of equivalent meta-data.
     *
     *  @param instr        The instruction whos operand should be converted.
     *  @param operand_idx  Index of the operand to represent as arg_t
     *  @param block        The block in which scope the argument will exist.
     */
    size_t map_operand(bh_instruction& instr, size_t operand_idx);
    
    /**
     * Maintain records of how many times an operand has been read, written,
     * and whether it is potentially a temporary operand.
     *
     * "Potentially Temporary" are operands which are subject to the FREE operator.
     * "Temporary" operands, are operands which within their life-time, that means
     * up until they are subject to FREE have the state:
     *
     * (reads[operand_symbol] == writes[operand_symbol] == 1)
     *
     * NOTE:
     * When a (in1 == in2) for binary operators then it only counts as a single "read".
     */
    void ref_count(const tac_t& tac);

    /**
     *  Turn an array operand into a scalar.
     */
    void turn_scalar(size_t symbol_idx);

    operand_t* table;   // The actual symbol-table

    //
    // The following are used to detect temporary arrays
    //
    std::set<size_t> disqualified;   // Operands which could be temps
    std::set<size_t> freed;         // Operands which are freed
    std::set<size_t> temps;         // Operands which are temps

    size_t* reads;              // Read-count of operands
    size_t* writes;             // Write-cout of operands

private:
    /**
     *  Initialization function used by constructors.
     */
    void init(void);

    size_t reserved;    // Capacity reserved
    size_t nsymbols;    // The current number of symbols in the table

};

}}}

#endif
