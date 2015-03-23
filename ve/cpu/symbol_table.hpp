#ifndef __BH_CORE_SYMBOLTABLE
#define __BH_CORE_SYMBOLTABLE
#include "bh.h"
#include "tac.h"
#include <string>
#include <set>

namespace bohrium {
namespace core {

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
     *  Import the given operand into the symbol_table.
     *  This is used for copying operands between symbol tables.
     */
    size_t import(operand_t& operand);

    /**
     *  Add instruction operand as argument to block.
     *
     *  Reuses operands of equivalent meta-data.
     *  NOTE: When mapping "BH_R123" a single operand is mapped to
     *        to two operands. In such cases the tac will be mapped like::
     *
     *        in1 = arg_idx-1
     *        in2 = arg_idx
     *
     *  @param instr        The instruction whos operand should be converted.
     *  @param operand_idx  Index of the operand to represent as arg_t
     *  @return arg_idx     Index of the operand in symbol table
     */
    size_t map_operand(bh_instruction& instr, size_t operand_idx);

    /**
     *  Turn an array operand into a contractable array.
     */
    void turn_contractable(size_t operand_idx);

    /**
     *  Reset refcounted / temp information and operands.
     */
    void clear(void);

    /**
     *  Return a reference to the operand with operand_idx.
     */
    operand_t& operator[](size_t operand_idx);

    operand_t* operands(void);

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

private:
    SymbolTable(void);  // We do not want to be able to create a symbol_table
                        // with assumptions on capacity.

    operand_t* table_;           // The actual symbol-table
    size_t capacity_;    // Capacity reserved
    size_t nsymbols_;    // The current number of symbols in the table

    static const char TAG[];
};

}}

#endif
