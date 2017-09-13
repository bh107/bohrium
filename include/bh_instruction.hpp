#ifndef __BH_INSTRUCTION_H
#define __BH_INSTRUCTION_H

#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>
#include <set>

#include "bh_opcode.h"
#include <bh_view.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

// Maximum number of operands in a instruction.
#define BH_MAX_NO_OPERANDS (3)

//Memory layout of the Bohrium instruction
struct bh_instruction
{
    // Opcode: Identifies the operation
    bh_opcode  opcode;
    // Id of each operand
    std::vector<bh_view> operand;
    // Constant included in the instruction (Used if one of the operands == NULL)
    bh_constant constant;
    // Flag that indicates whether this instruction construct the output array (i.e. is the first operation on that array)
    // For now, this flag is only used by the code generators.
    bool constructor;
    // An identifier to track the original source of instruction transformations thus transformations such as
    // copy, transpose, and reshape does not change the 'origin_id'.
    // For now, this flag is only used by the code generators.
    int64_t origin_id = -1; // -1 indicates: unset

    // Constructors
    bh_instruction(){}
    bh_instruction(bh_opcode opcode, const std::vector<bh_view> &operands) : opcode(opcode), operand(operands) {}
    bh_instruction(const bh_instruction& instr)
    {
        opcode      = instr.opcode;
        constant    = instr.constant;
        constructor = instr.constructor;
        origin_id   = instr.origin_id;
        operand     = instr.operand;
    }

    // Return a set of all bases used by the instruction
    std::set<const bh_base *> get_bases_const() const;
    std::set<bh_base *> get_bases();

    // Return a vector of views in this instruction.
    // The first element is the output and the rest are inputs (the constant is ignored)
    std::vector<const bh_view*> get_views() const;

    // Returns true when one of the operands of 'instr' is a constant
    bool has_constant() const {
        for (const bh_view &v: operand) {
            if (bh_is_constant(&v)) {
                return true;
            }
        }
        return false;
    }

    // Check if all views in this instruction is contiguous
    bool is_contiguous() const;

    // Check if all view in this instruction have the same shape
    bool all_same_shape() const;

    // Is this instruction (and all its views) reshapable?
    bool reshapable() const;

    // Returns the principal shape of this instructions, which is the shape of the computation that constitute
    // this instruction. E.g. in reduce, this function returns the shape of the reduced array
    std::vector<int64_t> shape() const;

    // Returns the principal number of dimension of this instruction, which is the number of dimension of the
    // computation that constitute this instruction
    int64_t ndim() const;

    // Returns the axis this instruction reduces over or 'BH_MAXDIM' if 'instr' isn't a reduction
    int sweep_axis() const;

    // Reshape the views of the instruction to 'shape'
    void reshape(const std::vector<int64_t> &shape);

    // Reshape the views of the instruction to 'shape' (no checks!)
    void reshape_force(const std::vector<int64_t> &shape);

    // Remove 'axis' from all views in this instruction.
    // Notice that 'axis' is based on the 'dominating shape' thus remove_axis() will correct
    // the axis value when handling reductions automatically
    void remove_axis(int64_t axis);

    // Transposes by swapping the two axes 'axis1' and 'axis2'
    void transpose(int64_t axis1, int64_t axis2);

    // Returns the type of the operand at given index (support constants)
    bh_type operand_type(int operand_index) const;

    // Returns a pretty print of this instruction (as a string)
    std::string pprint(bool python_notation = true) const;

    // Equality
    bool operator==(const bh_instruction& other) const {
        if (opcode != other.opcode) {
            return false;
        }
        for (size_t i = 0; i < operand.size(); ++i) {
            if (bh_is_constant(&operand[i]) xor bh_is_constant(&other.operand[i])) {
                return false;
            } else if (bh_is_constant(&operand[i])) { // Both are constant
                if (constant != other.constant)
                    return false;
            } else {
                if (operand[i] != other.operand[i])
                    return false;
            }
        }
        return true;
    }

    // Inequality
    bool operator!=(const bh_instruction& other) const {
        return !(*this == other);
    }

    // Serialization using Boost
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & opcode;
        ar & operand;
        //We use make_array as a hack to make bh_constant BOOST_IS_BITWISE_SERIALIZABLE
        ar & boost::serialization::make_array(&constant, 1);
    }
};
BOOST_IS_BITWISE_SERIALIZABLE(bh_constant)

//Implements pprint of an instruction
DLLEXPORT std::ostream& operator<<(std::ostream& out, const bh_instruction& instr);

/* Retrive the operands of a instruction.
 *
 * @instruction  The instruction in question
 * @return The operand list
 */
DLLEXPORT bh_view *bh_inst_operands(bh_instruction *instruction);

/* Determines whether instruction 'a' depends on instruction 'b',
 * which is true when:
 *      'b' writes to an array that 'a' access
 *                        or
 *      'a' writes to an array that 'b' access
 *
 * @a The first instruction
 * @b The second instruction
 * @return The boolean answer
 */
DLLEXPORT bool bh_instr_dependency(const bh_instruction *a, const bh_instruction *b);


#endif
