#ifndef __BH_INSTRUCTION_H
#define __BH_INSTRUCTION_H

#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array.hpp>
#include "bh_opcode.h"
#include "bh_array.h"
#include "bh_error.h"

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

// Maximum number of operands in a instruction.
#define BH_MAX_NO_OPERANDS (3)

//Memory layout of the Bohrium instruction
typedef struct
{
    //Opcode: Identifies the operation
    bh_opcode  opcode;
    //Id of each operand
    bh_view  operand[BH_MAX_NO_OPERANDS];
    //Constant included in the instruction (Used if one of the operands == NULL)
    bh_constant constant;

protected:
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
} bh_instruction;

BOOST_IS_BITWISE_SERIALIZABLE(bh_constant)

#endif