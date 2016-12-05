/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __BH_IR_H
#define __BH_IR_H

#include <vector>
#include <map>
#include <set>
#include <boost/serialization/vector.hpp>

#include <bh_instruction.hpp>
#include <bh_seqset.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

/* The Bohrium Internal Representation (BhIR) represents an instruction
 * batch created by the Bridge component typically. */
class bh_ir
{
public:
    bh_ir(){};
    /* Constructs a Bohrium Internal Representation (BhIR)
     * from a instruction list.
     *
     * @ninstr      Number of instructions
     * @instr_list  The instruction list
     */
    bh_ir(bh_intp ninstr, const bh_instruction instr_list[]);

    // Special constructor for 1 instruction
    // Used by the GPU to send instructions to the CPU
    bh_ir(const bh_instruction& instr);

    /* Constructs a BhIR from a serialized BhIR.
    *
    * @bhir The BhIR serialized as a char array
    */
    bh_ir(const char bhir[], bh_intp size);

    /* Serialize the BhIR object into a char buffer
    *  (use the bh_ir constructor above to deserialization)
    *
    *  @buffer   The char vector to serialize into
    */
    void serialize(std::vector<char> &buffer) const;

    //The list of Bohrium instructions in topological order
    std::vector<bh_instruction> instr_list;

    // Should the ve tally after this bh_ir
    bool tally;

protected:
    // Serialization using Boost
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & instr_list;
    }
};

#endif
