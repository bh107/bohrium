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

#include "bh_type.h"
#include "bh_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The Bohrium Internal Representation (BhIR) represents an instruction
 * batch created by the Bridge component typically. */
class bh_ir
{
public:
    /* Constructs a Bohrium Internal Representation (BhIR)
     * from a instruction list.
     *
     * @ninstr      Number of instructions
     * @instr_list  The instruction list
     */
    bh_ir(bh_intp ninstr, const bh_instruction instr_list[]);

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
    void serialize(std::vector<char> &buffer);

    //The list of Bohrium instructions
    std::vector<bh_instruction> instr_list;
};

#ifdef __cplusplus
}
#endif

// The serialization code needs to be outside of extern "C".
namespace boost {
namespace serialization {
    // Serialize the BhIR object
    // Using the Boost serialization see:
    // <http://www.boost.org/doc/libs/1_47_0/libs/serialization/doc/tutorial.html#nonintrusiveversion>
    template<class Archive>
    void serialize(Archive &ar, bh_ir &bhir, const unsigned int version)
    {
        ar & bhir.instr_list;
    }
} // namespace serialization
} // namespace boost
#endif

