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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/foreach.hpp>
#include <bh.h>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "bh_ir.h"
#include "bh_fuse.h"

using namespace std;
using namespace boost;
namespace io = boost::iostreams;

/* Creates a Bohrium Internal Representation (BhIR) from a instruction list.
*
* @ninstr      Number of instructions
* @instr_list  The instruction list
*/
bh_ir::bh_ir(bh_intp ninstr, const bh_instruction instr_list[])
{
    bh_ir::instr_list = vector<bh_instruction>(instr_list, &instr_list[ninstr]);
}

/* Creates a BhIR from a serialized BhIR.
*
* @bhir The BhIr serialized as a char array or vector
*/
bh_ir::bh_ir(const char bhir[], bh_intp size)
{
    io::basic_array_source<char> source(bhir,size);
    io::stream<io::basic_array_source <char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);
    ia >> *this;
}

/* Serialize the BhIR object into a char buffer
*  (use the bh_ir constructor for deserialization)
*
*  @buffer   The char vector to serialize into
*/
void bh_ir::serialize(vector<char> &buffer) const
{
    io::stream<io::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);
    oa << *this;
    output_stream.flush();
}

/* Returns the cost of the BhIR */
uint64_t bh_ir::cost() const
{
    uint64_t sum = 0;
    BOOST_FOREACH(const bh_ir_kernel &k, kernel_list)
    {
        sum += k.cost();
    }
    return sum;
}

/* Pretty print the kernel list */
void bh_ir::pprint_kernel_list() const
{
    char msg[100]; int i=0;
    BOOST_FOREACH(const bh_ir_kernel &k, kernel_list)
    {
        snprintf(msg, 100, "kernel-%d", i++);
        // TODO: Change needed when switching kernel-instruction representation
        bh_pprint_instr_list(&k.get_instrs()[0], k.get_instrs().size(), msg);
    }
}

/* Returns the cost of a bh_view */
inline static uint64_t cost_of_view(const bh_view &v)
{
    return bh_nelements_nbcast(&v) * bh_type_size(v.base->type);
}

/* Returns the cost of the kernel */
uint64_t bh_ir_kernel::cost() const
{
    uint64_t sum = 0;
    BOOST_FOREACH(const bh_view &v, input_list())
    {
        sum += cost_of_view(v);
    }
    BOOST_FOREACH(const bh_view &v, output_list())
    {
        sum += cost_of_view(v);
    }
    return sum;
}

/* Add an instruction to the kernel
 *
 * @instr   The instruction to add
 * @return  The boolean answer
 */
void bh_ir_kernel::add_instr(const bh_instruction &instr)
{
    if(instr.opcode == BH_DISCARD)
    {
        const bh_base *base = instr.operand[0].base;
        for(vector<bh_view>::iterator it=outputs.begin(); it != outputs.end(); ++it)
        {
            if(base == it->base)
            {
                temps.push_back(base);
                outputs.erase(it);
                break;
            }
        }
    }
    else if(instr.opcode != BH_FREE)
    {
        {
            bool duplicates = false;
            const bh_view &v = instr.operand[0];
            BOOST_FOREACH(const bh_view &i, outputs)
            {
                if(bh_view_aligned(&v, &i))
                {
                    duplicates = true;
                    break;
                }
            }
            if(!duplicates)
                outputs.push_back(v);
        }
        const int nop = bh_operands(instr.opcode);
        for(int i=1; i<nop; ++i)
        {
            const bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
                continue;

            bool duplicates = false;
            BOOST_FOREACH(const bh_view &i, inputs)
            {
                if(bh_view_aligned(&v, &i))
                {
                    duplicates = true;
                    break;
                }
            }
            if(duplicates)
                continue;

            bool local_source = false;
            BOOST_FOREACH(const bh_instruction &i, instrs)
            {
                if(bh_view_aligned(&v, &i.operand[0]))
                {
                    local_source = true;
                    break;
                }
            }
            if(!local_source)
                inputs.push_back(v);
        }
    }
    instrs.push_back(instr);
};

/* Determines whether this kernel depends on 'other',
 * which is true when:
 *      'other' writes to an array that 'this' access
 *                        or
 *      'this' writes to an array that 'other' access
 *
 * @other The other kernel
 * @return The boolean answer
 */
bool bh_ir_kernel::dependency(const bh_ir_kernel &other) const
{
    // TODO: Change needed when switching kernel-instruction representation
    BOOST_FOREACH(const bh_instruction &i, get_instrs())
    {
        BOOST_FOREACH(const bh_instruction &o, other.get_instrs())
        {
            if(bh_instr_dependency(&i, &o))
                return true;
        }
    }
    return false;
}

