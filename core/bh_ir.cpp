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
        bh_pprint_instr_krnl(&k, msg);
    }
}

void bh_ir_kernel::add_instr(uint64_t instr_idx)
{

    /* Help function that checks if aligned view 'v' exist in 'views' */
    struct
    {
        bool operator()(const bh_view &v, const vector<bh_view> &views)
        {
            BOOST_FOREACH(const bh_view &i, views)
            {
                if(bh_view_aligned(&v, &i))
                    return true;
            }
            return false;
        }
    }aligned_view_exist;

    const bh_instruction& instr = bhir->instr_list[instr_idx];

    if(instr.opcode == BH_DISCARD)
    {
        //When discarding we might have to remove arrays from 'outputs' and 'temps'
        const bh_base *base = instr.operand[0].base;
        for(vector<bh_view>::iterator it=outputs.begin(); it != outputs.end(); ++it)
        {
            if(base == it->base)
            {
                outputs.erase(it);

                //If the discarded array isn't in 'inputs' (and not in 'outputs')
                //than it is a temp array
                if(not aligned_view_exist(*it, inputs))
                   temps.push_back(base);
                break;
            }
        }
    }
    else if(instr.opcode != BH_FREE)
    {
        //Add the output of the instruction to 'outputs'
        if(not aligned_view_exist(instr.operand[0], outputs))
            outputs.push_back(instr.operand[0]);

        //Add the inputs of the instruction to 'inputs'
        const int nop = bh_operands(instr.opcode);
        for(int i=1; i<nop; ++i)
        {
            const bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
                continue;

            //If 'v' is in 'inputs' already we can continue
            if(aligned_view_exist(v, inputs))
                continue;

            //Additionally, we shouldn't add 'v' to 'inputs' if it is
            //the output of an existing instruction.
            bool local_source = false;
            BOOST_FOREACH(uint64_t idx, instr_indexes)
            {
                if(bh_view_aligned(&v, &bhir->instr_list[idx].operand[0]))
                {
                    local_source = true;
                    break;
                }
            }
            if(!local_source)
                inputs.push_back(v);
        }
    }
    instr_indexes.push_back(instr_idx);
};

/* Determines whether all instructions in 'this' kernel
 * are system opcodes (e.g. BH_DISCARD, BH_FREE, etc.)
 *
 * @return The boolean answer
 */
bool bh_ir_kernel::only_system_opcodes() const
{
    BOOST_FOREACH(uint64_t this_idx, instr_indexes)
    {
        if(not bh_opcode_is_system(bhir->instr_list[this_idx].opcode))
            return false;
    }
    return true;
}

/* Determines whether the kernel fusible legal
 *
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible() const
{
    for(uint64_t i=0; i<instr_indexes.size(); ++i)
    {
        const bh_instruction *instr = &bhir->instr_list[instr_indexes[i]];
        for(uint64_t j=i+1; j<instr_indexes.size(); ++j)
        {
            if(not bohrium::check_fusible(instr, &bhir->instr_list[instr_indexes[j]]))
                return false;
        }
    }
    return true;
}

/* Determines whether it is legal to fuse with the instruction
 *
 * @instr_idx  The index of the instruction
 * @return     The boolean answer
 */
bool bh_ir_kernel::fusible(uint64_t instr_idx) const
{
    const bh_instruction *instr = &bhir->instr_list[instr_idx];
    BOOST_FOREACH(uint64_t i, instr_indexes)
    {
        if(not bohrium::check_fusible(instr, &bhir->instr_list[i]))
            return false;
    }
    return true;
}

/* Determines whether it is legal to fuse with the kernel
 *
 * @other  The other kernel
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible(const bh_ir_kernel &other) const
{
    BOOST_FOREACH(uint64_t idx1, instr_indexes)
    {
        const bh_instruction *instr = &bhir->instr_list[idx1];
        BOOST_FOREACH(uint64_t idx2, other.instr_indexes)
        {
            if(not bohrium::check_fusible(instr, &other.bhir->instr_list[idx2]))
                return false;
        }
    }
    return true;
}

/* Determines whether it is legal to fuse with the instruction
 * without changing 'this' kernel's dependencies.
 *
 * @instr_idx  The index of the instruction
 * @return     The boolean answer
 */
bool bh_ir_kernel::fusible_gently(uint64_t instr_idx) const
{
    const bh_instruction &instr = bhir->instr_list[instr_idx];

    //Check that 'instr' is gentle fusible with at least one existing instruction
    //that is not a system opcode (unless all instructions in 'this' kernel are
    //system opcodes)
    if(only_system_opcodes())
    {
        BOOST_FOREACH(uint64_t this_idx, instr_indexes)
        {
            const bh_instruction &this_instr = bhir->instr_list[this_idx];
            if(bh_instr_fusible_gently(&instr, &this_instr) &&
               bohrium::check_fusible(&instr, &this_instr))
                return true;
        }
    }
    else
    {
        BOOST_FOREACH(uint64_t this_idx, instr_indexes)
        {
            const bh_instruction &this_instr = bhir->instr_list[this_idx];
            if(bh_opcode_is_system(this_instr.opcode))
                continue;

            if(bh_instr_fusible_gently(&instr, &this_instr) &&
               bohrium::check_fusible(&instr, &this_instr))
                return true;
        }
    }
    return false;
}

/* Determines whether it is legal to fuse with the kernel without
 * changing 'this' kernel's dependencies.
 *
 * @other  The other kernel
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible_gently(const bh_ir_kernel &other) const
{

    //When all instructions in the 'other' kernel are system opcodes
    //only one of the instructions needs to be gentle fusible.
    if(other.only_system_opcodes())
    {
        BOOST_FOREACH(uint64_t other_idx, other.instr_indexes)
        {
            if(fusible_gently(other_idx))
                return true;
        }
        return false;
    }
    else
    {
        //Check that each instruction in 'other' is gentle fusible
        //with 'this' kernel while ignoring system opcodes.
        BOOST_FOREACH(uint64_t other_idx, other.instr_indexes)
        {
            if(bh_opcode_is_system(bhir->instr_list[other_idx].opcode))
                continue;
            if(not fusible_gently(other_idx))
                return false;
        }
    }
    return true;
}

/* Determines dependency between 'this' kernel and the instruction 'instr',
 * which is true when:
 *      'instr' writes to an array that 'this' access
 *                        or
 *      'this' writes to an array that 'instr' access
 *
 * @instr_idx  The index of the instruction
 * @return     0: no dependency
 *             1: this kernel depend on 'instr'
 *            -1: 'instr' depend on this kernel
 */
int bh_ir_kernel::dependency(uint64_t instr_idx) const
{
    int ret = 0;
    BOOST_FOREACH(uint64_t this_idx, instr_indexes)
    {
        if(bh_instr_dependency(&bhir->instr_list[instr_idx],
                               &bhir->instr_list[this_idx]))
        {
            if(this_idx >= instr_idx)
            {
                assert(ret == 0 or ret == 1);//Check for cyclic dependency
                ret = 1;
                //TODO: return 'ret' here, but for now we check all instructions
            }
            else
            {
                assert(ret == 0 or ret == -1);//Check for cyclic dependency
                ret = -1;
                //TODO: return 'ret' here, but for now we check all instructions
            }
        }
    }
    return ret;
}

/* Determines dependency between this kernel and 'other',
 * which is true when:
 *      'other' writes to an array that 'this' access
 *                        or
 *      'this' writes to an array that 'other' access
 *
 * @other    The other kernel
 * @return   0: no dependency
 *           1: this kernel depend on 'other'
 *          -1: 'other' depend on this kernel
 */
int bh_ir_kernel::dependency(const bh_ir_kernel &other) const
{
    int ret = 0;
    BOOST_FOREACH(uint64_t other_idx, other.instr_indexes)
    {
        const int dep = dependency(other_idx);
        if(dep != 0)
        {
            assert(ret == 0 or ret == dep);//Check for cyclic dependency
            ret = dep;
            //TODO: return 'ret' here, but for now we check all instructions
        }
    }
    return ret;
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

/* Returns the cost savings of merging with the 'other' kernel.
 * The cost savings is defined as the amount the BhIR will drop
 * in price if the two kernels are fused.
 * NB: This function determens the dependency order between the
 * two kernels and calculate the cost saving based on that order.
 *
 * @other  The other kernel
 * @return The cost value. Returns -1 if 'this' and the 'other'
 *         kernel isn't fusible.
 */
int64_t bh_ir_kernel::merge_cost_savings(const bh_ir_kernel &other) const
{
    if(this == &other)
        return 0;

    if(not fusible(other))
        return -1;

    const bh_ir_kernel *a;
    const bh_ir_kernel *b;
    //Lets make sure that 'a' depend on 'b'
    if(this->dependency(other) == 1)
    {
        a = this;
        b = &other;
    }
    else
    {
        a = &other;
        b = this;
    }

    int64_t price_drop = 0;

    //Subtract inputs in 'a' that comes from 'b' or is already an input in 'b'
    BOOST_FOREACH(const bh_view &i, a->input_list())
    {
        BOOST_FOREACH(const bh_view &o, b->output_list())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += cost_of_view(i);
        }
        BOOST_FOREACH(const bh_view &o, b->input_list())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += cost_of_view(i);
        }
    }
    //Subtract outputs from 'b' that are discared in 'a'
    BOOST_FOREACH(const bh_view &o, b->output_list())
    {
        BOOST_FOREACH(uint64_t a_instr_idx, a->instr_indexes)
        {
            const bh_instruction &a_instr = a->bhir->instr_list[a_instr_idx];
            if(a_instr.opcode == BH_DISCARD and a_instr.operand[0].base == o.base)
            {
                price_drop += cost_of_view(o);
                break;
            }
        }
    }
    return price_drop;
}

