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

/* Default constructor NB: the 'bhir' pointer is NULL in this case! */
bh_ir_kernel::bh_ir_kernel():bhir(NULL) {}

/* Kernel constructor, takes the bhir as constructor */
bh_ir_kernel::bh_ir_kernel(bh_ir &bhir) : bhir(&bhir) {}

/* Clear this kernel of all instructions */
void bh_ir_kernel::clear()
{
    instr_indexes.clear();
    input_set.clear();
    output_set.clear();
    input_map.clear();
    output_map.clear();
    temps.clear();
    views.clear();
}

void bh_ir_kernel::view_set::clear()
{
    maxid = 0;
    views.clear();
}

std::pair<bool,bh_view> bh_ir_kernel::view_set::insert(const bh_view &v)
{
    if (bh_is_constant(&v))
        throw bh_ir_kernel::view_exception(-1);
    
    bh_view sv = bh_view_simplify(&v);
    auto it = views.find(sv);
    if (it == views.end())
    {
        views.insert(std::make_pair(sv,maxid++));
        return std::make_pair(true,sv);
        
    } else {  
        return  std::make_pair(false,sv);
    }
}

size_t bh_ir_kernel::view_set::operator[] (const bh_view &v) const
{
    auto it = views.find(bh_view_simplify(&v));
    if (it == views.end())
        throw bh_ir_kernel::view_exception(-1);
    return it->second;
}

size_t bh_ir_kernel::get_view_id(const bh_view v) const
{
    return views[v];
}

/* Check f the 'base' is used in combination with the 'opcode' in this kernel  */
bool bh_ir_kernel::is_base_used_by_opcode(const bh_base *b, bh_opcode opcode) const
{
    BOOST_FOREACH(uint64_t idx, instr_indexes)
    {
        const bh_instruction &instr = bhir->instr_list[idx];
        if(instr.opcode == opcode and instr.operand[0].base == b)
            return true;
    }
    return false;
}

void bh_ir_kernel::add_instr(uint64_t instr_idx)
{
    
    const bh_instruction& instr = bhir->instr_list[instr_idx];
    switch (instr.opcode) { 
    case BH_SYNC:
        syncs.insert(instr.operand[0].base);
        break;
    case  BH_DISCARD:
    {
        //When discarding we might have to remove arrays from 'outputs' and add
        //them to 'temps' (if the discared array isn't synchronized)
        bh_base* base = instr.operand[0].base;
        if(syncs.find(base) == syncs.end())
        {
            auto range = output_map.equal_range(base);
            for (auto it = range.first; it != range.second; ++it)
                output_set.erase(it->second);
            output_map.erase(base);
            //If the discarded array isn't in 'inputs' (and not in 'outputs')
            //then it is a temp array
            if(input_map.find(base) == input_map.end())
                temps.insert(base);
        }
    }
        break;
    case BH_FREE:
        break;
    default:
    {
        const int nop = bh_operands(instr.opcode);
        //Add the inputs of the instruction to 'inputs'
        for(int i=1; i<nop; ++i)
        {
            const bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
                continue;
            std::pair<bool,bh_view> vid = views.insert(v);
            if (vid.first) // If we have not seen the view before add it to inputs
            {
                input_map.insert(std::make_pair(vid.second.base,vid.second));
                input_set.insert(vid.second);
            }
            shapes.insert(std::vector<bh_index>(v.shape,v.shape+v.ndim));
        }
        //Add the output of the instruction to 'outputs'
        {
            const bh_view &v = instr.operand[0];
            bh_view vid = views.insert(v).second;
            output_map.insert(std::make_pair(vid.base,vid));
            output_set.insert(vid);
            shapes.insert(std::vector<bh_index>(v.shape,v.shape+v.ndim));
        }
    }
    }
    instr_indexes.push_back(instr_idx);
}

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

/* Determines whether the in-/output of 'this' kernel is a subset of 'other'
 *
 * @other  The other kernel
 * @return The boolean answer
 */
bool bh_ir_kernel::input_and_output_subset_of(const bh_ir_kernel &other) const
{
    const std::set<bh_view>& other_input_set = other.get_input_set();
    if(input_set.size() > other_input_set.size())
        return false;
    const std::set<bh_view>& other_output_set = other.get_output_set();
    if(output_set.size() > other_output_set.size())
        return false;

    for (const bh_view& iv: other_input_set)
    {
        if (input_set.find(iv) == input_set.end())
            return false;
    }
    for (const bh_view& ov: other_output_set)
    {
        if (output_set.find(ov) == output_set.end())
            return false;
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
    BOOST_FOREACH(const bh_view &v, input_set)
    {
        sum += cost_of_view(v);
    }
    BOOST_FOREACH(const bh_view &v, output_set)
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
    BOOST_FOREACH(const bh_view &i, a->get_input_set())
    {
        BOOST_FOREACH(const bh_view &o, b->get_output_set())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += cost_of_view(i);
        }
        BOOST_FOREACH(const bh_view &o, b->get_input_set())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += cost_of_view(i);
        }
    }
    //Subtract outputs from 'b' that are discared in 'a'
    BOOST_FOREACH(const bh_view &o, b->get_output_set())
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
