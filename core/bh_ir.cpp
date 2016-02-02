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
#include "bh_fuse_price.h"

using namespace std;
using namespace boost;
namespace io = boost::iostreams;

/* Creates a Bohrium Internal Representation (BhIR) from a instruction list.
*
* @ninstr      Number of instructions
* @instr_list  The instruction list
*/
bh_ir::bh_ir(bh_intp ninstr, const bh_instruction instr_list[])
    : tally(false)
{
    this->instr_list = vector<bh_instruction>(instr_list, &instr_list[ninstr]);
}

bh_ir::bh_ir(const bh_instruction& instr)
    : tally(false)
{
    instr_list.push_back(instr);
    bh_ir_kernel kernel(*this);
    kernel.add_instr(0);
    kernel_list.push_back(kernel);
}

/* Creates a BhIR from a serialized BhIR.
*
* @bhir The BhIr serialized as a char array or vector
*/
bh_ir::bh_ir(const char bhir[], bh_intp size)
    : tally(false)
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

bh_ir_kernel::bh_ir_kernel()
    : scalar(false)
    , bhir(NULL)
{}

bh_ir_kernel::bh_ir_kernel(bh_ir &bhir)
    : scalar(false)
    , bhir(&bhir)
{}


/* Clear this kernel of all instructions */
void bh_ir_kernel::clear()
{
    _instr_indexes.clear();
    input_set.clear();
    output_set.clear();
    input_map.clear();
    output_map.clear();
    temps.clear();
    views.clear();
    syncs.clear();
    frees.clear();
    discards.clear();
    parameters.clear();
    input_shape.clear();
    constants.clear();
    scalar = false;
}

size_t bh_ir_kernel::get_view_id(const bh_view& v) const
{
    return views[bh_view_simplify(v)];
}

/* Check f the 'base' is used in combination with the 'opcode' in this kernel  */
bool bh_ir_kernel::is_base_used_by_opcode(const bh_base *b, bh_opcode opcode) const
{
    assert(bhir != NULL);
    BOOST_FOREACH(uint64_t idx, instr_indexes())
    {
        const bh_instruction &instr = bhir->instr_list[idx];
        if(instr.opcode == opcode and instr.operand[0].base == b)
            return true;
    }
    return false;
}

void bh_ir_kernel::add_instr(uint64_t instr_idx)
{
    assert(bhir != NULL);
    assert(instr_idx < bhir->instr_list.size());
    const bh_instruction& instr = bhir->instr_list[instr_idx];
    const int nop = bh_operands(instr.opcode);
    for(int i=0; i<nop; ++i)
    {
        if(not bh_is_constant(&instr.operand[i]))
            bases.insert(instr.operand[i].base);
    }
    switch (instr.opcode) {
    case BH_NONE:
        break;
    case BH_TALLY:
        bhir->tally = true;
        break;
    case BH_SYNC:
        syncs.insert(instr.operand[0].base);
        break;
    case  BH_DISCARD:
    {
        bool temp = false;
        //When discarding we might have to remove arrays from 'outputs' and add
        //them to 'temps' (if the discared array isn't synchronized)
        bh_base* base = instr.operand[0].base;
        if(syncs.find(base) == syncs.end())
        {
            auto range = output_map.equal_range(base);
            if (range.first != range.second)
                temp = true;
            for (auto it = range.first; it != range.second; ++it)
                output_set.erase(it->second);
            output_map.erase(base);
            //If the discarded array isn't in 'inputs' (and not in 'outputs')
            //then it is a temp array
            if(temp && input_map.find(base) == input_map.end())
            {
                temps.insert(base);
                frees.erase(base);
                parameters.erase(base);
            }
            else
                temp = false;

        }
        if (!temp) // It is a discard of an array created elsewhere
            discards.insert(base);
    }
    break;
    case BH_FREE:
    {
        bh_base* base = instr.operand[0].base;
        if (temps.find(base) == temps.end())
            // It is a free of an array created elsewhere
            frees.insert(base);
    }
    break;
    default:
    {
        bool sweep = bh_opcode_is_sweep(instr.opcode);
        //Add the inputs of the instruction to 'inputs'
        for(int i=1; i<nop; ++i)
        {
            const bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
            {
                if (!sweep)
                    constants[instr_idx] = instr.constant;
                continue;
            }
            bh_view sv = bh_view_simplify(v);
            std::pair<size_t,bool> vid = views.insert(sv);
            if (vid.second) // If we have not seen the view before add it to inputs
            {
                input_map.insert(std::make_pair(v.base,v));
                input_set.insert(v);
                parameters.insert(v.base);
            }
            if (v.ndim > (bh_intp)input_shape.size())
                input_shape = std::vector<bh_index>(v.shape,v.shape+v.ndim);
        }
        //Add the output of the instruction to 'outputs'
        {
            const bh_view &v = instr.operand[0];
            bh_view sv = bh_view_simplify(v);
            views.insert(sv);
            output_map.insert(std::make_pair(v.base,v));
            output_set.insert(v);
            parameters.insert(v.base);
            if (bh_is_scalar(&v))
                scalar = true;
            // For now we treat a 1D accumulate like a 1D reduce i.e. the kernel is a scalar kernel
            if (bh_opcode_is_accumulate(instr.opcode) && sv.ndim == 1)
                scalar = true;
            if (v.ndim > (bh_intp)input_shape.size())
                input_shape = std::vector<bh_index>(v.shape,v.shape+v.ndim);
        }
        if (sweep)
        {
            sweeps[instr.operand[1].ndim] = instr.constant.value.int64;
        }
    }
    }
    _instr_indexes.push_back(instr_idx);
}

// Smallest output shape used in the kernel
std::vector<bh_index> bh_ir_kernel::get_output_shape() const
{
    bh_index nelem = INT64_MAX;
    std::vector<bh_index> res;
    for (const bh_view& v: output_set)
    {
        if (bh_nelements(v) < nelem)
        {
            nelem = bh_nelements(v);
            res = std::vector<bh_index>(v.shape,v.shape+v.ndim);
        }
    }
    return res;
}

/* Determines whether all instructions in 'this' kernel
 * are system opcodes (e.g. BH_DISCARD, BH_FREE, etc.)
 *
 * @return The boolean answer
 */
bool bh_ir_kernel::only_system_opcodes() const
{
    assert(bhir != NULL);
    BOOST_FOREACH(uint64_t this_idx, instr_indexes())
    {
        if(not bh_opcode_is_system(bhir->instr_list[this_idx].opcode))
            return false;
    }
    return true;
}

/* Determines whether all instructions in 'this' kernel
 * are BH_NONE
 *
 * @return The boolean answer
 */
bool bh_ir_kernel::is_noop() const
{
    assert(bhir != NULL);
    BOOST_FOREACH(uint64_t this_idx, instr_indexes())
    {
        if(bhir->instr_list[this_idx].opcode != BH_NONE)
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
    assert(bhir != NULL);
    for(uint64_t i=0; i<instr_indexes().size(); ++i)
    {
        const bh_instruction *instr = &bhir->instr_list[instr_indexes()[i]];
        for(uint64_t j=i+1; j<instr_indexes().size(); ++j)
        {
            if(not bohrium::check_fusible(instr, &bhir->instr_list[instr_indexes()[j]]))
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
    assert(bhir != NULL);
    const bh_instruction *instr = &bhir->instr_list[instr_idx];
    BOOST_FOREACH(uint64_t i, instr_indexes())
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
    assert(bhir != NULL);
    BOOST_FOREACH(uint64_t idx1, instr_indexes())
    {
        const bh_instruction *instr = &bhir->instr_list[idx1];
        BOOST_FOREACH(uint64_t idx2, other.instr_indexes())
        {
            if(not bohrium::check_fusible(instr, &other.bhir->instr_list[idx2]))
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
    assert(bhir != NULL);
    int ret = 0;
    BOOST_FOREACH(uint64_t this_idx, instr_indexes())
    {
        if(bh_instr_dependency(&bhir->instr_list[instr_idx],
                               &bhir->instr_list[this_idx]))
        {
            if(this_idx >= instr_idx)
            {
                assert(ret == 0 or ret == 1);//Check for cyclic dependency
                ret = 1;
                #ifdef NDEBUG
                    return ret; //We only check all instructions when in debug mode
                #endif
            }
            else
            {
                assert(ret == 0 or ret == -1);//Check for cyclic dependency
                ret = -1;
                #ifdef NDEBUG
                    return ret; //We only check all instructions when in debug mode
                #endif
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
    BOOST_FOREACH(uint64_t other_idx, other.instr_indexes())
    {
        const int dep = dependency(other_idx);
        if(dep != 0)
        {
            assert(ret == 0 or ret == dep);//Check for cyclic dependency
            ret = dep;
            #ifdef NDEBUG
                return ret; //We only check all instructions when in debug mode
            #endif
        }
    }
    return ret;
}

/* Returns the cost of the kernel */
uint64_t bh_ir_kernel::cost() const
{
    return bohrium::kernel_cost(*this);
}

/* Returns the cost of the kernel using the unique views cost model */
uint64_t bh_ir_kernel::cost_unique_views() const
{
    return bohrium::kernel_cost_unique_views(*this);
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
    return bohrium::cost_savings(*b, *a);
}

// We use the current lowest instruction index in '_instr_indexes'
// as kernel ID. Empty kernels have ID '-1'
int64_t bh_ir_kernel::id() const
{
    if(_instr_indexes.size() == 0)
        return -1;
    return *std::min_element(_instr_indexes.begin(), _instr_indexes.end());
}

