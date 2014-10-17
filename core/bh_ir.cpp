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
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <bh.h>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "bh_ir.h"
#include "bh_dag.h"

using namespace std;
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
    boost::archive::binary_iarchive ia(input_stream);
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
    boost::archive::binary_oarchive oa(output_stream);
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
        bh_pprint_instr_list(&k.instr_list()[0], k.instr_list().size(), msg);
    }
}

/* Pretty write the kernel DAG as a DOT file
*
*  @filename   Name of the DOT file
*/
void bh_ir::pprint_kernel_dag(const char filename[]) const
{
    using namespace boost;
    typedef adjacency_list<setS, vecS, bidirectionalS, bh_ir_kernel> Graph;
    Graph dag;
    bh_dag_from_kernels(kernel_list, dag);
    bh_dag_transitive_reduction(dag);
    bh_dag_pprint(dag, filename);
}

/* Determines whether there are cyclic dependencies between the kernels in the BhIR
*
*  @index_map  A map from an instruction in the kernel_list (a pair of a kernel and
*              an instruction index) to an index into the original instruction list
*  @return     True when no cycles was found
*/
bool bh_ir::check_kernel_cycles(const map<pair<int,int>,int> index_map) const
{
    //Create a DAG of the kernels and check for cycles
    using namespace boost;
    typedef adjacency_list<setS, vecS, bidirectionalS, bh_ir_kernel> Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;

    Graph dag(kernel_list.size());

    //First we build a map from an index in the original instruction list
    //to a vertex in the 'dag'
    map<int,Vertex> instr2vertex;
    unsigned int k=0;
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        const vector<bh_instruction> &kernel = kernel_list[k].instr_list();
        dag[v] = kernel_list[k];
        for(unsigned int i=0; i<kernel.size(); ++i)
        {
            int instr_index = index_map.at(make_pair(k,i));
            instr2vertex.insert(pair<int,Vertex>(instr_index,v));
        }
        ++k;
    }
    //Then we add edges to the 'dag' for each dependency in the original
    //instruction list
    for(unsigned int i=0; i<instr_list.size(); ++i)
    {
        Vertex v = instr2vertex[i];
        for(unsigned int j=0; j<i; ++j)//We may depend on any previous instruction
        {
            if(bh_instr_dependency(&instr_list[i], &instr_list[j]))
            {
                if(instr2vertex[j] != v)
                    add_edge(instr2vertex[j], v, dag);
            }
        }
    }
    //Finally we check for cycles between the kernel dependencies
    vector<Vertex> topological_order;
    try
    {
        //TODO: topological sort is an efficient method for finding cycles,
        //but we should avoid allocating a vector
        topological_sort(dag, back_inserter(topological_order));
        return true;
    }
    catch (const not_a_dag &e)
    {
        cout << "Writing the failed kernel list: check_kernel_cycles-fail.dot" << endl;
        bh_dag_pprint(dag, "check_kernel_cycles-fail.dot");
        return false;
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
    BOOST_FOREACH(const bh_instruction &i, instr_list())
    {
        BOOST_FOREACH(const bh_instruction &o, other.instr_list())
        {
            if(bh_instr_dependency(&i, &o))
                return true;
        }
    }
    return false;
}

/* Returns the cost of this kernel's dependency on the 'other' kernel.
 * The cost of a dependency is defined as the amount the BhIR will drop
 * in price if the two kernels are fused.
 * Note that a zero cost dependency is possible because of system
 * instructions such as BH_FREE and BH_DISCARD.
 *
 * @other  The other kernel
 * @return The cost value. Returns -1 if this and the 'other'
 *         kernel isn't fusible.
 */
int64_t bh_ir_kernel::dependency_cost(const bh_ir_kernel &other) const
{
    if(not fusible(other))
        return -1;

    int64_t price_drop = 0;

    //Subtract inputs that comes from 'other' or is already an input in 'other'
    BOOST_FOREACH(const bh_view &i, input_list())
    {
        BOOST_FOREACH(const bh_view &o, other.output_list())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += cost_of_view(i);
        }
        BOOST_FOREACH(const bh_view &o, other.input_list())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += cost_of_view(i);
        }
    }
    //Subtract outputs that are discared in 'this'
    BOOST_FOREACH(const bh_view &o, other.output_list())
    {
        BOOST_FOREACH(const bh_instruction &i, instr_list())
        {
            if(i.opcode == BH_DISCARD and i.operand[0].base == o.base)
            {
                price_drop += cost_of_view(o);
                break;
            }
        }
    }
    return price_drop;
}

/* Determines whether it is legal to fuse with the kernel
 *
 * @other The other kernel
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible(const bh_ir_kernel &other) const
{
    BOOST_FOREACH(const bh_instruction &a, instr_list())
    {
        BOOST_FOREACH(const bh_instruction &b, other.instr_list())
        {
            if(not bh_instr_fusible(&a, &b))
                return false;
        }
    }
    return true;
}

/* Determines whether it is legal to fuse with the instruction
 *
 * @instr  The instruction
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible(const bh_instruction &instr) const
{
    BOOST_FOREACH(const bh_instruction &i, instr_list())
    {
        if(not bh_instr_fusible(&i, &instr))
            return false;
    }
    return true;
}

/* Determines whether it is legal to fuse with the instruction
 * without changing this kernel's dependencies.
 *
 * @instr  The instruction
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible_gently(const bh_instruction &instr) const
{
    if(bh_opcode_is_system(instr.opcode))
        return true;

    //Check that 'instr' is fusible with least one existing instruction
    BOOST_FOREACH(const bh_instruction &i, instr_list())
    {
        if(bh_opcode_is_system(i.opcode))
            continue;

        if(bh_instr_fusible_gently(&instr, &i))
            return true;
    }
    return false;
}

/* Determines whether it is legal to fuse with the kernel without
 * changing this kernel's dependencies.
 *
 * @other  The other kernel
 * @return The boolean answer
 */
bool bh_ir_kernel::fusible_gently(const bh_ir_kernel &other) const
{
    BOOST_FOREACH(const bh_instruction &i, other.instr_list())
    {
        if(not fusible_gently(i))
            return false;
    }
    return true;
}
