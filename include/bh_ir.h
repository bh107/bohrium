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

#include "bh.h"
#include "bh_seqset.h"

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

class bh_ir_kernel; // Forward declaration

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

    /* Returns the cost of the BhIR */
    uint64_t cost() const;

    /* Pretty print the kernel list */
    void pprint_kernel_list() const;

    //The list of Bohrium instructions in topological order
    std::vector<bh_instruction> instr_list;

    //The list of kernels in topological order
    std::vector<bh_ir_kernel> kernel_list;

    bool tally;  // Should the ve tally after this bh_ir

protected:
    // Serialization using Boost
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & instr_list;
    }
};

/* A kernel is a list of instructions that are fusible. That is, a SIMD
 * machine can theoretically execute all the instructions in a single
 * operation.
*/
class bh_ir_kernel
{
private:
    // Set of all base-arrays in this kernel.
    std::set<bh_base*> bases;

    // Set of temporary base-arrays in this kernel.
    std::set<bh_base*> temps;

    // Set of base-arrays that are freed and not in temps i.e. created elsewhere
    std::set<bh_base*> frees;

    // Set of base-arrays that are discarded and not in temps i.e. created elsewhere
    std::set<bh_base*> discards;

    // Set of base-arrays that are synced by this kernel.
    std::set<bh_base*> syncs;

    // Sequence set of views used in this kernel
    seqset<bh_view> views;

    // Sequence set of base arrays used for input and output
    seqset<bh_base*> parameters;

    // List of input and output to this kernel.
    // NB: system instruction (e.g. BH_DISCARD) is
    // never part of kernel input or output
    std::multimap<bh_base*,bh_view> output_map;
    std::set<bh_view>               output_set;
    std::multimap<bh_base*,bh_view> input_map;
    std::set<bh_view>               input_set;

    // Largest input shape used in the kernel
    std::vector<bh_index> input_shape;

    bool scalar; // Indicate whether there is a scalar output from the kernel or not

    // sweep (reduce and accumulate) dimensions
    std::map<bh_intp, bh_int64> sweeps;

    // map of constants used in this kernel key is instruction id
    std::map<uint64_t, bh_constant> constants;

    /* Check f the 'base' is used in combination with the 'opcode' in this kernel  */
    bool is_base_used_by_opcode(const bh_base *b, bh_opcode opcode) const;

    // Topologically ordered list of instruction indexes
    std::vector<uint64_t> _instr_indexes;

public:

    /* Default constructor NB: the 'bhir' pointer is NULL in this case! */
    bh_ir_kernel();

    /* Kernel constructor, takes the bhir as constructor */
    bh_ir_kernel(bh_ir &bhir);

    // The program representation that the kernel is subset of
    bh_ir *bhir;

    /* Clear this kernel of all instructions */
    void clear();

    const std::vector<uint64_t>& instr_indexes() const {return _instr_indexes;}
    const std::multimap<bh_base*,bh_view>& get_output_map() const {return output_map;}
    const std::multimap<bh_base*,bh_view>& get_input_map() const {return input_map;}
    const std::set<bh_view>& get_output_set() const {return output_set;}
    const std::set<bh_view>& get_input_set() const {return input_set;}
    const std::set<bh_base*>& get_bases() const {return bases;}
    const std::set<bh_base*>& get_temps() const {return temps;}
    const std::set<bh_base*>& get_frees() const {return frees;}
    const std::set<bh_base*>& get_discards() const {return discards;}
    const std::set<bh_base*>& get_syncs() const {return syncs;}
    const seqset<bh_base*>& get_parameters() const {return parameters;}
    const std::map<uint64_t, bh_constant>& get_constants() const {return constants;}
    const std::map<bh_intp, bh_int64>& get_sweeps() const {return sweeps;}
    const std::vector<bh_index>& get_input_shape() const {return input_shape;}
    std::vector<bh_index> get_output_shape() const;

    bool is_output(bh_base* base) const {return output_map.find(base) != output_map.end();}
    bool is_output(const bh_view& view) const {return output_set.find(view) != output_set.end();}
    bool is_input(const bh_view& view) const {return input_set.find(view) != input_set.end();}
    bool is_scalar() const { return scalar;}

    size_t get_view_id(const bh_view& v) const;
    const bh_view& get_view(size_t id) const {return views[id];}

    /* Add an instruction to the kernel
     *
     *
     * @instr   The instruction to add
     * @return  The boolean answer
     */
    void add_instr(uint64_t instr_idx);

    /* Determines whether all instructions in 'this' kernel
     * are system opcodes (e.g. BH_DISCARD, BH_FREE, etc.)
     *
     * @return The boolean answer
     */
    bool only_system_opcodes() const;

    /* Determines whether all instructions in 'this' kernel
     * are BH_NONE
     *
     * @return The boolean answer
     */
    bool is_noop() const;

    /* Determines whether the kernel fusible legal
     *
     * @return The boolean answer
     */
    bool fusible() const;

    /* Determines whether it is legal to fuse with the instruction
     *
     * @instr_idx  The index of the instruction
     * @return     The boolean answer
     */
    bool fusible(uint64_t instr_idx) const;

    /* Determines whether it is legal to fuse with the kernel
     *
     * @other  The other kernel
     * @return The boolean answer
     */
    bool fusible(const bh_ir_kernel &other) const;

    /* Determines dependency between this kernel and the instruction 'instr',
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
    int dependency(uint64_t instr_idx) const;

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
    int dependency(const bh_ir_kernel &other) const;

    /* Returns the cost of the kernel */
    uint64_t cost() const;

    /* Returns the cost of the kernel using the unique views cost model */
    uint64_t cost_unique_views() const;

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
    int64_t merge_cost_savings(const bh_ir_kernel &other) const;

    // We use the current lowest instruction index in '_instr_indexes'
    // as kernel ID. Empty kernels have ID '-1'
    int64_t id() const;
};

#endif

