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
#include <boost/serialization/vector.hpp>

#include "bh_type.h"
#include "bh_error.h"

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

/* A kernel is a list of instructions that are fusible. That is, a SIMD
 * machine can theoretically execute all the instructions in a single
 * operation.
*/
class bh_ir_kernel
{
protected:
    // Serialization using Boost
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & instrs;
    }

public:
    //The list of Bohrium instructions in this kernel
    std::vector<bh_instruction> instrs;
    // NOTE: I guess this should be "shielded" behind protected
    //       but for "right now" this makes it a lot easier to integrate
    //       with the current CPU-VE codebase.
    //       For the future 'instrs' should be hidden behind protected
    //       and accessed via const getters along with inputs, 
    //       outputs and temps?

    //List of input and output to this kernel.
    //NB: system instruction (e.g. BH_DISCARD) is
    //never part of kernel input or output
    std::vector<bh_view> inputs;
    std::vector<bh_view> outputs;

    //Lets of temporary base-arrays in this kernel.
    std::vector<const bh_base*> temps;

public:

    /* Returns the instructions in this kernel (read-only) */
    const std::vector<bh_instruction>& instr_list() const {return instrs;};

    /* Returns a list of inputs to this kernel (read-only) */
    const std::vector<bh_view>& input_list() const {return inputs;};

    /* Returns a list of outputs from this kernel (read-only) */
    const std::vector<bh_view>& output_list() const {return outputs;};

    /* Returns a list of temporary base-arrays in this kernel (read-only) */
    const std::vector<const bh_base*>& temp_list() const {return temps;};

    /* Returns the cost of the kernel */
    uint64_t cost() const;

    /* Add an instruction to the kernel
     *
     * @instr   The instruction to add
     * @return  The boolean answer
     */
    void add_instr(const bh_instruction &instr);

    /* Determines whether this kernel depends on 'other',
     * which is true when:
     *      'other' writes to an array that 'this' access
     *                        or
     *      'this' writes to an array that 'other' access
     *
     * @other The other kernel
     * @return The boolean answer
     */
    bool dependency(const bh_ir_kernel &other) const;

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
    int64_t dependency_cost(const bh_ir_kernel &other) const;

    /* Determines whether it is legal to fuse with the kernel
     *
     * @other   The other kernel
     * @return  The boolean answer
     */
    bool fusible(const bh_ir_kernel &other) const;

    /* Determines whether it is legal to fuse with the instruction
     *
     * @instr  The instruction
     * @return The boolean answer
     */
    bool fusible(const bh_instruction &instr) const;

    /* Determines whether the kernel fusible legal
     *
     * @return The boolean answer
     */
    bool fusible() const;

    /* Determines whether it is legal to fuse with the instruction
     * without changing this kernel's dependencies.
     *
     * @instr  The instruction
     * @return The boolean answer
     */
    bool fusible_gently(const bh_instruction &instr) const;

    /* Determines whether it is legal to fuse with the kernel without
     * changing this kernel's dependencies.
     *
     * @other  The other kernel
     * @return The boolean answer
     */
    bool fusible_gently(const bh_ir_kernel &other) const;

};


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

protected:
    // Serialization using Boost
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & instr_list;
        ar & kernel_list;
    }
};

#endif

