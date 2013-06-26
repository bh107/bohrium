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
#include <string>
#include <vector>
#include <unordered_map>
#include <bh.h>
#include "utils.cpp"
#include "kernel.h"

bool is_temp(bh_array* op, ref_storage& reads, ref_storage& writes)
{
    return ((reads[op] == 1) && (writes[op] == 1));
}

void bh_pprint_list(bh_instruction* list, bh_intp start, bh_intp end)
{
    for(bh_intp i=start; i<=end; ++i) {
        switch((&list[i])->opcode) {
            case BH_NONE:
            case BH_DISCARD:
            case BH_SYNC:
            case BH_FREE:
                break;
            default:
                bh_pprint_instr(&list[i]);
        }
    }
}


int hash(bh_instruction *instr)
{
    uint64_t poly;
    int dims, nop,
        a0_type, a1_type, a2_type,
        a0_dense, a1_dense, a2_dense;

    dims     = instr->operand[0]->ndim;
    nop      = bh_operands(instr->opcode);
    a0_type  = instr->operand[0]->type;
    a0_dense = 1;
    if (3 == nop) {
        if (bh_is_constant(instr->operand[1])) {            // DDC
            a1_type  = instr->constant.type;
            a1_dense = 0;
            a2_type  = instr->operand[2]->type;
            a2_dense = 1;
        } else if (bh_is_constant(instr->operand[2])) {     // DCD
            a1_type  = instr->operand[1]->type;
            a1_dense = 1;
            a2_type  = instr->constant.type;
            a2_dense = 0;   
        } else {                                            // DDD
            a1_type  = instr->operand[1]->type;
            a1_dense = 1;
            a2_type  = instr->operand[2]->type;
            a2_dense = 1;
        }
    } else if (2 == nop) {
        if (bh_is_constant(instr->operand[1])) {            // DDC
            a1_type  = instr->constant.type;
            a1_dense = 0;
        } else {                                            // DDD
            a1_type  = instr->operand[1]->type;
            a1_dense = 1;
        }
        a2_type = 0;
        a2_dense = 0;
    } else {
        a1_type  = 0;
        a2_type  = 0;
        a1_dense = 0;
        a2_dense = 0;
    }

    poly  = (a0_type << 8) + (a1_type << 4) + (a2_type);
    poly += (a0_dense << 14) + (a1_dense << 13) + (a2_dense << 12);
    poly += (dims << 15);
    poly += (instr->opcode << 20);

    /*
    std::cout << "Opcode {" << instr->opcode << "}" << std::endl;
    std::cout << "Dims {" << dims << "}" << std::endl;
    std::cout << "Type {" << a0_type << ", " << a1_type << ", " << a2_type << "}" << std::endl;
    std::cout << "Struct {" << a0_dense << ", " << a1_dense << ", " << a2_dense << "}" << std::endl;
    std::cout << poly << std::endl;
    */
    return poly;
}

/**
 *  Deduce a set of kernels based
 *
 */
kernel_storage streaming(bh_intp count, bh_instruction* list)
{
    ref_storage reads;
    ref_storage writes;
    std::vector<bh_intp> potentials;
    kernel_storage kernels;

    kernel_t kernel;

    bh_instruction *instr;
    bh_opcode opcode;
    int noperands;

    for(bh_intp i=0; i<count; ++i) {
        instr       = &list[i];
        opcode      = instr->opcode;
        noperands   = bh_operands(opcode);

        switch(opcode) {
            case BH_NONE:                                       // Ignore these
            case BH_DISCARD:
            case BH_SYNC:
            case BH_FREE:
                break;

            case BH_ADD_REDUCE:                                 // POTENTIALS
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
                potentials.push_back(i);

            default:                                            // REFCOUNT
                if (writes.count(instr->operand[0])>0) {        // Output
                    ++(writes[instr->operand[0]]);
                } else {
                    writes.insert(std::make_pair(instr->operand[0], 1));
                }
                for(int op_i=1; op_i<noperands; ++op_i) {       // Input
                    if (bh_is_constant(instr->operand[op_i])) { // Skip constants
                        continue;
                    }
                    if (reads.count(instr->operand[op_i])>0) {
                        ++(reads[instr->operand[op_i]]);
                    } else {
                        reads.insert(std::make_pair(instr->operand[op_i], 1));
                    }
                }
        }
    }

    //
    // Now we have a list of potential endpoints for streaming kernels.
    // As well as a ref-count on all operands within current scope.
    //
    // Now; it is time to determine whether the potentials can become kernels.
    bh_array *operand = NULL;
    bh_intp border = 0;
    bh_intp potential = -1;
    for(bh_intp i=count-1; i>=0; --i) {
        instr       = &list[i];
        opcode      = instr->opcode;
        noperands   = bh_operands(opcode);
        operand     = instr->operand[0];

        if (-1==potential) {
            potential = potentials.back();
            potentials.pop_back();
        }

        if (i < potential) {
            if (!is_temp(operand, reads, writes)) {
                border = i+1;
                kernel.size  = potential-border+1;
                kernel.begin = border;
                kernel.end   = potential;
                kernels.insert(kernels.begin(), kernel);
                potential = -1;
                if (potentials.size() == 0) {   // No more potential...
                    break;                      // stop searching...
                }
            }
        } 
    }

    //
    // Now "kernels" contain pairs of instruction-list offsets which can be used
    // to create kernels with...
    //

    //
    // DEBUG PRINTING
    //
    /* 
    std::cout << "## Failed potential=" << potentials.size() << std::endl;
    for(auto it=potentials.begin(); it!=potentials.end(); ++it) {
        std::cout << (*it) << ",";
    }
    std::cout << "." << std::endl;

    if (kernels.size()>0) {
        std::cout << "## Kernels=" << kernels.size() << std::endl;
        for(auto it=kernels.begin();
            it!=kernels.end();
            ++it) {
            std::cout << "** kernel start " << (*it).first << " **" << std::endl;
            bh_pprint_list(list, (*it).first, (*it).second);
            std::cout << "** kernel end " << (*it).second << " **" << std::endl;
        }
    }

    if (potentials.size()>=1) {                             // POTENTIALS
        if (reads.size()>0) {
        std::cout << "** reads **" << std::endl;            // READS
            for(auto it=reads.begin();
                it != reads.end();
                ++it) {
                std::cout << (*it).first << ", read=";
                std::cout << (*it).second << "." << std::endl;
            }
        }
        if (writes.size()>0) {
            std::cout << "** writes **" << std::endl;            // WRITES
            for(auto it=writes.begin();
                it != writes.end();
                ++it) {
                std::cout << (*it).first << ", written=";
                std::cout << (*it).second << "." << std::endl;
            }
        }
    }
    */

    return kernels;
}

std::string name_input(kernel_t& kernel, bh_array* input)
{
    char varchar[20];
    sprintf(varchar, "*a%d_current", (int)(kernel.size));
    kernel.size++;
    kernel.inputs[kernel.size] = input;
    std::string varname(varchar);
    return varname;
}

std::string fused_expr(bh_instruction* list, bh_intp cur, bh_intp max, bh_array *input, kernel_t& kernel)
{
    if (cur == 0) { // Over the top! Find the non-fused input
        return "SHIT";
    } 
    bh_instruction *instr   = &list[cur];
    bh_opcode opcode        = instr->opcode;

    bh_intp  noperands  = bh_operands(opcode);
    bh_array *output    = instr->operand[0];
    bh_array *a1=NULL, *a2=NULL; // input operands
    std::string lh_str, rh_str;

    switch(opcode) {
        case BH_USERFUNC:               // Extract output from RANDOM
            bh_random_type *random_args;
            if ((opcode == BH_USERFUNC) && (instr->userfunc->id==random_impl_id)) {
                random_args = (bh_random_type*)instr->userfunc;
                output      = random_args->operand[0];
                break;
            }

        case BH_FREE:                   // Ignore these opcodes.
        case BH_SYNC:                   // AKA keep searching
        case BH_DISCARD:
        case BH_NONE:
        case BH_ADD_REDUCE:
            return fused_expr(list, cur-1, max, input, kernel);
            break;

        default:                        // No match, keep searching
            if (input!=output) {
                return fused_expr(list, cur-1, max, input, kernel);
            }
            break;
    }
                                        // Yeehaa we found it!
    if (cur <= max-1) {                 // Outside the kernel
        return name_input(kernel, output);
    } else {                            // Within the kernel

        if (!bh_is_constant(instr->operand[1])) {   // Assign operands
            a1 = instr->operand[1];
            lh_str = fused_expr(list, cur-1, max, a1, kernel);  // Lefthandside
        } else {
            lh_str = "("+std::string(bhtype_to_ctype(instr->constant.type))+")("+\
                      const_as_string(instr->constant)+")";    // Inline the constant value
        }

        if ((3==noperands) && (!bh_is_constant(instr->operand[2]))) {
            a2 = instr->operand[2];
            rh_str = fused_expr(list, cur-1, max, a2, kernel);  // Righthandside
        } else {
            rh_str = "("+std::string(bhtype_to_ctype(instr->constant.type))+")("+\
                      const_as_string(instr->constant)+")";    // Inline the constant value
        }

        switch(opcode) {
            case BH_ADD:
                return  "("+lh_str+") + ("+rh_str+")";
            case BH_SUBTRACT:
                return  "("+lh_str+") - ("+rh_str+")";
            case BH_MULTIPLY:
                return  "("+lh_str+") * ("+rh_str+")";
            case BH_DIVIDE:
                return  "("+lh_str+") / ("+rh_str+")";
            case BH_POWER:
                return  "pow("+lh_str+", "+rh_str+")";
            case BH_LESS_EQUAL:
                return  "("+lh_str+") <= ("+rh_str+")";
            case BH_EQUAL:
                return  "("+lh_str+") == ("+rh_str+")";
            case BH_SQRT:
                return "sqrt("+lh_str+")";
            case BH_IDENTITY:
                return "("+std::string(bhtype_to_ctype(output->type))+")("+lh_str+")";

            default:
                return "__ERROR__";
        }
    }
}

