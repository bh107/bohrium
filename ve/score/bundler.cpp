/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include "bundler.hpp"
#include <iostream>
#include <set>
#include "pp.h"

typedef cphvb_array* cphvb_array_ptr;

/* Calculates the bundleable instructions.
 *
 * @inst The instruction list
 * @size Size of the instruction list
 * @return Number of consecutive bundable instructions.
 */
cphvb_intp bundle(cphvb_instruction *insts[], cphvb_intp size)
{

    cphvb_intp bundle_len = 0;                                      // Number of cons. bundl. instr.

    std::set<cphvb_array_ptr> in, out;                               // Sets for classifying operands.
    std::set<cphvb_array_ptr>::iterator in_it;                       // NOTE: Use boost::set? It is O( k )!
    std::set<cphvb_array_ptr>::iterator out_it;
    std::pair<std::set<cphvb_array_ptr>::iterator, bool> ins_res;
                                                                    // Used as "iterator": 
    cphvb_instruction*  instr;                                      // - pointing to current instruction
    int op_count;                                                   // - opcount of the current instruction 
    cphvb_array* ops[3];

    std::cout << "BUNDLER {" << std::endl;
    for(cphvb_intp i=0; i<size; ++i)
    {

        instr       = insts[i];
        op_count    = cphvb_operands( instr->opcode );
        for(int j=0; j < op_count; j++) {
            ops[j] = instr->operand[j];
        }
        pp_instr( instr );

        switch(op_count)
        {

            case 3:

                std::cout << "\t" << "OP3 " << ops[2] << std::endl;
                if (out.count( ops[2] ) > 0)
                {
                    std::cout << "\t" << "Flushing... on account of INPUT " << ops[2] << " in OUTPUT-SET." << std::endl;
                } else {
                    ins_res = in.insert( ops[2] );
                    std::cout << "\t" << "[Added? = " << ins_res.second << "] " <<  ops[2] << " to INPUT-SET." << std::endl;
                }

            case 2:

                std::cout << "\t" << "OP2 " << ops[1] << std::endl;
                if (out.count( ops[1] ) > 0)
                {
                    std::cout << "\t" << "Flushing... on account of INPUT " << ops[1] << " in OUTPUT-SET." << std::endl;
                } else {
                    ins_res = in.insert( ops[1] );
                    std::cout << "\t" << "[Added? = " << ins_res.second << "] " << ops[1] << " to INPUT-SET." << std::endl;
                }

            case 1:

                std::cout << "\t" << "OP1 " << ops[0] << std::endl;
                if (in.count( ops[0] ) > 0)
                {
                    std::cout << "\t" << "Flushing... on account of OUTPUT " << ops[0] << " in INPUT-SET." << std::endl;
                } else {
                    ins_res = out.insert( ops[0] );
                    std::cout << "\t" << "[Added? = " << ins_res.second << "] " << ops[0] << " to OUTPUT-SET." << std::endl;
                }

            default:
                bundle_len++;
                break;

        }

    }

    std::cout << "} ... {" << std::endl;
    std::cout << "... result ..." << std::endl;
    std::cout << "Readers: ";
    for(in_it = in.begin(); in_it != in.end(); in_it++)
    {
        std::cout << *in_it << ",";
    }
    std::cout << "." << std::endl;

    std::cout << "Writers: ";
    for(out_it = out.begin(); out_it != out.end(); out_it++)
    {
        std::cout << *out_it << ",";
    }
    std::cout << std::endl <<  "} " << bundle_len  << std::endl;

    return bundle_len;

}
