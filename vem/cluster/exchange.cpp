/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb.h>
#include <StaticStore.hpp>
#include <map>

//Local array information and storage
static std::map<cphvb_intp, cphvb_array*> ary_map;
static StaticStore<cphvb_array> ary_store(512);

void exchange_inst_list(cphvb_intp count,
                        cphvb_instruction inst_list[])
{

    //TODO: Send the instruction list to all slave processes

    
    for(cphvb_intp i=0; i<count; ++i)
    {
        cphvb_instruction *inst = &inst_list[i];
        int nop = cphvb_operands_in_instruction(inst);

        for(cphvb_intp j=0; j<nop; ++j)
        {
            cphvb_array *bridge_a = inst->operand[j];
            cphvb_array *vem_a;
            if(cphvb_is_constant(bridge_a))
                continue;//No need to exchange constants

            if((vem_a = ary_map[(cphvb_intp)bridge_a]) != NULL)//We know the operand
            {
                inst->operand[j] = vem_a;
            }
            else//We don't know the operand
            {
                inst->operand[j] = ary_store.c_next();
                *inst->operand[j] = *bridge_a;
                ary_map[(cphvb_intp)bridge_a] = inst->operand[j];
            }
        }
    }
}
