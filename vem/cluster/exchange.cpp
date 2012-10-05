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
#include <cassert>

//Local array information and storage
static std::map<cphvb_intp, cphvb_array*> bridge2vem;
static std::map<cphvb_intp, cphvb_array*> vem2bridge;
static StaticStore<cphvb_array> ary_store(512);

void exchange_inst_bridge2vem(cphvb_intp count,
                              const cphvb_instruction bridge_inst[],
                              cphvb_instruction vem_inst[])
{

    //TODO: Send the instruction list to all slave processes

    //TODO: We also need to exchange the array-bases
    
    for(cphvb_intp i=0; i<count; ++i)
    {
        const cphvb_instruction *bridge = &bridge_inst[i];
        cphvb_instruction *vem = &vem_inst[i];
        *vem = *bridge;
        int nop = cphvb_operands_in_instruction(bridge);
        assert(nop == cphvb_operands_in_instruction(vem));
        for(cphvb_intp j=0; j<nop; ++j)
        {
            cphvb_array *bridge_op = bridge->operand[j];

            if(cphvb_is_constant(bridge_op))
                continue;//No need to exchange constants
            
            vem->operand[j] = bridge2vem[(cphvb_intp)bridge_op];
            if(vem->operand[j] == NULL)//We don't know the operand
            {
                vem->operand[j] = ary_store.c_next();
                *vem->operand[j] = *bridge_op;
                vem->operand[j]->data = NULL;
                bridge2vem[(cphvb_intp)bridge_op] = vem->operand[j];
                vem2bridge[(cphvb_intp)vem->operand[j]] = bridge_op;
                if(bridge_op->base != NULL)//Make sure the base is also known locally
                {
                    cphvb_array *bridge_base = bridge_op->base;
                    vem->operand[j]->base = bridge2vem[(cphvb_intp)bridge_base];
                    if(vem->operand[j]->base == NULL)
                    {
                        vem->operand[j]->base = ary_store.c_next();
                        *vem->operand[j]->base = *bridge_base;
                        vem->operand[j]->base->data = NULL;
                        bridge2vem[(cphvb_intp)bridge_base] = vem->operand[j]->base;
                        vem2bridge[(cphvb_intp)vem->operand[j]->base] = bridge_base;
                    }
                 }   
            }

            //TODO: Do data communication

            cphvb_array *bridge_base = cphvb_base_array(bridge->operand[j]);
            cphvb_array *vem_base = cphvb_base_array(vem->operand[j]);
            if(bridge_base->data != NULL)
            {
                cphvb_error e = cphvb_data_malloc(vem->operand[j]);
                assert(e == CPHVB_SUCCESS);

                cphvb_intp nelem = cphvb_nelements(vem_base->ndim, vem_base->shape);
                assert(bridge_base->data != vem_base->data);
                memcpy(vem_base->data, bridge_base->data, nelem * cphvb_type_size(vem_base->type));
            }
        }
    }
}

void exchange_inst_vem2bridge(cphvb_intp count,
                              const cphvb_instruction vem_inst[],
                              cphvb_instruction bridge_inst[])
{
     
    //TODO: Send the instruction list to all slave processes

    //TODO: We also need to exchange the array-bases
    
    for(cphvb_intp i=0; i<count; ++i)
    {
        const cphvb_instruction *vem = &vem_inst[i];
        cphvb_instruction *bridge = &bridge_inst[i];
        assert(bridge->opcode == vem->opcode);
        for(cphvb_intp j=0; j<cphvb_operands_in_instruction(vem); ++j)
        {
            cphvb_array *vem_op = vem->operand[j];

            if(cphvb_is_constant(vem_op))
                continue;//No need to exchange constants

            bridge->operand[j] = vem2bridge[(cphvb_intp)vem_op];
            assert(bridge->operand[j] != NULL);
            
           
            //TODO: Do data communication
            cphvb_array *vem_base = cphvb_base_array(vem->operand[j]);
            cphvb_array *bridge_base = cphvb_base_array(bridge->operand[j]);
            if(vem_base->data != NULL)
            {
                cphvb_error e = cphvb_data_malloc(bridge->operand[j]);
                assert(e == CPHVB_SUCCESS);

                cphvb_intp nelem = cphvb_nelements(vem_base->ndim, vem_base->shape);
                assert(bridge_base->data != vem_base->data); 
                memcpy(bridge_base->data, vem_base->data, nelem * cphvb_type_size(vem_base->type));
            }
        }
    }
}
