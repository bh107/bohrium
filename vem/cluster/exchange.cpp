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

static cphvb_error copy_data_bridge2vem(const cphvb_array *bridge_op, cphvb_array *vem_op)
{
    
    //TODO: Do data communication

    cphvb_array *bridge_base = cphvb_base_array(bridge_op);
    cphvb_array *vem_base = cphvb_base_array(vem_op);
    if(bridge_base->data != NULL && vem_base->data == NULL)
    {
        cphvb_error e = cphvb_data_malloc(vem_op);
        assert(e == CPHVB_SUCCESS);
        assert(bridge_base->type == vem_base->type);

        cphvb_intp nelem = cphvb_nelements(vem_base->ndim, vem_base->shape);
        assert(bridge_base->data != vem_base->data);
        memcpy(vem_base->data, bridge_base->data, nelem * cphvb_type_size(vem_base->type));
    }
    
    return CPHVB_SUCCESS;
}


void exchange_inst_bridge2vem(cphvb_intp count,
                              const cphvb_instruction bridge_inst[],
                              cphvb_instruction vem_inst[])
{

    //TODO: Send the instruction list to all slave processes

    //TODO: Send userfunc instructions to all slave processes

    for(cphvb_intp i=0; i<count; ++i)
    {
        const cphvb_instruction *bridge = &bridge_inst[i];
        cphvb_instruction *vem = &vem_inst[i];
        *vem = *bridge;
        int nop = cphvb_operands_in_instruction(bridge);
        assert(nop == cphvb_operands_in_instruction(vem));

        //Allocate a new userfunc struct for the vem instruction
        if(bridge->opcode == CPHVB_USERFUNC)
        {
            vem->userfunc = (cphvb_userfunc *) malloc(bridge->userfunc->struct_size);
            assert(vem->userfunc != NULL);
            memcpy(vem->userfunc,bridge->userfunc,bridge->userfunc->struct_size);
        }

        for(cphvb_intp j=0; j<nop; ++j)
        {
            cphvb_array *bridge_op;
            if(bridge->opcode == CPHVB_USERFUNC)
                bridge_op = bridge->userfunc->operand[j];
            else
                bridge_op = bridge->operand[j];
 

            if(cphvb_is_constant(bridge_op))
                continue;//No need to exchange constants
            
            cphvb_array *vem_op = bridge2vem[(cphvb_intp)bridge_op];
            if(vem_op == NULL)//We don't know the operand
            {
                vem_op = ary_store.c_next();
                *vem_op = *bridge_op;
                vem_op->data = NULL;
                bridge2vem[(cphvb_intp)bridge_op] = vem_op;
                vem2bridge[(cphvb_intp)vem_op] = bridge_op;
                if(bridge_op->base != NULL)//Make sure the base is also known locally
                {
                    cphvb_array *bridge_base = bridge_op->base;
                    vem_op->base = bridge2vem[(cphvb_intp)bridge_base];
                    if(vem_op->base == NULL)
                    {
                        vem_op->base = ary_store.c_next();
                        *vem_op->base = *bridge_base;
                        vem_op->base->data = NULL;
                        bridge2vem[(cphvb_intp)bridge_base] = vem_op->base;
                        vem2bridge[(cphvb_intp)vem_op->base] = bridge_base;
                    }
                }   
            }
            copy_data_bridge2vem(bridge_op, vem_op);
            if(vem->opcode == CPHVB_USERFUNC)
                vem->userfunc->operand[j] = vem_op;
            else
                vem->operand[j] = vem_op;
        }
    }
}

void exchange_inst_vem2bridge(cphvb_intp count,
                              const cphvb_instruction vem_inst[],
                              cphvb_instruction bridge_inst[])
{
    for(cphvb_intp i=0; i<count; ++i)
    {
        const cphvb_instruction *vem = &vem_inst[i];
        cphvb_instruction *bridge = &bridge_inst[i];
        assert(bridge->opcode == vem->opcode);
        for(cphvb_intp j=0; j<cphvb_operands_in_instruction(vem); ++j)
        {
            cphvb_array *vem_op;
            if(vem->opcode == CPHVB_USERFUNC)
                vem_op = vem->userfunc->operand[j];
            else
                vem_op = vem->operand[j];

            if(cphvb_is_constant(vem_op))
                continue;//No need to exchange constants

            cphvb_array *bridge_op = vem2bridge[(cphvb_intp)vem_op];
            assert(bridge_op != NULL);
            
            //TODO: Do data communication
            cphvb_array *vem_base = cphvb_base_array(vem_op);
            cphvb_array *bridge_base = cphvb_base_array(bridge_op);
            if(vem_base->data != NULL)
            {
                cphvb_error e = cphvb_data_malloc(bridge_op);
                assert(e == CPHVB_SUCCESS);
                assert(bridge_base->type == vem_base->type);

                cphvb_intp nelem = cphvb_nelements(vem_base->ndim, vem_base->shape);
                assert(bridge_base->data != vem_base->data); 
                memcpy(bridge_base->data, vem_base->data, nelem * cphvb_type_size(vem_base->type));
            }
            if(bridge->opcode == CPHVB_USERFUNC)
                bridge->userfunc->operand[j] = bridge_op;
            else
                bridge->operand[j] = bridge_op;
        }
    }
}

void exchange_inst_discard(cphvb_array *vem_ary)
{
    cphvb_array *bridge_ary = vem2bridge[(cphvb_intp)vem_ary];
    if(bridge_ary != NULL)
    {
        bridge2vem.erase((cphvb_intp)bridge_ary);
        vem2bridge.erase((cphvb_intp)vem_ary);
        ary_store.erase(vem_ary);
    }
}



