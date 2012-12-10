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

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <cphvb.h>

#include "cphvb_vem_cluster.h"
#include "exchange.h"
#include "mapping.h"
#include "array.h"
#include "pgrid.h"
#include "dispatch.h"
#include "comm.h"


//Function pointers to the Node VEM.
static cphvb_init vem_init;
static cphvb_execute vem_execute;
static cphvb_shutdown vem_shutdown;
static cphvb_reg_func vem_reg_func;

//The VE components
static cphvb_component **my_components;

//Our self
static cphvb_component *myself;

//Number of user-defined functions registered.
static cphvb_intp userfunc_count = 0;

/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error exec_init(const char *component_name)
{
    cphvb_intp children_count;
    cphvb_error err;
    myself = cphvb_component_setup(component_name);
    if(myself == NULL)
        return CPHVB_ERROR;
    
    
    err = cphvb_component_children(myself, &children_count, &my_components);
    if (children_count != 1) 
    {
		std::cerr << "Unexpected number of child nodes for VEM, must be 1" << std::endl;
		return CPHVB_ERROR;
    }
    
    if (err != CPHVB_SUCCESS)
	    return err;
    
    vem_init = my_components[0]->init;
    vem_execute = my_components[0]->execute;
    vem_shutdown = my_components[0]->shutdown;
    vem_reg_func = my_components[0]->reg_func;

    //Let us initiate the Node VEM.
    if((err = vem_init(my_components[0])) != 0)
        return err;

    return CPHVB_SUCCESS;
}


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error exec_shutdown(void)
{
    cphvb_error err;
    if((err = vem_shutdown()) != CPHVB_SUCCESS)
        return err;
    cphvb_component_free(my_components[0]);//Only got one child.
    vem_init     = NULL;
    vem_execute  = NULL;
    vem_shutdown = NULL;
    vem_reg_func = NULL;
    cphvb_component_free_ptr(my_components);
    my_components = NULL;

    //Finalize the process grid
    if((err = pgrid_finalize()) != CPHVB_SUCCESS)
        return err;

    //Finalize the process grid
    if((err = dispatch_finalize()) != CPHVB_SUCCESS)
        return err;

    return CPHVB_SUCCESS;
}


/* Register a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error exec_reg_func(char *fun, cphvb_intp *id)
{
    cphvb_error e;
    
    if(*id == 0)//Only if parent didn't set the ID.
        *id = ++userfunc_count;

    if((e = vem_reg_func(fun, id)) != CPHVB_SUCCESS)
    {
        *id = 0;
        return e;
    }
    return e;
}


/* Execute one instruction.
 *
 * @opcode The opcode of the instruction
 * @operands  The operands in the instruction
 * @inst_status The returned status of the instruction (output)
 * @return Error codes of vem_execute()
 */
cphvb_error exec_inst(cphvb_opcode opcode, cphvb_array *operands[], 
                      cphvb_error *inst_status)
{
    cphvb_error e;

    cphvb_instruction new_inst;
    new_inst.opcode = opcode;
    new_inst.status = CPHVB_INST_PENDING;
    int nop = cphvb_operands(opcode);
    
    for(int i=0; i<nop; ++i)
        new_inst.operand[i] = operands[i];

    if((e = vem_execute(1, &new_inst)) != CPHVB_SUCCESS)
        *inst_status = new_inst.status;
    else 
        *inst_status = CPHVB_SUCCESS; 
    return e;
}


/* Execute to instruction locally at the master-process
 *
 * @instruction The instructionto execute
 * @return Error codes (CPHVB_SUCCESS)
 */
static cphvb_error fallback_exec(cphvb_instruction *inst)
{
    cphvb_error e, stat;
    int nop = cphvb_operands_in_instruction(inst);

    //Gather all data at the master-process
    cphvb_array **oprands = cphvb_inst_operands(inst);
    for(cphvb_intp o=0; o < nop; ++o)
    {
        cphvb_array *op = oprands[o];
        if(cphvb_is_constant(op))
            continue;

        cphvb_array *base = cphvb_base_array(op);
        if((e = exec_inst(CPHVB_SYNC, &base, &stat)) != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Error in fallback_exec() when executing "
            "CPHVB_SYNC: instruction status: %s\n",cphvb_error_text(stat));
            return e;
        }

        if((e = comm_slaves2master(base)) != CPHVB_SUCCESS)
            return e;
    }
    
    //Do global instruction
    if(pgrid_myrank == 0)
    {
        if((e = vem_execute(1, inst)) != CPHVB_SUCCESS)
            return e;
    }

    //Scatter all data back to all processes
    for(cphvb_intp o=0; o < nop; ++o)
    {
        cphvb_array *op = oprands[o];
        if(cphvb_is_constant(op))
            continue;
        cphvb_array *base = cphvb_base_array(op);
        
        if((e = exec_inst(CPHVB_SYNC, &base, &stat)) != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Error in fallback_exec() when executing "
            "CPHVB_SYNC: instruction status: %s\n",cphvb_error_text(stat));
            return e;
        }
        
        if((e = comm_master2slaves(base)) != CPHVB_SUCCESS)
            return e;

        //TODO: need to discard all bases aswell
        if((e = exec_inst(CPHVB_DISCARD, &op, &stat)) != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Error in fallback_exec() when executing "
            "CPHVB_DISCARD: instruction status: %s\n",cphvb_error_text(stat));
            return e;
        }
    }
    return CPHVB_SUCCESS; 
}


/* Execute a regular computation instruction
 *
 * @instruction The regular computation instruction
 * @return Error codes (CPHVB_SUCCESS)
 */
static cphvb_error execute_regular(cphvb_instruction *inst)
{
    cphvb_error e, stat; 
    std::vector<cphvb_array> chunks;
    std::vector<array_ext> chunks_ext;
    e = mapping_chunks(inst, chunks, chunks_ext);
    if(e != CPHVB_SUCCESS)
    {
        inst->status = CPHVB_INST_PENDING;
        return e;
    }

    assert(chunks.size() > 0);
    int nop = cphvb_operands_in_instruction(inst);
    //Handle one chunk at a time.
    for(std::vector<cphvb_array>::size_type c=0; c < chunks.size();c += nop)
    {
        //The process where the output chunk is located will do the computation.
        int owner_rank = chunks_ext[0+c].rank;

        //Create a local instruction based on the array-chunks
        cphvb_instruction local_inst = *inst;
        for(cphvb_intp k=0; k < nop; ++k)
        {
            if(!cphvb_is_constant(inst->operand[k]))
            {
                cphvb_array *ary = &chunks[k+c];
                array_ext *ary_ext = &chunks_ext[k+c];
                local_inst.operand[k] = ary;
                comm_array_data(ary, ary_ext, owner_rank);
            }
        }

        //Check if we should do the computation
        if(pgrid_myrank != owner_rank)
            continue;

        //Apply the local computation
        local_inst.status = CPHVB_INST_PENDING;
        e = vem_execute(1, &local_inst);
        inst->status = local_inst.status;
        if(e != CPHVB_SUCCESS)
            return e;

        //Sync and discard the local arrays
        for(cphvb_intp k=0; k < nop; ++k)
        {
            if(cphvb_is_constant(inst->operand[k]))
                continue;
            
            cphvb_array *ary = cphvb_base_array(&chunks[k+c]);
            if((e = exec_inst(CPHVB_SYNC, &ary, &stat)) != CPHVB_SUCCESS)
            {
                fprintf(stderr, "Error in execute_regular() when executing "
                "CPHVB_SYNC: instruction status: %s\n",cphvb_error_text(stat));
                return e;
            }

            //TODO: need to discard all bases aswell
            ary = &chunks[k+c];
            if((e = exec_inst(CPHVB_DISCARD, &ary, &stat)) != CPHVB_SUCCESS)
            {
                fprintf(stderr, "Error in execute_regular() when executing "
                "CPHVB_DISCARD: instruction status: %s\n",cphvb_error_text(stat));
                return e;
            }
        }
    }
    return CPHVB_SUCCESS;
}



/* Execute a list of instructions where all operands are global arrays
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error exec_execute(cphvb_intp count, cphvb_instruction inst_list[])
{
    cphvb_error e;
    if(count <= 0)
        return CPHVB_SUCCESS;
    
//    cphvb_pprint_instr_list(inst_list, count, "GLOBAL");

    for(cphvb_intp i=0; i < count; ++i)
    {
        cphvb_instruction* inst = &inst_list[i];
        assert(inst->opcode >= 0);
        switch(inst->opcode) 
        {
            case CPHVB_DISCARD:
            {
                dispatch_slave_known_remove(inst->operand[0]);
                if(inst->operand[0]->base == NULL)
                    array_rm_local(inst->operand[0]); 
                break;
            }
            case CPHVB_USERFUNC:
            {
                if((e = fallback_exec(inst)) != CPHVB_SUCCESS)
                    return e;
                break;
            }
            case CPHVB_FREE:
            {
                cphvb_array *base = cphvb_base_array(inst->operand[0]);
                cphvb_data_free(base);
                cphvb_data_free(array_get_local(base));
                break;
            }
            case CPHVB_NONE:
            {
                break;
            }
            case CPHVB_SYNC:
            {
                if((e = comm_slaves2master(inst->operand[0])) != CPHVB_SUCCESS)
                    return e;
                break;
            }
            default:
            {
                if((e = execute_regular(inst)) != CPHVB_SUCCESS)
                    return e;
            }
        }
    }
    return CPHVB_SUCCESS;
}

/*
    //Local copy of the instruction list
    cphvb_instruction local_inst[count];
    cphvb_error err;
    int NPROC = 3;


    //Exchange the instruction list between all processes and
    //update all operand pointers to local pointers
//    cphvb_pprint_instr_list(inst_list, count, "CLUSTER GLOBAL");
    exchange_inst_bridge2vem(count, inst_list, local_inst);
//    cphvb_pprint_instr_list(local_inst, count, "CLUSTER LOCAL");

    for(cphvb_intp i=0; i < count; ++i)
    {
        cphvb_instruction* inst = &local_inst[i];
        assert(inst->opcode >= 0);
        switch(inst->opcode) 
        {
            case CPHVB_DISCARD:
            {
                cphvb_array *op = inst->operand[0];
                if(cphvb_base_array(op) == op &&//This is a base array
                   op->base != NULL)//and has local allocated memory
                {
                    cphvb_instruction new_inst;
                    new_inst.opcode     = CPHVB_FREE;
                    new_inst.status     = CPHVB_INST_PENDING;
                    new_inst.operand[0] = op;
                    err = vem_execute(1, &new_inst);
                    local_inst[i].status = new_inst.status;
                    inst_list[i].status  = new_inst.status;
                    if(err != CPHVB_SUCCESS)
                        return err; 
                }
                exchange_inst_discard(op);
            }
            case CPHVB_USERFUNC:
            case CPHVB_FREE:
            case CPHVB_NONE:
            {
                err = vem_execute(1, inst);
                local_inst[i].status = inst->status;
                inst_list[i].status = inst->status;
                if(err != CPHVB_SUCCESS)
                    return err; 
                break;
            }
            case CPHVB_SYNC:
            {
                err = vem_execute(1, inst);
                local_inst[i].status = inst->status;
                inst_list[i].status = inst->status;
                if(err != CPHVB_SUCCESS)
                    return err; 
                
                //Update the Bridge's instruction list
                exchange_inst_vem2bridge(1,inst,&inst_list[i]);                
                break;
            }
            default:
            {
                cphvb_instruction new_inst;
                std::vector<cphvb_array> chunks;
                std::vector<array_ext> chunks_ext;
                err = mapping_chunks(NPROC, inst, chunks, chunks_ext);
                if(err != CPHVB_SUCCESS)
                {
                    inst->status = CPHVB_INST_PENDING;
                    return err;
                }

                assert(chunks.size() > 0);
                for(std::vector<cphvb_array>::size_type c=0; c < chunks.size();
                    c += cphvb_operands_in_instruction(inst))
                {
                    new_inst = *inst;
                    for(cphvb_intp k=0; k < cphvb_operands_in_instruction(inst); ++k)
                    {
                        if(!cphvb_is_constant(inst->operand[k]))
                        {
                            cphvb_array *ary = &chunks[k+c];
                            //Only for the local test.
                            {
                                cphvb_intp totalsize=1;
                                for(cphvb_intp d=0; d < cphvb_base_array(ary)->ndim; ++d)
                                    totalsize *= cphvb_base_array(ary)->shape[d];
                                cphvb_intp localsize = totalsize / NPROC;
                                if(localsize == 0)
                                    localsize = 1;
                                ary->start += localsize * chunks_ext[k+c].rank;
                            } 
                            new_inst.operand[k] = ary;
                        }
                    }
                    new_inst.status = CPHVB_INST_PENDING;
                    err = vem_execute(1, &new_inst);
                    local_inst[i].status = new_inst.status;
                    inst_list[i].status = new_inst.status;                            
                    if(err != CPHVB_SUCCESS)
                        return err;
    
                    //Discard the created array-views
                    for(cphvb_intp k=0; k < cphvb_operands_in_instruction(inst); ++k)
                    {
                        if(!cphvb_is_constant(inst->operand[k]))
                        {
                            new_inst.opcode = CPHVB_DISCARD;
                            new_inst.status = CPHVB_INST_PENDING;
                            new_inst.operand[0] = &chunks[k+c];
                            err = vem_execute(1, &new_inst);
                            local_inst[i].status = new_inst.status;
                            inst_list[i].status = new_inst.status;                            
                            if(err != CPHVB_SUCCESS)
                                return err;
                        }
                    }
                }
            }
        }
    }
    return CPHVB_SUCCESS;
*/
