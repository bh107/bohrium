#include <cphvb.h>
#include "get_traverse.hpp"
#include "dispatch.h"
#include <assert.h>

//The data type of cphvb_array extended with temporary info.
typedef struct
{
    CPHVB_ARRAY_HEAD
    //Saved original values
    cphvb_index      org_start;
    cphvb_index      org_shape[CPHVB_MAXDIM];
}dispatch_ary;

//Dispatch the instruction.
cphvb_error dispatch( cphvb_instruction *instr ) {

    traverse_ptr traverser = get_traverse( instr );

    if (traverser == NULL) {
        return CPHVB_ERROR;
    } else {
        return traverser( instr );
    }

}

//Dispatch the bundle of instructions.
cphvb_error dispatch_bundle(cphvb_instruction** inst_bundle,
                            cphvb_intp size,
                            cphvb_intp nblocks)
{
    cphvb_error ret = CPHVB_SUCCESS;
    //Current offset of each instruction.
    cphvb_intp offsets[CPHVB_MAX_NO_INST];
    memset(offsets, 0, size * sizeof(cphvb_intp));

    //Get all traverser function -- one per instruction.
    traverse_ptr traverses[CPHVB_MAX_NO_INST];
    for(cphvb_intp j=0; j<size; ++j)
    {
        traverses[j] = get_traverse( inst_bundle[j] );
        if(traverses[j] == NULL)
        {
            inst_bundle[j]->status = CPHVB_OUT_OF_MEMORY;
            ret = CPHVB_INST_NOT_SUPPORTED;
            goto finish;
        }
    }
    //Save the original array information.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->org_start = ary->start;
            memcpy(ary->org_shape, ary->shape, ary->ndim * sizeof(cphvb_index));
        }
    }
    //Make sure that all array-data is allocated.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            if(cphvb_malloc_array_data(inst->operand[i]) != CPHVB_SUCCESS)
            {
                inst->status = CPHVB_OUT_OF_MEMORY;
                ret = CPHVB_PARTIAL_SUCCESS;
                goto finish;
            }
        }
    }
    //Initiate the first block.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            //We block over the most significant dimension.
            //NB: the first block gets the reminder.
            ary->shape[0] = ary->org_shape[0] / nblocks;
            ary->shape[0] += ary->org_shape[0] % nblocks;
        }
    }
    //Dispatch the first block.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        inst->status = traverses[j](inst);
        if(inst->status != CPHVB_SUCCESS)
        {
            ret = CPHVB_PARTIAL_SUCCESS;
            goto finish;
        }
        //Update the offsets to the next block.
        offsets[j] += inst->operand[0]->shape[0];
    }
    //Iterate to the second block.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->shape[0] = ary->org_shape[0] / nblocks;
            ary->start = ary->org_start + ary->stride[0] * offsets[j];
        }
    }
    //Handle the rest of the blocks.
    for(cphvb_intp b=1; b<nblocks; ++b)
    {
        if(inst_bundle[0]->operand[0]->shape[0] <= 0)
            break;//We a finished.

        //Dispatch a block.
        for(cphvb_intp j=0; j<size; ++j)
        {
            cphvb_instruction *inst = inst_bundle[j];
            assert(inst->operand[0]->shape[0] > 0);
            inst->status = traverses[j](inst);
            if(inst->status != CPHVB_SUCCESS)
            {
                ret = CPHVB_PARTIAL_SUCCESS;
                goto finish;
            }
            //Update the offsets to the next block.
            offsets[j] += inst->operand[0]->shape[0];
        }
        //Iterate to the next block.
        for(cphvb_intp j=0; j<size; ++j)
        {
            cphvb_instruction *inst = inst_bundle[j];
            for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
            {
                dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
                ary->start = ary->org_start + ary->stride[0] * offsets[j];
            }
        }
    }

finish:
    //Restore the original arrays.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->start = ary->org_start;
            memcpy(ary->shape, ary->org_shape, ary->ndim * sizeof(cphvb_index));
        }
    }

    return ret;
}
