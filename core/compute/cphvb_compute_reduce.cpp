#include <cphvb.h>
#include <cphvb_compute.h>

/**
 * cphvb_compute_reduce
 *
 * Implementation of the user-defined funtion "reduce".
 * Note that we follow the function signature defined by cphvb_userfunc_impl.
 *
 */
cphvb_error cphvb_compute_reduce(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;
    cphvb_instruction inst;
    cphvb_error err;
    cphvb_intp i,j;
    cphvb_index coord[CPHVB_MAXDIM];
    memset(coord, 0, a->operand[1]->ndim * sizeof(cphvb_index));

    cphvb_array *out, *in, tmp; // We need a tmp copy of the arrays.

    if(cphvb_operands(a->opcode) != 3)
    {
        fprintf(stderr, "Reduce only support binary operations.\n");
        exit(-1);
    }

    // Make sure that the array memory is allocated.
    if(cphvb_data_malloc(a->operand[0]) != CPHVB_SUCCESS ||
       cphvb_data_malloc(a->operand[1]) != CPHVB_SUCCESS)
    {
        return CPHVB_OUT_OF_MEMORY;
    }

    //We need a tmp copy of the arrays.
    out     = a->operand[0];
    in      = a->operand[1];

    // WARN: This can create a ndim = 0.
    // it seems to work though...
    tmp         = *in;
    tmp.base    = cphvb_base_array(in);
    cphvb_intp step = in->stride[a->axis];
    tmp.start = 0;
    j=0;
    for(i=0; i<in->ndim; ++i) //Remove the 'axis' dimension from in
        if(i != a->axis)
        {
            tmp.shape[j]    = in->shape[i];
            tmp.stride[j]   = in->stride[i];
            ++j;
        }
    --tmp.ndim;

    //We copy the first element to the output.
    inst.status = CPHVB_INST_UNDONE;
    inst.opcode = CPHVB_IDENTITY;
    inst.operand[0] = out;
    inst.operand[1] = &tmp;
    inst.operand[2] = NULL;

    err = cphvb_compute_apply( &inst );    // execute the instruction...
    if(err != CPHVB_SUCCESS)
        return err;
    tmp.start += step;

    //Reduce over the 'axis' dimension.
    //NB: the first element is already handled.
    inst.status = CPHVB_INST_UNDONE;
    inst.opcode = a->opcode;
    inst.operand[0] = out;
    inst.operand[1] = out;
    inst.operand[2] = &tmp;
    cphvb_intp axis_size = in->shape[a->axis];

    for(i=1; i<axis_size; ++i)
    {
        // One block per thread.
        err = cphvb_compute_apply( &inst );
        if(err != CPHVB_SUCCESS)
            return err;
        tmp.start += step;
    }

    return CPHVB_SUCCESS;
}
